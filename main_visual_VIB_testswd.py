# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import time
from networks.mlp import SimpleMLP
from networks.pgl_sum.pgl_sum import PGL_SUM
from networks.vasnet.vasnet import VASNet
from networks.sl_module.SWVI_test import *
from networks.graph_fusion import graph_fusion
from model.utils.evaluation_metrics import evaluate_summary
from model.utils.generate_summary import generate_summary
from model.utils.evaluate_map import generate_mrhisum_seg_scores, top50_summary, top15_summary
from networks.atfuse.ATFuse import FactorAtt_ConvRelPosEnc, MHCABlock, UpScale
from networks.CrossAttentional.cam import CAM
from networks.sl_module.BottleneckTransformer import BottleneckTransformer
import os
import torch
import argparse
import gc
from model.configs import Config, str2bool
from torch.utils.data import DataLoader
from model.mrhisum_dataset_fixed import MrHiSumDataset, BatchCollator


# 添加新的时间感知距离计算类
class TimeAwareCost(nn.Module):
    """时间感知成本函数：量化序列间的几何差异"""
    def __init__(self, w1=0.4, w2=0.4, w3=0.2):
        super().__init__()
        self.w = [w1, w2, w3]

    def soft_dtw(self, x, y, gamma=0.1):
        """改进版软DTW实现，确保正值输出（支持自动微分）"""
        n, m = x.shape[0], y.shape[0]

        x_norm = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        y_norm = y / (torch.norm(y, dim=1, keepdim=True) + 1e-8)

        if n == 1 and m == 1:
            D = torch.norm(x_norm - y_norm, p=2).unsqueeze(0).unsqueeze(0)
        elif n == 1:
            D = torch.norm(x_norm.unsqueeze(0) - y_norm, p=2, dim=1).unsqueeze(0)
        elif m == 1:
            D = torch.norm(x_norm - y_norm.unsqueeze(0), p=2, dim=1).unsqueeze(1)
        else:
            D = torch.cdist(x_norm, y_norm, p=2)

        D = D / (D.max() + 1e-8)

        R = torch.zeros(n+1, m+1, device=x.device) + float('inf')
        R[0, 0] = 0

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = D[i-1, j-1]
                min_val = torch.min(torch.min(R[i-1, j], R[i, j-1]), R[i-1, j-1])
                R[i, j] = cost + min_val

        return torch.abs(R[-1, -1])

    def autocorr(self, z):
        """计算序列的自相关函数（ACF）"""
        z_centered = z - torch.mean(z, dim=0)
        acf = torch.stack([
            torch.sum(z_centered[:-lag] * z_centered[lag:], dim=0)
            for lag in range(1, 6)
        ])
        return acf / torch.max(torch.abs(acf), dim=0)[0]

    def forward(self, x_i, x_j, z_i, z_j):
        """计算两个序列对的Wasserstein传输成本"""
        is_same_sample = torch.all(x_i == x_j) and torch.all(z_i == z_j)
        if is_same_sample:
            return torch.tensor(0.0, device=x_i.device)

        d_time = self.soft_dtw(x_i, x_j)
        d_latent = torch.norm(z_i - z_j, p=2) / z_i.shape[0]
        acf_i = self.autocorr(z_i)
        acf_j = self.autocorr(z_j)
        d_acf = torch.norm(acf_i - acf_j, p=1)

        return (self.w[0] * d_time +
                self.w[1] * d_latent +
                self.w[2] * d_acf)

def sinkhorn_cost(X, Z, cost_fn, epsilon=0.1, n_iters=50):
    """Sinkhorn 算法计算 Wasserstein 距离"""
    batch_size = X.shape[0]
    device = X.device

    def log_sum_exp(x, axis=-1):
        max_x = torch.max(x, axis=axis, keepdim=True)[0]
        return max_x + torch.log(torch.sum(
            torch.exp(x - max_x), axis=axis, keepdim=True)+1e-8)

    C = torch.zeros(batch_size, batch_size, device=device)
    for i in range(batch_size):
        for j in range(batch_size):
            C[i, j] = cost_fn(X[i], X[j], Z[i], Z[j])

    u = torch.zeros(batch_size, device=device)
    v = torch.zeros(batch_size, device=device)

    K = torch.exp(-C / epsilon)
    for _ in range(n_iters):
        u = epsilon * (torch.log((torch.ones(batch_size, device=device) / batch_size) + 1e-8) -
                       torch.log((torch.sum(K * v.exp().unsqueeze(0), dim=1))) + 1e-8)
        v = epsilon * (torch.log((torch.ones(batch_size, device=device) / batch_size) + 1e-8) -
                       torch.log((torch.sum(K * u.exp().unsqueeze(1), dim=0))) + 1e-8)

    P = torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - C) / epsilon)
    W = torch.sum(P * C)

    return W

class Solver(object):
    def __init__(self, config=None, train_loader=None, val_loader=None, test_loader=None, device=None, modal=None):

        self.model, self.optimizer, self.writer, self.scheduler = None, None, None, None
        self.device = device
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.modal = modal
        self.global_step = 0

        self.criterion = nn.MSELoss(reduction='none').to(self.device)

    def build(self):
        """ Define your own summarization model here """
        # Model creation
        cuda_device = self.device
        if self.config.type == 'base':
            self.model = SL_module(input_dim=1024, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5)
        elif self.config.type == 'ib':
            self.model = SL_module_SWD(input_dim=1024, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5)
        self.model.to(cuda_device)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.l2_reg)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        # 修改优化器设置，分离变分参数和其他参数
        ib_params = []
        other_params = []

        # 检查模型中的所有参数
        for name, param in self.model.named_parameters():
            if 'post_z_mu' in name or 'post_z_logD' in name:
                print(f"VIB parameter detected: {name}")
                ib_params.append(param)
            else:
                other_params.append(param)

        # 为变分参数使用更小的学习率（原始学习率的1/10）
        ib_lr = self.config.lr
        print(f"使用学习率: 一般参数 {self.config.lr}, VIB参数 {ib_lr}")

        self.optimizer = optim.SGD([
            {'params': ib_params, 'lr': ib_lr, 'weight_decay': self.config.l2_reg},  # 变分参数
            {'params': other_params, 'lr': self.config.lr, 'weight_decay': self.config.l2_reg}  # 其他参数
        ], momentum=0.9)
        # self.optimizer = optim.Adam([
        #     {'params': ib_params, 'lr': ib_lr, 'weight_decay': self.config.l2_reg},  # 变分参数
        #     {'params': other_params, 'lr': self.config.lr, 'weight_decay': self.config.l2_reg}  # 其他参数
        # ], betas=(0.9, 0.999), eps=1e-8)

        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        # 使用ReduceLROnPlateau调度器根据验证损失自动调整学习率
        # 当验证损失在patience个epoch没有改善时，将学习率乘以factor
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=10,
            verbose=True, threshold=0.0001, threshold_mode='rel',
            cooldown=0, min_lr=1e-6, eps=1e-08
        )

    def train(self):
        path50 = []
        path100 = []
        path150 = []
        path200 = []
        path = []
        # proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 每次使用 10%, 20%, ..., 100%
        proportion = 1.0
        model = self.model
        cuda_device = self.device
        # for proportion in proportions:
        # time.sleep(0.5)
        best_f1score = -1.0
        best_map50 = -1.0
        best_map15 = -1.0
        best_map = -1.0
        best_pre = -1.0
        best_f1score_epoch = 0
        best_map50_epoch = 0
        best_map15_epoch = 0
        best_map_epoch = 0
        best_pre_epoch = 0
        # print(f"Training with {int(proportion * 100)}% of the training data...")

        for epoch_i in range(self.config.epochs):
            # time.sleep(0.5)
            print("[Epoch: {0:6}]".format(str(epoch_i + 1) + "/" + str(self.config.epochs)))
            model.train()
            loss_history = []
            kl_loss_history = []
            loss_v_history = []
            loss_a_history = []
            num_batches = int(len(self.train_loader))
            iterator = iter(self.train_loader)

            for _ in tqdm(range(num_batches)):

                self.optimizer.zero_grad()
                time.sleep(0.2)
                data = next(iterator)
                # 'video_name' : video_name, 'features' : frame_feat_visual, 'audio':frame_feat_audio, 'gtscore':gtscore, 'mask':mask_visual, 'mask_audio':mask_audio
                visual = data['features'].to(cuda_device)
                gtscore = data['gtscore'].to(cuda_device)
                audio = data['audio'].to(cuda_device)
                mask = data['mask'].to(cuda_device)

                # 一般---------------------------
                if self.config.type == "base":
                    score, weights = model(visual, mask)
                    loss = self.criterion(score[mask], gtscore[mask]).mean()
                    loss.backward()
                    loss_history.append(loss.detach().item())
                    self.optimizer.step()
                # IB+KL-----------------------------------------------------
                # **调用 SL_module**
                # print(input_feature)
                elif self.config.type == "ib":
                    score, kl_loss = model(visual, mask)

                    # 计算预测损失 (主任务损失)
                    prediction_loss = self.criterion(score[mask], gtscore[mask]).mean()

                    # 确保KL损失有合理的值（避免inf/nan但不裁剪范围）
                    if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                        print(
                            f"WARNING: KL loss is {kl_loss.item() if hasattr(kl_loss, 'item') else kl_loss}, replacing with 0")
                        kl_loss = torch.tensor(0.0, device=score.device)

                    # 分开考虑KL损失和预测损失（类似VIBNet）
                    beta = self.config.beta  # KL 损失的权重系数

                    # 确保记录正确的损失值（即使条件为0）
                    if beta > 0:
                        total_loss = prediction_loss + beta * kl_loss
                    else:
                        total_loss = prediction_loss

                    # 计算梯度并更新
                    # self.optimizer.zero_grad()
                    total_loss.backward()

                    # 记录损失
                    pred_loss_val = prediction_loss.detach().item()
                    kl_loss_val = kl_loss.detach().item()
                    loss_v_history.append(pred_loss_val)
                    kl_loss_history.append(kl_loss_val)
                    loss_history.append(total_loss.detach().item())

                    # 如果KL损失组件为0或很小，打印警告
                    if kl_loss_val < 1e-8 and beta > 0:
                        print(f"WARNING: KL loss is nearly zero: {kl_loss_val}")

                    # 更新模型参数
                    self.optimizer.step()

                # time.sleep(1.5)
            if not loss_history == []:
                loss = np.mean(np.array(loss_history))
            else:
                loss = 0
            if not kl_loss_history == []:
                kl_loss = np.mean(np.array(kl_loss_history))
            else:
                kl_loss = 0
            if not loss_v_history == []:
                v_loss = np.mean(np.array(loss_v_history))
            else:
                v_loss = 0
            if not loss_a_history == []:
                a_loss = np.mean(np.array(loss_a_history))
            else:
                a_loss = 0
            val_recon_loss = 0
            val_kl_loss = 0
            val_loss = 0
            val_f1score = 0
            val_map50 = 0
            val_map15 = 0
            val_map = 0
            val_precision = 0
            if self.config.type == "base":
                val_f1score, val_map50, val_map15, val_loss, val_precision, val_map = self.evaluate(
                    dataloader=self.val_loader)
            elif self.config.type == "ib":
                val_f1score, val_map50, val_map15, val_loss, val_precision, val_loss_v, val_loss_a, val_kl_loss, val_map = self.evaluate(
                    dataloader=self.val_loader)
            # final_f_score, final_map50, final_map15, final_loss, final_precision, final_prediction_loss_v, final_prediction_loss_a, final_kl_loss

            # 保存每次比例的日志
            proportion_dir = os.path.join(self.config.save_dir_root, f'logs/proportion_{int(proportion * 100)}')
            os.makedirs(proportion_dir, exist_ok=True)
            #
            f = open(os.path.join(proportion_dir, 'results.txt'), 'a')
            print(f"proportion: {proportion}, type: {type(proportion)}")
            print(f"epoch_i: {epoch_i}, type: {type(epoch_i)}")

            f.write(f'[Proportion {str(proportion * 100)}% | Epoch {str(epoch_i + 1)}], \n'
                    f'Train Loss: {loss:.5f}, Val Loss: {val_loss:.5f}, \n'
                    f'Visual loss: {v_loss:.5f}, Audio loss: {a_loss:.5f}\n'
                    f'Val F1 Score: {val_f1score:.5f}, Val MAP50: {val_map50:.5f}, \n'
                    f'Val MAP15: {val_map15:.5f}, Val MAP: {val_map:.5f}, KL loss: {kl_loss:.5f}\n')
            f.flush()
            f.close()
            f = open(os.path.join(self.config.save_dir_root, 'logs/loss.txt'), 'a')
            f.write(f'Epoch: {epoch_i + 1}, loss: {loss}, val_loss: {val_loss}\n')
            f.flush()
            f.close()
            state_dict = model.state_dict()

            if best_f1score <= val_f1score:
                best_f1score = val_f1score
                best_f1score_epoch = epoch_i
                f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, f'best_f1.pkl')
                if os.path.exists(f1_save_ckpt_path):
                    os.remove(f1_save_ckpt_path)
                torch.save(state_dict, f1_save_ckpt_path)
                if f1_save_ckpt_path not in path:
                    path.append(f1_save_ckpt_path)

            if best_map50 <= val_map50:
                best_map50 = val_map50
                best_map50_epoch = epoch_i
                map50_save_ckpt_path = os.path.join(self.config.best_map50_save_dir, f'best_map50.pkl')
                if os.path.exists(map50_save_ckpt_path):
                    os.remove(map50_save_ckpt_path)
                torch.save(state_dict, map50_save_ckpt_path)
                if map50_save_ckpt_path not in path:
                    path.append(map50_save_ckpt_path)

            if best_map15 <= val_map15:
                best_map15 = val_map15
                best_map15_epoch = epoch_i
                map15_save_ckpt_path = os.path.join(self.config.best_map15_save_dir, f'best_map15.pkl')
                if os.path.exists(map15_save_ckpt_path):
                    os.remove(map15_save_ckpt_path)
                torch.save(state_dict, map15_save_ckpt_path)
                if map15_save_ckpt_path not in path:
                    path.append(map15_save_ckpt_path)

            if best_map <= val_map:
                best_map = val_map
                best_map_epoch = epoch_i
                best_map_ckpt_path = os.path.join(self.config.best_map_save_dir,
                                                  f'Proportion_{int(proportion * 100)}%_best_map.pkl')
                if os.path.exists(best_map_ckpt_path):
                    os.remove(best_map_ckpt_path)
                torch.save(state_dict, best_map_ckpt_path)
                if best_map_ckpt_path not in path:
                    path.append(best_map_ckpt_path)

            # 调用学习率调度器，传入验证损失作为监控指标
            # 当验证损失不再下降时，ReduceLROnPlateau会自动降低学习率
            self.scheduler.step(val_loss)

            # print("   [Epoch {0}] Train loss: {1:.05f}".format(epoch_i+1, loss))
            # print('    VAL  F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5}'.format(val_f1score, val_map50, val_map15))
            print(f'[Proportion {str(proportion * 100)}% | Epoch {str(epoch_i + 1)}], \n'
                  f'Train Loss: {loss:.5f}, Val Loss: {val_loss:.5f}, \n'
                  f'Visual loss: {v_loss:.5f}, Audio loss: {a_loss:.5f}\n'
                  f'Val F1 Score: {val_f1score:.5f}, Val MAP50: {val_map50:.5f}, \n'
                  f'Val MAP15: {val_map15:.5f}, Val MAP: {val_map:.5f}, KL loss: {kl_loss:.5f}\n')
            f = open(os.path.join(self.config.save_dir_root, 'logs/all_result.txt'), 'a')
            f.write(
                f'Val_f1score: {val_f1score}, Val_map50: {val_map50}, Val_map15: {val_map15}, Val_precision: {val_precision}\n')
            f.flush()
            f.close()
            # del data, visual, gtscore, audio, mask, score, weights, loss
            gc.collect()
        print(f'  [Proportion {int(proportion * 100)}%]')
        print('   Best Val F1 score {0:0.5} @ epoch{1}'.format(best_f1score, best_f1score_epoch + 1))
        print('   Best Val MAP-50   {0:0.5} @ epoch{1}'.format(best_map50, best_map50_epoch + 1))
        print('   Best Val MAP-15   {0:0.5} @ epoch{1}'.format(best_map15, best_map15_epoch + 1))
        print('   Best Val MAP   {0:0.5} @ epoch{1}'.format(best_map, best_map_epoch + 1))
        print('   Best Val PRECISION   {0:0.5} @ epoch{1}'.format(best_pre, best_pre_epoch + 1))

        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write(f'    [Proportion {int(proportion * 100)}%]\n')
        f.write('   Best Val F1 score {0:0.5} @ epoch{1}\n'.format(best_f1score, best_f1score_epoch + 1))
        f.write('   Best Val MAP-50   {0:0.5} @ epoch{1}\n'.format(best_map50, best_map50_epoch + 1))
        f.write('   Best Val MAP-15   {0:0.5} @ epoch{1}\n'.format(best_map15, best_map15_epoch + 1))
        f.write('   Best Val MAP   {0:0.5} @ epoch{1}\n'.format(best_map, best_map_epoch + 1))
        f.write('   Best Val PRECISION   {0:0.5} @ epoch{1}\n\n'.format(best_pre, best_pre_epoch + 1))
        f.flush()
        f.close()
        torch.cuda.empty_cache()
        return path, path50, path100, path150, path200
        # return f1_save_ckpt_path, map50_save_ckpt_path, map15_save_ckpt_path

    def evaluate(self, dataloader=None):
        """ Saves the frame's importance scores for the test videos in json format.

        :param int epoch_i: The current training epoch.
        """
        model = self.model
        cuda_device = self.device
        model.eval()
        loss_history = []
        kl_loss_history = []
        loss_v_history = []
        loss_a_history = []
        fscore_history = []
        map50_history = []
        map15_history = []
        map_history = []
        precision_history = []

        dataloader = iter(dataloader)

        for data in dataloader:
            visual = data['features'].to(cuda_device)
            gtscore = data['gtscore'].to(cuda_device)
            audio = data['audio'].to(cuda_device)
            input_mask = 'mask'
            # multi_feature = data['multi'].to(cuda_device)
            # 一般-------------------------------------
            if self.config.type == "base":
                input_feature = visual
                if len(input_feature.shape) == 2:
                    seq = seq.unsqueeze(0)
                elif len(input_feature.shape) == 4:
                    input_feature = input_feature.squeeze(0)
                if len(gtscore.shape) == 1:
                    gtscore = gtscore.unsqueeze(0)

                B = input_feature.shape[0]
                mask = None
                if input_mask in data:
                    mask = data[input_mask].to(cuda_device)
                with torch.no_grad():
                    score, weights = model(input_feature, mask)
                # print(f"multi, score, gtscore, mask_multi={multi_feature.shape, score.shape, gtscore.shape, mask_multi.shape}")
                loss = self.criterion(score[mask], gtscore[mask]).mean()
                torch.cuda.synchronize()
                # loss.backward()
                loss_history.append(loss.detach().cpu().item())

            # -------------------------------------------

            # IB------------------------------------------------------------
            elif self.config.type == "ib":
                input_feature = visual
                if len(input_feature.shape) == 2:
                    seq = seq.unsqueeze(0)
                elif len(input_feature.shape) == 4:
                    input_feature = input_feature.squeeze(0)
                if len(gtscore.shape) == 1:
                    gtscore = gtscore.unsqueeze(0)

                B = input_feature.shape[0]
                mask = None
                if input_mask in data:
                    mask = data[input_mask].to(cuda_device)
                with torch.no_grad():
                    score, kl_loss = model(input_feature, mask)

                    # 计算预测损失
                    prediction_loss = self.criterion(score[mask], gtscore[mask]).mean()

                    # 处理可能的异常KL损失值
                    if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                        print(
                            f"WARNING in evaluation: KL loss is {kl_loss.item() if hasattr(kl_loss, 'item') else kl_loss}, replacing with 0")
                        kl_loss = torch.tensor(0.0, device=score.device)

                    # 分开考虑KL损失和预测损失
                    beta = self.config.beta  # KL损失权重系数

                    # 计算总损失
                    if beta > 0:
                        total_loss = prediction_loss + beta * kl_loss
                    else:
                        total_loss = prediction_loss

                    # torch.cuda.synchronize()

                    # 记录损失值
                    pred_loss_val = prediction_loss.detach().cpu().item()
                    kl_loss_val = kl_loss.detach().cpu().item()

                    loss_history.append(total_loss.detach().cpu().item())
                    kl_loss_history.append(kl_loss_val)

                    # 检测KL损失是否接近零
                    if kl_loss_val < 1e-8 and beta > 0:
                        print(f"WARNING in evaluation: KL loss is nearly zero: {kl_loss_val}")
                # self.optimizer.step()

            loss = np.mean(np.array(loss_history))

            # Calculate precision
            predictions = (score > 0.5).float()  # Example threshold, modify as needed
            true_positives = (predictions[mask] * gtscore[mask]).sum().detach().item()
            predicted_positives = predictions[mask].sum().detach().item()
            precision = true_positives / (predicted_positives + 1e-7)  # Avoid division by zero
            precision_history.append(precision)

            # Summarization metric
            score = score.squeeze().cpu()
            gt_summary = data['gt_summary'][0]
            cps = data['change_points'][0]
            n_frames = data['n_frames']
            nfps = data['n_frame_per_seg'][0].tolist()
            picks = data['picks'][0].numpy()
            # print("score",len(score), "cps",len(cps), "nframe",len(n_frames), "nfps",len(nfps), "picks",len(picks))
            machine_summary = generate_summary(score, cps, n_frames, nfps, picks)
            # print("MACHINE", machine_summary, machine_summary.shape)
            # print("GT SUMMARY",gt_summary, gt_summary.shape)
            try:
                f_score, kTau, sRho = evaluate_summary(machine_summary, gt_summary, eval_method='avg')
            except:
                machine_summary = np.delete(machine_summary, -1)
                f_score, kTau, sRho = evaluate_summary(machine_summary, gt_summary, eval_method='avg')
            fscore_history.append(f_score)

            # Highlight Detection Metric
            gt_seg_score = generate_mrhisum_seg_scores(gtscore.squeeze(0), uniform_clip=5)
            gt_top50_summary = top50_summary(gt_seg_score)
            gt_top15_summary = top15_summary(gt_seg_score)

            highlight_seg_machine_score = generate_mrhisum_seg_scores(score, uniform_clip=5)
            highlight_seg_machine_score = torch.exp(highlight_seg_machine_score) / (
                        torch.exp(highlight_seg_machine_score).sum() + 1e-7)

            clone_machine_summary = highlight_seg_machine_score.clone().detach().cpu()
            clone_machine_summary = clone_machine_summary.numpy()

            # 获取gt_seg_score的numpy格式用于计算
            gt_seg_score_np = gt_seg_score.clone().detach().cpu().numpy()

            # 归一化真实标签分数（如果尚未归一化）
            gt_scores_normalized = gt_seg_score_np / np.max(gt_seg_score_np) if np.max(
                gt_seg_score_np) > 0 else gt_seg_score_np

            # 使用一系列阈值将连续标签转换为二元标签
            thresholds = np.arange(0.05, 1.0, 0.05)
            ap_values = []

            for threshold in thresholds:
                # 将连续真实标签转换为二元标签
                binary_gt = (gt_scores_normalized >= threshold).astype(int)

                # 计算当前阈值下的AP
                ap_at_threshold = average_precision_score(binary_gt, clone_machine_summary)
                ap_values.append(ap_at_threshold)

            # 计算mAP（所有阈值下AP的平均值）
            mAP = np.mean(ap_values) if ap_values else 0
            aP50 = average_precision_score(gt_top50_summary, clone_machine_summary)
            aP15 = average_precision_score(gt_top15_summary, clone_machine_summary)
            map_history.append(mAP)
            map50_history.append(aP50)
            map15_history.append(aP15)

        final_f_score = np.mean(fscore_history)
        final_map50 = np.mean(map50_history)
        final_map15 = np.mean(map15_history)
        final_map = np.mean(map_history)
        final_precision = np.mean(precision_history)
        final_loss = np.mean(loss_history)
        if not loss_v_history == []:
            final_prediction_loss_v = np.mean(loss_v_history)
        else:
            final_prediction_loss_v = 0
        if not loss_a_history == []:
            final_prediction_loss_a = np.mean(loss_a_history)
        else:
            final_prediction_loss_a = 0
        if not kl_loss_history == []:
            final_kl_loss = np.mean(kl_loss_history)
        else:
            final_kl_loss = 0
        if self.config.type == "base":
            return final_f_score, final_map50, final_map15, final_loss, final_precision, final_map
        elif self.config.type == "ib" or self.config.type == "cib" or self.config.type == "eib" or self.config.type == "lib":
            return final_f_score, final_map50, final_map15, final_loss, final_precision, final_prediction_loss_v, final_prediction_loss_a, final_kl_loss, final_map

    def test(self, ckpt_path):
        model = self.model
        cuda_device = self.device
        if ckpt_path != None:
            print("Testing Model: ", ckpt_path)
            print("Device: ", cuda_device)
            model.load_state_dict(torch.load(ckpt_path))
        if self.config.type == "base":
            test_fscore, test_map50, test_map15, _, _, test_map = self.evaluate(dataloader=self.test_loader)
        elif self.config.type == "ib" or self.config.type == "cib" or self.config.type == "eib" or self.config.type == "lib":
            test_fscore, test_map50, test_map15, _, _, _, _, _, test_map = self.evaluate(dataloader=self.test_loader)
        print("------------------------------------------------------")
        print(f"   TEST RESULT on {ckpt_path}: ")
        print('   TEST MRHiSum F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5} | MAP {3:0.5}'.format(test_fscore,
                                                                                                     test_map50,
                                                                                                     test_map15,
                                                                                                     test_map))
        print("------------------------------------------------------")

        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write("Testing on Model " + ckpt_path + '\n')
        f.write('Test F-score ' + str(test_fscore) + '\n')
        f.write('Test MAP50   ' + str(test_map50) + '\n')
        f.write('Test MAP15   ' + str(test_map15) + '\n')
        f.write('Test MAP     ' + str(test_map) + '\n\n')
        f.flush()

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """ Initialize 'net' network weights, based on the chosen 'init_type' and 'init_gain'.

        :param nn.Module net: Network to be initialized.
        :param str init_type: Name of initialization method: normal | xavier | kaiming | orthogonal.
        :param float init_gain: Scaling factor for normal.
        """
        for name, param in net.named_parameters():
            if 'weight' in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))  # ReLU activation function
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))  # ReLU activation function
                else:
                    raise NotImplementedError(f"initialization method {init_type} is not implemented.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SL_module', help='the name of the model')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='the learning rate')
    parser.add_argument('--l2_reg', type=float, default=5e-5, help='l2 regularizer')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='the dropout ratio')
    parser.add_argument('--batch_size', type=int, default=64, help='the batch size')
    parser.add_argument('--tag', type=str, default='test_swd_beta1e4_lr0.05', help='A tag for experiments')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='checkpoint path for inference or weight initialization')
    parser.add_argument('--train', type=str2bool, default='true', help='when use Train')
    parser.add_argument('--path', type=str, default='dataset/mr_hisum_split.json', help='path')
    parser.add_argument('--device', type=str, default='4', help='gpu')
    parser.add_argument('--modal', type=str, default='visual', help='visual,audio,multi')
    parser.add_argument('--beta', type=float, default=0.0001, help='beta')
    parser.add_argument('--type', type=str, default='ib', help='base,ib,cib,eib,lib')  # cib,eib,lib

    opt = parser.parse_args()
    # print(type(opt))
    # print(opt)
    kwargs = vars(opt)
    config = Config(**kwargs)
    # print(config)
    # print(type(config))
    # os._exit()
    train_dataset = MrHiSumDataset(mode='train', path=config.path)
    val_dataset = MrHiSumDataset(mode='val', path=config.path)
    test_dataset = MrHiSumDataset(mode='test', path=config.path)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4,
                              collate_fn=BatchCollator())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    device = torch.device("cuda:" + config.device)
    print("Device being used:", device)
    solver = Solver(config, train_loader, val_loader, test_loader, device, config.modal)

    solver.build()
    test_model_ckpt_path = None
    # proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if config.train:
        # best_path, best_f1_ckpt_path, best_map50_ckpt_path, best_map15_ckpt_path = solver.train()
        best_path, best_path50, best_path100, best_path150, best_path200, = solver.train()
        # print(best_path, best_path50, best_path100)
        for eachpath in best_path:
            solver.test(eachpath)
        for eachpath in best_path50:
            solver.test(eachpath)
        for eachpath in best_path100:
            solver.test(eachpath)
        for eachpath in best_path150:
            solver.test(eachpath)
        for eachpath in best_path200:
            solver.test(eachpath)
        # solver.test(best_f1_ckpt_path)
        # solver.test(best_map50_ckpt_path)
        # solver.test(best_map15_ckpt_path)
        # solver.test(best_pre_ckpt_path)
        print(best_path, best_path50, best_path100)
    else:
        test_model_ckpt_path = config.ckpt_path
        if test_model_ckpt_path == None:
            print("Trained model checkpoint requried. Exit program")
            exit()
        else:
            solver.test(test_model_ckpt_path)
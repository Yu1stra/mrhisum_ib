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
from networks.sl_module.sl_module import *
from networks.graph_fusion import graph_fusion
from model.utils.evaluation_metrics import evaluate_summary
from model.utils.generate_summary import generate_summary
from model.utils.evaluate_map import generate_mrhisum_seg_scores, top50_summary, top15_summary, top_summary
from networks.atfuse.ATFuse import FactorAtt_ConvRelPosEnc, MHCABlock, UpScale 
from networks.CrossAttentional.cam import CAM
from networks.sl_module.BottleneckTransformer import BottleneckTransformer
from model.mrhisum_dataset import MrHiSumDataset, BatchCollator
import gc
import tracemalloc
import objgraph
import sys
import os
import torch
import argparse
import gc
from model.configs import Config, str2bool
from torch.utils.data import DataLoader

def compute_hit_at_10(score, gtscore):
    """
    計算 Hit@10（適用於單一標註者）。
    
    score: (batch_size, num_clips) 預測的亮點分數
    gtscore: (batch_size, num_clips) Ground Truth 標註
    """
    # 取得預測最高的 Top-10 片段索引
    pred_top10 = torch.argsort(score, dim=-1, descending=True)[:, :10]  # (batch_size, 10)

    # 取得 Ground Truth 最高分的片段索引
    gt_top1 = torch.argmax(gtscore, dim=-1, keepdim=True)  # (batch_size, 1)

    # 計算 Hit@10 命中率：如果 GT 片段索引在預測的 Top-10 內，則算命中
    hit10 = (gt_top1 == pred_top10).any(dim=-1).float().mean().item()

    return hit10

def evaluate(model, criterion, dataloader=None):
    model.eval()
    loss_history = []
    fscore_history = []
    map50_history = []
    map15_history = []
    map_history = []
    precision_history = []
    hit1_history = []  # 新增 Hit@1

    dataloader = iter(dataloader)
    for data in dataloader:
        visual = data['features'].to("cpu")
        gtscore = data['gtscore'].to("cpu")
        input_feature = visual

        if len(input_feature.shape) == 2:
            input_feature = input_feature.unsqueeze(0)
        elif len(input_feature.shape) == 4:
            input_feature = input_feature.squeeze(0)
        if len(gtscore.shape) == 1:
            gtscore = gtscore.unsqueeze(0)

        mask = data.get('mask', None)
        if mask is not None:
            mask = mask.to("cpu")

        with torch.no_grad():
            score, weights = model(input_feature, mask)
        loss = criterion(score[mask], gtscore[mask]).mean()
        loss_history.append(loss.item())

        # Precision 計算
        predictions = (score > 0.5).float()
        true_positives = (predictions[mask] * gtscore[mask]).sum().item()
        predicted_positives = predictions[mask].sum().item()
        precision = true_positives / (predicted_positives + 1e-7)
        precision_history.append(precision)

        # **Hit@1 計算**
        pred_top1 = torch.argmax(score, dim=-1)
        gt_top1 = torch.argmax(gtscore, dim=-1)
        hit1 = (pred_top1 == gt_top1).float().mean().item()
        hit1_history.append(hit1)

        final_hit10=compute_hit_at_10(score, gtscore)

        # F-score 計算
        score = score.squeeze().cpu()
        gt_summary = data['gt_summary'][0]
        cps = data['change_points'][0]
        n_frames = data['n_frames']
        nfps = data['n_frame_per_seg'][0].tolist()
        picks = data['picks'][0].numpy()
        machine_summary = generate_summary(score, cps, n_frames, nfps, picks)

        try:
            f_score, kTau, sRho = evaluate_summary(machine_summary, gt_summary, eval_method='avg')
        except:
            machine_summary = np.delete(machine_summary, -1)
            f_score, kTau, sRho = evaluate_summary(machine_summary, gt_summary, eval_method='avg')
        fscore_history.append(f_score)

        # MAP@50 和 MAP@15 計算
        gt_seg_score = generate_mrhisum_seg_scores(gtscore.squeeze(0), uniform_clip=5)
        gt_top50_summary = top50_summary(gt_seg_score)
        gt_top15_summary = top15_summary(gt_seg_score)
        gt_top_summary = top_summary(gt_seg_score)

        highlight_seg_machine_score = generate_mrhisum_seg_scores(score, uniform_clip=5)
        highlight_seg_machine_score = torch.exp(highlight_seg_machine_score) / (torch.exp(highlight_seg_machine_score).sum() + 1e-7)

        clone_machine_summary = highlight_seg_machine_score.clone().detach().cpu().numpy()
        aP50 = average_precision_score(gt_top50_summary, clone_machine_summary)
        aP15 = average_precision_score(gt_top15_summary, clone_machine_summary)
        aP = average_precision_score(gt_top_summary, clone_machine_summary)

        map50_history.append(aP50)
        map15_history.append(aP15)
        map_history.append(aP)

    # 計算最終的平均指標
    final_f_score = np.mean(fscore_history)
    final_map50 = np.mean(map50_history)
    final_map15 = np.mean(map15_history)
    final_map = np.mean(map_history)
    final_precision = np.mean(precision_history)
    final_loss = np.mean(loss_history)
    final_hit1 = np.mean(hit1_history)  # **Hit@1**

    return final_f_score, final_map50, final_map15, final_map, final_loss, final_precision, final_hit1, final_hit10




def test(self, ckpt_path):
        model=self.model
        cuda_device=self.device
        if ckpt_path != None:
            print("Testing Model: ", ckpt_path)
            print("Device: ",  cuda_device)
            model.load_state_dict(torch.load(ckpt_path))
        if self.config.type=="base":
            test_fscore, test_map50, test_map15, _, _, = self.evaluate(dataloader=self.test_loader)
        elif self.config.type=="ib" or self.config.type=="cib" or self.config.type=="eib" or self.config.type=="lib":
            test_fscore, test_map50, test_map15, _, _, _, _, _ = self.evaluate(dataloader=self.test_loader)

        print("------------------------------------------------------")
        print(f"   TEST RESULT on {ckpt_path}: ")
        print('   TEST MRHiSum F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5}'.format(test_fscore, test_map50, test_map15))
        print("------------------------------------------------------")
        
        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write("Testing on Model " + ckpt_path + '\n')
        f.write('Test F-score ' + str(test_fscore) + '\n')
        f.write('Test MAP50   ' + str(test_map50) + '\n')
        f.write('Test MAP15   ' + str(test_map15) + '\n\n')
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
                nn.init.orthogonal_(param, gain=np.sqrt(2.0))      # ReLU activation function
            else:
                raise NotImplementedError(f"initialization method {init_type} is not implemented.")
        elif 'bias' in name:
            nn.init.constant_(param, 0.1)

def main():
    model=SL_module(input_dim=1024, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5)
    model.to("cpu")
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    criterion = nn.MSELoss(reduction='none').to("cpu")

    train_dataset = MrHiSumDataset(mode='train', path="dataset/Finance_split.json")
    val_dataset = MrHiSumDataset(mode='val', path="dataset/Finance_split.json")
    test_dataset = MrHiSumDataset(mode='test', path="dataset/Finance_split.json")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=BatchCollator())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    epoch=50
    for epoch_i in range(epoch):
        print("[Epoch: {0:6}]".format(str(epoch_i+1)+"/"+str(epoch)))
        model.train()
        loss_history = []
        kl_loss_history = []
        loss_v_history = []
        loss_a_history = []
        num_batches = int(len(train_loader))
        iterator = iter(train_loader)
        for _ in tqdm(range(num_batches)):
            optimizer.zero_grad()
            data = next(iterator)
            visual = data['features'].to("cpu")
            gtscore = data['gtscore'].to("cpu")
            gtscore=(gtscore > 0.7).float()
            #audio = data['audio'].to("cpu")
            mask = data['mask'].to("cpu")

            score, weights = model(visual, mask)
            loss = criterion(score[mask], gtscore[mask]).mean()
            print(f"score[mask], gtscore[mask]= {score[mask], gtscore[mask]}")
            loss.backward()
            loss_history.append(loss.detach().item())
            optimizer.step()
            if not loss_history==[]:
                loss = np.mean(np.array(loss_history))
            else:
                loss = 0
            if not kl_loss_history==[]:
                kl_loss = np.mean(np.array(kl_loss_history))
            else:
                kl_loss = 0 
            if not loss_v_history==[]:
                v_loss = np.mean(np.array(loss_v_history))
            else:
                v_loss = 0
            if not loss_a_history==[]:
                a_loss = np.mean(np.array(loss_a_history))
            else:
                a_loss = 0
            val_recon_loss=0
            val_kl_loss=0
            val_loss=0
            val_f1score=0
            val_map50=0
            val_map15=0
            val_map=0
            val_precision=0
            #final_f_score, final_map50, final_map15, final_map, final_loss, final_precision, final_hit1
            val_f1score, val_map50, val_map15, val_map, val_loss, val_precision, test_hit1, test_hit10 = evaluate(model, criterion, dataloader=val_loader)
            print(f'[Proportion {str(1 * 100)}% | Epoch {str(epoch_i+1)}] \n'
                        f'Train Loss: {loss:.5f}, Val Loss: {val_loss:.5f} \n'
                        f'Visual loss: {v_loss:.5f}, Audio loss: {a_loss:.5f}\n'
                        f'Val F1 Score: {val_f1score:.5f}, Val MAP50: {val_map50:.5f} \n'
                        f'Val MAP15: {val_map15:.5f}, KL loss: {kl_loss:.5f}\n'
                        f'Hit@1:{test_hit1}, Hit@10:{test_hit10} \n'
                        f'MAP: {val_map:.5f}\n') 





if __name__=='__main__':
    main()
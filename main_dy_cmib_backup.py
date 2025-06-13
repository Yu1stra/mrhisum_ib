# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from networks.mlp import SimpleMLP
from networks.pgl_sum.pgl_sum import PGL_SUM
from networks.vasnet.vasnet import VASNet
from networks.sl_module.sl_module import *
from networks.graph_fusion import graph_fusion
from model.utils.evaluation_metrics import evaluate_summary
from model.utils.generate_summary import generate_summary
from model.utils.evaluate_map import generate_mrhisum_seg_scores, top50_summary, top15_summary, top_n_summary
from model.SMI import sliceMI
from networks.atfuse.ATFuse import FactorAtt_ConvRelPosEnc, MHCABlock, UpScale 
from networks.CrossAttentional.cam import CAM
from networks.sl_module.BottleneckTransformer import BottleneckTransformer
import gc
import tracemalloc
import objgraph
import sys
import argparse
from model.configs import Config, str2bool
from torch.utils.data import DataLoader
from model.mrhisum_dataset_fixed import MrHiSumDataset, BatchCollator

import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(42)

#SL_module_CIB_dy
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
        cuda_device=self.device
        self.model = SL_module_CIB( 
            visual_input_dim=1024, 
            audio_input_dim=128, 
            depth=5, heads=8, 
            mlp_dim=3072, 
            dropout_ratio=0.5, 
            visual_bottleneck_dim=128, 
            audio_bottleneck_dim=128).to(cuda_device) #(self, visual_input_dim, audio_input_dim, depth, heads, mlp_dim, dropout_ratio, visual_bottleneck_dim, audio_bottleneck_dim)
        # Model already moved to device in initialization
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.l2_reg)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
    
    def train(self):
        path50=[]
        path100=[]
        path150=[]
        path200=[]
        path=[]
        #proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 每次使用 10%, 20%, ..., 100%
        proportions = [1.0]
        proportion=1.0
        model=self.model
        cuda_device=self.device
        #for proportion in proportions:
        #time.sleep(0.5)
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
        print(f"Training with {int(proportion * 100)}% of the training data...")
        
        """# 创建子集数据加载器
        subset_size = max(1, int(len(self.train_loader.dataset) * proportion))
        if subset_size == 0:
            print(f"Skipping proportion {int(proportion * 100)}% because the dataset is too small.")
            continue
        subset_indices = torch.randperm(len(self.train_loader))[:subset_size]
        subset_dataset = torch.utils.data.Subset(self.train_loader.dataset, subset_indices.tolist())
        subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=self.train_loader.batch_size, shuffle=True)"""

        # Create GradScaler with enabled memory-efficient optimizations
        scaler = torch.cuda.amp.GradScaler(growth_interval=100, enabled=True)

        for epoch_i in range(self.config.epochs):
            # Already commented out sleep
            print("[Epoch: {0:6}]".format(str(epoch_i+1)+"/"+str(self.config.epochs)))
            model.train()
            loss_history = []
            kl_loss_history_v = []
            kl_loss_history_a = []
            kl_loss_history_m = []
            loss_v_history = []
            loss_a_history = []
            loss_m_history = []
            v_SMI_history = []
            a_SMI_history = []
            m_SMI_history = []
            num_batches = int(len(self.train_loader))

            num_smi_batches = 3
            # 平均取樣 10 個 batch index，分散在整個 epoch
            smi_batch_indices = np.linspace(0, num_batches - 1, num=num_smi_batches, dtype=int)

            iterator = iter(self.train_loader)

            for batch_idx in tqdm(range(num_batches)):
                
                # More memory-efficient way to zero gradients
                self.optimizer.zero_grad(set_to_none=True)
                # Removed unnecessary sleep
                try:
                    data = next(iterator)
                except StopIteration:
                    iterator = iter(self.train_loader)
                    data = next(iterator)
                #'video_name' : video_name, 'features' : frame_feat_visual, 'audio':frame_feat_audio, 'gtscore':gtscore, 'mask':mask_visual, 'mask_audio':mask_audio
                visual = data['features'].to(cuda_device, non_blocking=True)
                gtscore = data['gtscore'].to(cuda_device, non_blocking=True)
                audio = data['audio'].to(cuda_device, non_blocking=True)
                mask = data['mask'].to(cuda_device, non_blocking=True)
                
                #print(f"visual.shape={visual.shape},audio.shape={audio.shape},gtscore.shape={gtscore.shape}")
                #score_v, score_m,  kl_loss_v+kl_loss_m
                with torch.cuda.amp.autocast():
                    score_v, score_a, score_m,  kl_loss_v, kl_loss_a, kl_loss_m = model(visual, audio, mask)
                    prediction_loss_v = self.criterion(score_v[mask], gtscore[mask]).mean()
                    prediction_loss_a = self.criterion(score_a[mask], gtscore[mask]).mean()
                    prediction_loss_m = self.criterion(score_m[mask], gtscore[mask]).mean()
                    vbeta = self.config.vbeta  # KL 损失的权重系数
                    abeta = self.config.abeta  # KL 损失的权重系数
                    mbeta = self.config.mbeta  # KL 损失的权重系数
                    total_loss = prediction_loss_v + prediction_loss_m  + vbeta * kl_loss_v + abeta * kl_loss_a + mbeta * kl_loss_m
                #total_loss = prediction_loss + reconstruction_loss + beta * kl_loss
                scaler.scale(total_loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                # total_loss.backward()
                # === forward 執行後馬上做 SMI ===
                if self.config.SMI:
                    if batch_idx in smi_batch_indices:# 每個 epoch 抽十個 batch 算然後是平均在前中後
                        with torch.no_grad():
                            for name, feature in {
                                'visual': self.model.visual_feature,
                                'audio': self.model.audio_feature,
                                'fused': self.model.fused_feature,
                            }.items():
                                mask_bool = mask.flatten().bool()
                                feature_flat = feature.view(-1, feature.shape[-1])  # [B*T, D]
                                gtscore_flat = gtscore.view(-1, 1)
                                X = feature_flat[mask_bool].cpu()
                                Y = gtscore_flat[mask_bool].cpu()
                                del mask_bool, feature_flat, gtscore_flat
                                torch.cuda.empty_cache()
                                # Reduce SMI computation parameters for better efficiency
                                smi_val = sliceMI(X, Y, M=400, n=4000, DY=False)
                                if name == 'visual':
                                    v_SMI_history.append(smi_val)
                                elif name == 'audio':
                                    a_SMI_history.append(smi_val)
                                elif name == 'fused':
                                    m_SMI_history.append(smi_val)

                loss_v_history.append(prediction_loss_v.detach().item())
                loss_a_history.append(prediction_loss_a.detach().item())
                loss_m_history.append(prediction_loss_m.detach().item())
                kl_loss_history_v.append(kl_loss_v.detach().item())
                kl_loss_history_a.append(kl_loss_a.detach().item())
                kl_loss_history_m.append(kl_loss_m.detach().item())
                loss_history.append(total_loss.detach().item())
                # self.optimizer.step()

                
                #time.sleep(1.5)
            if not loss_history==[]:
                loss = np.mean(np.array(loss_history))
            else:
                loss = 0
            if not kl_loss_history_v==[]:
                kl_loss_v = np.mean(np.array(kl_loss_history_v))
            else:
                kl_loss_v = 0 
            if not kl_loss_history_a==[]:
                kl_loss_a = np.mean(np.array(kl_loss_history_a))
            else:
                kl_loss_a = 0 
            if not kl_loss_history_m==[]:
                kl_loss_m = np.mean(np.array(kl_loss_history_m))
            else:
                kl_loss_m = 0 
            if not loss_v_history==[]:
                v_loss = np.mean(np.array(loss_v_history))
            else:
                v_loss = 0
            if not loss_a_history==[]:
                a_loss = np.mean(np.array(loss_a_history))
            else:
                a_loss = 0
            if not loss_m_history==[]:
                m_loss = np.mean(np.array(loss_m_history))
            else:
                m_loss = 0
            val_recon_loss=0
            val_kl_loss=0
            val_loss=0
            val_f1score=0
            val_map50=0
            val_map15=0
            val_map=0
            val_precision=0
            val_f1score, val_map50, val_map15, val_map, val_loss, val_precision, _, _, _, _, _, _, = self.evaluate(dataloader=self.val_loader)
            #final_f_score, final_map50, final_map15, final_loss, final_precision, final_prediction_loss_v, final_prediction_loss_a, final_kl_loss
            #final_f_score, final_map50, final_map15, final_loss, final_precision, final_prediction_loss_v, final_prediction_loss_a, final_prediction_loss_m, final_kl_loss_v, final_kl_loss_a, final_kl_loss_m        
            # 保存每次比例的日志
            proportion_dir = os.path.join(self.config.save_dir_root, f'logs/proportion_{int(proportion * 100)}')
            os.makedirs(proportion_dir, exist_ok=True)
            v_SMI = np.mean(v_SMI_history)
            a_SMI = np.mean(a_SMI_history)
            m_SMI = np.mean(m_SMI_history)

            f = open(os.path.join(self.config.save_dir_root, 'logs/SMI_result.txt'), 'a')
            if self.config.SMI:
                print(f"Epoch: {epoch_i+1}, v_SMI: {v_SMI:.5f}, a_SMI: {a_SMI:.5f}, m_SMI: {m_SMI:.5f}")
                if epoch_i == 0:
                    f.write(f"Epoch     v_SMI     a_SMI     m_SMI\n")
                f.write(f"   {epoch_i + 1}     {v_SMI:.5f}     {a_SMI:.5f}     {m_SMI:.5f}\n")
            else:
                print("No use SMI")
                if epoch_i == 0:
                    f.write("No use SMI")
            f.flush()
            f.close()


            f = open(os.path.join(proportion_dir, 'results.txt'), 'a')
            print(f"proportion: {proportion}, type: {type(proportion)}")
            print(f"epoch_i: {epoch_i}, type: {type(epoch_i)}")

            f.write(f'[Proportion {str(proportion * 100)}% | Epoch {str(epoch_i+1)}], \n'
                    f'Train Loss: {loss:.5f}, Val Loss: {val_loss:.5f}, \n'
                    f'Visual loss: {v_loss:.5f}, Audio loss: {a_loss:.5f}\n'
                    f'Multi loss: {m_loss:.5f}\n'
                    f'Visual KL loss: {kl_loss_v:.5f}, Audio KL loss: {kl_loss_a:.5f}\n'
                    f'Multi KL loss: {kl_loss_m:.5f}\n'
                    f'Val F1 Score: {val_f1score:.5f}, Val MAP50: {val_map50:.5f}, \n'
                    f'Val MAP15: {val_map15:.5f}\n'
                    f'Val MAP: {val_map:.5f}\n')    
            f.flush()
            f.close()
            f = open(os.path.join(self.config.save_dir_root, 'logs/loss.txt'), 'a')
            
            if epoch_i==0:
                f.write(f'Epoch     loss     val_loss     \n')
            f.write(f'{epoch_i + 1}     {loss}     {val_loss}     \n')
            f.flush()
            f.close()
            state_dict=model.state_dict()
            """
            if best_f1score <= val_f1score:
                best_f1score = val_f1score
                best_f1score_epoch = epoch_i
                f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, f'best_f1.pkl')
                torch.save(state_dict, f1_save_ckpt_path)

            if best_map50 <= val_map50:
                best_map50 = val_map50
                best_map50_epoch = epoch_i
                map50_save_ckpt_path = os.path.join(self.config.best_map50_save_dir, f'best_map50.pkl')
                torch.save(state_dict, map50_save_ckpt_path)
            
            if best_map15 <= val_map15:
                best_map15 = val_map15
                best_map15_epoch = epoch_i
                map15_save_ckpt_path = os.path.join(self.config.best_map15_save_dir, f'best_map15.pkl')
                torch.save(state_dict, map15_save_ckpt_path)
            """
            if best_f1score <= val_f1score:
                best_f1score = val_f1score
                best_f1score_epoch = epoch_i
                epoch=epoch_i+1
                if epoch <= 50:
                    f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, f'Proportion_{int(proportion * 100)}%_best_f1_epoch50.pkl')
                    if os.path.exists(f1_save_ckpt_path):
                        os.remove(f1_save_ckpt_path)
                    torch.save(state_dict, f1_save_ckpt_path)
                    if f1_save_ckpt_path not in path50:
                        path50.append(f1_save_ckpt_path)
                if epoch>50 and epoch <= 100:
                    f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, f'Proportion_{int(proportion * 100)}%_best_f1_epoch100.pkl')
                    if os.path.exists(f1_save_ckpt_path):
                        os.remove(f1_save_ckpt_path)
                    torch.save(state_dict, f1_save_ckpt_path)
                    if f1_save_ckpt_path not in path100:
                        path100.append(f1_save_ckpt_path)
                if epoch>100 and epoch <= 150:
                    f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, f'Proportion_{int(proportion * 100)}%_best_f1_epoch150.pkl')
                    if os.path.exists(f1_save_ckpt_path):
                        os.remove(f1_save_ckpt_path)
                    torch.save(state_dict, f1_save_ckpt_path)
                    if f1_save_ckpt_path not in path150:
                        path150.append(f1_save_ckpt_path)
                if epoch>150 and epoch <= 200:
                    f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, f'Proportion_{int(proportion * 100)}%_best_f1_epoch200.pkl')
                    if os.path.exists(f1_save_ckpt_path):
                        os.remove(f1_save_ckpt_path)
                    torch.save(state_dict, f1_save_ckpt_path)
                    if f1_save_ckpt_path not in path200:
                        path200.append(f1_save_ckpt_path)
                if epoch == self.config.epochs:
                    f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, f'Proportion_{int(proportion * 100)}%_best_f1_epochlast.pkl')
                    if os.path.exists(f1_save_ckpt_path):
                        os.remove(f1_save_ckpt_path)
                    torch.save(state_dict, f1_save_ckpt_path)
                    if f1_save_ckpt_path not in path:
                        path.append(f1_save_ckpt_path)
            if best_map50 <= val_map50:
                best_map50 = val_map50
                best_map50_epoch = epoch_i
                epoch=epoch_i+1
                if epoch <= 50:
                    map50_save_ckpt_path = os.path.join(self.config.best_map50_save_dir, f'Proportion_{int(proportion * 100)}%_best_map50_epoch50.pkl')
                    if os.path.exists(map50_save_ckpt_path):
                        os.remove(map50_save_ckpt_path)
                    torch.save(state_dict, map50_save_ckpt_path)
                    if map50_save_ckpt_path not in path50:
                        path50.append(map50_save_ckpt_path)
                if epoch>50 and epoch <= 100:
                    map50_save_ckpt_path = os.path.join(self.config.best_map50_save_dir, f'Proportion_{int(proportion * 100)}%_best_map50_epoch100.pkl')
                    if os.path.exists(map50_save_ckpt_path):
                        os.remove(map50_save_ckpt_path)
                    torch.save(state_dict, map50_save_ckpt_path)
                    if map50_save_ckpt_path not in path100:
                        path100.append(map50_save_ckpt_path)
                if epoch>100 and epoch <= 150:
                    map50_save_ckpt_path = os.path.join(self.config.best_map50_save_dir, f'Proportion_{int(proportion * 100)}%_best_map50_epoch150.pkl')
                    if os.path.exists(map50_save_ckpt_path):
                        os.remove(map50_save_ckpt_path)
                    torch.save(state_dict, map50_save_ckpt_path)
                    if map50_save_ckpt_path not in path150:
                        path150.append(map50_save_ckpt_path)
                if epoch>150 and epoch <= 200:
                    map50_save_ckpt_path = os.path.join(self.config.best_map50_save_dir, f'Proportion_{int(proportion * 100)}%_best_map50_epoch200.pkl')
                    if os.path.exists(map50_save_ckpt_path):
                        os.remove(map50_save_ckpt_path)
                    torch.save(state_dict, map50_save_ckpt_path)
                    if map50_save_ckpt_path not in path200:
                        path200.append(map50_save_ckpt_path)
                if epoch == self.config.epochs:
                    map50_save_ckpt_path = os.path.join(self.config.best_map50_save_dir, f'Proportion_{int(proportion * 100)}%_best_map50_epochlast.pkl')
                    if os.path.exists(map50_save_ckpt_path):
                        os.remove(map50_save_ckpt_path)
                    torch.save(state_dict, map50_save_ckpt_path)
                    if map50_save_ckpt_path not in path:
                        path.append(map50_save_ckpt_path)
            
            if best_map15 <= val_map15:
                best_map15 = val_map15
                best_map15_epoch = epoch_i
                epoch=epoch_i+1
                if epoch <= 50:
                    map15_save_ckpt_path = os.path.join(self.config.best_map15_save_dir, f'Proportion_{int(proportion * 100)}%_best_map15_epoch50.pkl')
                    if os.path.exists(map15_save_ckpt_path):
                        os.remove(map15_save_ckpt_path)
                    torch.save(state_dict, map15_save_ckpt_path)
                    if map15_save_ckpt_path not in path50:
                        path50.append(map15_save_ckpt_path)
                if epoch>50 and epoch <= 100:
                    map15_save_ckpt_path = os.path.join(self.config.best_map15_save_dir, f'Proportion_{int(proportion * 100)}%_best_map15_epoch100.pkl')
                    if os.path.exists(map15_save_ckpt_path):
                        os.remove(map15_save_ckpt_path)
                    torch.save(state_dict, map15_save_ckpt_path)
                    if map15_save_ckpt_path not in path100:
                        path100.append(map15_save_ckpt_path)
                if epoch>100 and epoch <= 150:
                    map15_save_ckpt_path = os.path.join(self.config.best_map15_save_dir, f'Proportion_{int(proportion * 100)}%_best_map15_epoch150.pkl')
                    if os.path.exists(map15_save_ckpt_path):
                        os.remove(map15_save_ckpt_path)
                    torch.save(state_dict, map15_save_ckpt_path)
                    if map15_save_ckpt_path not in path150:
                        path150.append(map15_save_ckpt_path)
                if epoch>150 and epoch <= 200:
                    map15_save_ckpt_path = os.path.join(self.config.best_map15_save_dir, f'Proportion_{int(proportion * 100)}%_best_map15_epoch200.pkl')
                    if os.path.exists(map15_save_ckpt_path):
                        os.remove(map15_save_ckpt_path)
                    torch.save(state_dict, map15_save_ckpt_path)
                    if map15_save_ckpt_path not in path200:
                        path200.append(map15_save_ckpt_path)
                if epoch == self.config.epochs:
                    map15_save_ckpt_path = os.path.join(self.config.best_map15_save_dir, f'Proportion_{int(proportion * 100)}%_best_map15_epochlast.pkl')
                    if os.path.exists(map15_save_ckpt_path):
                        os.remove(map15_save_ckpt_path)
                    torch.save(state_dict, map15_save_ckpt_path)
                    if map15_save_ckpt_path not in path:
                        path.append(map15_save_ckpt_path)
                
            if best_pre <= val_precision:
                best_pre = val_precision
                best_pre_epoch = epoch_i
                epoch=epoch_i+1
                if epoch <= 50:
                    pre_save_ckpt_path = os.path.join(self.config.best_pre_save_dir, f'Proportion_{int(proportion * 100)}%_best_precision_epoch50.pkl')
                    if os.path.exists(pre_save_ckpt_path):
                        os.remove(pre_save_ckpt_path)
                    torch.save(state_dict, pre_save_ckpt_path)
                    if pre_save_ckpt_path not in path50:
                        path50.append(pre_save_ckpt_path)
                if epoch>50 and epoch <= 100:
                    pre_save_ckpt_path = os.path.join(self.config.best_pre_save_dir, f'Proportion_{int(proportion * 100)}%_best_precision_epoch100.pkl')
                    if os.path.exists(pre_save_ckpt_path):
                        os.remove(pre_save_ckpt_path)
                    torch.save(state_dict, pre_save_ckpt_path)
                    if pre_save_ckpt_path not in path100:
                        path100.append(pre_save_ckpt_path)
                if epoch>100 and epoch <= 150:
                    pre_save_ckpt_path = os.path.join(self.config.best_pre_save_dir, f'Proportion_{int(proportion * 100)}%_best_precision_epoch150.pkl')
                    if os.path.exists(pre_save_ckpt_path):
                        os.remove(pre_save_ckpt_path)
                    torch.save(state_dict, pre_save_ckpt_path)
                    if pre_save_ckpt_path not in path150:
                        path150.append(pre_save_ckpt_path)
                if epoch>150 and epoch <= 200:
                    pre_save_ckpt_path = os.path.join(self.config.best_pre_save_dir, f'Proportion_{int(proportion * 100)}%_best_precision_epoch200.pkl')
                    if os.path.exists(pre_save_ckpt_path):
                        os.remove(pre_save_ckpt_path)
                    torch.save(state_dict, pre_save_ckpt_path)
                    if pre_save_ckpt_path not in path200:
                        path200.append(pre_save_ckpt_path)
                if epoch == self.config.epochs:
                    pre_save_ckpt_path = os.path.join(self.config.best_pre_save_dir, f'Proportion_{int(proportion * 100)}%_best_precision_epochlast.pkl')
                    if os.path.exists(pre_save_ckpt_path):
                        os.remove(pre_save_ckpt_path)
                    torch.save(state_dict, pre_save_ckpt_path)
                    if pre_save_ckpt_path not in path:
                        path.append(pre_save_ckpt_path)
            if best_map <= val_map:
                best_map = val_map
                best_map_epoch = epoch_i
                best_map_ckpt_path = os.path.join(self.config.best_map_save_dir, f'Proportion_{int(proportion * 100)}%_best_map.pkl')
                if os.path.exists(best_map_ckpt_path):
                    os.remove(best_map_ckpt_path)
                torch.save(state_dict, best_map_ckpt_path)
                if best_map_ckpt_path not in path:
                    path.append(best_map_ckpt_path)
            
            #print("   [Epoch {0}] Train loss: {1:.05f}".format(epoch_i+1, loss))
            #print('    VAL  F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5}'.format(val_f1score, val_map50, val_map15))
            # print(f'[Proportion {str(proportion * 100)}% | Epoch {str(epoch_i+1)}], \n'
            #             f'Train Loss: {loss:.5f}, Val Loss: {val_loss:.5f}, \n'
            #             f'Visual loss: {v_loss:.5f}, Audio loss: {a_loss:.5f}\n'
            #             f'Val F1 Score: {val_f1score:.5f}, Val MAP50: {val_map50:.5f}, \n'
            #             f'Val MAP15: {val_map15:.5f}, KL loss: {kl_loss:.5f}\n') 
            print(f'[Proportion {str(proportion * 100)}% | Epoch {str(epoch_i+1)}], \n'
                    f'Train Loss: {loss:.5f}, Val Loss: {val_loss:.5f}, \n'
                    f'Visual loss: {v_loss:.5f}, Audio loss: {a_loss:.5f}\n'
                    f'Multi loss: {m_loss:.5f}\n'
                    f'Visual KL loss: {kl_loss_v:.5f}, Audio KL loss: {kl_loss_a:.5f}\n'
                    f'Multi KL loss: {kl_loss_m:.5f}\n'
                    f'Val F1 Score: {val_f1score:.5f}, Val MAP50: {val_map50:.5f}, \n'
                    f'Val MAP15: {val_map15:.5f}\n'
                    f'Val MAP: {val_map:.5f}\n')
            f = open(os.path.join(self.config.save_dir_root, 'logs/all_result.txt'), 'a')
            if epoch_i == 0:
                f.write(f'Epoch     Val_f1score     Val_map50     Val_map15     Val_map     \n')
            f.write(f'{epoch_i+1}     {val_f1score}     {val_map50}     {val_map15}     {val_map}     \n')
            f.flush()
            f.close()
            #del data, visual, gtscore, audio, mask, score, weights, loss
            # === SMI 分析（只在 mbeta = 1e-4 時執行） ===

            gc.collect()
        print(f'  [Proportion {int(proportion * 100)}%]')
        print('   Best Val F1 score {0:0.5} @ epoch{1}'.format(best_f1score, best_f1score_epoch+1))
        print('   Best Val MAP-50   {0:0.5} @ epoch{1}'.format(best_map50, best_map50_epoch+1))
        print('   Best Val MAP-15   {0:0.5} @ epoch{1}'.format(best_map15, best_map15_epoch+1))
        print('   Best Val MAP     {0:0.5} @ epoch{1}'.format(best_map, best_map_epoch+1))
        print('   Best Val PRECISION   {0:0.5} @ epoch{1}'.format(best_pre, best_pre_epoch+1))

        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write(f'    [Proportion {int(proportion * 100)}%]\n')
        f.write('   Best Val F1 score {0:0.5} @ epoch{1}\n'.format(best_f1score, best_f1score_epoch+1))
        f.write('   Best Val MAP-50   {0:0.5} @ epoch{1}\n'.format(best_map50, best_map50_epoch+1))
        f.write('   Best Val MAP-15   {0:0.5} @ epoch{1}\n'.format(best_map15, best_map15_epoch+1))
        f.write('   Best Val MAP     {0:0.5} @ epoch{1}\n'.format(best_map, best_map_epoch+1))
        f.write('   Best Val PRECISION   {0:0.5} @ epoch{1}\n\n'.format(best_pre, best_pre_epoch+1))
        f.flush()
        f.close()
        torch.cuda.empty_cache()
        return path, path50, path100, path150, path200
        #return f1_save_ckpt_path, map50_save_ckpt_path, map15_save_ckpt_path

    def evaluate(self, dataloader=None):
        """ Saves the frame's importance scores for the test videos in json format.

        :param int epoch_i: The current training epoch.
        """
        model=self.model
        cuda_device=self.device
        model.eval()
        loss_history = []
        kl_loss_history_v = []
        kl_loss_history_a = []
        kl_loss_history_m = []
        loss_v_history = []
        loss_a_history = []
        loss_m_history = []
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
            #multi_feature = data['multi'].to(cuda_device)
            #一般-------------------------------------
            
            for input_feature in (visual,audio):
                if len(input_feature.shape) == 2:
                        input_feature = input_feature.unsqueeze(0)
                elif len(input_feature.shape) == 4:
                        input_feature = input_feature.squeeze(0)
            if len(gtscore.shape) == 1:
                    gtscore = gtscore.unsqueeze(0)
    
            B = input_feature.shape[0]
            mask=None
            if input_mask in data:
                mask = data[input_mask].to(cuda_device)
            with torch.no_grad():
                score_v, score_a, score_m,  kl_loss_v, kl_loss_a, kl_loss_m = model(visual, audio, mask)
            prediction_loss_v = self.criterion(score_v[mask], gtscore[mask]).mean()
            prediction_loss_a = self.criterion(score_a[mask], gtscore[mask]).mean()
            prediction_loss_m = self.criterion(score_m[mask], gtscore[mask]).mean()
            vbeta = self.config.vbeta  # KL 损失的权重系数
            abeta = self.config.abeta  # KL 损失的权重系数
            mbeta = self.config.mbeta  # KL 损失的权重系数
            total_loss = prediction_loss_v + prediction_loss_m  + vbeta * kl_loss_v + abeta * kl_loss_a + mbeta * kl_loss_m
            #total_loss = prediction_loss + reconstruction_loss + beta * kl_loss
            torch.cuda.synchronize() 
            #total_loss.backward()
            loss_v_history.append(prediction_loss_v.detach().item())
            #loss_a_history.append(prediction_loss_a.detach().item())
            loss_v_history.append(prediction_loss_v.detach().item())
            loss_a_history.append(prediction_loss_a.detach().item())
            loss_m_history.append(prediction_loss_m.detach().item())
            kl_loss_history_v.append(kl_loss_v.detach().item())
            kl_loss_history_a.append(kl_loss_a.detach().item())
            kl_loss_history_m.append(kl_loss_m.detach().item())
            loss_history.append(total_loss.detach().item())
            #self.optimizer.step()
            # Calculate precision
            predictions = (score_m > 0.5).float()  # Example threshold, modify as needed
            true_positives = (predictions[mask] * gtscore[mask]).sum().detach().item()
            predicted_positives = predictions[mask].sum().detach().item()
            precision = true_positives / (predicted_positives + 1e-7)  # Avoid division by zero
            precision_history.append(precision)
        
            # Summarization metric
            score_m = score_m.squeeze().cpu()
            gt_summary = data['gt_summary'][0]
            cps = data['change_points'][0]
            n_frames = data['n_frames']
            nfps = data['n_frame_per_seg'][0].tolist()
            picks = data['picks'][0].numpy()
            #print("score",len(score), "cps",len(cps), "nframe",len(n_frames), "nfps",len(nfps), "picks",len(picks))
            machine_summary = generate_summary(score_m, cps, n_frames, nfps, picks)
            #print("MACHINE", machine_summary, machine_summary.shape)
            #print("GT SUMMARY",gt_summary, gt_summary.shape)
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
            
            highlight_seg_machine_score = generate_mrhisum_seg_scores(score_m, uniform_clip=5)
            highlight_seg_machine_score = torch.exp(highlight_seg_machine_score) / (torch.exp(highlight_seg_machine_score).sum() + 1e-7)
            
            clone_machine_summary = highlight_seg_machine_score.clone().detach().cpu()
            clone_machine_summary = clone_machine_summary.numpy()
            
            # 获取gt_seg_score的numpy格式用于计算
            gt_seg_score_np = gt_seg_score.clone().detach().cpu().numpy()
            
            # 归一化真实标签分数（如果尚未归一化）
            gt_scores_normalized = gt_seg_score_np / np.max(gt_seg_score_np) if np.max(gt_seg_score_np) > 0 else gt_seg_score_np
            
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
        final_map = np.mean(map_history) if map_history else 0
        final_precision = np.mean(precision_history)
        final_loss = np.mean(loss_history)
        if not loss_v_history==[]:
            final_prediction_loss_v = np.mean(loss_v_history)
        else:
            final_prediction_loss_v = 0
        if not loss_a_history==[]:
            final_prediction_loss_a = np.mean(loss_a_history)
        else:
            final_prediction_loss_a = 0
        if not loss_m_history==[]:
            final_prediction_loss_m = np.mean(loss_m_history)
        else:
            final_prediction_loss_m = 0
        if not kl_loss_history_v==[]:
            final_kl_loss_v = np.mean(kl_loss_history_v)
        else:
            final_kl_loss_v = 0
        if not kl_loss_history_a==[]:
            final_kl_loss_a = np.mean(kl_loss_history_a)
        else:
            final_kl_loss_a = 0
        if not kl_loss_history_m==[]:
            final_kl_loss_m = np.mean(kl_loss_history_m)
        else:
            final_kl_loss_m = 0
        return final_f_score, final_map50, final_map15, final_map, final_loss, final_precision, final_prediction_loss_v, final_prediction_loss_a, final_prediction_loss_m, final_kl_loss_v, final_kl_loss_a, final_kl_loss_m
            
    def test(self, ckpt_path):
        model=self.model
        cuda_device=self.device
        if ckpt_path != None:
            print("Testing Model: ", ckpt_path)
            print("Device: ",  cuda_device)
            model.load_state_dict(torch.load(ckpt_path))
        test_fscore, test_map50, test_map15, test_map, _, _, _, _, _, _, _, _ = self.evaluate(dataloader=self.test_loader)

        print("------------------------------------------------------")
        print(f"   TEST RESULT on {ckpt_path}: ")
        print('   TEST MRHiSum F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5} | MAP {3:0.5}'.format(test_fscore, test_map50, test_map15, test_map))
        print("------------------------------------------------------")
        
        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write("Testing on Model " + ckpt_path + '\n')
        f.write('Test F-score ' + str(test_fscore) + '\n')
        f.write('Test MAP50   ' + str(test_map50) + '\n')
        f.write('Test MAP15   ' + str(test_map15) + '\n\n')
        f.write('Test MAP     ' + str(test_map) + '\n\n')
        f.flush()

    
    def save_epoch_smi(self, epoch_i, v_smi, a_smi, m_smi):
        """保存每個epoch的SMI數據 (4.1風格)"""
        import pickle
        import os
        
        # 將三種特徵的SMI添加到smi_all_epochs
        self.smi_all_epochs[f'Epoch{epoch_i+1:02d}'] = [v_smi, a_smi, m_smi]
        
        # 保存到文件
        smi_file = os.path.join(self.config.save_dir_root, 'smi_training_history.pkl')
        with open(smi_file, 'wb') as f:
            pickle.dump(self.smi_all_epochs, f, pickle.HIGHEST_PROTOCOL)
        
        print(f"Epoch {epoch_i+1} SMI data saved to {smi_file}")
    
    def analyze_final_model_smi(self, mode="val"):
        """分析最終模型的所有層的SMI (4.2風格)"""
        print(f"Analyzing final model SMI for all layers using {mode} data...")
        import pickle
        import os
        
        model = self.model
        cuda_device = self.device
        
        # 註冊鉤子獲取中間層輸出
        activations = {}
        hooks = []
        
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()
            return hook
        
        # 註冊鉤子到所有有權重的層
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)
        
        # 使用指定數據集進行前向傳播
        with torch.no_grad():
            smi_all_layers = {}
            
            # 選擇數據集
            eval_loader = self.val_loader if mode == "val" else self.test_loader
            
            # 只使用一個批次的數據
            for batch_idx, data in enumerate(eval_loader):
                if batch_idx > 0:  # 只使用第一個批次
                    break
                    
                # 準備數據
                visual = data['features'].to(cuda_device)
                gtscore = data['gtscore'].to(cuda_device)
                audio = data['audio'].to(cuda_device)
                mask = data['mask'].to(cuda_device)
                
                # 前向傳播
                model(visual, audio, mask)
                
                # 計算每層的SMI
                mask_bool = mask.flatten().bool()
                gtscore_flat = gtscore.view(-1, 1)
                Y = gtscore_flat[mask_bool].cpu()
                
                for name, activation in activations.items():
                    try:
                        # 處理不同形狀的激活
                        if len(activation.shape) > 2:
                            activation_flat = activation.view(activation.size(0), -1, activation.size(-1))
                            activation_flat = activation_flat.view(-1, activation_flat.size(-1))
                        else:
                            activation_flat = activation
                        
                        # 使用掩碼獲取有效數據
                        X = activation_flat[mask_bool].cpu()
                        
                        # 使用較高的M值計算SMI
                        smi_val = sliceMI(X, Y, M=1000, n=4000, DY=False)
                        smi_all_layers[name] = smi_val
                        print(f'Layer {name}: SI(T;Y)={smi_val:.3f}')
                    except Exception as e:
                        print(f"Skipping layer {name} due to error: {str(e)}")
        
        # 移除鉤子
        for hook in hooks:
            hook.remove()
        
        # 保存結果
        smi_file = os.path.join(self.config.save_dir_root, f'smi_final_model_{mode}.pkl')
        with open(smi_file, 'wb') as f:
            pickle.dump(smi_all_layers, f, pickle.HIGHEST_PROTOCOL)
        
        print(f"Final model SMI analysis saved to {smi_file}")
        return smi_all_layers
        
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

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    g = torch.Generator()
    g.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'SL_module', help = 'the name of the model')
    parser.add_argument('--epochs', type = int, default = 5, help = 'the number of training epochs')
    parser.add_argument('--lr', type = float, default = 5e-5, help = 'the learning rate')
    parser.add_argument('--l2_reg', type = float, default = 1e-4, help = 'l2 regularizer')
    parser.add_argument('--dropout_ratio', type = float, default = 0.5, help = 'the dropout ratio')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'the batch size')
    parser.add_argument('--tag', type = str, default = 'test', help = 'A tag for experiments')
    parser.add_argument('--ckpt_path', type = str, default = None, help = 'checkpoint path for inference or weight initialization')
    parser.add_argument('--train', type=str2bool, default='true', help='when use Train')
    parser.add_argument('--path', type=str, default='dataset/mr_hisum_split.json', help='path')
    parser.add_argument('--device', type=str, default='1', help='gpu')
    parser.add_argument('--modal', type=str, default='visual', help='visual,audio,multi')
    parser.add_argument('--vbeta', type=float, default=0, help='beta_visual')
    parser.add_argument('--abeta', type=float, default=0, help='beta_audio')
    parser.add_argument('--mbeta', type=float, default=0, help='beta_multi')
    parser.add_argument('--type',type = str, default='base', help='base,ib,cib,eib,lib')#cib,eib,lib
    parser.add_argument('--SMI', type=bool, default=True, help='use SMI eval')
    parser.add_argument('--smi_layers', type=bool, default=True, help='analyze all model layers with SMI (4.2 style)')
    parser.add_argument('--smi_epochs', type=bool, default=True, help='track SMI across epochs (4.1 style)')
    arser.add_argument('--gpus', type=int, default=1, help='number of GPUs to use (1=single GPU, 2=two GPUs)')
    #parser.add_argument('--savepath', type = str, default='default', help='save folder path')

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
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=BatchCollator(),worker_init_fn=seed_worker,generator=g)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    device = torch.device("cuda:"+config.device)
    print ("Device being used:", device)
    solver = Solver(config, train_loader, val_loader, test_loader, device, config.modal)

    solver.build()
    test_model_ckpt_path = None
    #proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if config.train:
        #best_path, best_f1_ckpt_path, best_map50_ckpt_path, best_map15_ckpt_path = solver.train()
        best_path, best_path50, best_path100, best_path150, best_path200, = solver.train()
        #print(best_path, best_path50, best_path100)
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
        #solver.test(best_f1_ckpt_path)
        #solver.test(best_map50_ckpt_path)
        #solver.test(best_map15_ckpt_path)
        #solver.test(best_pre_ckpt_path)
        print(best_path, best_path50, best_path100)
    else:
        test_model_ckpt_path = config.ckpt_path
        if test_model_ckpt_path == None:
            print("Trained model checkpoint requried. Exit program")
            exit()
        else:
            solver.test(test_model_ckpt_path)
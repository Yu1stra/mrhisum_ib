#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
視頻摘要訓練模塊 - 基於 Transformer-IB 的變體
提供固定精度計算，優化記憶體使用
"""

# 標準庫導入
import os
import gc
import random
import argparse

# 科學計算和數據處理
import numpy as np
import psutil  # 恢復 psutil 導入
from sklearn.metrics import average_precision_score
from tqdm import tqdm

# PyTorch 相關
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn  # 正確導入 cudnn 模組
from torch.utils.data import DataLoader
from contextlib import nullcontext  # 用於處理上下文管理

# 自定義模組
from model.configs import Config, str2bool
from model.mrhisum_dataset import MrHiSumDataset, BatchCollator
from model.SL_module import SL_module_tran_IB
from model.metric.sequence_metrics import evaluate_summary
from model.metric.hm_metric import generate_summary, generate_mrhisum_seg_scores, top50_summary, top15_summary

# ===== 全局配置 =====

# 固定精度計算相關設定
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 避免某些 Intel 庫的重複加載問題

# 設置默認張量類型為 float32，確保計算精度統一
torch.set_default_dtype(torch.float32)

def seed_worker(worker_id):
    """為 DataLoader 工作線程設置種子，確保數據加載的可復現性
   
    Args:
        worker_id (int): 工作線程 ID
    """
    # 使用 worker_id 來確保每個工作線程有不同的種子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed=42, deterministic=True):
    """設置隨機種子並啟用確定性模式以確保結果可復現
   
    Args:
        seed (int): 隨機種子值，默認為 42
        deterministic (bool): 是否啟用確定性計算模式，默認為 True
    """
    # 設置 Python、NumPy 和 PyTorch 的隨機種子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
   
    # 設置 CUDA 相關的隨機種子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
   
    # 強制啟用確定性模式以固定計算精度
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
       
        # 使用確定性演算法（支持不同版本的 PyTorch）
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        else:  # 相容較舊版本的 PyTorch
            torch.backends.cudnn.enabled = True
    else:
        # 非確定性模式（提高性能但結果不固定）
        torch.backends.cudnn.benchmark = True

# 初始化全局隨機性
set_seed(42, deterministic=True)

# 禁用 TF32 精度，確保浮點運算的一致性
# 注意：這會影響性能，但可確保結果一致性
if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
    torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def cleanup_memory():
    """清理未使用的記憶體，釋放 CPU 和 GPU 資源
   
    同時執行 Python 垃圾回收和 CUDA 緩存清理（如果可用）
    """
    # 強制執行 Python 垃圾回收
    gc.collect()
   
    # 清理 CUDA 緩存（如果有 GPU）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
       
    # 返回處理後的可用記憶體信息（僅供參考）
    if hasattr(psutil, 'virtual_memory'):
        return {
            'available_ram': psutil.virtual_memory().available / (1024**3),
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if torch.cuda.is_available() else 0
        }
    return None


class Solver(object):
    """視頻摘要訓練與評估模型的管理類
   
    負責模型的初始化、訓練、評估和測試。實現了基於 Transformer-IB 的架構。
    """
   
    def __init__(self, config=None, train_loader=None, val_loader=None, test_loader=None, device=None, modal=None):
        """初始化 Solver 類
       
        Args:
            config (Config): 配置對象，包含模型參數和訓練設置
            train_loader (DataLoader): 訓練數據加載器
            val_loader (DataLoader): 驗證數據加載器
            test_loader (DataLoader): 測試數據加載器
            device (torch.device): 運算設備
            modal (str): 模態類型（'visual' 或 'audio'）
        """
        # 模型及優化器相關屬性
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None  # TensorBoard 寫入器（如需要）
       
        # 設備及配置相關屬性
        self.device = device
        self.config = config
        self.modal = modal
       
        # 數據加載器
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
       
        # 訓練相關屬性
        self.global_step = 0
        self.criterion = nn.MSELoss(reduction='none').to(self.device)
    def build(self):
        """构建視頻摘要模型
       
        根據選擇的模態（視覺或音頳）初始化相應的模型、優化器和學習率調節器
        """
        # 根據模態類型選擇輸入維度
        if self.modal == 'visual':
            input_dim = 1024  # 視覺特徵維度
            self.model = SL_module_tran_IB(
                input_dim=input_dim,
                depth=5,
                heads=8,
                mlp_dim=3072,
                dropout_ratio=0.5,
                bottleneck_dim=128
            )
        elif self.modal == 'audio':
            input_dim = 128   # 音頳特徵維度
            self.model = SL_module_tran_IB(
                input_dim=input_dim,
                depth=5,
                heads=8,
                mlp_dim=3072,
                dropout_ratio=0.5,
                bottleneck_dim=128
            )
        else:
            raise ValueError(f"Unsupported modal type: {self.modal}. Must be 'visual' or 'audio'.")
           
        # 將模型移動到指定設備上
        self.model.to(self.device)
       
        # 初始化模型權重
        self.init_weights(self.model)
       
        # 創建優化器及學習率調整器
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.lr,
            momentum=0.9,
            weight_decay=self.config.l2_reg
        )
       
        # 學習率每 100 個 epoch 調整一次，每次降低爲原來的 0.1 倍
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,
            gamma=0.1
        )

    def train(self):
        # 使用字典管理儲存路徑，減少冗餘變數
        checkpoints = {
            'final': [],
            'epoch50': [],
            'epoch100': [],
            'epoch150': [],
            'epoch200': []
        }
       
        proportion = 1.0
        model = self.model
        cuda_device = self.device
       
        # 使用字典統一管理最佳指標和對應的 epoch
        best_metrics = {
            'f1score': {'value': -1.0, 'epoch': 0},
            'map50': {'value': -1.0, 'epoch': 0},
            'map15': {'value': -1.0, 'epoch': 0},
            'pre': {'value': -1.0, 'epoch': 0}
        }
        
        # 初始化變量，避免'使用前賦值'警告
        best_f1score = best_map50 = best_map15 = best_pre = -1.0
        best_f1score_epoch = best_map50_epoch = best_map15_epoch = best_pre_epoch = 0
        
        # 初始化路徑變量，避免「使用前賦值」警告
        path = []
        path50 = []
        path100 = []
        path150 = []
        path200 = []
       
        # 啟用 Python 的垃圾回收
        gc.enable()
        
        # 初始化評估指標用於追蹤最佳值
        kl_loss = 0.0  # 初始化 kl_loss 以避免警告
       
        print(f"Training with {int(proportion * 100)}% of the training data...")
       
        """# 创建子集数据加载器
        subset_size = max(1, int(len(self.train_loader.dataset) * proportion))
        if subset_size == 0:
            print(f"Skipping proportion {int(proportion * 100)}% because the dataset is too small.")
            continue
        subset_indices = torch.randperm(len(self.train_loader))[:subset_size]
        subset_dataset = torch.utils.data.Subset(self.train_loader.dataset, subset_indices.tolist())
        subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=self.train_loader.batch_size, shuffle=True)"""

        # 使用 amp 自動混合精度加速訓練
        scaler = torch.cuda.amp.GradScaler()
       
        # 建立預先分配的記憶體空間來存放損失值，避免動態擴展
        max_batch_count = len(self.train_loader)
       
        for epoch_i in range(self.config.epochs):
            print("[Epoch: {0:6}]".format(str(epoch_i+1)+"/"+str(self.config.epochs)))
            model.train()
           
            # 使用固定大小的 Numpy 陣列預先分配記憶體
            loss_history = np.zeros(max_batch_count, dtype=np.float32)
            kl_loss_history = np.zeros(max_batch_count, dtype=np.float32)
            loss_v_history = np.zeros(max_batch_count, dtype=np.float32)
            loss_a_history = np.zeros(max_batch_count, dtype=np.float32)
           
            batch_count = 0  # 實際處理的批次數量
            iterator = iter(self.train_loader)

            # 使用 tqdm 顯示進度，並設置 mininterval 減少刷新頻率
            for batch_idx in tqdm(range(max_batch_count), mininterval=1.0):
                # 使用 set_to_none=True 更有效率地清除梯度
                self.optimizer.zero_grad(set_to_none=True)
               
                try:
                    data = next(iterator)
                except StopIteration:
                    # 迭代結束
                    break
               
                # 移除 time.sleep 來避免不必要的 CPU 等待
               
                # 優化數據載入，使用暫存以減少 GPU 和 CPU 之間的傳輸
                visual = data['features'].to(cuda_device, non_blocking=True)
                gtscore = data['gtscore'].to(cuda_device, non_blocking=True)
                audio = data['audio'].to(cuda_device, non_blocking=True)
                mask = data['mask'].to(cuda_device, non_blocking=True)
               
                # 定期釋放 Python 的 CPU 緩存
                if batch_idx % 10 == 0:
                    gc.collect()
               
                #一般---------------------------
                with torch.cuda.amp.autocast():
                    #IB+KL-----------------------------------------------------
                        if self.config.modal == "visual":
                            score,  kl_loss = model(visual, mask)
                        elif self.config.modal == "audio":
                            score,  kl_loss = model(audio, mask)
                       
                        # 計算預測損失
                        prediction_loss = self.criterion(score[mask], gtscore[mask]).mean()
                       
                        # 計算 KL 損失
                        beta = self.config.beta 
                        total_loss = prediction_loss + beta * kl_loss
                       
                        scaler.scale(total_loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                       
                        # 直接存入預分配的陣列
                        loss_history[batch_count] = total_loss.detach().item()
                        kl_loss_history[batch_count] = kl_loss.detach().item()
                        batch_count += 1
               
                # 清理變數以節省記憶體
                # 每處理 3 個批次後清理一次記憶體
                if batch_idx % 3 == 0:
                    del data, visual, audio, gtscore, score, mask
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 計算實際批次的平均損失值
            if batch_count > 0:
                loss = np.mean(loss_history[:batch_count])
                kl_loss = np.mean(kl_loss_history[:batch_count]) if np.any(kl_loss_history[:batch_count]) else 0
                v_loss = np.mean(loss_v_history[:batch_count]) if np.any(loss_v_history[:batch_count]) else 0
                a_loss = np.mean(loss_a_history[:batch_count]) if np.any(loss_a_history[:batch_count]) else 0
            else:
                loss, kl_loss, v_loss, a_loss = 0, 0, 0, 0
               
            # 主動釋放記憶體
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            val_recon_loss=0
            val_kl_loss=0
            val_loss=0
            val_f1score=0
            val_map50=0
            val_map15=0
            val_precision=0
            val_f1score, val_map50, val_map15, val_loss, val_precision, val_loss_v, val_loss_a, val_kl_loss = self.evaluate(dataloader=self.val_loader)
           
            # 保存每次比例的日志
            proportion_dir = os.path.join(self.config.save_dir_root, f'logs/proportion_{int(proportion * 100)}')
            os.makedirs(proportion_dir, exist_ok=True)
            #
            f = open(os.path.join(proportion_dir, 'results.txt'), 'a', encoding='utf-8')
            print(f"proportion: {proportion}, type: {type(proportion)}")
            print(f"epoch_i: {epoch_i}, type: {type(epoch_i)}")

            f.write(f'[Proportion {str(proportion * 100)}% | Epoch {str(epoch_i+1)}], \n'
                    f'Train Loss: {loss:.5f}, Val Loss: {val_loss:.5f}, \n'
                    f'Visual loss: {v_loss:.5f}, Audio loss: {a_loss:.5f}\n'
                    f'Val F1 Score: {val_f1score:.5f}, Val MAP50: {val_map50:.5f}, \n'
                    f'Val MAP15: {val_map15:.5f}, KL loss: {kl_loss:.5f}\n')   
            f.flush()
            f.close()
            f = open(os.path.join(self.config.save_dir_root, 'logs/loss.txt'), 'a')
            f.write(f'Epoch: {epoch_i+1}, loss: {loss}, val_loss: {val_loss}\n')
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
           
            #print("   [Epoch {0}] Train loss: {1:.05f}".format(epoch_i+1, loss))
            #print('    VAL  F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5}'.format(val_f1score, val_map50, val_map15))
            print(f'[Proportion {str(proportion * 100)}% | Epoch {str(epoch_i+1)}], \n'
                        f'Train Loss: {loss:.5f}, Val Loss: {val_loss:.5f}, \n'
                        f'Visual loss: {v_loss:.5f}, Audio loss: {a_loss:.5f}\n'
                        f'Val F1 Score: {val_f1score:.5f}, Val MAP50: {val_map50:.5f}, \n'
                        f'Val MAP15: {val_map15:.5f}, KL loss: {kl_loss:.5f}\n') 
            f = open(os.path.join(self.config.save_dir_root, 'logs/all_result.txt'), 'a', encoding='utf-8')
            f.write(f'Val_f1score: {val_f1score}, Val_map50: {val_map50}, Val_map15: {val_map15}, Val_precision: {val_precision}\n')
            f.flush()
            f.close()
            #del data, visual, gtscore, audio, mask, score, weights, loss
            gc.collect()
        print(f'  [Proportion {int(proportion * 100)}%]')
        print('   Best Val F1 score {0:0.5} @ epoch{1}'.format(best_f1score, best_f1score_epoch+1))
        print('   Best Val MAP-50   {0:0.5} @ epoch{1}'.format(best_map50, best_map50_epoch+1))
        print('   Best Val MAP-15   {0:0.5} @ epoch{1}'.format(best_map15, best_map15_epoch+1))
        print('   Best Val PRECISION   {0:0.5} @ epoch{1}'.format(best_pre, best_pre_epoch+1))

        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write(f'    [Proportion {int(proportion * 100)}%]\n')
        f.write('   Best Val F1 score {0:0.5} @ epoch{1}\n'.format(best_f1score, best_f1score_epoch+1))
        f.write('   Best Val MAP-50   {0:0.5} @ epoch{1}\n'.format(best_map50, best_map50_epoch+1))
        f.write('   Best Val MAP-15   {0:0.5} @ epoch{1}\n'.format(best_map15, best_map15_epoch+1))
        f.write('   Best Val PRECISION   {0:0.5} @ epoch{1}\n\n'.format(best_pre, best_pre_epoch+1))
        f.flush()
        f.close()
           
        return path, path50, path100, path150, path200
        #return f1_save_ckpt_path, map50_save_ckpt_path, map15_save_ckpt_path

    def evaluate(self, dataloader=None):
        """ 評估模型在驗證或測試集上的性能

        :param dataloader: 要評估的數據加載器
        """
        model = self.model
        cuda_device = self.device
        model.eval()
       
        # 預先分配固定大小的陣列
        max_samples = len(dataloader)
        loss_history = np.zeros(max_samples, dtype=np.float32)
        kl_loss_history = np.zeros(max_samples, dtype=np.float32)
        loss_v_history = np.zeros(max_samples, dtype=np.float32)
        loss_a_history = np.zeros(max_samples, dtype=np.float32)
        fscore_history = np.zeros(max_samples, dtype=np.float32)
        map50_history = np.zeros(max_samples, dtype=np.float32)
        map15_history = np.zeros(max_samples, dtype=np.float32)
        precision_history = np.zeros(max_samples, dtype=np.float32)
       
        # 記錄實際處理的樣本數
        sample_count = 0
        dataloader = iter(dataloader)
       
        # 使用 tqdm 顯示進度，設置較長的更新間隔
        for sample_idx in tqdm(range(max_samples), desc="Evaluating", mininterval=2.0):
            try:
                data = next(dataloader)
            except StopIteration:
                # 數據迭代完成
                break
           
            # 使用 non_blocking=True 來提高數據傳輸效率
            visual = data['features'].to(cuda_device, non_blocking=True)
            gtscore = data['gtscore'].to(cuda_device, non_blocking=True)
            audio = data['audio'].to(cuda_device, non_blocking=True)
            input_mask = 'mask'
            #IB------------------------------------------------------------
            if self.config.modal == "visual":
                input_feature = visual
                if len(input_feature.shape) == 2:
                    seq = seq.unsqueeze(0)
                elif len(input_feature.shape) == 4:
                    input_feature = input_feature.squeeze(0)
                if len(gtscore.shape) == 1:
                    gtscore = gtscore.unsqueeze(0)
   
                B = input_feature.shape[0]
                mask=None
                if input_mask in data:
                    mask = data[input_mask].to(cuda_device)
                with torch.no_grad():
                    score,  kl_loss = model(input_feature, mask)
            elif self.config.modal == "audio":
                input_feature = audio
                if len(input_feature.shape) == 2:
                    seq = seq.unsqueeze(0)
                elif len(input_feature.shape) == 4:
                    input_feature = input_feature.squeeze(0)
                if len(gtscore.shape) == 1:
                    gtscore = gtscore.unsqueeze(0)
   
                B = input_feature.shape[0]
                mask=None
                if input_mask in data:
                    mask = data[input_mask].to(cuda_device)
                with torch.no_grad():
                    score,  kl_loss = model(input_feature, mask)
            # **计算预测损失**
                # 計算預測損失
                prediction_loss = self.criterion(score[mask], gtscore[mask]).mean()
            
                # 將 KL 損失加權後與預測損失結合
                beta = self.config.beta  # KL 損失的權重系數
                total_loss = prediction_loss + beta * kl_loss
                
                # 確保 GPU操作已經完成
                torch.cuda.synchronize() 
                
                # 記錄損失值
                loss_history[sample_count] = total_loss.detach().cpu().item()
                kl_loss_history[sample_count] = kl_loss.detach().cpu().item()
           
            loss = np.mean(np.array(loss_history))
            
            # 計算精確度指標
            predictions = (score > 0.5).float()  # 使用 0.5 閾值
            true_positives = (predictions[mask] * gtscore[mask]).sum().detach().item()
            predicted_positives = predictions[mask].sum().detach().item()
            precision = true_positives / (predicted_positives + 1e-7)  # 避免除零錯誤
            precision_history[sample_count] = precision
       
            # Summarization metric
            score = score.squeeze().cpu()
            gt_summary = data['gt_summary'][0]
            cps = data['change_points'][0]
            n_frames = data['n_frames']
            nfps = data['n_frame_per_seg'][0].tolist()
            picks = data['picks'][0].numpy()
            #print("score",len(score), "cps",len(cps), "nframe",len(n_frames), "nfps",len(nfps), "picks",len(picks))
            machine_summary = generate_summary(score, cps, n_frames, nfps, picks)
            #print("MACHINE", machine_summary, machine_summary.shape)
            #print("GT SUMMARY",gt_summary, gt_summary.shape)
            try:
                f_score, kTau, sRho = evaluate_summary(machine_summary, gt_summary, eval_method='avg')
            except:
                machine_summary = np.delete(machine_summary, -1)
                f_score, kTau, sRho = evaluate_summary(machine_summary, gt_summary, eval_method='avg')
            fscore_history[sample_count] = f_score

            # Highlight Detection Metric
            gt_seg_score = generate_mrhisum_seg_scores(gtscore.squeeze(0), uniform_clip=5)
            gt_top50_summary = top50_summary(gt_seg_score)
            gt_top15_summary = top15_summary(gt_seg_score)
           
            highlight_seg_machine_score = generate_mrhisum_seg_scores(score, uniform_clip=5)
            highlight_seg_machine_score = torch.exp(highlight_seg_machine_score) / (torch.exp(highlight_seg_machine_score).sum() + 1e-7)
           
            clone_machine_summary = highlight_seg_machine_score.clone().detach().cpu()
            clone_machine_summary = clone_machine_summary.numpy()
            aP50 = average_precision_score(gt_top50_summary, clone_machine_summary)
            aP15 = average_precision_score(gt_top15_summary, clone_machine_summary)
            map50_history[sample_count] = aP50
            map15_history[sample_count] = aP15
           
            # 增加計數器
            sample_count += 1
           
            # 每處理 3 個樣本清理一次記憶體
            if sample_idx % 3 == 0:
                gc.collect()

        # 只計算實際處理的樣本統計值
        if sample_count > 0:
            final_f_score = np.mean(fscore_history[:sample_count])
            final_map50 = np.mean(map50_history[:sample_count])
            final_map15 = np.mean(map15_history[:sample_count])
            final_precision = np.mean(precision_history[:sample_count])
            final_loss = np.mean(loss_history[:sample_count])
            final_prediction_loss_v = np.mean(loss_v_history[:sample_count]) if np.any(loss_v_history[:sample_count]) else 0
            final_prediction_loss_a = np.mean(loss_a_history[:sample_count]) if np.any(loss_a_history[:sample_count]) else 0
            final_kl_loss = np.mean(kl_loss_history[:sample_count]) if np.any(kl_loss_history[:sample_count]) else 0
        else:
            final_f_score = final_map50 = final_map15 = final_precision = final_loss = 0
            final_prediction_loss_v = final_prediction_loss_a = final_kl_loss = 0
           
        # 清理記憶體
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return final_f_score, final_map50, final_map15, final_loss, final_precision, final_prediction_loss_v, final_prediction_loss_a, final_kl_loss
           
    def test(self, ckpt_path):
        model=self.model
        cuda_device=self.device
        if ckpt_path != None:
            print("Testing Model: ", ckpt_path)
            print("Device: ",  cuda_device)
            model.load_state_dict(torch.load(ckpt_path))
        test_fscore, test_map50, test_map15, _, _, _, _, _ = self.evaluate(dataloader=self.test_loader)

        print("------------------------------------------------------")
        print(f"   TEST RESULT on {ckpt_path}: ")
        print('   TEST MRHiSum F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5}'.format(
            test_fscore, test_map50, test_map15
        ))
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


if __name__ == '__main__':
    # 啟用垃圾收集並先清理一次記憶體
    gc.enable()
    cleanup_memory()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # 固定隨機生成器種子
    g = torch.Generator()
    g.manual_seed(42)
    # 設定可控制的記憶體使用
    if torch.cuda.is_available():
        # 限制 PyTorch 快取的最大記憶體大小
        torch.cuda.set_per_process_memory_fraction(0.8)  # 使用最多 80% 的 GPU 記憶體
       
    # 全局禁用自動混合精度，確保使用完整精度
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
    # pylint: enable=protected-access

    # 建立命令行參數解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SL_module',
                        help='the name of the model')
    parser.add_argument('--epochs', type=int, default=200,
                        help='the number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='the learning rate')
    parser.add_argument('--precision', type=str, default='float32',
                        help='computation precision: float32 or float64')
    parser.add_argument('--fixed_precision', type=str2bool, default='true',
                        help='whether to use fixed precision computation')
    parser.add_argument('--memory_efficient', type=str2bool, default='true',
                        help='use memory-efficient training techniques')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='override default batch size for lower memory usage')
    parser.add_argument('--l2_reg', type=float,
                        default=1e-4, help='l2 regularizer')
    parser.add_argument('--dropout_ratio', type=float,
                        default=0.5, help='the dropout ratio')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='the batch size')
    parser.add_argument('--num_workers', type=int,
                        default=2, help='number of workers for data loading')
    parser.add_argument('--pin_memory', type=str2bool,
                        default='true', help='pin memory for faster GPU transfer')
    parser.add_argument('--deterministic', type=str2bool,
                        default='false', help='use deterministic algorithms')
    parser.add_argument('--tag', type=str, default='dev',
                        help='A tag for experiments')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='checkpoint path for inference or weight initialization')
    parser.add_argument('--train', type=str2bool,
                        default='true', help='when use Train')
    parser.add_argument('--path', type=str,
                        default='dataset/metadata_split.json', help='path')
    parser.add_argument('--device', type=str, default='0', help='gpu')
    parser.add_argument('--modal', type=str,
                        default='visual', help='visual,audio,multi')
    parser.add_argument('--beta', type=float, default=0, help='beta')
    parser.add_argument('--type', type=str, default='base',
                        help='base,ib,cib,eib,lib')  # cib,eib,lib

    opt = parser.parse_args()
    # print(type(opt))
    # print(opt)
    kwargs = vars(opt)
    config = Config(**kwargs)
    # 使用用戶指定的確定性設定執行隨機種子初始化
    set_seed(42, deterministic=config.fixed_precision)
    # 使用延遲式加載來減少記憶體使用
    print("Loading training dataset...")
    train_dataset = MrHiSumDataset(mode='train', path=config.path) if config.train else None
    # 優化 DataLoader 設定
    pin_memory = config.pin_memory and torch.cuda.is_available()
    num_workers = config.num_workers
    # 如果指定了記憶體效率模式，則調整批次大小
    if hasattr(config, 'memory_efficient') and config.memory_efficient:
        # 根據系統記憶體自動調整批次大小
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # 可用記憶體（GB）
        if available_memory < 8:  # 如果可用記憶體少於 8GB
            # 安全地設置批次大小（當為None時設為預設值）
            config.batch_size = 32 if config.batch_size is None else min(config.batch_size, 32)
            num_workers = min(num_workers, 2)  # 減少工作線程數
    # 只在需要時建立資料加載器
    train_loader = None
    if config.train:
        print("Creating training data loader...")
        # 安全檢查批次大小
        actual_batch_size = config.batch_size if config.batch_size else 64
       
        # 使用執行階段斷驗，會有助於清理中間記憶體
        train_loader = DataLoader(
            train_dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=BatchCollator(),
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None  # 預取因子，減少使用的記憶體
        )
       
        # 在加載資料集之間讓 CPU 和 GPU 有時間清理記憶體
        cleanup_memory()
       
        print("Loading validation dataset...")
        # 只在訓練時加載驗證集
        val_dataset = MrHiSumDataset(mode='val', path=config.path)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # 驗證集使用批次大小 1
            shuffle=False,
            num_workers=max(1, num_workers//2),  # 確保至少有一個工作線程
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None  # 預取因子
        )
    # 在建立測試資料加載器之前先清理記憶體
    cleanup_memory()
    print("Loading test dataset...")
    # 加載測試資料集以進行評估
    test_dataset = MrHiSumDataset(mode='test', path=config.path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 測試集始終使用批次大小 1
        shuffle=False,
        num_workers=max(1, num_workers//2),  # 確保至少有一個工作線程
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None  # 設定預取因子
    )
    # 檢查 CUDA 可用性並設定裝置
    if torch.cuda.is_available() and config.device.isdigit():
        cuda_device_id = int(config.device)
        device = torch.device(f"cuda:{cuda_device_id}")
       
        # 設定適當的 CUDA 裝置
        torch.cuda.set_device(cuda_device_id)
       
        # 顯示裝置信息
        print(f"Using CUDA device {cuda_device_id}: {torch.cuda.get_device_name(cuda_device_id)}")
        print(f"Total memory: {torch.cuda.get_device_properties(cuda_device_id).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU for computation")
    print("Device being used:", device)
    # Create context manager for optimizing memory
    amp_context = torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext()
    # 再次清理記憶體，以預備全新的環境建立模型
    cleanup_memory()
    print("Initializing solver...")
    solver = Solver(config, train_loader, val_loader,
                    test_loader, device, config.modal)

    solver.build()
    test_model_ckpt_path = None
    # Training mode
    if config.train:
        print("Starting training...")
        # 使用 try/finally 區塊確保程式結束時清理記憶體
        try:
            # Using amp_context for mixed precision training (faster computation)
            with amp_context:
                path, path50, path100, path150, path200, = solver.train()
           
            print("\nTesting best checkpoints...")
            # 組合所有檢查點路徑並去除重複
            all_paths = []
            for path_list in [path, path50, path100, path150, path200]:
                for ckpt_path in path_list:
                    if ckpt_path not in all_paths:
                        all_paths.append(ckpt_path)
           
            # 為了給系統時間清理記憶體，在每次測試之間加入記憶體清理
            for ckpt_path in all_paths:
                cleanup_memory()
                solver.test(ckpt_path)
               
            # 顯示最終結果摘要
            print("\nTraining completed.")
            print("Best checkpoints:")
            if path: print(f"Final: {path}")
            if path50: print(f"Epoch 50: {path50}")
            if path100: print(f"Epoch 100: {path100}")
        finally:
            # 確保大型對象被清理
            del solver
            cleanup_memory()
    # 測試模式
    else:
        test_model_ckpt_path = config.ckpt_path
        if test_model_ckpt_path is None:
            print("Trained model checkpoint required. Exit program")
            exit()
        else:
            try:
                print(f"Testing model: {test_model_ckpt_path}")
                solver.test(test_model_ckpt_path)
            finally:
                # 確保清理記憶體
                del solver
                cleanup_memory()
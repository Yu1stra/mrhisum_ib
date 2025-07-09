import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.signal import correlate
from networks.sl_module import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sl_module.transformer import Transformer, Transformer_with_ib, Transformer_with_ib_post
from networks.sl_module.score_net import ScoreFCN
from networks.sl_module.sl_module import InformationBottleneck_VIB
import math
from torch.nn.parameter import Parameter
import numpy as np
from torch.autograd import Variable

# ======================= 核心损失组件 =======================
class TimeAwareCost(nn.Module):
    """时间感知成本函数：量化序列间的几何差异"""
    def __init__(self, w1=0.4, w2=0.4, w3=0.2):
        super().__init__()
        self.w = [w1, w2, w3]  # 添加 KL 散度權重

    def soft_dtw(self, x, y, gamma=0.1):
        """改进版软DTW实现，确保正值输出（支持自动微分）"""
        # x, y: [seq_len, features]
        n, m = x.shape[0], y.shape[0]
        
        # 正规化序列，减小大值的影响
        x_norm = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        y_norm = y / (torch.norm(y, dim=1, keepdim=True) + 1e-8)
        
        # 处理维度问题，确保D始终是二维的
        if n == 1 and m == 1:
            # 如果两个序列各只有一个元素，直接计算距离
            D = torch.norm(x_norm - y_norm, p=2).unsqueeze(0).unsqueeze(0)
        elif n == 1:
            # 如果x只有一个元素
            D = torch.norm(x_norm.unsqueeze(0) - y_norm, p=2, dim=1).unsqueeze(0)
        elif m == 1:
            # 如果y只有一个元素
            D = torch.norm(x_norm - y_norm.unsqueeze(0), p=2, dim=1).unsqueeze(1)
        else:
            # 正常情况下使用cdist
            D = torch.cdist(x_norm, y_norm, p=2)  # [n, m]
        
        # 将距离缩放到合理范围，防止数值问题
        D = D / (D.max() + 1e-8)  # 范围0-1
        
        # 动态规划计算软路径
        R = torch.zeros(n+1, m+1, device=x.device) + float('inf')
        R[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = D[i-1, j-1]
                min_val = torch.min(torch.min(R[i-1, j], R[i, j-1]), R[i-1, j-1])
                R[i, j] = cost + min_val
        
        # 使用DTW的最终值作为距离
        dtw_dist = R[-1, -1]
        
        # 确保返回非负值
        return torch.abs(dtw_dist)
    
    def autocorr(self, z):
        """计算序列的自相关函数（ACF）"""
        # z: [seq_len, latent_dim]
        z_centered = z - torch.mean(z, dim=0)
        acf = torch.stack([
            torch.sum(z_centered[:-lag] * z_centered[lag:], dim=0) 
            for lag in range(1, 6)  # 计算前5个滞后
        ])  # [5, latent_dim]
        return acf / torch.max(torch.abs(acf), dim=0)[0]  # 归一化
    
    def forward(self, x_i, x_j, z_i, z_j, **kwargs):
        """
        计算两个序列对的Wasserstein传输成本
        x_i, x_j: [seq_len, input_dim] - 原始序列
        z_i, z_j: [seq_len, latent_dim] - 隐变量序列
        """
        # 判断是否为相同样本
        is_same_sample = torch.all(x_i == x_j) and torch.all(z_i == z_j)
        if is_same_sample:
            return torch.tensor(0.0, device=x_i.device)
            
        # 1. Soft-DTW计算原始序列距离
        d_time = self.soft_dtw(x_i, x_j)
        
        # 2. 隐空间欧氏距离
        d_latent = torch.norm(z_i - z_j, p=2) / z_i.shape[0]

        # 3. 自相关一致性
        acf_i = self.autocorr(z_i)
        acf_j = self.autocorr(z_j)
        d_acf = torch.norm(acf_i - acf_j, p=1)

        # 加权组合Wasserstein成本
        return (self.w[0] * d_time +
                self.w[1] * d_latent + 
                self.w[2] * d_acf)

def sinkhorn_cost(X, Z, cost_fn, epsilon=0.1, n_iters=50):
    """改進的 Sinkhorn 算法"""
    batch_size = X.shape[0]
    device = X.device
    
    # 使用 log-space 實現以提高數值穩定性
    def log_sum_exp(x, axis=-1):
        max_x = torch.max(x, axis=axis, keepdim=True)[0]
        return max_x + torch.log(torch.sum(
            torch.exp(x - max_x), axis=axis, keepdim=True))
    
    # 構建成本矩陣
    C = torch.zeros(batch_size, batch_size, device=device)
    for i in range(batch_size):
        for j in range(batch_size):
            C[i, j] = cost_fn(X[i], X[j], Z[i], Z[j])
    
    # 初始化 dual variables
    u = torch.zeros(batch_size, device=device)
    v = torch.zeros(batch_size, device=device)
    
    # Sinkhorn iterations in log-space
    K = torch.exp(-C / epsilon)
    for _ in range(n_iters):
        u = epsilon * (torch.log(torch.ones(batch_size, device=device) / batch_size) - 
                      torch.log(torch.sum(K * v.exp().unsqueeze(0), dim=1)))
        v = epsilon * (torch.log(torch.ones(batch_size, device=device) / batch_size) - 
                      torch.log(torch.sum(K * u.exp().unsqueeze(1), dim=0)))
    
    # 計算最終的傳輸計劃
    P = torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - C) / epsilon)
    
    # 計算 Wasserstein 距離
    W = torch.sum(P * C)
    
    return W

class WassersteinIBLoss(nn.Module):
    """整合几何感知压缩与动态稀疏保护的损失函数"""
    def __init__(self, beta=0.1, gamma=0.01):
        super().__init__()
        self.beta = beta      # Wasserstein压缩强度
        self.gamma = gamma    # 稀疏保护强度
        self.cost_fn = TimeAwareCost(0.4, 0.4, 0.2)

    def forward(self, scores, y_true, X, Z, mask=None):
        # Ensure total_loss is always a tensor for autograd
        device = 'cpu'
        if isinstance(scores, dict) and len(scores) > 0:
            first_key = next(iter(scores))
            device = scores[first_key].device
        else:
            device = 'cpu'
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_components = {}
        # Only iterate over modalities that are keys in scores (scores should be a dict)
        modalities = scores.keys() if isinstance(scores, dict) else []

        # If no modalities, fill default values
        if not modalities:
            for key in ['visual_mse', 'visual_wasserstein', 'visual_sparsity', 'visual_total']:
                loss_components[key] = 0.0
            return total_loss, loss_components

        for modality in modalities:
            # 1. 輸入檢查和預處理
            pred = scores[modality][mask] if mask is not None else scores[modality]
            target = y_true[mask] if mask is not None else y_true

            # 數據清理和正規化
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
            pred = torch.clamp(pred, min=-100, max=100)
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)

            # 2. MSE loss with safety checks
            pred_loss = F.mse_loss(pred, target, reduction='none')
            pred_loss = pred_loss.mean()

            # 3. Wasserstein loss with numerical stability
            X[modality] = F.normalize(X[modality], dim=-1)
            Z[modality] = F.normalize(Z[modality], dim=-1)
            w_loss = sinkhorn_cost(X[modality], Z[modality], self.cost_fn)
            w_loss = torch.clamp(w_loss, min=0.0, max=100.0)
            sparse_loss = dynamic_sparsity_regularizer(Z[modality])
            sparse_loss = torch.clamp(sparse_loss, min=0.0, max=100.0)
            modality_loss = (
                pred_loss +
                self.beta * torch.tanh(w_loss) +
                self.gamma * torch.tanh(sparse_loss)
            )
            weight = self.modality_weights[modality]
            weighted_loss = weight * modality_loss
            if not torch.isnan(weighted_loss) and not torch.isinf(weighted_loss):
                total_loss = total_loss + weighted_loss

            # Store components
            loss_components[f"{modality}_mse"] = pred_loss.item()
            loss_components[f"{modality}_wasserstein"] = w_loss.item()
            loss_components[f"{modality}_sparsity"] = sparse_loss.item()
            loss_components[f"{modality}_total"] = weighted_loss.item()

        return total_loss, loss_components


# 定义多模态简单模型
class SimpleMultiModalModel(nn.Module):
    def __init__(self, visual_dim=1024, audio_dim=128, hidden_dim=256, bottleneck_dim=128, num_heads=4):
        super().__init__()
        # Visual pathway
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.visual_bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )

        # Audio pathway
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.audio_bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )

        # Multi-head attention for cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Decoders
        self.visual_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.audio_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.multi_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, visual, audio):
        batch_size = visual.shape[0]

        # Encode and create bottlenecks
        visual_h = self.visual_encoder(visual)
        visual_z = self.visual_bottleneck(visual_h)

        audio_h = self.audio_encoder(audio)
        audio_z = self.audio_bottleneck(audio_h)

        # Cross-modal attention
        attn_output, _ = self.cross_attention(
            visual_z, audio_z, audio_z
        )

        # Fusion
        multi_z = self.fusion_layer(
            torch.cat([visual_z, attn_output], dim=-1)
        )

        # Generate scores
        visual_score = self.visual_decoder(visual_z).squeeze(-1)
        audio_score = self.audio_decoder(audio_z).squeeze(-1)
        multi_score = self.multi_decoder(multi_z).squeeze(-1)

        scores = {
            'visual': visual_score,
            'audio': audio_score,
            'multi': multi_score
        }

        latents = {
            'visual': visual_z,
            'audio': audio_z,
            'multi': multi_z
        }

        return scores, latents


class SL_module_VIB(nn.Module):
    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_VIB, self).__init__()

        # 加入 IB 层
        self.ib = InformationBottleneck_VIB(input_dim)

        # Transformer 处理瓶颈后的特征
        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)

        # Score 计算
        self.score_model = ScoreFCN(emb_dim=input_dim)

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys():
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_VIB, self).load_state_dict(state_dict)

    def forward(self, x, mask):

        # **Transformer 提取特征**\
        z = self.ib(x)
        transformed_emb = self.transformer(z)
        # **信息瓶颈处理**

        # **最终评分**
        score = self.score_model(transformed_emb).squeeze(-1)
        score = torch.sigmoid(score)
        kl_loss = self.ib.kl_closed_form(x)

        return score, z  # 返回 KL 损失所需的 mu & logvar

def test_with_simple_model():
    """
    使用 SL_module_VIB 和 WassersteinIBLoss 進行測試 (僅視覺特徵)
    """
    print("\n-----使用 SL_module_VIB 和 WassersteinIBLoss 測試 (視覺特徵)-----")
    
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset

    print("正在加載數據集...")
    from model.mrhisum_dataset_fixed import MrHiSumDataset, BatchCollator

    # 創建小型數據集
    dataset = MrHiSumDataset(mode="train", path='dataset/mr_hisum_split.json')
    print(f"完整數據集大小: {len(dataset)}")

    # 只使用前10個樣本進行測試
    num_samples = min(10, len(dataset))
    indices = list(range(num_samples))
    small_dataset = Subset(dataset, indices)
    print(f"使用的樣本數量: {num_samples}")

    # 配置 dataloader
    batch_size = 2
    dataloader = DataLoader(
        small_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=BatchCollator(),
        num_workers=0
    )
    print(f"Batch size: {batch_size}")
    print(f"每個epoch的batch數量: {len(dataloader)}")

    # 初始化模型
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    model = SL_module_VIB(
        input_dim=1024,  # 視覺特徵維度
        depth=5,
        heads=8,
        mlp_dim=2048,
        dropout_ratio=0.1
    ).to(device)

    # 初始化優化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 初始化損失函數 (僅使用視覺模態的權重)
    loss_fn = WassersteinIBLoss(
        beta=0.1,
        gamma=0.01,
        modality_weights={'visual': 1.0}  # 只使用視覺模態
    )

    # 開始訓練
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(dataloader):
            print(f"\n處理批次 {batch_idx+1}/{len(dataloader)}")

            # 只準備視覺特徵數據
            visual = batch['features'].to(device)
            gtscore = batch['gtscore'].to(device)
            mask = batch['mask'].to(device)

            print(f"Batch 形狀: visual={visual.shape}, gtscore={gtscore.shape}")

            # 清零梯度
            optimizer.zero_grad()

            # 前向傳播 (只處理視覺特徵)
            scores, latents = model(visual, mask)

            # 準備特徵字典
            X = {'visual': visual}

            # 計算損失
            loss, components = loss_fn(
                scores=scores,
                y_true=gtscore,
                X=X,
                Z=latents,
                mask=mask
            )

            # 反向傳播
            loss.backward()

            # 更新參數
            optimizer.step()
            print(components)
            # 輸出損失組件
            print(f"\n當前批次損失組件:")
            if 'visual_sparsity' in components:
                print(f"  Sparsity Loss: {components['visual_sparsity']:.6f}")
            else:
                print(f"  Sparsity Loss: [Key 'visual_sparsity' not found, available: {components}]")
            print(f"  Total Loss: {loss.item():.6f}")

            epoch_loss += loss.item()
            batch_count += 1


            # 打印 epoch 平均損失
            avg_loss = epoch_loss / max(1, batch_count)
            print(f"\nEpoch {epoch+1}/{num_epochs} 完成")
            print(f"平均損失: {avg_loss:.6f}")

        print("\n模型訓練完成!")


if __name__ == "__main__":
    #test_wasserstein_ib_loss()
    # 嘗試使用真實數據進行測試
    #test_with_real_data()
    # 使用合成數據測試
    #test_synthetic_data()
    # 測試個別組件
    #test_individual_components()
    # 執行數值梯度檢查
    #numerical_gradient_check()
    # 檢查中間計算值
    #test_intermediate_values()
    # 使用簡單模型測試
    test_with_simple_model()
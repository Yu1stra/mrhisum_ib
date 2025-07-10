import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sl_module.transformer import Transformer, Transformer_with_ib, Transformer_with_ib_post, Transformer_with_ib_after_ff
from networks.sl_module.score_net import ScoreFCN
import math
from torch.nn.parameter import Parameter
import numpy as np
from torch.autograd import Variable

# ------------------------使用multi-modality via information bottleneck 的IB layer 設計---------------------
class InformationBottleneck(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(InformationBottleneck, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, input_dim),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) ) 
        self.fc_mu = nn.Linear(input_dim, bottleneck_dim)  # 均值映射
        self.fc_std = nn.Linear(input_dim, bottleneck_dim)  # 方差映射
        self.decoder = nn.Linear(bottleneck_dim, input_dim)  # 还原

    def reparameterize(self, mu, std):
        #变分重参数技巧 (Reparameterization Trick) 
        #logvar = torch.clamp(logvar, min=-10, max=10)
        #std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 采样标准正态分布
        return mu + eps * std  # 生成随机样本
        
    def encode(self, x):
        #获取瓶颈层的均值和方差 
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = F.softplus(self.fc_std(x) - 5, beta=1)
        return mu, logvar
        
    def forward(self, x):
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)  # 采样瓶颈特征
        x_reconstructed = self.decoder(z)  # 还原输入
        return z, x_reconstructed, mu, std  # 额外返回 mu 和 logvar 用于计算 KL 损失

def kl_divergence_a(mu, std):
    #计算 KL 散度，使分布逼近标准正态 
    #KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2 * std.log() - 1)
    #KL = -0.5 * torch.sum(1+std - mu.pow(2) - std.exp(), dim=-1).mean()
    # 标准KL散度（与正态分布）
    KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - torch.log(std.pow(2)) - 1)
    return KL
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


#--------------------------使用wasserstein loss取代IB中kl loss--------------------------------
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

    C = torch.zeros(batch_size, batch_size, device=device)
    for i in range(batch_size):
        for j in range(batch_size):
            C[i, j] = cost_fn(X[i], X[j], Z[i], Z[j])

    u = torch.zeros(batch_size, device=device)
    v = torch.zeros(batch_size, device=device)

    K = torch.exp(-C / epsilon)
    for _ in range(n_iters):
        u = epsilon * (torch.log(torch.ones(batch_size, device=device) / batch_size) -
                      torch.log(torch.sum(K * v.exp().unsqueeze(0), dim=1)))
        v = epsilon * (torch.log(torch.ones(batch_size, device=device) / batch_size) -
                      torch.log(torch.sum(K * u.exp().unsqueeze(1), dim=0)))

    P = torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - C) / epsilon)
    W = torch.sum(P * C)

    return W

class SL_module_SWD(nn.Module):
    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_SWD, self).__init__()

        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.score_model = ScoreFCN(emb_dim=input_dim)

        # Replace simple encoder with InformationBottleneck
        self.encoder = InformationBottleneck(input_dim=input_dim, bottleneck_dim=input_dim)

    def compute_swd(self, x, z, num_projections=50):
        """计算 Sliced Wasserstein Distance"""
        # 获取维度信息
        batch_size, seq_len, dim = x.shape
        device = x.device

        # 確保 z 的維度與 x 相同
        if z.shape != x.shape:
            raise ValueError(f"Shape mismatch: x shape is {x.shape}, z shape is {z.shape}")

        # 生成隨機投影向量
        theta = torch.randn(num_projections, dim, device=device)
        theta = theta / torch.norm(theta, dim=1, keepdim=True)

        # 將輸入展平為 2D 張量以便投影
        x_flat = x.reshape(-1, dim)  # (batch_size * seq_len, dim)
        z_flat = z.reshape(-1, dim)  # (batch_size * seq_len, dim)

        # 計算投影
        x_proj = torch.mm(x_flat, theta.t())  # (batch_size * seq_len, num_projections)
        z_proj = torch.mm(z_flat, theta.t())  # (batch_size * seq_len, num_projections)

        # 分別為每個批次計算
        total_distance = 0
        for i in range(batch_size):
            start_idx = i * seq_len
            end_idx = (i + 1) * seq_len

            # 獲取當前批次的投影
            x_proj_batch = x_proj[start_idx:end_idx]  # (seq_len, num_projections)
            z_proj_batch = z_proj[start_idx:end_idx]  # (seq_len, num_projections)

            # 檢查張量是否為空
            if x_proj_batch.numel() == 0 or z_proj_batch.numel() == 0:
                continue

            # 在序列維度上排序
            x_sort = torch.sort(x_proj_batch, dim=0)[0]  # (seq_len, num_projections)
            z_sort = torch.sort(z_proj_batch, dim=0)[0]  # (seq_len, num_projections)

            # 計算每個批次的距離
            batch_distance = torch.mean((x_sort - z_sort) ** 2)
            total_distance += batch_distance

        # 計算平均距離（避免除以零）
        num_valid_batches = max(batch_size, 1)
        wasserstein_dist = total_distance / num_valid_batches

        return wasserstein_dist

    def forward(self, x, mask=None):
        # Get encoded features using InformationBottleneck
        z, x_reconstructed, mu, std = self.encoder(x)

        # Use transformer to process encoded features
        transformed_emb = self.transformer(z)

        # Calculate final score
        score = self.score_model(transformed_emb).squeeze(-1)
        score = torch.sigmoid(score)

        # Calculate Sliced Wasserstein Distance as regularization
        wasserstein_loss = self.compute_swd(x, z)

        return score, wasserstein_loss

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys():
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_SWD, self).load_state_dict(state_dict)

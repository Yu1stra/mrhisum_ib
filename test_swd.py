import torch
import numpy as np
from model.mrhisum_dataset import MrHiSumDataset, BatchCollator
from networks.sl_module.sl_module import SL_module_VIB
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm


def get_random_projections(dimension, num_projections):
    """
    Generate random projection vectors on a unit sphere.

    Args:
        dimension (int): Dimensionality of the space
        num_projections (int): Number of random projections to generate

    Returns:
        torch.Tensor: Random projection vectors of shape (num_projections, dimension)
    """
    # Generate random vectors
    projections = torch.randn(num_projections, dimension)
    # Normalize the vectors to lie on unit sphere
    projections = projections / torch.norm(projections, dim=1, keepdim=True)
    return projections

def sliced_wasserstein_distance(distribution1, distribution2, num_projections=50, device='cuda'):
    """
    Calculate the Sliced Wasserstein Distance between two video feature distributions.

    Args:
        distribution1 (torch.Tensor): First distribution samples (batch_size, seq_len, dimension)
        distribution2 (torch.Tensor): Second distribution samples (batch_size, seq_len, dimension)
        num_projections (int): Number of random projections to use
        device (str): Device to use for computation ('cuda' or 'cpu')

    Returns:
        torch.Tensor: The computed Sliced Wasserstein Distance
    """
    # Move inputs to specified device if they're not already there
    distribution1 = distribution1.to(device)
    distribution2 = distribution2.to(device)

    # Get shapes
    batch_size, seq_len, dimension = distribution1.size()

    # Reshape distributions to (batch_size * seq_len, dimension)
    flat_dist1 = distribution1.reshape(-1, dimension)
    flat_dist2 = distribution2.reshape(-1, dimension)

    # Generate random projections
    projections = get_random_projections(dimension, num_projections).to(device)

    # Project the distributions onto random directions
    proj_dist1 = torch.matmul(flat_dist1, projections.t())  # (batch_size * seq_len, num_projections)
    proj_dist2 = torch.matmul(flat_dist2, projections.t())  # (batch_size * seq_len, num_projections)

    # Reshape projections back to include sequence dimension
    proj_dist1 = proj_dist1.view(batch_size, seq_len, num_projections)
    proj_dist2 = proj_dist2.view(batch_size, seq_len, num_projections)

    # Sort the projected distributions along the sequence dimension
    sorted_proj1, _ = torch.sort(proj_dist1, dim=1)
    sorted_proj2, _ = torch.sort(proj_dist2, dim=1)

    # Compute L2 distance between sorted projections
    # Average over sequence length and projections
    wasserstein_distance = torch.mean((sorted_proj1 - sorted_proj2) ** 2)

    return wasserstein_distance

class SlicedWassersteinLoss(torch.nn.Module):
    """
    Sliced Wasserstein Distance Loss module for video features.
    """
    def __init__(self, num_projections=50, device='cuda'):
        super().__init__()
        self.num_projections = num_projections
        self.device = device

    def forward(self, predicted_distribution, target_distribution):
        """
        Compute the Sliced Wasserstein Loss between predicted and target distributions.

        Args:
            predicted_distribution (torch.Tensor): Predicted distribution (batch_size, seq_len, dimension)
            target_distribution (torch.Tensor): Target distribution (batch_size, seq_len, dimension)

        Returns:
            torch.Tensor: The computed loss
        """
        return sliced_wasserstein_distance(
            predicted_distribution,
            target_distribution,
            num_projections=self.num_projections,
            device=self.device
        )

# Example usage:
if __name__ == "__main__":
    import torch.utils.data as data
    device = torch.device("cuda:5")
    # 初始化数据集和数据加载器
    data_path = 'dataset/mr_hisum_split.json'
    batch_size = 128
    train_dataset = MrHiSumDataset(mode='train', path=data_path)
    val_dataset = MrHiSumDataset(mode='val', path=data_path)
    test_dataset = MrHiSumDataset(mode='test', path=data_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              collate_fn=BatchCollator())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


    # 初始化 SWD loss
    swd_loss = SlicedWassersteinLoss(num_projections=100)
    # 获取一个批次的数据
    for batch in test_loader:
        # 假设我们要比较视频特征的分布
        # 这里我们使用同一批次的特征，但对其中一个进行扰动来模拟两个不同的分布
        features = batch['features']  # 应该是 (batch_size, seq_len, dim) 的形状

        # 创建一个扰动的版本作为目标分布

        perturbed_features = features + 0.5 * torch.randn_like(features)

        # 计算 SWD loss
        loss = swd_loss(features, perturbed_features)
        print(f"Sliced Wasserstein Loss for real video features: {loss.item()}")

        # 打印特征的形状以验证
        print(f"Feature shape: {features.shape}")
        break  # 只测试一个批次

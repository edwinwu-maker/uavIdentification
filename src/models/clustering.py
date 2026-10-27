"""DEC/DCEC 的聚类层与目标分布计算。"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClusteringLayer(nn.Module):
    """DEC/DCEC 的 Student-t 软分配层；B 为样本数，K 为簇数，D 为潜在向量维数。"""

    def __init__(self, num_clusters: int, latent_dim: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        # 可训练的簇中心：(K, D)，其中 K = num_clusters，D = latent_dim。
        self.centers = nn.Parameter(torch.empty(num_clusters, latent_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """输入潜在向量 (B, D)，返回各样本对 K 个簇的软分配概率 (B, K)。"""
        # (B, 1, D) 与 (K, D) 广播为 (B, K, D)，沿 D 求和得到平方距离 (B, K)。
        distance = torch.sum((latent.unsqueeze(1) - self.centers) ** 2, dim=2)
        # Student-t 权重为 (B, K)，随后沿簇维度归一化，使每个样本的分配概率之和为 1。
        numerator = (1.0 + distance / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        return numerator / numerator.sum(dim=1, keepdim=True).clamp_min(1e-12)


def target_distribution(soft_assignments: torch.Tensor) -> torch.Tensor:
    """输入软分配概率 (B, K)，返回同形状的 DEC 目标分布；B 为样本数，K 为簇数。"""

    # 沿样本维度求和得到簇频率 (K,)，广播相除后权重仍为 (B, K)。
    weights = soft_assignments.square() / soft_assignments.sum(dim=0).clamp_min(1e-12)
    # 每行权重和为 (B, 1)，沿簇维度归一化得到目标分布 (B, K)。
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

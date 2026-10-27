"""组合自编码器与聚类层的 STFT DCEC 模型。"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.autoencoder import StftAutoencoder
from src.models.clustering import ClusteringLayer


class StftDcec(nn.Module):
    """STFT DCEC 模型；B 为批次大小，D = autoencoder.latent_dim，K = num_clusters。"""

    def __init__(self, autoencoder: StftAutoencoder, num_clusters: int) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.clustering = ClusteringLayer(num_clusters, autoencoder.latent_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """输入单通道 STFT (B, 1, 256, 256)，按以下顺序返回：

        reconstruction: 重建图 (B, 1, 256, 256)。
        latent: 自编码器输出的潜在向量 (B, D)。
        soft_assignments: 各样本对 K 个簇的软分配概率 (B, K)，每行概率之和为 1。
        """
        reconstruction, latent = self.autoencoder(inputs)
        return reconstruction, latent, self.clustering(latent)

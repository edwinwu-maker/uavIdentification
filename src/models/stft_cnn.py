"""用于 STFT DeepCluster 的独立 CNN 特征提取器。"""

from __future__ import annotations

import torch
import torch.nn as nn


class StftCNN(nn.Module):
    """五层 Conv-BN-ReLU 与全局平均池化，用于快速验证深度聚类。"""

    def __init__(self, latent_dim: int = 128, in_channels: int = 1) -> None:
        super().__init__()
        if in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        channels = (in_channels, 32, 64, 128, 256, 256)
        self.encoder_conv = nn.Sequential(*(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for in_channels, out_channels in zip(channels[:-1], channels[1:])
        ))
        # 256×256 输入经五次下采样得到 256×8×8，全局汇聚后投影为有符号特征。
        self.encoder_pool = nn.AdaptiveAvgPool2d(1)
        self.encoder_fc = nn.Linear(256, latent_dim)
        # 首轮编码使用 BN 的初始运行统计，避免默认初始化使低幅值特征逐层衰减。
        for module in self.encoder_conv.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """输入 (B, in_channels, 256, 256)，返回 (B, latent_dim) 特征。"""
        return self.encoder_fc(self.encoder_pool(self.encoder_conv(inputs)).flatten(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encode(inputs)

"""监督分类与 STFT DeepCluster 使用的两个独立网络。"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

from src.data.drone_rfa_io import RESEARCH_LABELS

NUM_CLASSES = len(RESEARCH_LABELS)


class DroneRFaResNet18(nn.Module):
    """ResNet-18 adapted for DroneRFa single-channel input (CPP / STFT).

    Input: (B, 1, H, W) — feature map from the selected RF channel.
    Output: (B, num_classes) — 默认 16 类：背景与 15 类无人机。
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = resnet18(weights=None)
        # Replace first conv: 3-channel RGB → single-channel feature
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # 按当前研究类别数设置分类层。
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class StftCNN(nn.Module):
    """五层 Conv-BN-ReLU 与全局平均池化，用于快速验证深度聚类。"""

    # latent_dim 为输出特征维度（默认 128），in_channels 为输入通道数（灰度 1、RGB 3）。
    def __init__(self, latent_dim: int = 128, in_channels: int = 1) -> None:
        super().__init__()
        if in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        channels = (in_channels, 32, 64, 128, 256, 256)
        # 堆叠五层步长为 2 的 Conv-BN-ReLU，逐级下采样并提取时频特征。
        self.encoder_conv = nn.Sequential(*(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            # channels[:-1] -> (1, 32, 64, 128, 256)
            # channels[1:] -> (32, 64, 128, 256, 256)
            for in_channels, out_channels in zip(channels[:-1], channels[1:])
        ))
        # 256×256 输入经五次下采样得到 256×8×8，全局汇聚后投影为有符号特征。
        self.encoder_pool = nn.AdaptiveAvgPool2d(1)
        self.encoder_fc = nn.Linear(256, latent_dim)
        # 给 CNN 的五个卷积层设置训练前的初始权重
        for module in self.encoder_conv.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """输入 (B, in_channels, 256, 256)，返回 (B, latent_dim) 特征。"""
        return self.encoder_fc(self.encoder_pool(self.encoder_conv(inputs)).flatten(1))

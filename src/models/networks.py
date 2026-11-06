"""监督分类与 STFT DeepCluster 使用的三个独立网络。"""

from __future__ import annotations

from functools import partial

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


class StftResNet18(nn.Module):
    """小卷积入口、16 倍下采样与 GroupNorm，不包含聚类分类头。"""

    def __init__(self, latent_dim: int = 128, in_channels: int = 1) -> None:
        super().__init__()
        if in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.backbone = resnet18(
            weights=None, num_classes=latent_dim, norm_layer=partial(nn.GroupNorm, 8),
        )
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False,
        )
        # 替换后的卷积不经过 torchvision 构造时的初始化，需单独初始化。
        nn.init.kaiming_normal_(self.backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.backbone.maxpool = nn.Identity()
        # 256×256 输入的末层为 512×16×16；保留时频粗分区，再投影为有符号特征。
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.backbone.fc = nn.Linear(512 * 2 * 2, latent_dim)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """输入 (B, in_channels, 256, 256)，返回 (B, latent_dim) 特征。"""
        return self.backbone(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encode(inputs)

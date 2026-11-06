"""保留粗略时频位置的 STFT DeepCluster ResNet-18 特征提取器。"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from torchvision.models import resnet18


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

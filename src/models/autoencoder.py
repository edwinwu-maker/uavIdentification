"""Convolutional autoencoder for STFT reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn


def _encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """
    输入 (B, in_channels, H, W)
    输出 (B, out_channels, H // 2, W // 2)
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """
    输入 (B, in_channels, H, W)
    输出 (B, out_channels, 2H, 2W)
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class StftAutoencoder(nn.Module):
    """单通道 STFT 自编码器；形状中的 B 表示批次大小，latent_dim 表示潜在向量维数。"""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            _encoder_block(1, 32),     # (B, 1, 256, 256) → (B, 32, 128, 128)
            _encoder_block(32, 64),    # → (B, 64, 64, 64)
            _encoder_block(64, 128),   # → (B, 128, 32, 32)
            _encoder_block(128, 256),  # → (B, 256, 16, 16)
            _encoder_block(256, 256),  # → (B, 256, 8, 8)
        )

        # 卷积特征展平后编码为潜在向量：(B, 16384) → (B, latent_dim)。
        self.encoder_fc = nn.Linear(256 * 8 * 8, latent_dim)
        # 潜在向量映射回展平特征：(B, latent_dim) → (B, 16384)。
        self.decoder_fc = nn.Linear(latent_dim, 256 * 8 * 8)

        self.decoder_conv = nn.Sequential(
            _decoder_block(256, 256),  # (B, 256, 8, 8) → (B, 256, 16, 16)
            _decoder_block(256, 128),  # → (B, 128, 32, 32)
            _decoder_block(128, 64),   # → (B, 64, 64, 64)
            _decoder_block(64, 32),    # → (B, 32, 128, 128)
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # → (B, 1, 256, 256)
            nn.Tanh(),  # 形状不变，重建值范围为 [-1, 1]。
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """输入 STFT (B, 1, 256, 256)，返回潜在向量 (B, latent_dim)。"""
        encoded = self.encoder_conv(inputs)
        return self.encoder_fc(encoded.flatten(1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """输入潜在向量 (B, latent_dim)，返回重建图 (B, 1, 256, 256)。"""
        decoded = self.decoder_fc(latent).reshape(-1, 256, 8, 8)
        return self.decoder_conv(decoded)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """输入 (B, 1, 256, 256)，依次返回重建图 (B, 1, 256, 256) 和潜在向量 (B, latent_dim)。"""
        latent = self.encode(inputs)
        return self.decode(latent), latent

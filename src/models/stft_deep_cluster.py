"""Convolutional autoencoder and DCEC model for STFT clustering."""

from __future__ import annotations

import torch
import torch.nn as nn


def _encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class StftAutoencoder(nn.Module):
    """Encode a 256x256 single-channel STFT into a compact latent vector."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            _encoder_block(1, 32),
            _encoder_block(32, 64),
            _encoder_block(64, 128),
            _encoder_block(128, 256),
            _encoder_block(256, 256),
        )
        self.encoder_fc = nn.Linear(256 * 8 * 8, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder_conv = nn.Sequential(
            _decoder_block(256, 256),
            _decoder_block(256, 128),
            _decoder_block(128, 64),
            _decoder_block(64, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_conv(inputs)
        return self.encoder_fc(encoded.flatten(1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_fc(latent).reshape(-1, 256, 8, 8)
        return self.decoder_conv(decoded)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(inputs)
        return self.decode(latent), latent


class ClusteringLayer(nn.Module):
    """Student-t soft assignment layer used by DEC/DCEC."""

    def __init__(self, num_clusters: int, latent_dim: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.centers = nn.Parameter(torch.empty(num_clusters, latent_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        distance = torch.sum((latent.unsqueeze(1) - self.centers) ** 2, dim=2)
        numerator = (1.0 + distance / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        return numerator / numerator.sum(dim=1, keepdim=True).clamp_min(1e-12)


class StftDcec(nn.Module):
    def __init__(self, autoencoder: StftAutoencoder, num_clusters: int) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.clustering = ClusteringLayer(num_clusters, autoencoder.latent_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction, latent = self.autoencoder(inputs)
        return reconstruction, latent, self.clustering(latent)


def target_distribution(soft_assignments: torch.Tensor) -> torch.Tensor:
    """Compute the sharpened DEC target distribution."""

    weights = soft_assignments.square() / soft_assignments.sum(dim=0).clamp_min(1e-12)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

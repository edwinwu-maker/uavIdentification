"""BYOL model for single-channel 512x512 STFT matrices."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def _encoder() -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
    model.fc = nn.Identity()
    return model


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


class StftByol(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.online_encoder = _encoder()
        self.online_projector = _mlp(512, 1024, 256)
        self.predictor = _mlp(256, 1024, 256)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        self.classifier = nn.Linear(512, 2)
        for parameter in self.target_parameters():
            parameter.requires_grad = False

    def target_parameters(self):
        return [*self.target_encoder.parameters(), *self.target_projector.parameters()]

    def online_parameters(self):
        return [*self.online_encoder.parameters(), *self.online_projector.parameters(),
                *self.predictor.parameters()]

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(inputs)

    def classify(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(inputs))

    def byol_loss(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first_encoding = self.online_encoder(first)
        second_encoding = self.online_encoder(second)
        first_prediction = self.predictor(self.online_projector(first_encoding))
        second_prediction = self.predictor(self.online_projector(second_encoding))
        with torch.no_grad():
            first_target = self.target_projector(self.target_encoder(first))
            second_target = self.target_projector(self.target_encoder(second))
        first_loss = 2 - 2 * F.cosine_similarity(first_prediction, second_target.detach(), dim=1)
        second_loss = 2 - 2 * F.cosine_similarity(second_prediction, first_target.detach(), dim=1)
        encodings = torch.cat((first_encoding, second_encoding)).detach()
        return (first_loss + second_loss).mean(), encodings

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        online = [*self.online_encoder.parameters(), *self.online_projector.parameters()]
        for target, source in zip(self.target_parameters(), online):
            target.mul_(momentum).add_(source, alpha=1.0 - momentum)
        target_buffers = [*self.target_encoder.buffers(), *self.target_projector.buffers()]
        online_buffers = [*self.online_encoder.buffers(), *self.online_projector.buffers()]
        for target, source in zip(target_buffers, online_buffers):
            if torch.is_floating_point(target):
                target.mul_(momentum).add_(source, alpha=1.0 - momentum)
            else:
                target.copy_(source)

    def train(self, mode: bool = True):
        super().train(mode)
        # target 分支始终使用稳定的推理态统计量。
        self.target_encoder.eval()
        self.target_projector.eval()
        return self

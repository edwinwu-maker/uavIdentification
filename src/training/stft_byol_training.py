"""Training utilities for STFT BYOL."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim != 4 or inputs.shape[1] != 1:
        raise ValueError(f"Expected (B, 1, H, W), got {tuple(inputs.shape)}")
    output = []
    for item in inputs:
        shifts = tuple(int(value) for value in torch.randint(-8, 9, (2,), device=inputs.device))
        shifted = torch.roll(item, shifts=shifts, dims=(-2, -1))
        contrast = torch.empty((), device=inputs.device).uniform_(0.9, 1.1)
        output.append((shifted * contrast + torch.randn_like(shifted) * 0.015).clamp(-1, 1))
    result = torch.stack(output)
    if not torch.isfinite(result).all():
        raise ValueError("Augmentation produced non-finite values")
    return result


def ema_momentum(step: int, total_steps: int, base: float = 0.996) -> float:
    if not 0 <= step < total_steps or not 0 <= base < 1:
        raise ValueError("Invalid EMA schedule arguments")
    return 1.0 - (1.0 - base) * (math.cos(math.pi * step / total_steps) + 1.0) / 2.0


def representation_stats(features: torch.Tensor) -> dict[str, float]:
    # 在 CPU 计算诊断量，避免不同 CUDA/MPS 后端的 SVD 支持差异。
    normalized = F.normalize(features.detach().float().cpu(), dim=1)
    std = float(normalized.std(dim=0).mean())
    similarity = normalized @ normalized.T
    mask = ~torch.eye(len(normalized), dtype=torch.bool, device=normalized.device)
    mean_cosine = float(similarity[mask].mean()) if len(normalized) > 1 else 1.0
    centered = normalized - normalized.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular.square()
    probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
    effective_rank = float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))
    return {"embedding_std": std, "mean_cosine_similarity": mean_cosine,
            "effective_rank": effective_rank}


class BalancedSampler(Sampler[int]):
    def __init__(self, labels: list[int], seed: int) -> None:
        values = np.asarray(labels)
        self.groups = {label: np.flatnonzero(values == label) for label in (0, 1)}
        if any(len(group) == 0 for group in self.groups.values()):
            raise ValueError("Balanced sampling requires both classes")
        self.size = max(len(group) for group in self.groups.values())
        self.seed, self.epoch = seed, 0

    def __len__(self) -> int:
        return self.size * 2

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        draws = {
            label: rng.choice(group, self.size, replace=len(group) < self.size)
            for label, group in self.groups.items()
        }
        return iter([int(draws[label][index]) for index in range(self.size) for label in (0, 1)])

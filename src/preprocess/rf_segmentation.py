"""ST-ESER based RF predominant segment selection."""

from __future__ import annotations

import math

import torch

from src.utils.device import default_device


def segment_predominant_rf(
    x: torch.Tensor,
    *,
    frame_len: int,
    target_len: int | None = 100_000,
    top_k: int | None = None,
    sort_by_time: bool = True,
    eps: float = 1e-12,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select predominant RF frames by short-time energy-to-spectral-entropy ratio."""

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if frame_len <= 0:
        raise ValueError("frame_len must be positive")
    if top_k is None:
        if target_len is None:
            raise ValueError("top_k or target_len must be provided")
        if target_len <= 0:
            raise ValueError("target_len must be positive")
        top_k = math.ceil(target_len / frame_len)
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    input_dtype = x.dtype
    requested_device = default_device() if device is None else device
    # MPS 对 complex FFT 支持不稳定；ST-ESER 预选段保持 CPU 计算，后续 FAM 仍可用原 device。
    compute_device = "cpu" if requested_device.lower().startswith("mps") else requested_device
    signal = x.to(compute_device)

    squeeze_batch = False
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)    # (signal_length,) -> (1, signal_length)
        squeeze_batch = True
    if signal.ndim != 2:
        raise ValueError("x must be a one-dimensional or two-dimensional RF sequence")

    batch_size, sample_count = signal.shape
    frame_count = sample_count // frame_len
    if frame_count == 0:
        raise ValueError("x is shorter than frame_len")
    selected_count = min(top_k, frame_count)

    frames = signal[:, : frame_count * frame_len].reshape(batch_size, frame_count, frame_len)
    spectrum = torch.fft.fft(frames, dim=-1)
    power = torch.abs(spectrum) ** 2
    energy = power.sum(dim=-1)
    probability = power / energy.unsqueeze(-1).clamp_min(eps)
    entropy = -(probability * torch.log(probability.clamp_min(eps))).sum(dim=-1)
    eser = energy / entropy.clamp_min(eps)

    _, indices = torch.topk(eser, k=selected_count, dim=1, largest=True)
    if sort_by_time:
        indices = torch.sort(indices, dim=1).values

    gather_idx = indices.unsqueeze(-1).expand(-1, -1, frame_len)
    selected = torch.gather(frames, dim=1, index=gather_idx).reshape(batch_size, selected_count * frame_len)

    if squeeze_batch:
        selected = selected.squeeze(0)
        indices = indices.squeeze(0)
        eser = eser.squeeze(0)

    return selected.to(dtype=input_dtype), indices, eser

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

    # 输入 RF 信号只接受 Tensor；分帧长度、目标长度和 top-k 必须能得到有效片段数。
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
    # signal: 一维输入为 (sample_count,)，二维输入为 (B, sample_count)。
    signal = x.to(compute_device)

    squeeze_batch = False
    if signal.ndim == 1:
        # 单条 RF 序列补 batch 维度，后续统一按 (B, sample_count) 处理。
        signal = signal.unsqueeze(0)    # (signal_length,) -> (1, signal_length)
        squeeze_batch = True
    if signal.ndim != 2:
        raise ValueError("x must be a one-dimensional or two-dimensional RF sequence")

    batch_size, sample_count = signal.shape
    frame_count = sample_count // frame_len
    if frame_count == 0:
        raise ValueError("x is shorter than frame_len")
    selected_count = min(top_k, frame_count)

    # 丢弃尾部不足一帧的采样点，并 reshape 为 (B, frame_count, frame_len)。
    frames = signal[:, : frame_count * frame_len].reshape(batch_size, frame_count, frame_len)
    # spectrum/power: (B, frame_count, frame_len)，保留每帧的频域采样点。
    spectrum = torch.fft.fft(frames, dim=-1)
    power = torch.abs(spectrum) ** 2
    # energy: (B, frame_count)，每个 frame 对应一个短时能量值。
    energy = power.sum(dim=-1)
    # probability: (B, frame_count, frame_len)，每个 frame 内沿频率维归一化。
    probability = power / energy.unsqueeze(-1).clamp_min(eps)
    # entropy/eser: (B, frame_count)，每个 frame 各有一个频谱熵和 ESER 分数。
    entropy = -(probability * torch.log(probability.clamp_min(eps))).sum(dim=-1)
    # ESER 越大，表示该帧能量更强且频谱熵更低，通常包含更突出的 RF 特征。
    eser = energy / entropy.clamp_min(eps)

    # 选出 ESER 最高的 top-k 帧；需要时再按原始时间顺序拼接。
    # indices: (B, selected_count)，存放被选中 frame 在原序列中的下标。
    _, indices = torch.topk(eser, k=selected_count, dim=1, largest=True)
    if sort_by_time:
        indices = torch.sort(indices, dim=1).values

    # gather_idx: (B, selected_count, frame_len)，用于在 frame 维复制索引并 gather。
    gather_idx = indices.unsqueeze(-1).expand(-1, -1, frame_len)
    # selected: (B, selected_count * frame_len)，把选中 frames 按时间维拼接回 RF 片段。
    selected = torch.gather(frames, dim=1, index=gather_idx).reshape(batch_size, selected_count * frame_len)

    if squeeze_batch:
        # 输入是一维信号时，返回也去掉临时 batch 维度。
        selected = selected.squeeze(0)
        indices = indices.squeeze(0)
        eser = eser.squeeze(0)

    return selected.to(dtype=input_dtype), indices, eser

"""STFT precompute helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.preprocess.window import create_window

# 声明模块的公开接口
__all__ = ["compute_stft"]

# 进程内缓存 Hann window，避免重复调用 STFT 时每次都重新创建。
# `_WINDOW_DEVICE` 记录缓存所在设备；同一个设备、同一个窗口长度
# 可以直接复用。窗口长度变化时会重新生成，避免旧缓存误用。
_WINDOW: torch.Tensor | None = None
_WINDOW_DEVICE: str | None = None

def _get_window(device: str, win_length: int) -> torch.Tensor:
    """
    获取指定设备上的周期 Hann 窗，并缓存最近一次结果。
    设备和窗口长度均与缓存一致时，直接返回缓存张量；否则重新创建。

    输入：
    device 为设备字符串（如 "cpu"、"cuda:0"、"mps"）
    win_length 为窗口长度。
    输出：
    形状为 (win_length,) 的实数张量，位于 device ，创建时使用 torch 默认浮点类型。
    """
    global _WINDOW, _WINDOW_DEVICE
    if _WINDOW is None or _WINDOW_DEVICE != device or _WINDOW.numel() != win_length:
        _WINDOW = create_window(
            "hann", win_length, device=device, dtype=torch.get_default_dtype(), periodic=True
        )
        _WINDOW_DEVICE = device
    return _WINDOW

def compute_stft(
    iq_batch: torch.Tensor,
    device: str = "cpu",
    *,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int | None = None,
    output_freq_bins: int | None = None,
    output_time_bins: int | None = None,
) -> torch.Tensor:
    """
    iq_batch: (B, 1, L) complex64
    device:  torch device string, e.g. "cpu", "cuda:0", "mps"
    returns: torch.Tensor with shape (B, 1, output_freq_bins, output_time_bins), float32,
             z-score normalized per channel
    B: batch size, 1: selected RF channel, output_freq_bins: freq bins, output_time_bins: time bins
    """
    if not isinstance(iq_batch, torch.Tensor):
        raise TypeError("iq_batch must be a torch.Tensor")

    B, C, L = iq_batch.shape
    if C != 1:
        raise ValueError(f"iq_batch must have exactly one RF channel, got shape {tuple(iq_batch.shape)}")
    if hop_length is not None and hop_length <= 0:
        raise ValueError("hop_length must be positive")
    hop_length = hop_length or win_length // 2
    out_f = output_freq_bins or n_fft
    out_t = output_time_bins
    window = _get_window(device, win_length)
    with torch.no_grad():
        sig = iq_batch.reshape(B * C, L).to(device=device, dtype=torch.complex64)
        Zxx = torch.stft(
            sig,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True,
        )
        Zxx = torch.fft.fftshift(Zxx, dim=1)
        eps = torch.finfo(torch.float32).eps
        Zxx_db = 20.0 * torch.log10(Zxx.abs() + eps)
        target_time_bins = out_t or Zxx_db.shape[-1]
        if out_f != n_fft or target_time_bins != Zxx_db.shape[-1]:
            # 先在 dB 域规整尺寸，再做每通道归一化，避免池化改变归一化统计。
            Zxx_db = F.adaptive_avg_pool2d(
                Zxx_db.unsqueeze(1),
                output_size=(out_f, target_time_bins),
            ).squeeze(1)
        mean = Zxx_db.mean(dim=(1, 2), keepdim=True)
        std = Zxx_db.std(dim=(1, 2), keepdim=True)
        Zxx_norm = (Zxx_db - mean) / (std + 1e-8)
        Zxx_norm = Zxx_norm.reshape(B, C, out_f, target_time_bins)
    return Zxx_norm

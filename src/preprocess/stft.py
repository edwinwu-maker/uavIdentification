"""STFT precompute helpers."""

from __future__ import annotations

import numpy as np
import torch

# 声明模块的公开接口
__all__ = ["compute_stft"]

# 进程内缓存 Hann window，避免重复调用 STFT 时每次都重新创建。
# `_WINDOW_DEVICE` 记录缓存所在设备；同一个设备、同一个窗口长度
# 可以直接复用。窗口长度变化时会重新生成，避免旧缓存误用。
_WINDOW: torch.Tensor | None = None
_WINDOW_DEVICE: str | None = None


def _get_window(device: str, win_length: int) -> torch.Tensor:
    global _WINDOW, _WINDOW_DEVICE
    if _WINDOW is None or _WINDOW_DEVICE != device or _WINDOW.numel() != win_length:
        _WINDOW = torch.hann_window(win_length, device=device)
        _WINDOW_DEVICE = device
    return _WINDOW

def compute_stft(
    iq_batch: np.ndarray,
    device: str = "cpu",
    *,
    n_fft: int = 1024,
    win_length: int = 1024,
    spec_time_bins: int = 1024,
) -> np.ndarray:
    """
    iq_batch: (B, 2, L) complex64
    device:  torch device string, e.g. "cpu", "cuda:0", "mps"
    returns: (B, 2, n_fft, spec_time_bins) float32, z-score normalized per channel
    B: batch size, 2: channels, n_fft: freq bins, spec_time_bins: time bins
    """
    B, C, L = iq_batch.shape
    hop_length = L // (spec_time_bins - 1)
    window = _get_window(device, win_length)
    with torch.no_grad():
        sig = torch.from_numpy(iq_batch.reshape(B * C, L)).to(device)
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
        Zxx = Zxx[:, :, :spec_time_bins]
        eps = torch.finfo(torch.float32).eps
        Zxx_db = 20.0 * torch.log10(Zxx.abs() + eps)
        mean = Zxx_db.mean(dim=(1, 2), keepdim=True)
        std = Zxx_db.std(dim=(1, 2), keepdim=True)
        Zxx_norm = (Zxx_db - mean) / (std + 1e-8)
        Zxx_norm = Zxx_norm.reshape(B, C, n_fft, spec_time_bins)
    return Zxx_norm.cpu().numpy().astype(np.float32)

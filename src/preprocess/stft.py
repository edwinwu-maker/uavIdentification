"""STFT precompute helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

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
    iq_batch: torch.Tensor,
    device: str = "cpu",
    *,
    n_fft: int = 1024,
    win_length: int = 1024,
    spec_time_bins: int = 1024,
    output_freq_bins: int | None = None,
    output_time_bins: int | None = None,
) -> torch.Tensor:
    """
    iq_batch: (B, 2, L) complex64
    device:  torch device string, e.g. "cpu", "cuda:0", "mps"
    returns: torch.Tensor with shape (B, 2, output_freq_bins, output_time_bins), float32,
             z-score normalized per channel
    B: batch size, 2: channels, output_freq_bins: freq bins, output_time_bins: time bins
    """
    if not isinstance(iq_batch, torch.Tensor):
        raise TypeError("iq_batch must be a torch.Tensor")

    B, C, L = iq_batch.shape
    hop_length = L // (spec_time_bins - 1)
    out_f = output_freq_bins or n_fft
    out_t = output_time_bins or spec_time_bins
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
        Zxx = Zxx[:, :, :spec_time_bins]
        eps = torch.finfo(torch.float32).eps
        Zxx_db = 20.0 * torch.log10(Zxx.abs() + eps)
        if out_f != n_fft or out_t != spec_time_bins:
            # 先在 dB 域规整尺寸，再做每通道归一化，避免池化改变归一化统计。
            Zxx_db = F.adaptive_avg_pool2d(
                Zxx_db.unsqueeze(1),
                output_size=(out_f, out_t),
            ).squeeze(1)
        mean = Zxx_db.mean(dim=(1, 2), keepdim=True)
        std = Zxx_db.std(dim=(1, 2), keepdim=True)
        Zxx_norm = (Zxx_db - mean) / (std + 1e-8)
        Zxx_norm = Zxx_norm.reshape(B, C, out_f, out_t)
    return Zxx_norm

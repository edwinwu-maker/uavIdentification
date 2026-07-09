from __future__ import annotations

from collections.abc import Iterator
from typing import Final, Optional

import numpy as np
import torch

from src.preprocess.fam_defaults import (
    FAM_ALPHA_RANGE,
    FAM_F_RANGE,
)


# 声明模块的公开接口，IDE 和静态检查工具依此识别公开 API。
__all__ = ["fam_grid_torch"]

_FAM_HAMMING_WINDOWS: Final[tuple[str, ...]] = ("hamming", "hamm")
_FAM_HANN_WINDOWS: Final[tuple[str, ...]] = ("hann", "hanning")
_FAM_RECT_WINDOWS: Final[tuple[str, ...]] = ("rect", "boxcar", "rectangle")
_PRINCIPAL_DOMAIN_BOUNDARY: Final[float] = 0.5
_FAM_NYQUIST_BIN: Final[float] = -0.5


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    return resolved


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex128:
        return torch.float64
    if dtype == torch.complex64:
        return torch.float32
    raise ValueError("dtype must be torch.complex64 or torch.complex128")


def _window(
    name: str,
    nfft: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    name = name.lower()
    if name in _FAM_HAMMING_WINDOWS:
        # Match NumPy's np.hamming/np.hanning definitions used by the CPU FAM path.
        return torch.hamming_window(nfft, periodic=False, device=device, dtype=dtype)
    if name in _FAM_HANN_WINDOWS:
        return torch.hann_window(nfft, periodic=False, device=device, dtype=dtype)
    if name in _FAM_RECT_WINDOWS:
        return torch.ones(nfft, device=device, dtype=dtype)
    raise ValueError(f"unsupported window: {name}")


def _make_blocks(
    x: torch.Tensor,
    nfft: int,
    hop: int,
    *,
    n_blocks: Optional[int],
) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional complex IQ sequence")
    if nfft <= 0:
        raise ValueError("nfft must be positive")
    if hop <= 0:
        raise ValueError("hop must be positive")

    if n_blocks is None:
        if x.numel() <= nfft:
            n_blocks = 1
        else:
            n_blocks = int(np.ceil((x.numel() - nfft) / hop)) + 1

    total_needed = (n_blocks - 1) * hop + nfft
    if x.numel() < total_needed:
        pad = torch.zeros(total_needed - x.numel(), dtype=x.dtype, device=x.device)
        x = torch.cat((x, pad))

    # Build all block sample indices on the target device to avoid CPU-side slicing loops.
    starts = torch.arange(n_blocks, device=x.device) * hop
    sample_index = starts[:, None] + torch.arange(nfft, device=x.device)[None, :]
    return x[sample_index]


def _channelize(
    x: torch.Tensor,
    *,
    nfft: int,
    hop: int,
    window: str,
    n_blocks: Optional[int],
    real_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks = _make_blocks(x, nfft, hop, n_blocks=n_blocks)
    p_count = blocks.shape[0]

    w = _window(window, nfft, device=x.device, dtype=real_dtype)
    window_energy = torch.sum(torch.abs(w) ** 2)
    blocks_w = blocks * w[None, :]

    spectrum = torch.fft.fft(blocks_w, dim=1)
    # Keep frequency coordinates in the requested real precision; creating them
    # as float32 first would break close agreement with the NumPy reference.
    freqs = torch.fft.fftfreq(nfft, d=1.0, device=x.device, dtype=real_dtype)
    spectrum = torch.fft.fftshift(spectrum, dim=1)
    freqs = torch.fft.fftshift(freqs)

    p = torch.arange(p_count, device=x.device, dtype=real_dtype)
    # Restore the global time-origin phase relation between adjacent blocks.
    phase = torch.exp(-1j * 2.0 * torch.pi * p[:, None] * hop * freqs[None, :])
    x_tilde = spectrum * phase

    return x_tilde, freqs, window_energy


def _iter_fam_point_batches_torch(
    x: torch.Tensor,
    *,
    nfft: int,
    hop: int,
    n_blocks: Optional[int],
    window: str,
    keep_principal_domain: bool,
    normalize: bool,
    device: str | torch.device,
    dtype: torch.dtype,
    pair_chunk_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")

    resolved_device = _resolve_device(device)
    real_dtype = _real_dtype(dtype)
    x_tensor = x.to(device=resolved_device, dtype=dtype)
    x_tilde, freqs, window_energy = _channelize(
        x_tensor,
        nfft=nfft,
        hop=hop,
        window=window,
        n_blocks=n_blocks,
        real_dtype=real_dtype,
    )
    # x_tilde has shape [P, nfft]: P time blocks by nfft frequency channels.
    p_count = x_tilde.shape[0]

    beta = torch.fft.fftfreq(p_count, d=hop, device=resolved_device, dtype=real_dtype)
    beta = torch.fft.fftshift(beta)

    channel = torch.arange(nfft, device=resolved_device)
    """
        k_all =                             l_all =
    tensor([                            tensor([
        [0, 0,..., 0],                      [0, 1,..., nfft-1],
        [1, 1,..., 1],                      [0, 1,..., nfft-1],
        ...,                                ...,
        [nfft-1, nfft-1,..., nfft-1]        [0, 1,..., nfft-1],
    ])                                  ])
    """
    k_all, l_all = torch.meshgrid(channel, channel, indexing="ij")
    k_all = k_all.reshape(-1)
    l_all = l_all.reshape(-1)

    # Keep the same Nyquist-bin convention as the CPU FAM path to avoid asymmetric
    # edge artifacts near the principal-domain boundary.
    nyquist = torch.tensor(_FAM_NYQUIST_BIN, device=resolved_device, dtype=real_dtype)
    valid = (~torch.isclose(freqs[k_all], nyquist)) & (
        ~torch.isclose(freqs[l_all], nyquist)
    )
    k_all = k_all[valid]
    l_all = l_all[valid]

    scale = p_count * window_energy if normalize else 1.0
    for start in range(0, k_all.numel(), pair_chunk_size):
        k = k_all[start : start + pair_chunk_size]
        l = l_all[start : start + pair_chunk_size]
        fk = freqs[k]
        fl = freqs[l]

        product = x_tilde[:, k] * torch.conj(x_tilde[:, l])
        z = torch.fft.fft(product.T, dim=1)
        z = torch.fft.fftshift(z, dim=1)
        if normalize:
            z = z / scale

        f_value = 0.5 * (fk + fl)
        alpha = (fk - fl)[:, None] + beta[None, :]
        f = f_value[:, None].expand_as(alpha)

        # f and alpha have shape [current_pair_count, p_count]: 当前通道对批次的
        # z has shape [current_pair_count, p_count]: 同一网格上的复数 SCF 估计值。
        # current_pair_count <= pair_chunk_size。
        # f[mask], alpha[mask], and z[mask] have shape [kept_point_count]:
        # principal domain 内保留下来的展平 FAM 点。
        if keep_principal_domain:
            mask = torch.abs(f) + 0.5 * torch.abs(alpha) < _PRINCIPAL_DOMAIN_BOUNDARY
            if not torch.any(mask):
                continue
            yield f[mask], alpha[mask], z[mask]
        else:
            yield f.reshape(-1), alpha.reshape(-1), z.reshape(-1)


def fam_grid_torch(
    x: torch.Tensor,
    *,
    nfft: int,
    hop: int,
    f_bins: int,
    alpha_bins: int,
    n_blocks: Optional[int] = None,
    window: str = "hamming",
    keep_principal_domain: bool = True,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.complex64,
    pair_chunk_size: int = 8192,
    f_range: tuple[float, float] = FAM_F_RANGE,
    alpha_range: tuple[float, float] = FAM_ALPHA_RANGE,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Use FAM to estimate |SCF| and aggregate it into a grid on the torch device.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")

    resolved_device = _resolve_device(device)
    real_dtype = _real_dtype(dtype)
    grid_size = alpha_bins * f_bins
    image_sum = torch.zeros(grid_size, device=resolved_device, dtype=real_dtype)
    image_count = torch.zeros(grid_size, device=resolved_device, dtype=real_dtype)
    f_min, f_max = f_range
    alpha_min, alpha_max = alpha_range
    x_tensor = x.to(device=resolved_device, dtype=dtype)

    for f, alpha, value in _iter_fam_point_batches_torch(
        x_tensor,
        nfft=nfft,
        hop=hop,
        n_blocks=n_blocks,
        window=window,
        keep_principal_domain=keep_principal_domain,
        normalize=True,
        device=resolved_device,
        dtype=dtype,
        pair_chunk_size=pair_chunk_size,
    ):
        f_idx = torch.floor((f - f_min) / (f_max - f_min) * (f_bins - 1)).to(
            torch.long
        )
        a_idx = torch.floor(
            (alpha - alpha_min) / (alpha_max - alpha_min) * (alpha_bins - 1)
        ).to(torch.long)
        mask = (f_idx >= 0) & (f_idx < f_bins) & (a_idx >= 0) & (a_idx < alpha_bins)
        if not torch.any(mask):
            continue

        flat_idx = (a_idx[mask] * f_bins + f_idx[mask]).reshape(-1)
        mag = torch.abs(value[mask]).to(real_dtype).reshape(-1)
        image_sum.scatter_add_(0, flat_idx, mag)
        image_count.scatter_add_(0, flat_idx, torch.ones_like(mag))

    image = torch.zeros_like(image_sum)
    count_mask = image_count > 0
    image[count_mask] = image_sum[count_mask] / image_count[count_mask]
    image = image.reshape(alpha_bins, f_bins)

    f_axis = np.linspace(f_range[0], f_range[1], f_bins)
    alpha_axis = np.linspace(alpha_range[0], alpha_range[1], alpha_bins)
    return image, f_axis, alpha_axis

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.utils.fam import FAMResult

# 声明模块的公开接口，IDE 和静态检查工具依此识别公开 API。
__all__ = ["fam_scf_points_gpu"]

def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return resolved


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex128:
        return torch.float64
    if dtype == torch.complex64:
        return torch.float32
    raise ValueError("dtype must be torch.complex64 or torch.complex128")


def _window(name: str, nfft: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    name = name.lower()
    if name in {"hamming", "hamm"}:
        # Match NumPy's np.hamming/np.hanning definitions used by the CPU FAM path.
        return torch.hamming_window(nfft, periodic=False, device=device, dtype=dtype)
    if name in {"hann", "hanning"}:
        return torch.hann_window(nfft, periodic=False, device=device, dtype=dtype)
    if name in {"rect", "boxcar", "rectangle"}:
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


def fam_scf_points_gpu(
    x: np.ndarray,
    *,
    nfft: int = 256,
    hop: int = 64,
    n_blocks: Optional[int] = None,
    window: str = "hamming",
    keep_principal_domain: bool = True,
    normalize: bool = True,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.complex64,
    pair_chunk_size: int = 8192,
) -> FAMResult:
    """Estimate sparse FAM SCF points with batched PyTorch FFTs.

    Arguments:
        pair_chunk_size: Number of (k, l) channel pairs to process in each batch. Adjust based on available GPU memory;

    The public result mirrors ``fam_scf_points``: NumPy arrays are returned so
    existing plotting and gridding code can consume the GPU implementation.
    """

    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")

    resolved_device = _resolve_device(device)
    real_dtype = _real_dtype(dtype)
    x_tensor = torch.as_tensor(np.asarray(x), dtype=dtype, device=resolved_device)
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

    channel = torch.arange(nfft, device=resolved_device)    # tensor([0, 1,..., nfft-1], device='cuda:0')
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
    # k_all = tensor([0, 0,..., 0, 1, 1,..., 1,..., nfft-1, nfft-1,..., nfft-1], device='cuda:0')
    # l_all = tensor([0, 1,..., nfft-1, 0, 1,..., nfft-1,..., 0, 1,..., nfft-1], device='cuda:0')
    k_all = k_all.reshape(-1)
    l_all = l_all.reshape(-1)

    # Keep the same Nyquist-bin convention as fam_scf_points to avoid asymmetric
    # edge artifacts near the principal-domain boundary.
    # 舍去频率为-0.5的点，避免不对称。
    nyquist = torch.tensor(-0.5, device=resolved_device, dtype=real_dtype)
    valid = (~torch.isclose(freqs[k_all], nyquist)) & (~torch.isclose(freqs[l_all], nyquist))
    k_all = k_all[valid]
    l_all = l_all[valid]

    scale = p_count * window_energy if normalize else 1.0
    f_chunks: list[torch.Tensor] = []
    alpha_chunks: list[torch.Tensor] = []
    value_chunks: list[torch.Tensor] = []

    # start = [0, pair_chunk_size, 2*pair_chunk_size,...]
    for start in range(0, k_all.numel(), pair_chunk_size):
        # k 和 l 是长度为 B（≤ pair_chunk_size）的 1 维整数张量，值在 [0, nfft-1] 之间
        k = k_all[start : start + pair_chunk_size]
        l = l_all[start : start + pair_chunk_size]
        # fk 和 fl 是长度为 B（≤ pair_chunk_size）的 1 维整数张量， 值在 (-0.5, 0.5) 之间
        fk = freqs[k]
        fl = freqs[l]

        # x_tilde[:, k]->形状 [P, B], x_tilde[:, l]->形状 [P, B], product->形状[P, B], *—>逐元素相乘
        product = x_tilde[:, k] * torch.conj(x_tilde[:, l])
        # transforms of length P.
        z = torch.fft.fft(product.T, dim=1)
        z = torch.fft.fftshift(z, dim=1)
        if normalize:
            z = z / scale

        # f_value 是 [B] 形状的浮点张量，值域 (-0.5, 0.5)
        f_value = 0.5 * (fk + fl)   
        # alpha 是 [B, P] 形状的浮点张量，dtype 为 real_dtype，值域 (-1.5, 1.5)
        # 由 (fk - fl)[:, None] + beta[None, :] 广播得到
        alpha = (fk - fl)[:, None] + beta[None, :]
        # f 是 [B, P] 形状的浮点张量，值域 (-0.5, 0.5)
        f = f_value[:, None].expand_as(alpha)

        if keep_principal_domain:
            # mask 是 [B, P] 形状的bool张量
            mask = torch.abs(f) + 0.5 * torch.abs(alpha) < 0.5
            if not torch.any(mask):
                continue
            # f[mask] 布尔索引取出 mask 为 True 位置的 f 值
            # f_chunks 每次循环将一个 chunk 的有效 f 值 append 进去
            f_chunks.append(f[mask])
            alpha_chunks.append(alpha[mask])
            value_chunks.append(z[mask])
        else:
            f_chunks.append(f.reshape(-1))
            alpha_chunks.append(alpha.reshape(-1))
            value_chunks.append(z.reshape(-1))

    if not f_chunks:
        value_dtype = np.complex128 if dtype == torch.complex128 else np.complex64
        return FAMResult(
            f=np.empty(0, dtype=float),
            alpha=np.empty(0, dtype=float),
            value=np.empty(0, dtype=value_dtype),
        )

    return FAMResult(
        f=torch.cat(f_chunks).detach().cpu().numpy(),
        alpha=torch.cat(alpha_chunks).detach().cpu().numpy(),
        value=torch.cat(value_chunks).detach().cpu().numpy(),
    )

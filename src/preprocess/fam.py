from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

import numpy as np

from src.preprocess.fam_constants import (
    FAM_ALPHA_RANGE,
    FAM_F_RANGE,
    FAM_NYQUIST_BIN,
    SUPPORTED_FAM_WINDOWS,
    principal_domain_mask,
)

# 声明模块的公开接口
__all__ = ["fam_scf_grid"]


def _make_blocks(
    x: np.ndarray,
    nfft: int,
    hop: int,
    *,
    n_blocks: Optional[int] = None,
    pad: bool = True,
) -> np.ndarray:
    """Step 1: 将一维 IQ 信号切成 data blocks。

    pad:
        当输入信号长度不够凑齐最后一个 block 时，是否在末尾补零。

    返回 shape = (P, nfft) 的矩阵，每一行是一个 block。
    """

    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional complex IQ sequence")
    if nfft <= 0:
        raise ValueError("nfft must be positive")
    if hop <= 0:
        raise ValueError("hop must be positive")

    if n_blocks is None:
        if len(x) <= nfft:
            n_blocks = 1
        else:
            n_blocks = int(np.ceil((len(x) - nfft) / hop)) + 1

    total_needed = (n_blocks - 1) * hop + nfft
    if len(x) < total_needed:
        if not pad:
            raise ValueError("input is too short for requested n_blocks without padding")
        x = np.pad(x, (0, total_needed - len(x)))

    starts = np.arange(n_blocks) * hop
    # 通过广播生成每个 block 的采样索引，shape = (P, nfft)。
    sample_index = starts[:, None] + np.arange(nfft)[None, :]
    return x[sample_index]


def _window(name: str, nfft: int) -> np.ndarray:
    """Step 2: 生成 channelizer data-tapering window。"""

    name = name.lower()
    if name in SUPPORTED_FAM_WINDOWS[:2]:
        return np.hamming(nfft)
    if name in SUPPORTED_FAM_WINDOWS[2:4]:
        return np.hanning(nfft)
    if name in SUPPORTED_FAM_WINDOWS[4:]:
        return np.ones(nfft)
    raise ValueError(f"unsupported window: {name}")


def _channelize(
    x: np.ndarray,
    *,
    nfft: int,
    hop: int,
    window: str = "hamming",
    n_blocks: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Step 1-3: 分块、加窗、FFT、phase-shift。

    返回：
        X_tilde:
            shape = (P, nfft)，每一列对应一个频率通道。
        freqs:
            第一阶段 FFT 的 normalized frequency，单位 cycles/sample。
        window_energy:
            sum(abs(w)**2)，用于谱幅值归一化。
    """

    blocks = _make_blocks(x, nfft, hop, n_blocks=n_blocks, pad=True)
    p_count = blocks.shape[0]

    # Step 2: 对每个 block 乘以 tapering window。
    w = _window(window, nfft).astype(float)
    window_energy = float(np.sum(np.abs(w) ** 2))
    blocks_w = blocks * w[None, :]

    # Step 3: 对每个 block 做 FFT。
    spectrum = np.fft.fft(blocks_w, axis=1)
    freqs = np.fft.fftfreq(nfft, d=1.0)
    spectrum = np.fft.fftshift(spectrum, axes=1)
    freqs = np.fft.fftshift(freqs)

    # Step 3: phase-shift，恢复不同 blocks 之间的全局时间相位关系。
    # phase[p, k] = exp(-j 2pi f_k pL)
    p = np.arange(p_count)
    phase = np.exp(-1j * 2.0 * np.pi * p[:, None] * hop * freqs[None, :])
    x_tilde = spectrum * phase

    return x_tilde, freqs, window_energy


def _iter_fam_point_batches(
    x: np.ndarray,
    *,
    nfft: int,
    hop: int,
    n_blocks: Optional[int],
    window: str,
    keep_principal_domain: bool,
    normalize: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    x_tilde, freqs, window_energy = _channelize(
        x,
        nfft=nfft,
        hop=hop,
        window=window,
        n_blocks=n_blocks,
    )
    p_count = x_tilde.shape[0]

    # 第二阶段 FFT 的 cycle-frequency 细分量 beta，单位 cycles/sample。
    beta = np.fft.fftfreq(p_count, d=hop)
    beta = np.fft.fftshift(beta)

    channel_pairs = ((k, l) for k in range(nfft) for l in range(nfft))

    scale = 1.0
    if normalize:
        scale = p_count * window_energy

    for k, l in channel_pairs:
        fk = freqs[k]
        fl = freqs[l]

        # Drop the one-sided Nyquist bin from even-length FFTs. np.fft.fftfreq
        # represents Nyquist as -0.5, with no matching +0.5 bin, which can
        # create asymmetric edge artifacts near the principal-domain boundary.
        if np.isclose(fk, FAM_NYQUIST_BIN) or np.isclose(fl, FAM_NYQUIST_BIN):
            continue

        # Step 4: 构造长度 P 的 channelizer product vector。
        product = x_tilde[:, k] * np.conj(x_tilde[:, l])

        # Step 4: 沿窗口序号做第二次 FFT。
        z = np.fft.fft(product)
        z = np.fft.fftshift(z)

        if normalize:
            z = z / scale

        # Step 5: 映射到 (f, alpha)。
        f_value = 0.5 * (fk + fl)
        alpha = (fk - fl) + beta
        f = np.full_like(alpha, f_value, dtype=float)

        # 同一组 (k, l) 的 spectral frequency 固定，cycle frequency 随 beta 变化。
        if keep_principal_domain:
            mask = principal_domain_mask(f, alpha)
            if not np.any(mask):
                continue
            f = f[mask]
            alpha = alpha[mask]
            z = z[mask]

        yield f, alpha, z


def fam_scf_grid(
    x: np.ndarray,
    *,
    nfft: int = 256,
    hop: int = 64,
    n_blocks: Optional[int] = None,
    window: str = "hamming",
    keep_principal_domain: bool = True,
    normalize: bool = True,
    f_bins: int = 257,
    alpha_bins: int = 513,
    f_range: tuple[float, float] = FAM_F_RANGE,
    alpha_range: tuple[float, float] = FAM_ALPHA_RANGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate FAM and aggregate |SCF| directly into a grid."""

    image_sum = np.zeros((alpha_bins, f_bins), dtype=float)
    image_count = np.zeros((alpha_bins, f_bins), dtype=int)
    f_min, f_max = f_range
    alpha_min, alpha_max = alpha_range

    for f, alpha, value in _iter_fam_point_batches(
        x,
        nfft=nfft,
        hop=hop,
        n_blocks=n_blocks,
        window=window,
        keep_principal_domain=keep_principal_domain,
        normalize=True,
    ):
        f_idx = np.floor((f - f_min) / (f_max - f_min) * (f_bins - 1)).astype(int)
        a_idx = np.floor(
            (alpha - alpha_min) / (alpha_max - alpha_min) * (alpha_bins - 1)
        ).astype(int)
        valid = (
            (f_idx >= 0)
            & (f_idx < f_bins)
            & (a_idx >= 0)
            & (a_idx < alpha_bins)
        )
        mag = np.abs(value)
        np.add.at(image_sum, (a_idx[valid], f_idx[valid]), mag[valid])
        np.add.at(image_count, (a_idx[valid], f_idx[valid]), 1)

    image = np.zeros((alpha_bins, f_bins), dtype=float)
    np.divide(image_sum, image_count, out=image, where=image_count > 0)

    if normalize and image.max() > 0:
        image = image / image.max()

    f_axis = np.linspace(f_range[0], f_range[1], f_bins)
    alpha_axis = np.linspace(alpha_range[0], alpha_range[1], alpha_bins)
    return image, f_axis, alpha_axis

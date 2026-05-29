from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FAMResult:
    """FAM 输出点估计。

    f:
        spectral frequency 坐标。默认 normalized frequency，单位 cycles/sample。
    alpha:
        cycle frequency 坐标。默认 normalized frequency，单位 cycles/sample。
    value:
        对应的 complex SCF 估计值。
    """

    f: np.ndarray
    alpha: np.ndarray
    value: np.ndarray


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
    if name in {"hamming", "hamm"}:
        return np.hamming(nfft)
    if name in {"hann", "hanning"}:
        return np.hanning(nfft)
    if name in {"rect", "boxcar", "rectangle"}:
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


def _principal_domain_mask(f: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """non-conjugate SCF 常用 principal domain。

    normalized frequency 下：
        |f| + |alpha|/2 <= 1/2
    """

    return np.abs(f) + 0.5 * np.abs(alpha) <= 0.5


def fam_scf_points(
    x: np.ndarray,
    *,
    nfft: int = 256,
    hop: int = 64,
    n_blocks: Optional[int] = None,
    window: str = "hamming",
    keep_principal_domain: bool = True,
    normalize: bool = True,
) -> FAMResult:
    """按 FAM Step 1-5 估计 SCF 点。

    Parameters
    ----------
    x:
        输入 complex IQ 序列。
    nfft:
        N prime，channelizer 短时 FFT 点数。
    hop:
        L，data block hop size。
    n_blocks:
        P，使用的 data block 数。若为 None，则由输入长度自动决定。
    window:
        channelizer tapering window，支持 "hamming"、"hann"、"rect"。
    keep_principal_domain:
        是否丢弃 non-conjugate SCF principal domain 之外的点。
    normalize:
        若为 True，谱值除以 P * window_energy。函数始终使用 normalized
        frequency；如果需要 Hz 坐标，由调用者在函数外部乘以采样率 fs。

    Returns
    -------
    FAMResult:
        一组三元点估计 (f, alpha, value)。
    """

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

    f_chunks: list[np.ndarray] = []
    alpha_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []

    scale = 1.0
    if normalize:
        scale = p_count * window_energy

    for k, l in channel_pairs:
        fk = freqs[k]
        fl = freqs[l]

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
        # principal-domain 过滤必须同时作用于 f、alpha 和对应的谱值 z。
        if keep_principal_domain:
            mask = _principal_domain_mask(f, alpha)
            if not np.any(mask):
                continue
            f = f[mask]
            alpha = alpha[mask]
            z = z[mask]

        f_chunks.append(f)
        alpha_chunks.append(alpha)
        value_chunks.append(z)

    if not f_chunks:
        return FAMResult(
            f=np.empty(0, dtype=float),
            alpha=np.empty(0, dtype=float),
            value=np.empty(0, dtype=np.complex128),
        )

    return FAMResult(
        f=np.concatenate(f_chunks),
        alpha=np.concatenate(alpha_chunks),
        value=np.concatenate(value_chunks),
    )
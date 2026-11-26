from __future__ import annotations

from collections.abc import Iterator
from typing import Final, Optional

import numpy as np
import torch

from src.preprocess.window import create_window
from src.utils.device import resolve_device
from src.utils.dtype import get_real_dtype


# 声明模块的公开接口，IDE 和静态检查工具依此识别公开 API。
__all__ = ["FAM_F_RANGE", "FAM_ALPHA_RANGE", "compute_fam_grid"]

# FAM/SCF 网格默认的归一化频率 f 与循环频率 alpha 范围。
FAM_F_RANGE: Final[tuple[float, float]] = (-0.5, 0.5)
FAM_ALPHA_RANGE: Final[tuple[float, float]] = (-1.0, 1.0)

_PRINCIPAL_DOMAIN_BOUNDARY: Final[float] = 0.5
_FAM_NYQUIST_BIN: Final[float] = -0.5


def _validate_fam_grid_args(
    x: torch.Tensor,
    *,
    nfft: int,
    hop: int,
    f_bins: int,
    alpha_bins: int,
    num_frames: Optional[int],
    pair_chunk_size: int,
    f_range: tuple[float, float],
    alpha_range: tuple[float, float],
) -> None:
    """统一校验 FAM 网格计算的输入张量、正整数参数和坐标范围。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional complex IQ sequence")
    if nfft <= 0:
        raise ValueError("nfft must be positive")
    if hop <= 0:
        raise ValueError("hop must be positive")
    if f_bins <= 0:
        raise ValueError("f_bins must be positive")
    if alpha_bins <= 0:
        raise ValueError("alpha_bins must be positive")
    if num_frames is not None and num_frames <= 0:
        raise ValueError("n_blocks must be positive")
    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")
    if f_range[0] >= f_range[1]:
        raise ValueError("f_range must be strictly increasing")
    if alpha_range[0] >= alpha_range[1]:
        raise ValueError("alpha_range must be strictly increasing")


def _frame_iq_signal(
    x: torch.Tensor,
    nfft: int,
    hop: int,
    *,
    num_frames: Optional[int],
) -> torch.Tensor:
    """按 hop 对一维 IQ 信号分帧，末帧不足时补零。

    输入：
        x: 形状为 (L,) 的一维 IQ 张量。
        nfft: 每帧的采样点数。
        hop: 相邻帧起点之间的采样点间隔。
        num_frames: 输出帧数；为 None 时自动计算以覆盖输入，且至少生成一帧。
    输出：
        形状为 (P, nfft) 的张量， P 为显式指定或自动计算的帧数；
        dtype 和 device 与 x 一致，末尾不足部分用零填充。
    """
    if num_frames is None:
        if x.numel() <= nfft:
            num_frames = 1
        else:
            num_frames = int(np.ceil((x.numel() - nfft) / hop)) + 1

    total_needed = (num_frames - 1) * hop + nfft
    if x.numel() < total_needed:
        pad = torch.zeros(total_needed - x.numel(), dtype=x.dtype, device=x.device)
        x = torch.cat((x, pad))

    starts = torch.arange(num_frames, device=x.device) * hop
    sample_index = starts[:, None] + torch.arange(nfft, device=x.device)[None, :]
    return x[sample_index]


def _channelize_iq_signal(
    x: torch.Tensor,
    *,
    nfft: int,
    hop: int,
    window: str,
    num_frames: Optional[int],
    real_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ 对 IQ 分块加窗、FFT 并校正时移相位，返回通道化频谱、频率坐标和窗能量。 """
    # Step1：对一维 IQ 信号分帧，得到数据矩阵
    blocks = _frame_iq_signal(x, nfft, hop, num_frames=num_frames)
    frame_count = blocks.shape[0]

    # Step2：对数据矩阵加窗
    w = create_window(window, nfft, device=x.device, dtype=real_dtype, periodic=False)
    window_energy = torch.sum(torch.abs(w) ** 2)
    blocks_w = blocks * w[None, :]

    # Step3：对加窗后的矩阵做FFT
    spectrum = torch.fft.fft(blocks_w, dim=1)
    freqs = torch.fft.fftfreq(nfft, d=1.0, device=x.device, dtype=real_dtype)
    spectrum = torch.fft.fftshift(spectrum, dim=1)
    freqs = torch.fft.fftshift(freqs)

    frame_indices = torch.arange(frame_count, device=x.device, dtype=real_dtype)
    # 补偿分帧引入的相位偏移，统一各帧的时间原点。
    phase = torch.exp(
        -1j * 2.0 * torch.pi * frame_indices[:, None] * hop * freqs[None, :]
    )
    x_tilde = spectrum * phase

    return x_tilde, freqs, window_energy


def _iter_fam_point_batches(
    x: torch.Tensor,
    *,
    nfft: int,
    hop: int,
    num_frames: Optional[int],
    window: str,
    real_dtype: torch.dtype,
    pair_chunk_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """分批处理频率通道对，生成供 FAM 网格聚合的 (f, alpha, SCF) 点。

    pair_chunk_size: 每批处理的 FFT 通道对数量；越大批次数越少，但内存/显存占用越高。
    """
    x_tilde, freqs, window_energy = _channelize_iq_signal(
        x,
        nfft=nfft,
        hop=hop,
        window=window,
        num_frames=num_frames,
        real_dtype=real_dtype,
    )
    # x_tilde 为 (num_frames, nfft) 的通道化频谱，行对应时间帧，列对应 FFT 频点。
    frame_count = x_tilde.shape[0]

    # 生成以零为中心的帧间循环频率细分坐标。
    beta = torch.fft.fftshift(
        torch.fft.fftfreq(frame_count, d=hop, device=x.device, dtype=real_dtype)
    )

    channel = torch.arange(nfft, device=x.device)

    # 枚举并展平所有 FFT 通道索引对 (k, l)，用于计算各通道对的 SCF。
    k_all, l_all = torch.meshgrid(channel, channel, indexing="ij")
    # [0, 0, ..., 0, 1, 1, ..., 1, ..., nfft-1, nfft-1, ..., nfft-1]
    k_all = k_all.reshape(-1)
    # [0, 1, ..., nfft-1, 0, 1, ..., nfft-1, ..., 0, 1, ..., nfft-1]
    l_all = l_all.reshape(-1)

    # 排除 Nyquist 频点（归一化频率 -0.5），避免主定义域边界附近出现不对称伪影。
    nyquist = torch.tensor(_FAM_NYQUIST_BIN, device=x.device, dtype=real_dtype)
    valid = (~torch.isclose(freqs[k_all], nyquist)) & (
        ~torch.isclose(freqs[l_all], nyquist)
    )
    k_all = k_all[valid]
    l_all = l_all[valid]

    scale = frame_count * window_energy

    # 根据 pair_chunk_size 分批处理频率通道对。
    for start in range(0, k_all.numel(), pair_chunk_size):
        k = k_all[start : start + pair_chunk_size]
        l = l_all[start : start + pair_chunk_size]
        fk = freqs[k]
        fl = freqs[l]

        # Step4：将 channelized subblocks 相乘并做 Fourier transform
        product = x_tilde[:, k] * torch.conj(x_tilde[:, l])
        z = torch.fft.fft(product.T, dim=1)
        z = torch.fft.fftshift(z, dim=1)
        z = z / scale

        f_value = 0.5 * (fk + fl)
        alpha = (fk - fl)[:, None] + beta[None, :]
        f = f_value[:, None].expand_as(alpha)

        # Step 5: 将每个 Fourier transform 输出关联到正确的 (f, alpha)
        mask = torch.abs(f) + 0.5 * torch.abs(alpha) < _PRINCIPAL_DOMAIN_BOUNDARY
        if not torch.any(mask):
            continue
        yield f[mask], alpha[mask], z[mask]


def compute_fam_grid(
    x: torch.Tensor,
    *,
    nfft: int,
    hop: int,
    f_bins: int,
    alpha_bins: int,
    n_blocks: Optional[int] = None,
    window: str = "hamming",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.complex64,
    pair_chunk_size: int = 8192,
    f_range: tuple[float, float] = FAM_F_RANGE,
    alpha_range: tuple[float, float] = FAM_ALPHA_RANGE,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Use FAM to estimate |SCF| and aggregate it into a grid on the torch device.
    """

    _validate_fam_grid_args(
        x,
        nfft=nfft,
        hop=hop,
        f_bins=f_bins,
        alpha_bins=alpha_bins,
        num_frames=n_blocks,
        pair_chunk_size=pair_chunk_size,
        f_range=f_range,
        alpha_range=alpha_range,
    )

    resolved_device = resolve_device(device)
    real_dtype = get_real_dtype(dtype)
    grid_size = alpha_bins * f_bins

    # 累计每个网格 bin 的 SCF 幅值总和及落点数量。
    bin_magnitude_sum = torch.zeros(grid_size, device=resolved_device, dtype=real_dtype)
    bin_point_count = torch.zeros(grid_size, device=resolved_device, dtype=real_dtype)

    f_min, f_max = f_range
    alpha_min, alpha_max = alpha_range
    x_tensor = x.to(device=resolved_device, dtype=dtype)

    for f, alpha, value in _iter_fam_point_batches(
        x_tensor,
        nfft=nfft,
        hop=hop,
        num_frames=n_blocks,
        window=window,
        real_dtype=real_dtype,
        pair_chunk_size=pair_chunk_size,
    ):
        f_idx = torch.floor((f - f_min) / (f_max - f_min) * (f_bins - 1)).to(
            torch.long
        )
        a_idx = torch.floor(
            (alpha - alpha_min) / (alpha_max - alpha_min) * (alpha_bins - 1)
        ).to(torch.long)

        # 过滤网格索引越界的点，当前批次无有效点时直接跳过。
        mask = (f_idx >= 0) & (f_idx < f_bins) & (a_idx >= 0) & (a_idx < alpha_bins)
        if not torch.any(mask):
            continue

        # 将有效二维网格坐标线性化为一维 bin 索引。
        flat_idx = (a_idx[mask] * f_bins + f_idx[mask]).reshape(-1)
        # 提取有效 SCF 值的幅值并转换为目标实数类型。
        mag = torch.abs(value[mask]).to(real_dtype).reshape(-1)
        # 按 bin 累加 SCF 幅值。
        bin_magnitude_sum.scatter_add_(0, flat_idx, mag)
        # 按 bin 累加落点数量。
        bin_point_count.scatter_add_(0, flat_idx, torch.ones_like(mag))

    # 初始化展平的 SCF 幅值网格，空 bin 保持为零。
    scf_magnitude_grid = torch.zeros_like(bin_magnitude_sum)
    # 标记至少包含一个 FAM 点的非空 bin。
    count_mask = bin_point_count > 0
    # 计算非空 bin 的平均 SCF 幅值，避免除零。
    scf_magnitude_grid[count_mask] = (
        bin_magnitude_sum[count_mask] / bin_point_count[count_mask]
    )
    # 恢复为 (alpha, f) 二维网格。
    scf_magnitude_grid = scf_magnitude_grid.reshape(alpha_bins, f_bins)

    f_axis = np.linspace(f_range[0], f_range[1], f_bins)
    alpha_axis = np.linspace(alpha_range[0], alpha_range[1], alpha_bins)
    return scf_magnitude_grid, f_axis, alpha_axis

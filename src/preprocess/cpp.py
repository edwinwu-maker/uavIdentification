"""CPP/FAM feature generation helpers."""

from __future__ import annotations

import numpy as np
import torch

from src.preprocess.fam import FAM_ALPHA_RANGE, FAM_F_RANGE, compute_fam_grid
from src.utils.samples import iter_padded_segments

SUPPORTED_FAM_MERGE_MODES = ("mean", "max")
DEFAULT_SEGMENT_SAMPLES = 262144


def _validate_cpp_args(
    x: torch.Tensor,
    *,
    segment_samples: int,
    segment_hop_samples: int | None,
    fam_merge: str,
) -> None:
    """校验 CPP 计算的输入张量、分段参数和合并模式。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive")
    if segment_hop_samples is not None and segment_hop_samples <= 0:
        raise ValueError("segment_hop_samples must be positive")
    if fam_merge not in SUPPORTED_FAM_MERGE_MODES:
        raise ValueError(f"unsupported merge mode: {fam_merge}")
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional complex IQ sequence")


def compute_cpp(
    x: torch.Tensor,
    *,
    segment_samples: int,
    segment_hop_samples: int | None,
    fam_merge: str,
    f_bins: int,
    alpha_bins: int,
    device: str,
    pair_chunk_size: int,
    fam_nfft: int = 256,
    fam_hop: int = 256,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Compute one single-channel CPP matrix and its axes from a complex IQ array."""

    _validate_cpp_args(
        x,
        segment_samples=segment_samples,
        segment_hop_samples=segment_hop_samples,
        fam_merge=fam_merge,
    )
    if segment_hop_samples is None:
        segment_hop_samples = segment_samples

    merged_fam_grid: torch.Tensor | None = None  # 所有分段 FAM 网格的合并结果
    segment_count = 0                       # 已处理的信号分段数量
    f_axis: np.ndarray | None = None        # FAM 网格的频率轴
    alpha_axis: np.ndarray | None = None    # FAM 网格的循环频率轴

    for segment in iter_padded_segments(
        x,
        segment_samples=segment_samples,
        segment_hop_samples=segment_hop_samples,
    ):
        segment_fam_grid, f_axis, alpha_axis = compute_fam_grid(
            segment,
            nfft=fam_nfft,
            hop=fam_hop,
            f_bins=f_bins,
            alpha_bins=alpha_bins,
            window="hann",
            device=device,
            dtype=torch.complex64,
            pair_chunk_size=pair_chunk_size,
            f_range=FAM_F_RANGE,
            alpha_range=FAM_ALPHA_RANGE,
        )

        # 按合并模式初始化结果：max 复制首个分段网格，mean 创建零张量用于累加。
        if merged_fam_grid is None:
            merged_fam_grid = (
                segment_fam_grid.clone()
                if fam_merge == "max"
                else torch.zeros_like(segment_fam_grid)
            )

        if fam_merge == "max":
            merged_fam_grid = torch.maximum(merged_fam_grid, segment_fam_grid)
        else:
            merged_fam_grid += segment_fam_grid
        segment_count += 1

    if (
        segment_count == 0
        or merged_fam_grid is None
        or f_axis is None
        or alpha_axis is None
    ):
        raise ValueError("input signal is empty")

    if fam_merge == "mean":
        merged_fam_grid = merged_fam_grid / segment_count

    # 最大值归一化
    max_value = (
        float(torch.max(merged_fam_grid)) if merged_fam_grid.numel() else 0.0
    )
    if max_value > 0:
        merged_fam_grid = merged_fam_grid / max_value

    # unsqueeze(0)：在最前面增加通道维度，使形状从 (alpha_bins, f_bins) 变为 (1, alpha_bins, f_bins)。
    cpp = merged_fam_grid.unsqueeze(0).to(dtype=torch.float32)
    return cpp, f_axis.astype(np.float32), alpha_axis.astype(np.float32)

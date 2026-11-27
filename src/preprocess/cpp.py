"""CPP/FAM feature generation helpers."""

from __future__ import annotations

import numpy as np
import torch

from src.preprocess.fam import FAM_ALPHA_RANGE, FAM_F_RANGE, compute_fam_grid
from src.utils.samples import iter_padded_segments

SUPPORTED_FAM_MERGE_MODES = ("mean", "max")
DEFAULT_SEGMENT_SAMPLES = 262144
DEFAULT_DYNAMIC_RANGE_DB = 50.0
_PRINCIPAL_DOMAIN_BOUNDARY = 0.5


def build_principal_domain_mask(
    f_axis: np.ndarray,
    alpha_axis: np.ndarray,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """根据 FAM 坐标轴构造与 floor 网格映射对齐的主定义域布尔掩码，形状为 (alpha_bins, f_bins)。"""
    f_arr = np.asarray(f_axis, dtype=np.float64)
    alpha_arr = np.asarray(alpha_axis, dtype=np.float64)
    # fam.py 使用 floor 将坐标落入 [axis[i], axis[i] + step) 网格，使用 bin 中心对齐四条菱形边界。
    df = (f_arr[-1] - f_arr[0]) / (len(f_arr) - 1) if len(f_arr) > 1 else 0.0
    da = (alpha_arr[-1] - alpha_arr[0]) / (len(alpha_arr) - 1) if len(alpha_arr) > 1 else 0.0
    f_centers = f_arr + 0.5 * df
    alpha_centers = alpha_arr + 0.5 * da
    mask_np = (np.abs(f_centers)[None, :] + 0.5 * np.abs(alpha_centers)[:, None]) < _PRINCIPAL_DOMAIN_BOUNDARY
    return torch.as_tensor(mask_np, dtype=torch.bool, device=device)


def _validate_cpp_args(
    x: torch.Tensor,
    *,
    segment_samples: int,
    segment_hop_samples: int | None,
    fam_merge: str,
    dynamic_range_db: float,
) -> None:
    """校验 CPP 计算的输入张量、分段参数、合并模式和动态范围。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive")
    if segment_hop_samples is not None and segment_hop_samples <= 0:
        raise ValueError("segment_hop_samples must be positive")
    if fam_merge not in SUPPORTED_FAM_MERGE_MODES:
        raise ValueError(f"unsupported merge mode: {fam_merge}")
    if dynamic_range_db <= 0:
        raise ValueError("dynamic_range_db must be positive")
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
    dynamic_range_db: float = DEFAULT_DYNAMIC_RANGE_DB,
    remove_dc: bool = True,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Compute one single-channel dB-compressed CPP matrix and its axes from a complex IQ array."""

    _validate_cpp_args(
        x,
        segment_samples=segment_samples,
        segment_hop_samples=segment_hop_samples,
        fam_merge=fam_merge,
        dynamic_range_db=dynamic_range_db,
    )
    if segment_hop_samples is None:
        segment_hop_samples = segment_samples
    if remove_dc and x.numel() > 0:
        # 去除 IQ 直流偏置，避免 (f=0, alpha=0) 尖峰抬高全局峰值并压扁其余循环谱特征。
        x = x - x.mean()

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

    # 相对峰值 dB 压缩：将 [-dynamic_range_db, 0] dB 映射到 [0, 1]，主定义域外无落点区域保持为 0。
    max_value = (
        float(torch.max(merged_fam_grid)) if merged_fam_grid.numel() else 0.0
    )
    if max_value > 0.0:
        floor_ratio = 10.0 ** (-dynamic_range_db / 10.0)
        valid_mask = merged_fam_grid > 0.0
        relative_mag = torch.clamp(merged_fam_grid / max_value, min=floor_ratio)
        db_normalized = torch.clamp(
            (10.0 * torch.log10(relative_mag) + dynamic_range_db) / dynamic_range_db,
            min=0.0,
            max=1.0,
        )
        merged_fam_grid = torch.where(
            valid_mask,
            db_normalized,
            torch.zeros_like(merged_fam_grid),
        )

    # unsqueeze(0)：在最前面增加通道维度，使形状从 (alpha_bins, f_bins) 变为 (1, alpha_bins, f_bins)。
    cpp = merged_fam_grid.unsqueeze(0).to(dtype=torch.float32)
    return cpp, f_axis.astype(np.float32), alpha_axis.astype(np.float32)

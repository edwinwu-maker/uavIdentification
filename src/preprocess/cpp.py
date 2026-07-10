"""CPP/FAM feature generation helpers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from src.preprocess.fam_defaults import FAM_ALPHA_RANGE, FAM_F_RANGE
from src.preprocess.fam_torch import fam_grid_torch

SUPPORTED_FAM_MERGE_MODES = ("mean", "max")
SUPPORTED_CPP_NORMALIZATION_MODES = (
    "max",
    "log-zscore-sample",
    "log_zscore_sample",
    "log-zscore-dataset",
    "log_zscore_dataset",
)
DEFAULT_SEGMENT_SAMPLES = 262144


def normalize_cpp(
    cpp: torch.Tensor,
    *,
    mode: str = "log-zscore-sample",
    eps: float = 1e-6,
) -> torch.Tensor:
    """对单个 CPP 样本执行归一化，输入形状为 (C, H, W)。"""

    if not isinstance(cpp, torch.Tensor):
        raise TypeError("cpp must be a torch.Tensor")

    cpp = cpp.to(dtype=torch.float32)
    if mode == "max":
        max_value = float(torch.max(cpp)) if cpp.numel() else 0.0
        if max_value > 0:
            return cpp / max_value
        return cpp.clone()
    if mode in ("log-zscore-sample", "log_zscore_sample"):
        logged = torch.log1p(cpp)
        std, mean = torch.std_mean(logged, dim=(1, 2), keepdim=True, unbiased=False)
        centered = logged - mean
        denominator = torch.where(std > eps, std, torch.ones_like(std))
        return torch.where(std > eps, centered / denominator, torch.zeros_like(centered))
    if mode in ("log-zscore-dataset", "log_zscore_dataset"):
        raise NotImplementedError("log_zscore_dataset requires dataset-level statistics and is not implemented here")
    raise ValueError(f"Unsupported CPP normalization mode: {mode}")


def compute_segmented_fam_grid(
    x: torch.Tensor,
    *,
    segment_samples: int = DEFAULT_SEGMENT_SAMPLES,
    segment_hop_samples: int | None = None,
    merge: str = "mean",
    f_bins: int = 257,
    alpha_bins: int = 513,
    f_range: tuple[float, float] = FAM_F_RANGE,
    alpha_range: tuple[float, float] = FAM_ALPHA_RANGE,
    device: str = "cpu",
    pair_chunk_size: int = 8192,
    fam_nfft: int = 64,
    fam_hop: int = 64,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Compute FAM grids over IQ segments and merge them into one |SCF| grid."""

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive")
    if segment_hop_samples is None:
        segment_hop_samples = segment_samples
    if segment_hop_samples <= 0:
        raise ValueError("segment_hop_samples must be positive")
    if merge not in SUPPORTED_FAM_MERGE_MODES:
        raise ValueError(f"unsupported merge mode: {merge}")

    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional complex IQ sequence")

    merged: torch.Tensor | None = None
    segment_count = 0
    f_axis: np.ndarray | None = None
    alpha_axis: np.ndarray | None = None

    for segment in _iter_padded_segments(
        x,
        segment_samples=segment_samples,
        segment_hop_samples=segment_hop_samples,
    ):
        # CPU/GPU/MPS 统一使用 torch FAM 路径，避免维护两套算法实现。
        image, f_axis, alpha_axis = fam_grid_torch(
            segment,
            nfft=fam_nfft,
            hop=fam_hop,
            f_bins=f_bins,
            alpha_bins=alpha_bins,
            window="hann",
            keep_principal_domain=True,
            device=device,
            dtype=torch.complex64,
            pair_chunk_size=pair_chunk_size,
            f_range=f_range,
            alpha_range=alpha_range,
        )

        if merged is None:
            merged = image.clone() if merge == "max" else torch.zeros_like(image)

        if merge == "max":
            merged = torch.maximum(merged, image)
        else:
            merged += image
        segment_count += 1

    if segment_count == 0 or merged is None or f_axis is None or alpha_axis is None:
        raise ValueError("input signal is empty")

    if merge == "mean":
        merged = merged / segment_count

    max_value = float(torch.max(merged)) if merged.numel() else 0.0
    if max_value > 0:
        merged = merged / max_value

    return merged, f_axis, alpha_axis


def _iter_padded_segments(
    x: torch.Tensor,
    *,
    segment_samples: int,
    segment_hop_samples: int,
) -> Iterator[torch.Tensor]:
    """Yield fixed-length IQ segments, padding the last segment with zeros."""

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")

    for start in range(0, x.numel(), segment_hop_samples):
        segment = x[start : start + segment_samples]
        if segment.numel() == 0:
            continue
        if segment.numel() < segment_samples:
            pad = torch.zeros(
                segment_samples - segment.numel(),
                dtype=segment.dtype,
                device=segment.device,
            )
            segment = torch.cat((segment, pad))
        yield segment


def compute_cpp(
    ch0: torch.Tensor,
    ch1: torch.Tensor,
    *,
    segment_samples: int,
    segment_hop_samples: int | None,
    fam_merge: str,
    f_bins: int,
    alpha_bins: int,
    device: str,
    pair_chunk_size: int,
    fam_nfft: int = 64,
    fam_hop: int = 64,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Compute one dual-channel CPP matrix and its axes from RF0/RF1 IQ arrays."""

    if not isinstance(ch0, torch.Tensor):
        raise TypeError("ch0 must be a torch.Tensor")
    if not isinstance(ch1, torch.Tensor):
        raise TypeError("ch1 must be a torch.Tensor")

    image0, f_axis, alpha_axis = compute_segmented_fam_grid(
        ch0,
        segment_samples=segment_samples,
        segment_hop_samples=segment_hop_samples,
        merge=fam_merge,
        f_bins=f_bins,
        alpha_bins=alpha_bins,
        device=device,
        pair_chunk_size=pair_chunk_size,
        fam_nfft=fam_nfft,
        fam_hop=fam_hop,
    )
    image1, _, _ = compute_segmented_fam_grid(
        ch1,
        segment_samples=segment_samples,
        segment_hop_samples=segment_hop_samples,
        merge=fam_merge,
        f_bins=f_bins,
        alpha_bins=alpha_bins,
        device=device,
        pair_chunk_size=pair_chunk_size,
        fam_nfft=fam_nfft,
        fam_hop=fam_hop,
    )
    cpp = torch.stack([image0, image1], dim=0).to(dtype=torch.float32)
    return cpp, f_axis.astype(np.float32), alpha_axis.astype(np.float32)

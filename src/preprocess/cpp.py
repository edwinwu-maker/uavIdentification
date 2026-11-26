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

    merged: torch.Tensor | None = None
    segment_count = 0
    f_axis: np.ndarray | None = None
    alpha_axis: np.ndarray | None = None

    for segment in iter_padded_segments(
        x,
        segment_samples=segment_samples,
        segment_hop_samples=segment_hop_samples,
    ):
        image, f_axis, alpha_axis = compute_fam_grid(
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

        if merged is None:
            merged = image.clone() if fam_merge == "max" else torch.zeros_like(image)

        if fam_merge == "max":
            merged = torch.maximum(merged, image)
        else:
            merged += image
        segment_count += 1

    if segment_count == 0 or merged is None or f_axis is None or alpha_axis is None:
        raise ValueError("input signal is empty")

    if fam_merge == "mean":
        merged = merged / segment_count

    max_value = float(torch.max(merged)) if merged.numel() else 0.0
    if max_value > 0:
        merged = merged / max_value

    cpp = merged.unsqueeze(0).to(dtype=torch.float32)
    return cpp, f_axis.astype(np.float32), alpha_axis.astype(np.float32)

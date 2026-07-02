"""CPP/FAM feature generation helpers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from src.preprocess.fam_constants import FAM_ALPHA_RANGE, FAM_F_RANGE
from src.preprocess.fam_torch import fam_scf_grid_torch

SUPPORTED_FAM_MERGE_MODES = ("mean", "max")
DEFAULT_SEGMENT_SAMPLES = 262144


def compute_fam_grid(
    x: torch.Tensor,
    *,
    f_bins: int = 257,
    alpha_bins: int = 513,
    f_range: tuple[float, float] = FAM_F_RANGE,
    alpha_range: tuple[float, float] = FAM_ALPHA_RANGE,
    normalize: bool = True,
    device: str = "cpu",
    pair_chunk_size: int = 8192,
    fam_nfft: int = 64,
    fam_hop: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FAM points and aggregate them into a normalized |SCF| grid."""

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")

    # CPU/GPU/MPS 统一使用 torch FAM 路径，避免维护两套算法实现。
    return fam_scf_grid_torch(
        x,
        nfft=fam_nfft,
        hop=fam_hop,
        window="hann",
        keep_principal_domain=True,
        device=device,
        dtype=torch.complex64,
        pair_chunk_size=pair_chunk_size,
        f_bins=f_bins,
        alpha_bins=alpha_bins,
        f_range=f_range,
        alpha_range=alpha_range,
        normalize=normalize,
    )


def compute_fam_grid_segmented(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute segmented FAM grids and merge them into one |SCF| grid."""

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

    merged = np.zeros((alpha_bins, f_bins), dtype=float)
    segment_count = 0

    for segment in _iter_padded_segments(
        x,
        segment_samples=segment_samples,
        segment_hop_samples=segment_hop_samples,
    ):
        image, f_axis, alpha_axis = compute_fam_grid(
            segment,
            f_bins=f_bins,
            alpha_bins=alpha_bins,
            f_range=f_range,
            alpha_range=alpha_range,
            normalize=False,
            device=device,
            pair_chunk_size=pair_chunk_size,
            fam_nfft=fam_nfft,
            fam_hop=fam_hop,
        )

        if merge == "max":
            merged = np.maximum(merged, image)
        else:
            merged += image
        segment_count += 1

    if segment_count == 0:
        raise ValueError("input signal is empty")

    if merge == "mean":
        merged = merged / segment_count

    if merged.max() > 0:
        merged = merged / merged.max()

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one dual-channel CPP matrix and its axes from RF0/RF1 IQ arrays."""

    if not isinstance(ch0, torch.Tensor):
        raise TypeError("ch0 must be a torch.Tensor")
    if not isinstance(ch1, torch.Tensor):
        raise TypeError("ch1 must be a torch.Tensor")

    image0, f_axis, alpha_axis = compute_fam_grid_segmented(
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
    image1, _, _ = compute_fam_grid_segmented(
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
    cpp = np.stack([image0, image1], axis=0).astype(np.float32)
    return cpp, f_axis.astype(np.float32), alpha_axis.astype(np.float32)

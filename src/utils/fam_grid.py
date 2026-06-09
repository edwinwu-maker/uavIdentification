"""Grid and segmented-grid helpers for FAM SCF point estimates."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from src.utils.fam import fam_scf_grid
from src.utils.fam_torch import fam_scf_grid_torch

SUPPORTED_FAM_MERGE_MODES = ("mean", "max")
FAM_NFFT = 64
FAM_HOP = 64
DEFAULT_SEGMENT_SAMPLES = 262144


def compute_fam_grid(
    x: np.ndarray,
    *,
    f_bins: int = 257,
    alpha_bins: int = 513,
    f_range: tuple[float, float] = (-0.5, 0.5),
    alpha_range: tuple[float, float] = (-1.0, 1.0),
    normalize: bool = True,
    device: str = "cpu",
    pair_chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FAM points and aggregate them into a normalized |SCF| grid."""

    if not device.lower().startswith("cpu"):
        # GPU/MPS paths aggregate on the torch device to avoid materializing and
        # copying all sparse FAM points before gridding.
        return fam_scf_grid_torch(
            x,
            nfft=FAM_NFFT,
            hop=FAM_HOP,
            window="hann",
            keep_principal_domain=True,
            device=device,
            pair_chunk_size=pair_chunk_size,
            f_bins=f_bins,
            alpha_bins=alpha_bins,
            f_range=f_range,
            alpha_range=alpha_range,
            normalize=normalize,
        )

    return fam_scf_grid(
        x,
        nfft=FAM_NFFT,
        hop=FAM_HOP,
        window="hann",
        keep_principal_domain=True,
        f_bins=f_bins,
        alpha_bins=alpha_bins,
        f_range=f_range,
        alpha_range=alpha_range,
        normalize=normalize,
    )


def iter_padded_segments(
    x: np.ndarray,
    *,
    segment_samples: int,
    segment_hop_samples: int,
) -> Iterator[np.ndarray]:
    """Yield fixed-length IQ segments, padding the last segment with zeros."""

    for start in range(0, len(x), segment_hop_samples):
        segment = x[start : start + segment_samples]
        if len(segment) == 0:
            continue
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))
        yield segment


def compute_fam_grid_segmented(
    x: np.ndarray,
    *,
    segment_samples: int = DEFAULT_SEGMENT_SAMPLES,
    segment_hop_samples: int | None = None,
    merge: str = "mean",
    f_bins: int = 257,
    alpha_bins: int = 513,
    f_range: tuple[float, float] = (-0.5, 0.5),
    alpha_range: tuple[float, float] = (-1.0, 1.0),
    device: str = "cpu",
    pair_chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute segmented FAM grids and merge them into one |SCF| grid."""

    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive")
    if segment_hop_samples is None:
        segment_hop_samples = segment_samples
    if segment_hop_samples <= 0:
        raise ValueError("segment_hop_samples must be positive")
    if merge not in SUPPORTED_FAM_MERGE_MODES:
        raise ValueError(f"unsupported merge mode: {merge}")

    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional complex IQ sequence")

    merged = np.zeros((alpha_bins, f_bins), dtype=float)
    segment_count = 0

    for segment in iter_padded_segments(
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

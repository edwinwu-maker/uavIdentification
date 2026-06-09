"""Grid and segmented-grid helpers for FAM SCF point estimates."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from src.utils.fam import fam_scf_points
from src.utils.fam_torch import fam_scf_points_torch

SUPPORTED_FAM_MERGE_MODES = ("mean", "max")
FAM_NFFT = 64
FAM_HOP = 64
DEFAULT_SEGMENT_SAMPLES = 262144


def points_to_grid(
    f: np.ndarray,
    alpha: np.ndarray,
    value: np.ndarray,
    *,
    f_bins: int = 257,
    alpha_bins: int = 513,
    f_range: tuple[float, float] = (-0.5, 0.5),
    alpha_range: tuple[float, float] = (-1.0, 1.0),
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate sparse FAM points into a drawable 2D |SCF| grid."""

    f_idx = np.floor(
        (f - f_range[0]) / (f_range[1] - f_range[0]) * (f_bins - 1)
    ).astype(int)
    a_idx = np.floor(
        (alpha - alpha_range[0])
        / (alpha_range[1] - alpha_range[0])
        * (alpha_bins - 1)
    ).astype(int)

    valid = (
        (f_idx >= 0)
        & (f_idx < f_bins)
        & (a_idx >= 0)
        & (a_idx < alpha_bins)
    )
    mag = np.abs(value)
    image_sum = np.zeros((alpha_bins, f_bins), dtype=float)
    image_count = np.zeros((alpha_bins, f_bins), dtype=int)

    np.add.at(image_sum, (a_idx[valid], f_idx[valid]), mag[valid])
    np.add.at(image_count, (a_idx[valid], f_idx[valid]), 1)

    image = np.zeros((alpha_bins, f_bins), dtype=float)
    np.divide(image_sum, image_count, out=image, where=image_count > 0)

    if normalize and image.max() > 0:
        image = image / image.max()

    f_axis = np.linspace(f_range[0], f_range[1], f_bins)
    alpha_axis = np.linspace(alpha_range[0], alpha_range[1], alpha_bins)
    return image, f_axis, alpha_axis


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

    if device.lower().startswith("cpu"):
        result = fam_scf_points(
            x,
            nfft=FAM_NFFT,
            hop=FAM_HOP,
            window="hann",
            keep_principal_domain=True,
        )
    else:
        # Non-CPU devices use the PyTorch implementation so callers can pass
        # explicit devices such as "cuda", "cuda:1", or "mps".
        result = fam_scf_points_torch(
            x,
            nfft=FAM_NFFT,
            hop=FAM_HOP,
            window="hann",
            keep_principal_domain=True,
            device=device,
            pair_chunk_size=pair_chunk_size,
        )
    return points_to_grid(
        result.f,
        result.alpha,
        result.value,
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

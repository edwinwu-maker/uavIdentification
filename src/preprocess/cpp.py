"""CPP/FAM precompute helpers."""

from __future__ import annotations

import numpy as np

from src.preprocess.fam_grid import compute_fam_grid_segmented


def compute_cpp_pair(
    ch0: np.ndarray,
    ch1: np.ndarray,
    *,
    segment_samples: int,
    segment_hop_samples: int | None,
    fam_merge: str,
    f_bins: int,
    alpha_bins: int,
    device: str,
    pair_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one dual-channel CPP matrix and its axes from RF0/RF1 IQ arrays."""

    image0, f_axis, alpha_axis = compute_fam_grid_segmented(
        ch0,
        segment_samples=segment_samples,
        segment_hop_samples=segment_hop_samples,
        merge=fam_merge,
        f_bins=f_bins,
        alpha_bins=alpha_bins,
        device=device,
        pair_chunk_size=pair_chunk_size,
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
    )
    cpp = np.stack([image0, image1], axis=0).astype(np.float32)
    return cpp, f_axis.astype(np.float32), alpha_axis.astype(np.float32)

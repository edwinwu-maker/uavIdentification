"""Shared constants and small helpers for the FAM/SCF pipeline."""

from __future__ import annotations

from typing import Final

import numpy as np

SUPPORTED_FAM_WINDOWS: Final[tuple[str, ...]] = (
    "hamming",
    "hamm",
    "hann",
    "hanning",
    "rect",
    "boxcar",
    "rectangle",
)
FAM_F_RANGE: Final[tuple[float, float]] = (-0.5, 0.5)
FAM_ALPHA_RANGE: Final[tuple[float, float]] = (-1.0, 1.0)
PRINCIPAL_DOMAIN_BOUNDARY: Final[float] = 0.5
FAM_NYQUIST_BIN: Final[float] = -0.5


def principal_domain_mask(f: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return the standard non-conjugate SCF principal-domain mask."""

    return np.abs(f) + 0.5 * np.abs(alpha) < PRINCIPAL_DOMAIN_BOUNDARY

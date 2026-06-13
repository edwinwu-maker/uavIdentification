"""Shared helpers for converting DroneRFa .mat files into feature .h5 files."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from tqdm import tqdm

from src.utils.logger import logger

__all__ = ["resolve_output_dir", "output_h5_path", "run_precompute_batch"]


def resolve_output_dir(data_dir: str, output_dir: str | None, default_subdir: str) -> str:
    return os.path.expanduser(output_dir) if output_dir else os.path.join(data_dir, default_subdir)


def output_h5_path(mat_path: str, output_dir: str) -> str:
    mat_name = os.path.basename(mat_path)
    return os.path.join(output_dir, mat_name.replace(".mat", ".h5"))


def run_precompute_batch(
    data_dir: str,
    output_dir: str,
    *,
    process_one_mat: Callable[..., tuple[str, int]],
    process_kwargs: dict[str, Any] | None = None,
    result_label: str,
) -> tuple[int, int]:
    """Run the shared .mat-to-.h5 batch loop and return file/sample counts."""

    mat_files = _list_mat_files(data_dir)
    logger.info("Found %d .mat files in %s", len(mat_files), data_dir)
    logger.info("Output directory: %s", output_dir)

    total_samples = 0
    process_kwargs = process_kwargs or {}
    for mat_file in tqdm(mat_files, desc="Processing .mat files"):
        _, num_samples = process_one_mat(
            os.path.join(data_dir, mat_file),
            output_dir,
            **process_kwargs,
        )
        total_samples += num_samples

    logger.info("All done. %d files -> %d %s", len(mat_files), total_samples, result_label)
    return len(mat_files), total_samples


def _list_mat_files(data_dir: str) -> list[str]:
    mat_files = sorted(filename for filename in os.listdir(data_dir) if filename.endswith(".mat"))
    return mat_files

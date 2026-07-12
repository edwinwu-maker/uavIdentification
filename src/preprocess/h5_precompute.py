"""Shared helpers for converting DroneRFa .mat files into feature .h5 files."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from tqdm import tqdm

from src.data.drone_rfa_io import select_mat_files
from src.utils.logger import logger

__all__ = ["output_h5_path", "run_precompute_batch"]


def output_h5_path(mat_path: str, output_dir: str) -> str:
    mat_name = os.path.basename(mat_path)
    return os.path.join(output_dir, mat_name.replace(".mat", ".h5"))


def run_precompute_batch(
    data_dir: str,
    output_dir: str,
    *,
    process_one_mat: Callable[..., tuple[str, int]],
    process_kwargs: dict[str, Any] | None = None,
    max_files: int | None = None,
    files_per_class: int | None = None,
    include_labels: list[int] | tuple[int, ...] | set[int] | None = None,
    log_label: str,
) -> tuple[int, int]:
    """
    Run the shared .mat-to-.h5 batch loop and return file/sample counts.

    process_one_mat: Callable[..., tuple[str, int]]:传入一个回调函数, 返回(字符串占位, 当前文件样本数量)
    process_kwargs: 传给 process_one_mat 的自定义参数字典
    log_label:日志文案后缀
    """

    mat_files = select_mat_files(
        data_dir,
        max_files=max_files,
        files_per_class=files_per_class,
        include_labels=include_labels,
    )
    logger.info("Found %d .mat files in %s", len(mat_files), data_dir)
    logger.info("Output directory: %s", output_dir)

    total_samples = 0
    process_kwargs = process_kwargs or {}
    for mat_file in tqdm(mat_files, desc="Processing .mat files"):
        _, num_samples = process_one_mat(
            os.path.join(data_dir, mat_file),
            output_dir,
            **process_kwargs,   # 字典解包语法
        )
        total_samples += num_samples

    logger.info("All done. %d files -> %d %s", len(mat_files), total_samples, log_label)
    return len(mat_files), total_samples

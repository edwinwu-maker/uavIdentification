"""Shared h5-to-PNG export helpers for feature visualization scripts."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterable
from typing import Any

import h5py
import tqdm

from src.utils.logger import logger


def default_num_workers() -> int:
    """Return the conservative worker count used by PNG export scripts."""

    return min(2, max(1, multiprocessing.cpu_count() // 2))


def limited_sample_count(total_samples: int, max_samples_per_file: int | None) -> int:
    if max_samples_per_file is None:
        return total_samples
    return min(total_samples, max_samples_per_file)


def sample_save_path(
    save_root: str,
    drone_code: str,
    name_no_extension: str,
    sample_idx: int,
) -> str:
    png_name = f"{name_no_extension}_sample_{sample_idx:04d}.png"
    return os.path.join(save_root, drone_code, png_name)


def list_h5_files(h5_dir: str, max_files: int | None = None) -> list[str]:
    h5_files = [
        os.path.join(h5_dir, filename)
        for filename in sorted(os.listdir(h5_dir))
        if filename.endswith(".h5")
    ]
    if max_files is not None:
        return h5_files[:max_files]
    return h5_files


def process_h5_samples(
    h5_path: str,
    save_root: str,
    *,
    feature_key: str,
    build_tasks: Callable[[h5py.File, str, str, str, int | None], Iterable[Any]],
    process_sample: Callable[[Any], None],
    max_samples_per_file: int | None = None,
    num_workers: int | None = None,
) -> None:
    """Read one h5 file and dispatch per-sample PNG export tasks."""

    h5_name = os.path.basename(h5_path)
    name_no_extension = os.path.splitext(h5_name)[0]
    drone_code = name_no_extension.split("_")[0]

    logger.info("Processing: %s", h5_name)
    with h5py.File(h5_path, "r") as h5f:
        feature_shape = h5f[feature_key].shape
        if len(feature_shape) != 4 or feature_shape[1] != 1:
            raise ValueError(
                f"Feature dataset must have shape (N, 1, H, W), got {feature_shape} in {h5_path}. "
                "Re-run precomputation; legacy dual-channel caches are not supported."
            )
        if "rf_channel" not in h5f.attrs or int(h5f.attrs["rf_channel"]) not in (0, 1):
            raise ValueError(f"Missing or invalid rf_channel attribute in {h5_path}; re-run precomputation")
        num_samples = limited_sample_count(h5f[feature_key].shape[0], max_samples_per_file)
        tasks = build_tasks(
            h5f,
            name_no_extension,
            drone_code,
            save_root,
            max_samples_per_file,
        )

        if num_workers is None:
            num_workers = default_num_workers()
        if num_workers == 1:
            for task in tqdm.tqdm(tasks, total=num_samples, desc=f"  {h5_name}"):
                process_sample(task)
        else:
            with multiprocessing.Pool(num_workers) as pool:
                list(
                    tqdm.tqdm(
                        pool.imap_unordered(process_sample, tasks, chunksize=1),
                        total=num_samples,
                        desc=f"  {h5_name}",
                    )
                )

    logger.info("Done: %s", h5_name)

"""Shared h5-to-PNG export helpers for feature visualization scripts."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterable
from typing import Any

import h5py
import tqdm

from src.utils.logger import logger
from src.utils.device import default_cpu_cores
from src.utils.samples import limited_samples


def build_sample_png_path(
    save_root: str,
    drone_code: str,
    name_no_extension: str,
    sample_idx: int,
    rf_channel: int,
) -> str:
    """按类别子目录生成 PNG 路径，文件名保留源文件名、RF 通道及样本序号。

    样本序号至少补齐四位；本函数只拼接路径，目录由绘图函数创建。
    """
    png_name = f"{name_no_extension}_RF{rf_channel}_sample_{sample_idx:04d}.png"
    return os.path.join(save_root, drone_code, png_name)


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
    """校验单个 H5 的单通道特征及逐样本 RF 通道信息，并分派 PNG 导出任务。

    build_tasks 在文件保持打开期间生成任务，并负责遵守每文件样本上限；
    process_sample 接收一个任务并完成绘图。num_workers 为 None 时使用
    默认进程数，为 1 时顺序执行，其余情况使用进程池并按完成数量更新进度。
    多进程模式要求处理函数可被子进程导入、任务可序列化，避免传递 H5 句柄。
    """

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
        if "rf_channel" not in h5f:
            raise ValueError(f"Missing rf_channel dataset in {h5_path}; re-run precomputation")
        rf_channels = h5f["rf_channel"][:]
        if rf_channels.shape != (feature_shape[0],) or not all(
            int(channel) in (0, 1) for channel in rf_channels
        ):
            raise ValueError(f"Invalid rf_channel dataset in {h5_path}; re-run precomputation")
        num_samples = limited_samples(h5f[feature_key].shape[0], max_samples_per_file)
        tasks = build_tasks(
            h5f,
            name_no_extension,
            drone_code,
            save_root,
            max_samples_per_file,
        )

        if num_workers is None:
            num_workers = default_cpu_cores()
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

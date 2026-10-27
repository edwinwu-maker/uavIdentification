"""H5 样本导图工具：根据样本信息生成 PNG 路径，并通过回调读取特征、保存图片。"""

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
    """根据源文件和样本信息构造 PNG 输出路径。

    参数：
        save_root: PNG 输出根目录。
        drone_code: 类别编码，如 "T0010"，用作根目录下的类别子目录名。
        name_no_extension: 源 H5 文件去掉目录和扩展名后的名称，用作 PNG 文件名前缀。
        sample_idx: 样本在源 H5 文件中的索引，从 0 开始，文件名中至少补齐四位。
        rf_channel: 当前样本所属的 RF 通道编号，通常为 0 或 1，写入文件名以区分通道。

    返回：
        字符串路径：
        <save_root>/<drone_code>/<name_no_extension>_RF<rf_channel>_sample_<sample_idx:04d>.png。
        仅拼接路径，不创建目录或写入图片，实际保存由绘图回调完成。
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
    """读取并校验单个 H5 文件，通过任务构建和绘图回调批量导出样本 PNG。

    参数：
        h5_path: 输入 H5 文件路径。feature_key 对应的数组须为 (N, 1, H, W)，
            N 为样本数，H、W 为特征图尺寸；rf_channel 数组须为 (N,)，元素为 0 或 1。
            源文件名去掉扩展名后作为图片名前缀，第一个下划线前的部分作为类别编码。
        save_root: PNG 输出根目录，传给 build_tasks，由回调确定具体路径并创建目录。
        feature_key: H5 中特征数组的键名，如 "stft" 或 "cpp"，用于校验形状和统计样本数。
        build_tasks: 任务构建回调，调用形式为
            build_tasks(h5f, name_no_extension, drone_code, save_root, max_samples_per_file)。
            h5f 是保持打开的只读 H5 句柄，其余参数分别为源文件名前缀、类别编码、
            输出根目录和样本上限；返回供 process_sample 逐项消费的可迭代任务对象。
            回调负责读取绘图数据并遵守样本上限，本函数不额外截断任务序列。
        process_sample: 单样本处理回调，接收 build_tasks 生成的一个任务对象，
            根据其中的数据和保存路径绘制、写入 PNG，返回 None。
        max_samples_per_file: 当前文件最多处理的样本数；None 表示全部样本。
            进度条总数取文件样本数与上限的较小值，实际任务数由 build_tasks 保证一致。
        num_workers: 导图进程数；None 时使用 default_cpu_cores() 的默认值，
            1 时顺序执行，大于 1 时使用进程池，按任务完成数更新进度。

    返回：
        None；PNG 文件由回调写入输出目录，本函数负责文件读取、任务分派及进度展示。

    多进程要求：
        process_sample 必须可被子进程导入，任务对象须可序列化，不应包含 H5 句柄；
        build_tasks 可惰性读取数据，任务生成和处理期间源 H5 文件始终保持打开。
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

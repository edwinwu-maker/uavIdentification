"""Generate CPP/FAM PNGs from pre-computed CPP .h5 files.

Usage:
  python scripts/generate_cpp_png.py --h5-dir ~/Desktop/dataset/droneRFa/cpp_h5
  python scripts/generate_cpp_png.py --h5-dir ~/Desktop/dataset/droneRFa/cpp_h5 --save-root outputs/figures/cpp_png
  python scripts/generate_cpp_png.py --h5-dir ~/Desktop/dataset/droneRFa/cpp_h5 --max-files 1 --max-samples-per-file 1

读取 precompute_cpp_h5.py 生成的缓存，直接绘制 CPP/FAM，不重新计算或加噪。
每张 PNG 对应一行样本，RF 通道取自该行的 rf_channel 元数据。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import tqdm

from src.data.drone_rfa_io import EXCLUDED_CLASS_CODES
from src.utils.cli import log_current_command
from src.utils.feature_config import get_feature_config
from src.utils.files import list_files_by_suffix
from src.utils.h5topng import (
    build_sample_png_path,
    validate_h5_samples,
)
from src.utils.logger import logger
from src.utils.plot import plot_cpp_sample
from src.utils.samples import limited_samples


def plot_single_channel_cpp(
    cpp_sample: np.ndarray,
    f_axis: np.ndarray,
    alpha_axis: np.ndarray,
    save_path: str,
    sample_idx: int,
    rf_channel: int,
    label: int | None = None,
) -> None:
    """将形状为 (1, H, W) 的归一化 CPP/FAM 保存为单通道特征图。"""
    title = f"RF{rf_channel} CPP - Sample {sample_idx}"
    if label is not None:
        title += f" (label={label})"
    plot_cpp_sample(cpp_sample, f_axis, alpha_axis, save_path, title=title)


def process_one_h5(
    h5_path: str,
    save_root: str,
    *,
    max_samples_per_file: int | None = None,
) -> None:
    """读取并校验 CPP H5，逐样本保存 PNG，绘图失败时记录并继续。"""
    h5_name = os.path.basename(h5_path)
    name_no_extension = os.path.splitext(h5_name)[0]
    drone_code = name_no_extension.split("_")[0]

    logger.info("Processing: %s", h5_name)
    with h5py.File(h5_path, "r") as h5f:
        validate_h5_samples(h5f, "cpp")
        cpp = h5f["cpp"]
        labels = h5f["labels"]
        f_axis = h5f["f_axis"][:]
        alpha_axis = h5f["alpha_axis"][:]
        rf_channels = h5f["rf_channel"]
        num_samples = limited_samples(cpp.shape[0], max_samples_per_file)

        for i in tqdm.tqdm(range(num_samples), total=num_samples, desc=f"  {h5_name}"):
            rf_channel = int(rf_channels[i])
            save_path = build_sample_png_path(save_root, drone_code, name_no_extension, i, rf_channel)
            # 逐行读取放在 try 外，读取错误向外传播，仅绘图失败时继续。
            cpp_slice = cpp[i]
            label = int(labels[i])
            try:
                plot_single_channel_cpp(cpp_slice, f_axis, alpha_axis, save_path, i, rf_channel, label)
            except Exception as e:
                logger.error("Sample %d: %s", i, e)

    logger.info("Done: %s", h5_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CPP/FAM PNGs from pre-computed CPP .h5 files")
    parser.add_argument("--h5-dir", type=str, default=get_feature_config("cpp").default_data_dir,
                        help="Directory containing CPP .h5 files")
    parser.add_argument("--save-root", type=str, default=None,
                        help="Directory for output PNG files (default: sibling cpp_png directory next to --h5-dir)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Process at most this many .h5 files")
    parser.add_argument("--max-samples-per-file", type=int, default=None,
                        help="Process at most this many samples from each .h5 file")
    return parser.parse_args(argv)


def main() -> None:
    # 解析运行参数并记录命令，便于复现导图配置。
    args = parse_args()
    log_current_command(logger)

    # 展开用户目录，默认在输入目录旁创建 cpp_png 输出目录。
    h5_dir = os.path.expanduser(args.h5_dir)
    if args.save_root is None:
        save_root = str(Path(h5_dir).parent / "cpp_png")
    else:
        save_root = os.path.expanduser(args.save_root)
    os.makedirs(save_root, exist_ok=True)

    # 按稳定文件顺序扫描，先排除研究范围外的类别，再应用文件数量限制。
    h5_files = [
        path for path in list_files_by_suffix(h5_dir, ".h5")
        if Path(path).stem.split("_")[0] not in EXCLUDED_CLASS_CODES
    ]
    if args.max_files is not None:
        h5_files = h5_files[:args.max_files]

    logger.info("Found %d .h5 files in %s", len(h5_files), h5_dir)
    logger.info("Output directory: %s", save_root)

    # 逐文件、逐样本顺序绘图；单文件处理函数负责校验及句柄关闭。
    for h5_path in tqdm.tqdm(h5_files, desc="Processing .h5 files"):
        process_one_h5(
            h5_path,
            save_root,
            max_samples_per_file=args.max_samples_per_file,
        )

    logger.info("All done!")


if __name__ == "__main__":
    main()

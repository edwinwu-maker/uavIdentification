"""Generate STFT PNGs from pre-computed .h5 files.

Usage:
  python scripts/generate_stft_png.py --h5-dir ~/Desktop/dataset/droneRFa/stft_h5
  python scripts/generate_stft_png.py --h5-dir ~/Desktop/dataset/droneRFa/stft_h5 --save-root outputs/figures/stft_png
  python scripts/generate_stft_png.py --h5-dir ~/Desktop/dataset/droneRFa/stft_h5 --max-files 1 --max-samples-per-file 1

读取 precompute_stft_h5.py 生成的缓存，直接绘制 STFT，不重新计算或加噪。
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
from src.utils.plot import plot_stft_sample, stft_plot_axes
from src.utils.samples import limited_samples

FS = 100e6
SAMPLE_LENGTH = 10_000_000


def plot_single_channel(
    stft_sample: np.ndarray,
    save_path: str,
    sample_idx: int,
    rf_channel: int,
    label: int | None = None,
    *,
    sample_length: int = SAMPLE_LENGTH,
    n_fft: int = 2048,
    hop_length: int = 1024,
    source_time_bins: int | None = None,
) -> None:
    """将形状为 (1, H, W) 的 z-score STFT 保存为单通道频谱图。"""
    # 从原始网格生成显示坐标；非整除时近似插值，单位换算由绘图工具完成。
    n_freqs, n_times = stft_sample.shape[1], stft_sample.shape[2]
    source_time_bins = source_time_bins or sample_length // hop_length + 1
    freqs, times = stft_plot_axes(
        sample_rate=FS,
        n_fft=n_fft,
        hop_length=hop_length,
        source_time_bins=source_time_bins,
        output_freq_bins=n_freqs,
        output_time_bins=n_times,
    )

    title = f"RF{rf_channel} — Sample {sample_idx}"
    if label is not None:
        title += f" (label={label})"
    plot_stft_sample(stft_sample, freqs, times, save_path, title=title)


def process_one_h5(
    h5_path: str,
    save_root: str,
    *,
    max_samples_per_file: int | None = None,
) -> None:
    """读取并校验 STFT H5，逐样本保存 PNG，绘图失败时记录并继续。"""
    h5_name = os.path.basename(h5_path)
    name_no_extension = os.path.splitext(h5_name)[0]
    drone_code = name_no_extension.split("_")[0]

    logger.info("Processing: %s", h5_name)
    with h5py.File(h5_path, "r") as h5f:
        validate_h5_samples(h5f, "stft")
        stft = h5f["stft"]
        labels = h5f["labels"]
        rf_channels = h5f["rf_channel"]
        # 缺省属性沿用预计算默认值；时间帧数缺省时由绘图函数推导。
        stft_metadata: dict[str, int | None] = {
            "sample_length": int(h5f.attrs.get("sample_length", SAMPLE_LENGTH)),
            "n_fft": int(h5f.attrs.get("n_fft", 2048)),
            "hop_length": int(h5f.attrs.get("hop_length", 1024)),
            "source_time_bins": int(h5f.attrs.get("source_time_bins", 0)) or None,
        }
        num_samples = limited_samples(stft.shape[0], max_samples_per_file)

        for i in tqdm.tqdm(range(num_samples), total=num_samples, desc=f"  {h5_name}"):
            rf_channel = int(rf_channels[i])
            save_path = build_sample_png_path(save_root, drone_code, name_no_extension, i, rf_channel)
            # 逐行读取放在 try 外，读取错误向外传播，仅绘图失败时继续。
            stft_slice = stft[i]
            label = int(labels[i])
            try:
                plot_single_channel(
                    stft_slice, save_path, i, rf_channel, label, **stft_metadata
                )
            except Exception as e:
                logger.error("Sample %d: %s", i, e)

    logger.info("Done: %s", h5_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate STFT PNGs from pre-computed STFT .h5 files")
    parser.add_argument("--h5-dir", type=str, default=get_feature_config("stft").default_data_dir,
                        help="Directory containing STFT .h5 files")
    parser.add_argument("--save-root", type=str, default=None,
                        help="Directory for output PNG files (default: sibling stft_png directory next to --h5-dir)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Process at most this many .h5 files")
    parser.add_argument("--max-samples-per-file", type=int, default=None,
                        help="Process at most this many samples from each .h5 file")
    return parser.parse_args(argv)


def main() -> None:
    # 解析运行参数并记录命令，便于复现导图配置。
    args = parse_args()
    log_current_command(logger)

    # 展开用户目录，默认在输入目录旁创建 stft_png 输出目录。
    h5_dir = os.path.expanduser(args.h5_dir)
    if args.save_root is None:
        save_root = str(Path(h5_dir).parent / "stft_png")
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

    logger.info("Found %d .h5 files", len(h5_files))
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

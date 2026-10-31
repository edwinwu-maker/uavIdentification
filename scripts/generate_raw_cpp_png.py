"""Generate single-channel CPP/FAM PNGs directly from raw DroneRFa IQ files.

Usage:
  python scripts/generate_raw_cpp_png.py --data-dir ~/Desktop/dataset/droneRFa --snrs -10 0 10
  python scripts/generate_raw_cpp_png.py --data-dir ~/Desktop/dataset/droneRFa --clean
  python scripts/generate_raw_cpp_png.py --data-dir ... --snrs 0 --device cuda:0 --pair-chunk-size 4096
  python scripts/generate_raw_cpp_png.py --data-dir ... --snrs -5 5 --max-files 1 --max-samples-per-file 1

由原始 IQ 计算 CPP/FAM；--clean 与 --snrs 必选其一，每张 PNG 对应一个 RF 通道样本。
T0000 背景文件分别处理 RF0/RF1，其他文件按 S 编码选择通道：
S0000-S0111 使用 RF0，S1000-S1111 使用 RF1；样本上限按通道分别生效。
噪声种子由文件名、RF 通道和样本索引派生，保持这些信息及 noise_seed 可复现噪声。
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
import torch
from tqdm import tqdm

from src.data.drone_rfa_io import (
    count_iq_samples,
    parse_label,
    read_iq_samples,
    rf_channels_from_filename,
    select_mat_files,
    signal_code_from_filename,
)
from src.preprocess.cpp import (
    DEFAULT_SEGMENT_SAMPLES,
    SUPPORTED_FAM_MERGE_MODES,
    compute_cpp,
    normalize_cpp,
)
from src.preprocess.random_snr_awgn import add_random_snr_awgn_with_snr
from src.utils.cli import log_current_command
from src.utils.device import default_device
from src.utils.logger import logger
from src.utils.paths import build_raw_sample_png_path, default_raw_data_dir
from src.utils.plot import plot_cpp_sample

SAMPLE_LENGTH = 1_000_000
F_BINS = 257
ALPHA_BINS = 257
FAM_NFFT = 256
FAM_HOP = 256


def plot_single_channel_cpp(
    cpp_sample: np.ndarray,
    f_axis: np.ndarray,
    alpha_axis: np.ndarray,
    save_path: Path,
    *,
    sample_idx: int,
    signal_code: str,
    rf_channel: int,
    label: int,
    snr_db: float | None,
) -> None:
    """构造样本及加噪条件标题，使用共享工具保存归一化 CPP/FAM。"""

    condition = "Clean" if snr_db is None else f"SNR {snr_db:g} dB"
    title = f"{signal_code} / RF{rf_channel} CPP — Sample {sample_idx} — {condition} (label={label})"
    plot_cpp_sample(cpp_sample, f_axis, alpha_axis, save_path, title=title)


def process_one_mat(
    mat_path: str,
    save_root: Path,
    *,
    snrs: tuple[float, ...],
    clean: bool,
    sample_length: int,
    segment_samples: int,
    segment_hop_samples: int | None,
    fam_merge: str,
    fam_nfft: int,
    fam_hop: int,
    f_bins: int,
    alpha_bins: int,
    device: str,
    pair_chunk_size: int,
    max_samples_per_file: int | None,
    noise_seed: int,
) -> int:
    """读取一个原始文件，为各样本和条件导图，返回所选通道的源样本数之和。

    max_samples_per_file 分别限制每个 RF 通道，不足一个完整样本的尾部不处理。
    返回值不乘以 SNR 条件数量；噪声身份包含文件名、RF 通道及样本索引。
    读取、计算或绘图异常向外传播，源文件句柄始终通过上下文管理器关闭。
    """

    # 背景双 RF 通道独立处理，其他文件按 S 编码选择单通道。
    filename = os.path.basename(mat_path)
    rf_channels = rf_channels_from_filename(filename)
    signal_code = signal_code_from_filename(filename) or "background"
    label = parse_label(filename)
    conditions: tuple[float | None, ...] = (None,) if clean else snrs

    logger.info("Processing: %s (%s -> %s)", filename, signal_code, rf_channels)
    total_samples = 0
    # 句柄覆盖整文件的所有通道和条件，异常退出也会关闭。
    with h5py.File(mat_path, "r") as src:
        for rf_channel in rf_channels:
            # 样本上限分别作用于每个所选通道。
            num_samples = count_iq_samples(
                src, rf_channel=rf_channel, sample_length=sample_length,
                max_samples=max_samples_per_file,
            )
            total_samples += num_samples
            for sample_idx in tqdm(
                range(num_samples), total=num_samples,
                desc=f"  {filename} RF{rf_channel}", leave=False,
            ):
                iq = read_iq_samples(
                    src, rf_channel=rf_channel, sample_length=sample_length,
                    start_sample_idx=sample_idx, end_sample_idx=sample_idx + 1,
                )
                iq_tensor = torch.as_tensor(iq, dtype=torch.complex64, device=device)

                for snr_db in conditions:
                    variant_iq = iq_tensor
                    if snr_db is not None:
                        # 噪声身份包含文件、RF 通道和样本索引，保留该身份以复现结果。
                        variant_iq, _ = add_random_snr_awgn_with_snr(
                            iq_tensor, file_id=f"{filename}:RF{rf_channel}",
                            start_idx=sample_idx, snr_min=snr_db, snr_max=snr_db,
                            noise_seed=noise_seed, device=device,
                        )
                    # 单样本 IQ (1, 1, L) 取为一维信号，供分段 FAM 计算。
                    cpp, f_axis, alpha_axis = compute_cpp(
                        variant_iq[0, 0, :], segment_samples=segment_samples,
                        segment_hop_samples=segment_hop_samples, fam_merge=fam_merge,
                        f_bins=f_bins, alpha_bins=alpha_bins, device=device,
                        pair_chunk_size=pair_chunk_size, fam_nfft=fam_nfft, fam_hop=fam_hop,
                    )
                    # 归一化后保留 (1, H, W) 的单通道 CPP，沿用计算返回的坐标轴。
                    cpp_sample = normalize_cpp(cpp).detach().cpu().numpy()
                    save_path = build_raw_sample_png_path(
                        save_root, filename=filename, sample_idx=sample_idx,
                        rf_channel=rf_channel, snr_db=snr_db,
                    )
                    plot_single_channel_cpp(
                        cpp_sample, f_axis, alpha_axis, save_path, sample_idx=sample_idx,
                        signal_code=signal_code, rf_channel=rf_channel, label=label, snr_db=snr_db,
                    )
    logger.info("Done: %s -> %d samples x %d conditions", filename, total_samples, len(conditions))
    return total_samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate specified-SNR single-channel CPP/FAM PNGs from raw DroneRFa .mat files"
    )
    parser.add_argument("--data-dir", type=str, default=default_raw_data_dir(),
                        help="Directory containing raw .mat files")
    # clean 与固定 SNR 必选其一，避免隐式选择加噪条件。
    noise_group = parser.add_mutually_exclusive_group(required=True)
    noise_group.add_argument("--snrs", type=float, nargs="+",
                             help="One or more fixed target SNR values in dB, e.g. --snrs -10 0 10")
    noise_group.add_argument("--clean", action="store_true",
                             help="Use the selected raw IQ channel without adding AWGN")
    parser.add_argument("--save-root", type=str, default=None,
                        help="Output root (default: <data-dir-parent>/DroneRFa_cpp_snr_png)")
    parser.add_argument("--sample-length", type=int, default=SAMPLE_LENGTH,
                        help="Number of IQ points per CPP sample")
    parser.add_argument("--segment-samples", type=int, default=DEFAULT_SEGMENT_SAMPLES,
                        help="Number of IQ samples per internal FAM segment")
    parser.add_argument("--segment-hop-samples", type=int, default=None,
                        help="Hop between FAM segments (default: --segment-samples)")
    parser.add_argument("--fam-merge", choices=SUPPORTED_FAM_MERGE_MODES, default="mean",
                        help="How to merge segmented FAM grids")
    parser.add_argument("--fam-nfft", type=int, default=FAM_NFFT,
                        help="FAM FFT size")
    parser.add_argument("--fam-hop", type=int, default=FAM_HOP,
                        help="FAM hop size")
    parser.add_argument("--f-bins", type=int, default=F_BINS,
                        help="Number of frequency bins in the CPP grid")
    parser.add_argument("--alpha-bins", type=int, default=ALPHA_BINS,
                        help="Number of cyclic-frequency bins in the CPP grid")
    parser.add_argument("--device", type=str, default=default_device(),
                        help='FAM compute device: "cpu", "cuda:0", or "mps"')
    parser.add_argument("--pair-chunk-size", type=int, default=8192,
                        help="Number of (k, l) channel pairs per torch batch")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Process at most this many .mat files")
    parser.add_argument("--max-samples-per-file", type=int, default=None,
                        help="Process at most this many samples from each .mat file")
    parser.add_argument("--noise-seed", type=int, default=42,
                        help="Seed for deterministic AWGN")
    return parser.parse_args(argv)


def main() -> None:
    # 解析并校验参数，记录命令以便复现实验配置。
    args = parse_args()
    log_current_command(logger)

    # 展开输入与输出路径，并统一 CUDA 设备写法。
    data_dir = os.path.expanduser(args.data_dir)
    if args.save_root:
        save_root = Path(os.path.expanduser(args.save_root))
    else:
        save_root = Path(data_dir).parent / "DroneRFa_cpp_snr_png"
    device = "cuda:0" if args.device == "cuda" else args.device
    snrs = tuple(args.snrs or ())

    # 复用稳定的文件筛选顺序与类别规则，再应用文件数量上限。
    mat_files = select_mat_files(data_dir, max_files=args.max_files)
    if not mat_files:
        raise ValueError(f"No .mat files found in {data_dir}")

    logger.info("Found %d .mat files", len(mat_files))
    if args.clean:
        logger.info("Noise condition: clean")
    else:
        logger.info("Target SNRs: %s dB", snrs)
    logger.info("Using device: %s", device)
    logger.info("Output directory: %s", save_root)

    # 逐文件生成各条件下的 PNG，源样本数按所选 RF 通道累计。
    total_samples = 0
    for filename in tqdm(mat_files, desc="Processing .mat files"):
        total_samples += process_one_mat(
            os.path.join(data_dir, filename),
            save_root,
            snrs=snrs,
            clean=args.clean,
            sample_length=args.sample_length,
            segment_samples=args.segment_samples,
            segment_hop_samples=args.segment_hop_samples,
            fam_merge=args.fam_merge,
            fam_nfft=args.fam_nfft,
            fam_hop=args.fam_hop,
            f_bins=args.f_bins,
            alpha_bins=args.alpha_bins,
            device=device,
            pair_chunk_size=args.pair_chunk_size,
            max_samples_per_file=args.max_samples_per_file,
            noise_seed=args.noise_seed,
        )

    # 每条通道内源样本在每种条件下各生成一张 PNG。
    logger.info(
        "All done. %d files, %d source samples, %d PNGs",
        len(mat_files),
        total_samples,
        total_samples * (1 if args.clean else len(snrs)),
    )


if __name__ == "__main__":
    main()

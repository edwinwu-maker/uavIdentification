"""Generate single-channel STFT PNGs directly from raw DroneRFa IQ files.

Usage:
  python scripts/generate_raw_stft_png.py --data-dir ~/Desktop/dataset/droneRFa --snrs -10 0 10
  python scripts/generate_raw_stft_png.py --data-dir ~/Desktop/dataset/droneRFa --clean
  python scripts/generate_raw_stft_png.py --data-dir ... --snrs 0 --device cuda:0 --batch-size 1
  python scripts/generate_raw_stft_png.py --data-dir ... --snrs -5 5 --max-files 1 --max-samples-per-file 1

由原始 IQ 计算 STFT；--clean 与 --snrs 必选其一，每张 PNG 对应一个 RF 通道样本。
T0000 背景文件分别处理 RF0/RF1，其他文件按 S 编码选择通道：
S0000-S0111 使用 RF0，S1000-S1111 使用 RF1；样本上限按通道分别生效。
相同种子和批次划分可复现加噪结果，改变 --batch-size 会改变噪声。
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
from src.preprocess.random_snr_awgn import add_random_snr_awgn_with_snr
from src.preprocess.stft import compute_stft
from src.utils.cli import log_current_command
from src.utils.device import default_device
from src.utils.logger import logger
from src.utils.paths import build_raw_sample_png_path, default_raw_data_dir
from src.utils.plot import plot_stft_sample, stft_plot_axes

FS = 100e6
SAMPLE_LENGTH = 10_000_000
N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = 1024
OUTPUT_FREQ_BINS = 1024
OUTPUT_TIME_BINS = 1024


def plot_single_channel(
    stft_sample: np.ndarray,
    save_path: Path,
    *,
    sample_idx: int,
    signal_code: str,
    rf_channel: int,
    label: int,
    snr_db: float | None,
    freqs: np.ndarray,
    times: np.ndarray,
) -> None:
    """构造样本及加噪条件标题，使用共享工具保存 z-score STFT。"""

    condition = "Clean" if snr_db is None else f"SNR {snr_db:g} dB"
    title = f"{signal_code} / RF{rf_channel} — Sample {sample_idx} — {condition} (label={label})"
    plot_stft_sample(stft_sample, freqs, times, save_path, title=title)
    logger.debug("Saved sample %d to %s", sample_idx, save_path)


def process_one_mat(
    mat_path: str,
    save_root: Path,
    *,
    snrs: tuple[float, ...],
    clean: bool,
    sample_length: int,
    batch_size: int,
    device: str,
    max_samples_per_file: int | None,
    noise_seed: int,
) -> int:
    """读取一个原始文件，为各样本和条件导图，返回所选通道的源样本数之和。

    max_samples_per_file 分别限制每个 RF 通道，不足一个完整样本的尾部不处理。
    返回值不乘以 SNR 条件数量；相同 noise_seed、文件和批次划分可复现噪声。
    读取、计算或绘图异常向外传播，源文件句柄始终通过上下文管理器关闭。
    """

    # 背景双 RF 通道独立处理，其他文件按 S 编码选择单通道。
    filename = os.path.basename(mat_path)
    rf_channels = rf_channels_from_filename(filename)
    signal_code = signal_code_from_filename(filename) or "background"
    label = parse_label(filename)
    freqs, times = stft_plot_axes(
        sample_rate=FS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        source_time_bins=sample_length // HOP_LENGTH + 1,
        output_freq_bins=OUTPUT_FREQ_BINS,
        output_time_bins=OUTPUT_TIME_BINS,
    )
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
            batch_starts = range(0, num_samples, batch_size)
            for start_idx in tqdm(
                batch_starts, total=len(batch_starts),
                desc=f"  {filename} RF{rf_channel}", leave=False,
            ):
                end_idx = min(start_idx + batch_size, num_samples)
                iq_batch = read_iq_samples(
                    src, rf_channel=rf_channel, sample_length=sample_length,
                    start_sample_idx=start_idx, end_sample_idx=end_idx,
                )
                # 保留单通道批次形状 (B, 1, L)，末批按实际样本数读取。
                iq_tensor = torch.as_tensor(iq_batch, dtype=torch.complex64, device=device)

                for snr_db in conditions:
                    variant_iq = iq_tensor
                    if snr_db is not None:
                        # 噪声身份包含文件、RF 通道和批次起点，批次边界影响复现结果。
                        variant_iq, _ = add_random_snr_awgn_with_snr(
                            iq_tensor, file_id=f"{filename}:RF{rf_channel}",
                            start_idx=start_idx, snr_min=snr_db, snr_max=snr_db,
                            noise_seed=noise_seed, device=device,
                        )
                    stft_batch = compute_stft(
                        variant_iq, device=device, n_fft=N_FFT, win_length=WIN_LENGTH,
                        hop_length=HOP_LENGTH, output_freq_bins=OUTPUT_FREQ_BINS,
                        output_time_bins=OUTPUT_TIME_BINS,
                    ).cpu().numpy()
                    for batch_offset, stft_sample in enumerate(stft_batch):
                        sample_idx = start_idx + batch_offset
                        save_path = build_raw_sample_png_path(
                            save_root, filename=filename, sample_idx=sample_idx,
                            rf_channel=rf_channel, snr_db=snr_db,
                        )
                        plot_single_channel(
                            stft_sample, save_path, sample_idx=sample_idx,
                            signal_code=signal_code, rf_channel=rf_channel, label=label,
                            snr_db=snr_db, freqs=freqs, times=times,
                        )
    logger.info("Done: %s -> %d samples x %d conditions", filename, total_samples, len(conditions))
    return total_samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate specified-SNR single-channel STFT PNGs from raw DroneRFa .mat files"
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
                        help="Output root (default: <data-dir-parent>/DroneRFa_stft_snr_png)")
    parser.add_argument("--sample-length", type=int, default=SAMPLE_LENGTH,
                        help="Number of IQ points per STFT sample")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="STFT batch size")
    parser.add_argument("--device", type=str, default=default_device(),
                        help='Torch device, e.g. "cpu", "cuda:0", or "mps"')
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
        save_root = Path(data_dir).parent / "DroneRFa_stft_snr_png"
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
            batch_size=args.batch_size,
            device=device,
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

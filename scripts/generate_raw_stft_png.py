"""Generate single-channel STFT PNGs directly from raw DroneRFa IQ files.

Usage:
  python scripts/generate_raw_stft_png.py --data-dir ~/Desktop/dataset/droneRFa --snrs -10 0 10
  python scripts/generate_raw_stft_png.py --data-dir ~/Desktop/dataset/droneRFa --clean
  python scripts/generate_raw_stft_png.py --data-dir ... --snrs 0 --device cuda:0 --batch-size 8
  python scripts/generate_raw_stft_png.py --data-dir ... --snrs -5 5 --max-files 1 --max-samples-per-file 1

The source filename selects one RF channel: S0000-S0111 uses RF0 and
S1000-S1111 uses RF1. T0000 background files keep the project convention of
using RF0. AWGN is added to the selected complex IQ signal before STFT.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.drone_rfa_io import (
    count_iq_samples,
    default_raw_data_dir,
    parse_label,
    read_iq_batch,
    rf_channel_for_file,
    select_mat_files,
)
from src.evaluation.snr_accuracy import format_snr_for_filename
from src.preprocess.random_snr_awgn import add_random_snr_awgn_with_snr
from src.preprocess.stft import compute_stft
from src.utils.cli import log_current_command
from src.utils.device import default_device
from src.utils.logger import logger

FS = 100e6
SAMPLE_LENGTH = 1_000_000
N_FFT = 1024
WIN_LENGTH = 1024
SPEC_TIME_BINS = 1024
OUTPUT_FREQ_BINS = 512
OUTPUT_TIME_BINS = 512


def _default_save_root(data_dir: str) -> str:
    return str(Path(data_dir).parent / "DroneRFa_stft_snr_png")


def _signal_code_for_title(filename: str) -> str:
    """返回用于标题显示的 S 编码；背景类使用明确的 background 标记。"""

    stem = Path(filename).stem
    if stem.split("_")[0] == "T0000":
        return "background"
    signal_codes = [token for token in stem.split("_") if re.fullmatch(r"S[01]{4}", token)]
    if len(signal_codes) != 1:
        # 通道解析函数会给出统一的文件名错误；这里不维护第二套校验文案。
        rf_channel_for_file(filename)
    return signal_codes[0]


def _pooled_axis(values: np.ndarray, output_bins: int) -> np.ndarray:
    """按 STFT 自适应平均池化的等宽分组计算坐标中心。"""

    if len(values) % output_bins != 0:
        return np.linspace(values[0], values[-1], output_bins)
    return values.reshape(output_bins, -1).mean(axis=1)


def stft_plot_axes(sample_length: int) -> tuple[np.ndarray, np.ndarray]:
    """生成与 compute_stft 的裁剪和池化过程对应的频率、时间坐标。"""

    hop_length = sample_length // (SPEC_TIME_BINS - 1)
    source_freqs = np.fft.fftshift(np.fft.fftfreq(N_FFT, 1.0 / FS))
    source_times = np.arange(SPEC_TIME_BINS, dtype=np.float64) * hop_length / FS
    freqs = _pooled_axis(source_freqs, OUTPUT_FREQ_BINS)
    times = _pooled_axis(source_times, OUTPUT_TIME_BINS)
    return freqs, times


def plot_single_channel(
    stft_sample: np.ndarray,
    save_path: Path,
    *,
    filename: str,
    sample_idx: int,
    signal_code: str,
    rf_channel: int,
    label: int,
    snr_db: float | None,
    freqs: np.ndarray,
    times: np.ndarray,
) -> None:
    """按现有 generate_stft_png.py 风格保存单通道 z-score STFT。"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 5), constrained_layout=True)
    image = ax.pcolormesh(
        freqs / 1e6,
        times * 1e3,
        stft_sample[0].T,
        shading="auto",
        cmap="jet",
    )
    condition = "Clean" if snr_db is None else f"SNR {snr_db:g} dB"
    ax.set_title(
        f"{signal_code} / RF{rf_channel} — Sample {sample_idx} — {condition} (label={label})",
        fontsize=12,
    )
    ax.set_xlabel("Frequency (MHz)", fontsize=10)
    ax.set_ylabel("Time (ms)", fontsize=10)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.92)
    colorbar.set_label("Normalized Power (z-score)", fontsize=9)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved %s sample %d to %s", filename, sample_idx, save_path)


def _sample_save_path(
    save_root: Path,
    *,
    filename: str,
    sample_idx: int,
    snr_db: float | None,
) -> Path:
    stem = Path(filename).stem
    class_code = stem.split("_")[0]
    if snr_db is None:
        return save_root / "clean" / class_code / stem / f"{stem}_sample_{sample_idx:04d}_clean.png"
    snr_name = format_snr_for_filename(snr_db)
    png_name = f"{stem}_sample_{sample_idx:04d}_snr_{snr_name}db.png"
    return save_root / f"snr_{snr_name}db" / class_code / stem / png_name


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
    """读取一个原始文件，并为每个样本和目标 SNR 输出一张 PNG。"""

    filename = os.path.basename(mat_path)
    rf_channel = rf_channel_for_file(filename)
    signal_code = _signal_code_for_title(filename)
    label = parse_label(filename)
    freqs, times = stft_plot_axes(sample_length)
    conditions: tuple[float | None, ...] = (None,) if clean else snrs

    logger.info("Processing: %s (%s -> RF%d)", filename, signal_code, rf_channel)
    with h5py.File(mat_path, "r") as src:
        num_samples = count_iq_samples(
            src,
            rf_channel=rf_channel,
            sample_length=sample_length,
            max_samples=max_samples_per_file,
        )
        batch_starts = range(0, num_samples, batch_size)
        for start_idx in tqdm(
            batch_starts,
            total=math.ceil(num_samples / batch_size),
            desc=f"  {filename}",
            leave=False,
        ):
            end_idx = min(start_idx + batch_size, num_samples)
            iq_batch = read_iq_batch(
                src,
                rf_channel=rf_channel,
                sample_length=sample_length,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            iq_tensor = torch.as_tensor(iq_batch, dtype=torch.complex64, device=device)

            for snr_db in conditions:
                variant_iq = iq_tensor
                if snr_db is not None:
                    # 所有 SNR 复用相同 seed 维度，使噪声形状一致，仅缩放不同。
                    variant_iq, _ = add_random_snr_awgn_with_snr(
                        iq_tensor,
                        file_id=filename,
                        start_idx=start_idx,
                        snr_min=snr_db,
                        snr_max=snr_db,
                        noise_seed=noise_seed,
                        device=device,
                    )
                stft_batch = compute_stft(
                    variant_iq,
                    device=device,
                    n_fft=N_FFT,
                    win_length=WIN_LENGTH,
                    spec_time_bins=SPEC_TIME_BINS,
                    output_freq_bins=OUTPUT_FREQ_BINS,
                    output_time_bins=OUTPUT_TIME_BINS,
                ).cpu().numpy()
                for batch_offset, stft_sample in enumerate(stft_batch):
                    sample_idx = start_idx + batch_offset
                    save_path = _sample_save_path(
                        save_root,
                        filename=filename,
                        sample_idx=sample_idx,
                        snr_db=snr_db,
                    )
                    plot_single_channel(
                        stft_sample,
                        save_path,
                        filename=filename,
                        sample_idx=sample_idx,
                        signal_code=signal_code,
                        rf_channel=rf_channel,
                        label=label,
                        snr_db=snr_db,
                        freqs=freqs,
                        times=times,
                    )
    logger.info("Done: %s -> %d samples x %d conditions", filename, num_samples, len(conditions))
    return num_samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate specified-SNR single-channel STFT PNGs from raw DroneRFa .mat files"
    )
    parser.add_argument("--data-dir", type=str, default=default_raw_data_dir(),
                        help="Directory containing raw .mat files")
    noise_group = parser.add_mutually_exclusive_group(required=True)
    noise_group.add_argument("--snrs", type=float, nargs="+",
                             help="One or more fixed target SNR values in dB, e.g. --snrs -10 0 10")
    noise_group.add_argument("--clean", action="store_true",
                             help="Use the selected raw IQ channel without adding AWGN")
    parser.add_argument("--save-root", type=str, default=None,
                        help="Output root (default: <data-dir-parent>/DroneRFa_stft_snr_png)")
    parser.add_argument("--sample-length", type=int, default=SAMPLE_LENGTH,
                        help="Number of IQ points per STFT sample")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="STFT batch size")
    parser.add_argument("--device", type=str, default=default_device(),
                        help='Torch device, e.g. "cpu", "cuda:0", or "mps"')
    parser.add_argument("--max-files", type=int, default=None,
                        help="Process at most this many .mat files")
    parser.add_argument("--max-samples-per-file", type=int, default=None,
                        help="Process at most this many samples from each .mat file")
    parser.add_argument("--noise-seed", type=int, default=42,
                        help="Seed for deterministic AWGN")
    args = parser.parse_args(argv)

    if args.sample_length < N_FFT:
        parser.error(f"--sample-length must be at least {N_FFT}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_files is not None and args.max_files <= 0:
        parser.error("--max-files must be positive")
    if args.max_samples_per_file is not None and args.max_samples_per_file <= 0:
        parser.error("--max-samples-per-file must be positive")
    if args.snrs is not None and any(not math.isfinite(snr) for snr in args.snrs):
        parser.error("--snrs must contain only finite values")
    if args.snrs is not None and len(set(args.snrs)) != len(args.snrs):
        parser.error("--snrs must not contain duplicate values")
    return args


def main() -> None:
    args = parse_args()
    log_current_command(logger)
    data_dir = os.path.expanduser(args.data_dir)
    save_root = Path(
        os.path.expanduser(args.save_root) if args.save_root else _default_save_root(data_dir)
    )
    device = "cuda:0" if args.device == "cuda" else args.device
    mat_files = select_mat_files(data_dir, max_files=args.max_files)
    if not mat_files:
        raise ValueError(f"No .mat files found in {data_dir}")

    logger.info("Found %d .mat files", len(mat_files))
    logger.info("Noise condition: clean" if args.clean else "Target SNRs: %s dB", *(() if args.clean else (args.snrs,)))
    logger.info("Using device: %s", device)
    logger.info("Output directory: %s", save_root)

    total_samples = 0
    for filename in tqdm(mat_files, desc="Processing .mat files"):
        total_samples += process_one_mat(
            os.path.join(data_dir, filename),
            save_root,
            snrs=tuple(args.snrs or ()),
            clean=args.clean,
            sample_length=args.sample_length,
            batch_size=args.batch_size,
            device=device,
            max_samples_per_file=args.max_samples_per_file,
            noise_seed=args.noise_seed,
        )
    logger.info(
        "All done. %d files, %d source samples, %d PNGs",
        len(mat_files),
        total_samples,
        total_samples * (1 if args.clean else len(args.snrs)),
    )


if __name__ == "__main__":
    main()

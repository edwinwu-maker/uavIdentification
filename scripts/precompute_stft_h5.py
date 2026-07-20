"""
Convert .mat IQ files -> one .h5 per file with pre-computed stfts.

Each .mat is converted independently to a same-named .h5 in the output
directory.

Output HDF5 structure (per file):
  /stft                 (N, 1, 1024, 1024) float32
  /labels               (N,) int64
  /is_clean             (N,) bool
  /snr_db               (N,) float32
  /source_sample_idx    (N,) int64
  /augmentation_variant (N,) S16
  /rf_channel           (N,) int8
  attrs/noise_profile   clean, random, or mixed-3x

Usage:
  python scripts/precompute_stft_h5.py
  python scripts/precompute_stft_h5.py --data-dir ~/Desktop/dataset/droneRFa
  python scripts/precompute_stft_h5.py --data-dir ... --output-dir ...
  python scripts/precompute_stft_h5.py --data-dir ... --device cuda:0
  python scripts/precompute_stft_h5.py --data-dir ... --device cuda:2 --batch-size 1
  python scripts/precompute_stft_h5.py --data-dir ... --sample-length 10000000
  python scripts/precompute_stft_h5.py --data-dir ... --device mps
  python scripts/precompute_stft_h5.py --data-dir ... --max-files 1
  python scripts/precompute_stft_h5.py --data-dir ... --files-per-class 1
  python scripts/precompute_stft_h5.py --data-dir ... --include-labels 0 1 2 3 4 --files-per-class 12
  python scripts/precompute_stft_h5.py --data-dir ... --max-samples-per-file 1
  python scripts/precompute_stft_h5.py --data-dir ... --noise-profile mixed-3x --snr-low-min -15 --snr-low-max 0 --snr-high-min 0 --snr-high-max 15
  python scripts/precompute_stft_h5.py --data-dir ... --snr-min -5 --snr-max 15 --noise-seed 42
  python scripts/precompute_stft_h5.py --data-dir ... --clean
"""

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
    default_raw_data_dir,
    parse_label,
    read_iq_batch,
    rf_channels_for_file,
)
from src.preprocess.h5_precompute import run_precompute_batch, output_h5_path
from src.preprocess.random_snr_awgn import add_random_snr_awgn_with_snr
from src.preprocess.stft import compute_stft
from src.utils.cli import log_current_command
from src.utils.device import default_device
from src.utils.logger import logger

# ── Paper parameters (match transforms.py) ──
SAMPLE_LENGTH = 10_000_000
N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = 1024
OUTPUT_FREQ_BINS = 1024
OUTPUT_TIME_BINS = 1024
SUPPORTED_NOISE_PROFILES = ("random", "mixed-3x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .mat IQ → .h5 stft files")
    parser.add_argument("--data-dir", type=str, default=default_raw_data_dir(),
                        help="Directory containing .mat files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output .h5 files (default: <data-dir-parent>/DroneRFa_stft_awgn_random_17class_h5, "
                             "DroneRFa_stft_awgn_mixed3x_17class_h5 for mixed-3x, or "
                             "DroneRFa_stft_17class_h5 with --clean)")
    parser.add_argument("--sample-length", type=int, default=SAMPLE_LENGTH,
                        help="Number of IQ samples per output stft")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="STFT batch size")
    parser.add_argument("--device", type=str, default=default_device(),
                        help='Torch device for STFT, e.g. "cpu", "cuda", "cuda:0", or "mps" '
                             '(default: auto-detect cuda/mps, fallback to cpu)')
    parser.add_argument("--max-files", type=int, default=None,
                        help="Process at most this many .mat files")
    parser.add_argument("--files-per-class", type=int, default=None,
                        help="Process at most this many .mat files per selected class")
    parser.add_argument("--include-labels", type=int, nargs="+", default=None,
                        help="Only process these original class labels, e.g. --include-labels 0 1 2 3 4")
    parser.add_argument("--max-samples-per-file", type=int, default=None,
                        help="Process at most this many samples from each .mat file")
    parser.add_argument("--clean", action="store_true",
                        help="Skip AWGN noise; use clean IQ data for STFT precompute")
    parser.add_argument("--noise-profile", choices=SUPPORTED_NOISE_PROFILES, default="random",
                        help="Noise strategy: one random-AWGN version or clean/low/high mixed 3x")
    parser.add_argument("--snr-min", type=float, default=-5.0,
                        help="Minimum SNR in dB for AWGN")
    parser.add_argument("--snr-max", type=float, default=15.0,
                        help="Maximum SNR in dB for AWGN")
    parser.add_argument("--snr-low-min", type=float, default=-15.0,
                        help="Minimum SNR for the mixed-3x low-SNR version")
    parser.add_argument("--snr-low-max", type=float, default=0.0,
                        help="Maximum SNR for the mixed-3x low-SNR version (exclusive)")
    parser.add_argument("--snr-high-min", type=float, default=0.0,
                        help="Minimum SNR for the mixed-3x high-SNR version")
    parser.add_argument("--snr-high-max", type=float, default=15.0,
                        help="Maximum SNR for the mixed-3x high-SNR version (exclusive)")
    parser.add_argument("--noise-seed", type=int, default=42,
                        help="Seed for deterministic AWGN")
    args = parser.parse_args()
    if args.snr_min > args.snr_max:
        parser.error("--snr-min must be <= --snr-max")
    if args.snr_low_min >= args.snr_low_max:
        parser.error("--snr-low-min must be < --snr-low-max")
    if args.snr_high_min >= args.snr_high_max:
        parser.error("--snr-high-min must be < --snr-high-max")
    if args.clean and args.noise_profile != "random":
        parser.error("--clean cannot be used with --noise-profile mixed-3x")
    if args.files_per_class is not None and args.max_samples_per_file is not None:
        parser.error("--max-samples-per-file cannot be used with --files-per-class")
    return args


def process_one_mat(
    mat_path: str,
    output_dir: str,
    *,
    sample_length: int,
    batch_size: int,
    device: str,
    max_samples_per_file: int | None,
    noise_profile: str = "random",
    snr_min: float = -5.0,
    snr_max: float = 15.0,
    snr_low_min: float = -15.0,
    snr_low_max: float = 0.0,
    snr_high_min: float = 0.0,
    snr_high_max: float = 15.0,
    noise_seed: int = 42,
) -> tuple[str, int]:
    """Convert one DroneRFa .mat file into one STFT .h5 file."""

    if noise_profile not in (*SUPPORTED_NOISE_PROFILES, "clean"):
        raise ValueError(f"Unsupported noise profile: {noise_profile}")
    if snr_low_min >= snr_low_max:
        raise ValueError("snr_low_min must be less than snr_low_max")
    if snr_high_min >= snr_high_max:
        raise ValueError("snr_high_min must be less than snr_high_max")

    mat_file = os.path.basename(mat_path)
    out_path = output_h5_path(mat_path, output_dir)
    label = parse_label(mat_file)
    rf_channels = rf_channels_for_file(mat_file)

    logger.info("Processing: %s", mat_file)
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(mat_path, "r") as src:
        channel_sample_counts = {
            rf_channel: count_iq_samples(
                src,
                rf_channel=rf_channel,
                sample_length=sample_length,
                max_samples=max_samples_per_file,
            )
            for rf_channel in rf_channels
        }

        variants_per_sample = 3 if noise_profile == "mixed-3x" else 1
        num_samples = sum(channel_sample_counts.values()) * variants_per_sample
        if noise_profile == "mixed-3x":
            variant_specs = (
                ("clean", True, None, None),
                ("snr_low", False, snr_low_min, snr_low_max),
                ("snr_high", False, snr_high_min, snr_high_max),
            )
        elif noise_profile == "clean":
            variant_specs = (("clean", True, None, None),)
        else:
            variant_specs = (("random", False, snr_min, snr_max),)

        with h5py.File(out_path, "w") as h5f:
            h5f.create_dataset(
                "stft", shape=(num_samples, 1, OUTPUT_FREQ_BINS, OUTPUT_TIME_BINS),
                chunks=(1, 1, OUTPUT_FREQ_BINS, OUTPUT_TIME_BINS), dtype="f4",
            )
            h5f.create_dataset(
                "labels", shape=(num_samples,), chunks=None, dtype="i8",
            )
            h5f.create_dataset("is_clean", shape=(num_samples,), dtype="?")
            h5f.create_dataset("snr_db", shape=(num_samples,), dtype="f4")
            h5f.create_dataset("source_sample_idx", shape=(num_samples,), dtype="i8")
            h5f.create_dataset("rf_channel", shape=(num_samples,), dtype="i1")
            h5f.create_dataset("augmentation_variant", shape=(num_samples,), dtype="S16")
            h5f.attrs["noise_profile"] = noise_profile
            h5f.attrs["sample_length"] = sample_length
            h5f.attrs["n_fft"] = N_FFT
            h5f.attrs["win_length"] = WIN_LENGTH
            h5f.attrs["hop_length"] = HOP_LENGTH
            h5f.attrs["source_freq_bins"] = N_FFT
            h5f.attrs["source_time_bins"] = sample_length // HOP_LENGTH + 1
            h5f.attrs["output_freq_bins"] = OUTPUT_FREQ_BINS
            h5f.attrs["output_time_bins"] = OUTPUT_TIME_BINS

            channel_offset = 0
            for rf_channel, num_source_samples in channel_sample_counts.items():
                batch_starts = range(0, num_source_samples, batch_size)
                for sample_idx in tqdm(
                    batch_starts,
                    total=len(batch_starts),
                    desc=f"  {mat_file} RF{rf_channel}",
                ):
                    batch_end = min(sample_idx + batch_size, num_source_samples)
                    chunk_iq = read_iq_batch(
                        src,
                        rf_channel=rf_channel,
                        sample_length=sample_length,
                        start_idx=sample_idx,
                        end_idx=batch_end,
                    )
                    chunk_iq = torch.as_tensor(chunk_iq, dtype=torch.complex64, device=device)
                    source_indices = np.arange(sample_idx, batch_end, dtype=np.int64)
                    for variant_idx, (variant_name, is_clean, range_min, range_max) in enumerate(
                        variant_specs
                    ):
                        variant_iq = chunk_iq
                        sampled_snrs = torch.full(
                            (len(source_indices),), float("nan"), dtype=torch.float32, device=device
                        )
                        if not is_clean:
                            variant_iq, sampled_snrs = add_random_snr_awgn_with_snr(
                                chunk_iq,
                                file_id=f"{mat_file}:RF{rf_channel}",
                                start_idx=sample_idx,
                                snr_min=float(range_min),
                                snr_max=float(range_max),
                                noise_seed=noise_seed,
                                variant_idx=variant_idx,
                                device=device,
                            )

                        batch_stft = compute_stft(
                            variant_iq,
                            device,
                            n_fft=N_FFT,
                            win_length=WIN_LENGTH,
                            hop_length=HOP_LENGTH,
                            output_freq_bins=OUTPUT_FREQ_BINS,
                            output_time_bins=OUTPUT_TIME_BINS,
                        ).cpu().numpy()
                        output_indices = (
                            channel_offset + source_indices * variants_per_sample + variant_idx
                        )
                        h5f["stft"][output_indices] = batch_stft
                        h5f["labels"][output_indices] = label
                        h5f["is_clean"][output_indices] = is_clean
                        h5f["snr_db"][output_indices] = sampled_snrs.cpu().numpy()
                        h5f["source_sample_idx"][output_indices] = source_indices
                        h5f["rf_channel"][output_indices] = rf_channel
                        h5f["augmentation_variant"][output_indices] = variant_name
                channel_offset += num_source_samples * variants_per_sample

    logger.info("Done: %s -> %d stfts", mat_file, num_samples)
    return mat_file, num_samples


def main() -> None:
    args = parse_args() # 解析命令行参数并把解析结果保存到变量 args 中
    log_current_command(logger) # 打印一次执行命令
    data_dir = os.path.expanduser(args.data_dir)
    if args.output_dir:
        output_dir = os.path.expanduser(args.output_dir)
    elif args.clean:
        output_dir = str(Path(data_dir).parent / "DroneRFa_stft_17class_h5")
    elif args.noise_profile == "mixed-3x":
        output_dir = str(Path(data_dir).parent / "DroneRFa_stft_awgn_mixed3x_17class_h5")
    else:
        output_dir = str(Path(data_dir).parent / "DroneRFa_stft_awgn_random_17class_h5")

    device = "cuda:0" if args.device == "cuda" else args.device
    logger.info("Using device: %s", device)

    os.makedirs(output_dir, exist_ok=True)
    run_precompute_batch(
        data_dir,
        output_dir,
        process_one_mat=process_one_mat,
        process_kwargs={
            "sample_length": args.sample_length,
            "batch_size": args.batch_size,
            "device": device,
            "max_samples_per_file": args.max_samples_per_file,
            "noise_profile": "clean" if args.clean else args.noise_profile,
            "snr_min": args.snr_min,
            "snr_max": args.snr_max,
            "snr_low_min": args.snr_low_min,
            "snr_low_max": args.snr_low_max,
            "snr_high_min": args.snr_high_min,
            "snr_high_max": args.snr_high_max,
            "noise_seed": args.noise_seed,
        },
        max_files=args.max_files,
        files_per_class=args.files_per_class,
        include_labels=args.include_labels,
        log_label="stfts",
    )


if __name__ == "__main__":
    main()

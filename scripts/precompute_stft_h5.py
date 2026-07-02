"""
Convert .mat IQ files -> one .h5 per file with pre-computed stfts.

Each .mat is converted independently to a same-named .h5 in the output
directory.

Output HDF5 structure (per file):
  /stft  (N, 2, 512, 512) float32
  /labels        (N,) int64

Usage:
  python scripts/precompute_stft_h5.py
  python scripts/precompute_stft_h5.py --data-dir ~/Desktop/dataset/droneRFa
  python scripts/precompute_stft_h5.py --data-dir ... --output-dir ...
  python scripts/precompute_stft_h5.py --data-dir ... --device cuda:0
  python scripts/precompute_stft_h5.py --data-dir ... --device cuda:2 --batch-size 64
  python scripts/precompute_stft_h5.py --data-dir ... --sample-length 1000000
  python scripts/precompute_stft_h5.py --data-dir ... --device mps
  python scripts/precompute_stft_h5.py --data-dir ... --max-files 1
  python scripts/precompute_stft_h5.py --data-dir ... --max-samples-per-file 1
  python scripts/precompute_stft_h5.py --data-dir ... --random-snr --snr-min -5 --snr-max 15 --noise-seed 42
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
from tqdm import tqdm

from src.data.drone_rfa_io import count_iq_samples, default_raw_data_dir, parse_label, read_iq_batch
from src.preprocess.h5_precompute import run_precompute_batch, output_h5_path
from src.preprocess.random_snr_awgn import add_random_snr_awgn
from src.preprocess.stft import compute_stft
from src.utils.device import default_device
from src.utils.logger import logger

# ── Paper parameters (match transforms.py) ──
SAMPLE_LENGTH = 1_000_000
N_FFT = 1024
WIN_LENGTH = 1024
SPEC_TIME_BINS = 1024
OUTPUT_FREQ_BINS = 512
OUTPUT_TIME_BINS = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .mat IQ → .h5 stft files")
    parser.add_argument("--data-dir", type=str, default=default_raw_data_dir(),
                        help="Directory containing .mat files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output .h5 files (default: <data-dir-parent>/DroneRFa_stft_h5, "
                             "or DroneRFa_stft_awgn_random_h5 with --random-snr)")
    parser.add_argument("--sample-length", type=int, default=SAMPLE_LENGTH,
                        help="Number of IQ samples per output stft")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="STFT batch size")
    parser.add_argument("--device", type=str, default=default_device(),
                        help='Torch device for STFT, e.g. "cpu", "cuda", "cuda:0", or "mps" '
                             '(default: auto-detect cuda/mps, fallback to cpu)')
    parser.add_argument("--max-files", type=int, default=None,
                        help="Process at most this many .mat files")
    parser.add_argument("--max-samples-per-file", type=int, default=None,
                        help="Process at most this many samples from each .mat file")
    parser.add_argument("--random-snr", action="store_true",
                        help="Add random-SNR AWGN on raw IQ before STFT precompute")
    parser.add_argument("--snr-min", type=float, default=-5.0,
                        help="Minimum SNR in dB for --random-snr")
    parser.add_argument("--snr-max", type=float, default=15.0,
                        help="Maximum SNR in dB for --random-snr")
    parser.add_argument("--noise-seed", type=int, default=42,
                        help="Seed for deterministic random-SNR AWGN")
    args = parser.parse_args()
    if args.snr_min > args.snr_max:
        parser.error("--snr-min must be <= --snr-max")
    return args


def process_one_mat(
    mat_path: str,
    output_dir: str,
    *,
    sample_length: int,
    batch_size: int,
    device: str,
    max_samples_per_file: int | None,
    random_snr: bool,
    snr_min: float,
    snr_max: float,
    noise_seed: int,
) -> tuple[str, int]:
    """Convert one DroneRFa .mat file into one STFT .h5 file."""

    mat_file = os.path.basename(mat_path)
    out_path = output_h5_path(mat_path, output_dir)
    label = parse_label(mat_file)

    logger.info("Processing: %s", mat_file)
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(mat_path, "r") as src:
        num_samples = count_iq_samples(
            src,
            sample_length=sample_length,
            max_samples=max_samples_per_file,
        )

        with h5py.File(out_path, "w") as h5f:
            h5f.create_dataset(
                "stft", shape=(num_samples, 2, OUTPUT_FREQ_BINS, OUTPUT_TIME_BINS),
                chunks=(1, 2, OUTPUT_FREQ_BINS, OUTPUT_TIME_BINS), dtype="f4",
            )
            h5f.create_dataset(
                "labels", shape=(num_samples,), chunks=None, dtype="i8",
            )

            batch_starts = range(0, num_samples, batch_size)
            for sample_idx in tqdm(batch_starts, total=len(batch_starts), desc=f"  {mat_file}"):
                batch_end = min(sample_idx + batch_size, num_samples)
                chunk_iq = read_iq_batch(
                    src,
                    sample_length=sample_length,
                    start_idx=sample_idx,
                    end_idx=batch_end,
                )
                if random_snr:
                    chunk_iq = add_random_snr_awgn(
                        chunk_iq,
                        file_id=mat_file,
                        start_idx=sample_idx,
                        snr_min=snr_min,
                        snr_max=snr_max,
                        noise_seed=noise_seed,
                    )

                batch_stft = compute_stft(
                    chunk_iq,
                    device,
                    n_fft=N_FFT,
                    win_length=WIN_LENGTH,
                    spec_time_bins=SPEC_TIME_BINS,
                    output_freq_bins=OUTPUT_FREQ_BINS,
                    output_time_bins=OUTPUT_TIME_BINS,
                ).cpu().numpy()
                h5f["stft"][sample_idx:batch_end] = batch_stft
                h5f["labels"][sample_idx:batch_end] = label

    logger.info("Done: %s -> %d stfts", mat_file, num_samples)
    return mat_file, num_samples


def main() -> None:
    args = parse_args()
    data_dir = os.path.expanduser(args.data_dir)
    if args.output_dir:
        output_dir = os.path.expanduser(args.output_dir)
    elif args.random_snr:
        output_dir = str(Path(data_dir).parent / "DroneRFa_stft_awgn_random_h5")
    else:
        output_dir = str(Path(data_dir).parent / "DroneRFa_stft_h5")
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
            "random_snr": args.random_snr,
            "snr_min": args.snr_min,
            "snr_max": args.snr_max,
            "noise_seed": args.noise_seed,
        },
        max_files=args.max_files,
        log_label="stfts",
    )


if __name__ == "__main__":
    main()

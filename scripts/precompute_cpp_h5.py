"""Convert DroneRFa .mat IQ files to pre-computed CPP/FAM .h5 files.

Usage:
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --output-dir ~/Desktop/dataset/droneRFa/cpp_h5
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --max-files 1
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --max-samples-per-file 1
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --device mps
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --device cuda:0 --pair-chunk-size 4096
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --files-per-class 1 --cpp-normalization log-zscore-sample --fam-nfft 64 --fam-hop 64
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --use-rf-segmentation --rf-frame-len 10000
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --snr-min -5 --snr-max 15 --noise-seed 42
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --clean
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

from src.data.drone_rfa_io import default_raw_data_dir
from src.data.drone_rfa_io import count_iq_samples, parse_label, read_iq_batch
from src.preprocess.cpp import (
    compute_cpp,
    DEFAULT_SEGMENT_SAMPLES,
    normalize_cpp,
    SUPPORTED_FAM_MERGE_MODES,
)
from src.preprocess.h5_precompute import run_precompute_batch, output_h5_path
from src.preprocess.random_snr_awgn import add_random_snr_awgn
from src.preprocess.rf_segmentation import segment_predominant_rf
from src.utils.logger import logger
from src.utils.device import default_device

SAMPLE_LENGTH = 1_000_000
F_BINS = 257
ALPHA_BINS = 257
RF_FRAME_LEN = 10_000
RF_TARGET_LEN = 100_000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the .mat to CPP .h5 conversion script."""

    parser = argparse.ArgumentParser(description="Convert DroneRFa .mat IQ files to CPP/FAM .h5 files")
    parser.add_argument("--data-dir", type=str, default=default_raw_data_dir(),
                        help="Directory containing .mat files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output .h5 files (default: <data-dir-parent>/DroneRFa_cpp_awgn_random_h5, "
                             "or DroneRFa_cpp_h5 with --clean)")
    parser.add_argument("--sample-length", type=int, default=SAMPLE_LENGTH,
                        help="Number of IQ samples per output CPP sample")
    parser.add_argument("--segment-samples", type=int, default=DEFAULT_SEGMENT_SAMPLES,
                        help="Number of IQ samples per internal FAM segment")
    parser.add_argument("--segment-hop-samples", type=int, default=None,
                        help="Hop size between internal FAM segments. Defaults to --segment-samples")
    parser.add_argument("--fam-merge", choices=SUPPORTED_FAM_MERGE_MODES, default="mean",
                        help="How to merge segmented FAM grids: mean or max")
    parser.add_argument("--fam-nfft", type=int, default=256,
                        help="FAM FFT size")
    parser.add_argument("--fam-hop", type=int, default=256,
                        help="FAM hop size")
    parser.add_argument("--cpp-normalization", choices=("max", "log-zscore-sample"), default="max",
                        help="CPP normalization mode")
    parser.add_argument("--f-bins", type=int, default=F_BINS,
                        help="Number of frequency bins in the CPP grid")
    parser.add_argument("--alpha-bins", type=int, default=ALPHA_BINS,
                        help="Number of cyclic-frequency bins in the CPP grid")
    parser.add_argument("--device", type=str, default=default_device(),
                        help='FAM compute device: "cpu", "cuda", "cuda:0", or "mps"')
    parser.add_argument("--pair-chunk-size", type=int, default=8192,
                        help="Number of (k, l) channel pairs per torch batch")
    parser.add_argument("--use-rf-segmentation", action="store_true",
                        help="Apply ST-ESER predominant segment selection before CPP extraction")
    parser.add_argument("--rf-frame-len", type=int, default=RF_FRAME_LEN,
                        help="RF segmentation frame length for ST-ESER")
    parser.add_argument("--rf-target-len", type=int, default=RF_TARGET_LEN,
                        help="Target IQ length after RF segmentation; ignored when --rf-top-k is set")
    parser.add_argument("--rf-top-k", type=int, default=None,
                        help="Number of RF frames to retain; overrides --rf-target-len")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Process at most this many .mat files")
    parser.add_argument("--files-per-class", type=int, default=None,
                        help="Process at most this many .mat files per selected class")
    parser.add_argument("--max-samples-per-file", type=int, default=None,
                        help="Process at most this many samples from each .mat file")
    parser.add_argument("--clean", action="store_true",
                        help="Skip AWGN noise; use clean IQ data for CPP precompute")
    parser.add_argument("--snr-min", type=float, default=-5.0,
                        help="Minimum SNR in dB for AWGN")
    parser.add_argument("--snr-max", type=float, default=15.0,
                        help="Maximum SNR in dB for AWGN")
    parser.add_argument("--noise-seed", type=int, default=42,
                        help="Seed for deterministic AWGN")
    args = parser.parse_args()
    if args.snr_min > args.snr_max:
        parser.error("--snr-min must be <= --snr-max")
    if args.files_per_class is not None and args.max_samples_per_file is not None:
        parser.error("--max-samples-per-file cannot be used with --files-per-class")
    return args


def process_one_mat(
    mat_path: str,
    output_dir: str,
    *,
    sample_length: int,
    segment_samples: int,
    segment_hop_samples: int | None,
    fam_merge: str,
    f_bins: int,
    alpha_bins: int,
    device: str,
    pair_chunk_size: int,
    max_samples_per_file: int | None,
    cpp_normalization: str = "max",
    fam_nfft: int = 256,
    fam_hop: int = 256,
    use_rf_segmentation: bool = False,
    rf_frame_len: int = RF_FRAME_LEN,
    rf_target_len: int | None = RF_TARGET_LEN,
    rf_top_k: int | None = None,
    random_snr: bool = True,
    snr_min: float = -5.0,
    snr_max: float = 15.0,
    noise_seed: int = 42,
) -> tuple[str, int]:
    """Convert one DroneRFa .mat file into one CPP .h5 file."""

    mat_name = os.path.basename(mat_path)
    out_path = output_h5_path(mat_path, output_dir)
    label = parse_label(mat_name)

    logger.info("Processing: %s", mat_name)
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(mat_path, "r") as src:
        num_samples = count_iq_samples(
            src,
            sample_length=sample_length,
            max_samples=max_samples_per_file,
        )

        with h5py.File(out_path, "w") as h5f:
            h5f.create_dataset(
                "cpp",
                shape=(num_samples, 2, alpha_bins, f_bins),
                chunks=(1, 2, alpha_bins, f_bins),
                dtype="f4",
            )
            h5f.create_dataset("labels", shape=(num_samples,), dtype="i8")

            f_axis = np.linspace(-0.5, 0.5, f_bins, dtype=np.float32)
            alpha_axis = np.linspace(-1.0, 1.0, alpha_bins, dtype=np.float32)
            for sample_idx in tqdm(range(num_samples), total=num_samples, desc=f"  {mat_name}"):
                iq = read_iq_batch(
                    src,
                    sample_length=sample_length,
                    start_idx=sample_idx,
                    end_idx=sample_idx + 1,
                )
                iq = torch.as_tensor(iq, dtype=torch.complex64, device=device)
                if random_snr:
                    iq = add_random_snr_awgn(
                        iq,
                        file_id=mat_name,
                        start_idx=sample_idx,
                        snr_min=snr_min,
                        snr_max=snr_max,
                        noise_seed=noise_seed,
                        device=device,
                    )
                ch0 = iq[0, 0, :]
                ch1 = iq[0, 1, :]
                if use_rf_segmentation:
                    ch0, _, _ = segment_predominant_rf(
                        ch0,
                        frame_len=rf_frame_len,
                        target_len=rf_target_len,
                        top_k=rf_top_k,
                        device=device,
                    )
                    ch1, _, _ = segment_predominant_rf(
                        ch1,
                        frame_len=rf_frame_len,
                        target_len=rf_target_len,
                        top_k=rf_top_k,
                        device=device,
                    )
                cpp, f_axis, alpha_axis = compute_cpp(
                    ch0,
                    ch1,
                    segment_samples=segment_samples,
                    segment_hop_samples=segment_hop_samples,
                    fam_merge=fam_merge,
                    f_bins=f_bins,
                    alpha_bins=alpha_bins,
                    device=device,
                    pair_chunk_size=pair_chunk_size,
                    fam_nfft=fam_nfft,
                    fam_hop=fam_hop,
                )
                h5f["cpp"][sample_idx] = normalize_cpp(cpp, mode=cpp_normalization)
                h5f["labels"][sample_idx] = label

            h5f.create_dataset("f_axis", data=f_axis)
            h5f.create_dataset("alpha_axis", data=alpha_axis)

    logger.info("Done: %s -> %d CPP samples", mat_name, num_samples)
    return mat_name, num_samples


def main() -> None:
    """Run batch conversion from DroneRFa .mat files to CPP .h5 files."""

    args = parse_args()
    data_dir = os.path.expanduser(args.data_dir)
    if args.output_dir:
        output_dir = os.path.expanduser(args.output_dir)
    elif args.clean:
        output_dir = str(Path(data_dir).parent / "DroneRFa_cpp_h5")
    else:
        output_dir = str(Path(data_dir).parent / "DroneRFa_cpp_awgn_random_h5")
    os.makedirs(output_dir, exist_ok=True)
    run_precompute_batch(
        data_dir,
        output_dir,
        process_one_mat=process_one_mat,
        process_kwargs={
            "sample_length": args.sample_length,
            "segment_samples": args.segment_samples,
            "segment_hop_samples": args.segment_hop_samples,
            "fam_merge": args.fam_merge,
            "f_bins": args.f_bins,
            "alpha_bins": args.alpha_bins,
            "cpp_normalization": args.cpp_normalization,
            "fam_nfft": args.fam_nfft,
            "fam_hop": args.fam_hop,
            "device": args.device,
            "pair_chunk_size": args.pair_chunk_size,
            "max_samples_per_file": args.max_samples_per_file,
            "use_rf_segmentation": args.use_rf_segmentation,
            "rf_frame_len": args.rf_frame_len,
            "rf_target_len": args.rf_target_len,
            "rf_top_k": args.rf_top_k,
            "random_snr": not args.clean,
            "snr_min": args.snr_min,
            "snr_max": args.snr_max,
            "noise_seed": args.noise_seed,
        },
        max_files=args.max_files,
        files_per_class=args.files_per_class,
        log_label="CPP samples",
    )


if __name__ == "__main__":
    main()

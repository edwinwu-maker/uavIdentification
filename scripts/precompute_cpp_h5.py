"""Convert DroneRFa .mat IQ files to pre-computed CPP/FAM .h5 files.

Usage:
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --output-dir ~/Desktop/dataset/droneRFa/cpp_h5
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --max-files 1
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --max-samples-per-file 1
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --device mps
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --device cuda:0 --pair-chunk-size 4096
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --files-per-class 1 --fam-nfft 64 --fam-hop 64
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --include-labels 0 1 2 3 4 --files-per-class 12
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --noise-profile mixed-3x --snr-low-min -15 --snr-low-max 0 --snr-high-min 0 --snr-high-max 15
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
from src.data.drone_rfa_io import count_iq_samples, parse_label, read_iq_batch, rf_channels_for_file
from src.preprocess.cpp import (
    compute_cpp,
    DEFAULT_SEGMENT_SAMPLES,
    normalize_cpp,
    SUPPORTED_FAM_MERGE_MODES,
)
from src.preprocess.h5_precompute import run_precompute_batch, output_h5_path
from src.preprocess.random_snr_awgn import add_random_snr_awgn_with_snr
from src.utils.cli import log_current_command
from src.utils.logger import logger
from src.utils.device import default_device

SAMPLE_LENGTH = 1_000_000
F_BINS = 257
ALPHA_BINS = 257
SUPPORTED_NOISE_PROFILES = ("random", "mixed-3x")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the .mat to CPP .h5 conversion script."""

    parser = argparse.ArgumentParser(description="Convert DroneRFa .mat IQ files to CPP/FAM .h5 files")
    parser.add_argument("--data-dir", type=str, default=default_raw_data_dir(),
                        help="Directory containing .mat files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output .h5 files (default: <data-dir-parent>/DroneRFa_cpp_awgn_random_17class_h5, "
                             "DroneRFa_cpp_awgn_mixed3x_17class_h5 for mixed-3x, or "
                             "DroneRFa_cpp_17class_h5 with --clean)")
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
    parser.add_argument("--f-bins", type=int, default=F_BINS,
                        help="Number of frequency bins in the CPP grid")
    parser.add_argument("--alpha-bins", type=int, default=ALPHA_BINS,
                        help="Number of cyclic-frequency bins in the CPP grid")
    parser.add_argument("--device", type=str, default=default_device(),
                        help='FAM compute device: "cpu", "cuda", "cuda:0", or "mps"')
    parser.add_argument("--pair-chunk-size", type=int, default=8192,
                        help="Number of (k, l) channel pairs per torch batch")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Process at most this many .mat files")
    parser.add_argument("--files-per-class", type=int, default=None,
                        help="Process at most this many .mat files per selected class")
    parser.add_argument("--include-labels", type=int, nargs="+", default=None,
                        help="Only process these original class labels, e.g. --include-labels 0 1 2 3 4")
    parser.add_argument("--max-samples-per-file", type=int, default=None,
                        help="Process at most this many samples from each .mat file")
    parser.add_argument("--clean", action="store_true",
                        help="Skip AWGN noise; use clean IQ data for CPP precompute")
    parser.add_argument("--noise-profile", choices=SUPPORTED_NOISE_PROFILES, default="random",
                        help="Noise strategy: one random-AWGN version or clean/low/high mixed 3x")
    parser.add_argument("--snr-min", type=float, default=-5.0,
                        help="Minimum SNR in dB for AWGN")
    parser.add_argument("--snr-max", type=float, default=15.0,
                        help="Maximum SNR in dB for AWGN")
    parser.add_argument("--snr-low-min", type=float, default=-15.0,
                        help="Minimum SNR for the mixed-3x low-SNR version")
    parser.add_argument("--snr-low-max", type=float, default=0.0,
                        help="Maximum SNR for the mixed-3x low-SNR version (exclusive in intent)")
    parser.add_argument("--snr-high-min", type=float, default=0.0,
                        help="Minimum SNR for the mixed-3x high-SNR version")
    parser.add_argument("--snr-high-max", type=float, default=15.0,
                        help="Maximum SNR for the mixed-3x high-SNR version")
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
    segment_samples: int,
    segment_hop_samples: int | None,
    fam_merge: str,
    f_bins: int,
    alpha_bins: int,
    device: str,
    pair_chunk_size: int,
    max_samples_per_file: int | None,
    fam_nfft: int = 256,
    fam_hop: int = 256,
    noise_profile: str = "random",
    snr_min: float = -5.0,
    snr_max: float = 15.0,
    snr_low_min: float = -15.0,
    snr_low_max: float = 0.0,
    snr_high_min: float = 0.0,
    snr_high_max: float = 15.0,
    noise_seed: int = 42,
) -> tuple[str, int]:
    """Convert one DroneRFa .mat file into one CPP .h5 file."""

    if noise_profile not in (*SUPPORTED_NOISE_PROFILES, "clean"):
        raise ValueError(f"Unsupported noise profile: {noise_profile}")
    if snr_low_min >= snr_low_max:
        raise ValueError("snr_low_min must be less than snr_low_max")
    if snr_high_min >= snr_high_max:
        raise ValueError("snr_high_min must be less than snr_high_max")

    mat_name = os.path.basename(mat_path)
    out_path = output_h5_path(mat_path, output_dir)
    label = parse_label(mat_name)
    rf_channels = rf_channels_for_file(mat_name)

    logger.info("Processing: %s", mat_name)
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
        with h5py.File(out_path, "w") as h5f:
            h5f.create_dataset(
                "cpp",
                shape=(num_samples, 1, alpha_bins, f_bins),
                chunks=(1, 1, alpha_bins, f_bins),
                dtype="f4",
            )
            h5f.create_dataset("labels", shape=(num_samples,), dtype="i8")
            h5f.create_dataset("is_clean", shape=(num_samples,), dtype="?")
            h5f.create_dataset("snr_db", shape=(num_samples,), dtype="f4")
            h5f.create_dataset("source_sample_idx", shape=(num_samples,), dtype="i8")
            h5f.create_dataset("rf_channel", shape=(num_samples,), dtype="i1")
            h5f.create_dataset("augmentation_variant", shape=(num_samples,), dtype="S16")
            h5f.attrs["noise_profile"] = "clean" if noise_profile == "clean" else noise_profile

            f_axis = np.linspace(-0.5, 0.5, f_bins, dtype=np.float32)
            alpha_axis = np.linspace(-1.0, 1.0, alpha_bins, dtype=np.float32)
            channel_offset = 0
            for rf_channel, num_source_samples in channel_sample_counts.items():
                for sample_idx in tqdm(
                    range(num_source_samples),
                    total=num_source_samples,
                    desc=f"  {mat_name} RF{rf_channel}",
                ):
                    iq = read_iq_batch(
                        src,
                        rf_channel=rf_channel,
                        sample_length=sample_length,
                        start_idx=sample_idx,
                        end_idx=sample_idx + 1,
                    )
                    iq = torch.as_tensor(iq, dtype=torch.complex64, device=device)
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

                    for variant_idx, (variant_name, is_clean, range_min, range_max) in enumerate(
                        variant_specs
                    ):
                        variant_iq = iq
                        snr_value = np.nan
                        if not is_clean:
                            variant_iq, sampled_snr = add_random_snr_awgn_with_snr(
                                iq,
                                file_id=f"{mat_name}:RF{rf_channel}",
                                start_idx=sample_idx,
                                snr_min=float(range_min),
                                snr_max=float(range_max),
                                noise_seed=noise_seed,
                                variant_idx=variant_idx,
                                device=device,
                            )
                            snr_value = float(sampled_snr[0].item())

                        signal = variant_iq[0, 0, :]
                        cpp, f_axis, alpha_axis = compute_cpp(
                            signal,
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
                        output_idx = (
                            channel_offset + sample_idx * variants_per_sample + variant_idx
                        )
                        h5f["cpp"][output_idx] = normalize_cpp(cpp).detach().cpu().numpy()
                        h5f["labels"][output_idx] = label
                        h5f["is_clean"][output_idx] = is_clean
                        h5f["snr_db"][output_idx] = snr_value
                        h5f["source_sample_idx"][output_idx] = sample_idx
                        h5f["rf_channel"][output_idx] = rf_channel
                        h5f["augmentation_variant"][output_idx] = variant_name
                channel_offset += num_source_samples * variants_per_sample

            h5f.create_dataset("f_axis", data=f_axis)
            h5f.create_dataset("alpha_axis", data=alpha_axis)

    logger.info("Done: %s -> %d CPP samples", mat_name, num_samples)
    return mat_name, num_samples


def main() -> None:
    """Run batch conversion from DroneRFa .mat files to CPP .h5 files."""

    args = parse_args()
    log_current_command(logger)
    data_dir = os.path.expanduser(args.data_dir)
    if args.output_dir:
        output_dir = os.path.expanduser(args.output_dir)
    elif args.clean:
        output_dir = str(Path(data_dir).parent / "DroneRFa_cpp_17class_h5")
    elif args.noise_profile == "mixed-3x":
        output_dir = str(Path(data_dir).parent / "DroneRFa_cpp_awgn_mixed3x_17class_h5")
    else:
        output_dir = str(Path(data_dir).parent / "DroneRFa_cpp_awgn_random_17class_h5")
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
            "fam_nfft": args.fam_nfft,
            "fam_hop": args.fam_hop,
            "device": args.device,
            "pair_chunk_size": args.pair_chunk_size,
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
        log_label="CPP samples",
    )


if __name__ == "__main__":
    main()

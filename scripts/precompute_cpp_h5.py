"""Convert DroneRFa .mat IQ files to pre-computed CPP/FAM .h5 files.

Usage:
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --output-dir ~/Desktop/dataset/droneRFa/cpp_h5
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --max-files 1
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --max-samples-per-file 1
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --device mps
  python scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --device cuda:0 --pair-chunk-size 4096
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
from tqdm import tqdm

from src.data.drone_rfa_io import default_raw_data_dir
from src.data.drone_rfa_io import count_iq_samples, iter_iq_pairs, parse_label
from src.preprocess.cpp import (
    compute_cpp,
    DEFAULT_SEGMENT_SAMPLES,
    SUPPORTED_FAM_MERGE_MODES,
)
from src.preprocess.h5_precompute import resolve_output_dir, run_precompute_batch, output_h5_path
from src.utils.logger import logger
from src.utils.device import default_device

SAMPLE_LENGTH = 1_000_000
F_BINS = 257
ALPHA_BINS = 257


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the .mat to CPP .h5 conversion script."""

    parser = argparse.ArgumentParser(description="Convert DroneRFa .mat IQ files to CPP/FAM .h5 files")
    parser.add_argument("--data-dir", type=str, default=default_raw_data_dir(),
                        help="Directory containing .mat files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output .h5 files (default: <data-dir>/cpp_h5)")
    parser.add_argument("--sample-length", type=int, default=SAMPLE_LENGTH,
                        help="Number of IQ samples per output CPP sample")
    parser.add_argument("--segment-samples", type=int, default=DEFAULT_SEGMENT_SAMPLES,
                        help="Number of IQ samples per internal FAM segment")
    parser.add_argument("--segment-hop-samples", type=int, default=None,
                        help="Hop size between internal FAM segments. Defaults to --segment-samples")
    parser.add_argument("--fam-merge", choices=SUPPORTED_FAM_MERGE_MODES, default="mean",
                        help="How to merge segmented FAM grids: mean or max")
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
    parser.add_argument("--max-samples-per-file", type=int, default=None,
                        help="Process at most this many samples from each .mat file")
    return parser.parse_args()


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
            samples = iter_iq_pairs(
                src,
                sample_length=sample_length,
                num_samples=num_samples,
            )
            for sample_idx, ch0, ch1 in tqdm(samples, total=num_samples, desc=f"  {mat_name}"):
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
                    fam_nfft=256,
                    fam_hop=256
                )
                h5f["cpp"][sample_idx] = cpp
                h5f["labels"][sample_idx] = label

            h5f.create_dataset("f_axis", data=f_axis)
            h5f.create_dataset("alpha_axis", data=alpha_axis)

    logger.info("Done: %s -> %d CPP samples", mat_name, num_samples)
    return mat_name, num_samples


def main() -> None:
    """Run batch conversion from DroneRFa .mat files to CPP .h5 files."""

    args = parse_args()
    data_dir = os.path.expanduser(args.data_dir)
    output_dir = resolve_output_dir(data_dir, args.output_dir, "cpp_h5")
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
            "device": args.device,
            "pair_chunk_size": args.pair_chunk_size,
            "max_samples_per_file": args.max_samples_per_file,
        },
        max_files=args.max_files,
        log_label="CPP samples",
    )


if __name__ == "__main__":
    main()

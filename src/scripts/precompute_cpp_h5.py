"""Convert DroneRFa .mat IQ files to pre-computed CPP/FAM .h5 files.

Usage:
  python src/scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa
  python src/scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --output-dir ~/Desktop/dataset/droneRFa/cpp_h5
  python src/scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --max-files 1 --max-samples-per-file 1
  python src/scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --device mps
  python src/scripts/precompute_cpp_h5.py --data-dir ~/Desktop/dataset/droneRFa --device cuda:0 --pair-chunk-size 4096
"""

import argparse
import os
import sys
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
from tqdm import tqdm

from src.utils.fam_grid import (
    DEFAULT_SEGMENT_SAMPLES,
    SUPPORTED_FAM_MERGE_MODES,
    compute_fam_grid_segmented,
)
from src.utils.logger import logger

SAMPLE_LENGTH = 1_000_000
F_BINS = 257
ALPHA_BINS = 513
LABEL_MAPPING = {
    "T0000": 0, "T0001": 1, "T0010": 2, "T0011": 3,
    "T0100": 4, "T0101": 5, "T0110": 6, "T0111": 7,
    "T1000": 8, "T1001": 9, "T1010": 10, "T1011": 11,
    "T1100": 12, "T1101": 13, "T1110": 14, "T1111": 15,
    "T10000": 16, "T10001": 17, "T10010": 18, "T10011": 19,
    "T10100": 20, "T10101": 21, "T10110": 22, "T10111": 23,
    "T11000": 24,
}


def _default_data_dir() -> str:
    """Return the platform-specific default directory containing DroneRFa .mat files."""

    if os.name == "nt":
        return "E:/dataSet/DroneRFa"
    if sys.platform == "darwin":
        return os.path.expanduser("~/Desktop/dataset/droneRFa")
    return "/mnt/data/wurixin/DroneRFa"


def _parse_label(mat_file: str) -> int:
    """Parse a .mat file name and return its integer drone label."""

    drone_code = os.path.basename(mat_file).split("_")[0]
    return LABEL_MAPPING[drone_code]


def count_iq_samples(
    src: h5py.File,
    *,
    sample_length: int,
    max_samples: int | None = None,
) -> int:
    """Return the number of complete fixed-length IQ samples to process.

    Returns:
        Number of complete samples available after applying max_samples.
    """

    total_points = int(src["RF0_I"].shape[1])
    num_samples = total_points // sample_length
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)
    return num_samples


def iter_iq_pairs(
    src: h5py.File,
    *,
    sample_length: int,
    num_samples: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield fixed-length dual-channel complex IQ samples from one .mat file.

    Yields:
        Tuples of (sample_idx, ch0, ch1), where ch0 and ch1 are one-dimensional
        complex IQ arrays for RF0 and RF1.
    """

    for sample_idx in range(num_samples):
        offset = sample_idx * sample_length
        end = offset + sample_length
        ch0 = src["RF0_I"][0, offset:end] + 1j * src["RF0_Q"][0, offset:end]
        ch1 = src["RF1_I"][0, offset:end] + 1j * src["RF1_Q"][0, offset:end]
        yield sample_idx, ch0, ch1


def compute_cpp_pair(
    ch0: np.ndarray,
    ch1: np.ndarray,
    *,
    segment_samples: int,
    segment_hop_samples: int | None,
    fam_merge: str,
    f_bins: int,
    alpha_bins: int,
    device: str,
    pair_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one dual-channel CPP matrix and its axes from RF0/RF1 IQ arrays.

    Returns:
        (cpp, f_axis, alpha_axis), where cpp has shape (2, alpha_bins, f_bins).
    """

    image0, f_axis, alpha_axis = compute_fam_grid_segmented(
        ch0,
        segment_samples=segment_samples,
        segment_hop_samples=segment_hop_samples,
        merge=fam_merge,
        f_bins=f_bins,
        alpha_bins=alpha_bins,
        device=device,
        pair_chunk_size=pair_chunk_size,
    )
    image1, _, _ = compute_fam_grid_segmented(
        ch1,
        segment_samples=segment_samples,
        segment_hop_samples=segment_hop_samples,
        merge=fam_merge,
        f_bins=f_bins,
        alpha_bins=alpha_bins,
        device=device,
        pair_chunk_size=pair_chunk_size,
    )
    cpp = np.stack([image0, image1], axis=0).astype(np.float32)
    return cpp, f_axis.astype(np.float32), alpha_axis.astype(np.float32)


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
    """Convert one DroneRFa .mat file into one CPP .h5 file.

    Returns:
        (mat_name, num_samples), where num_samples is the number of CPP samples written.
    """

    mat_name = os.path.basename(mat_path)
    out_path = os.path.join(output_dir, mat_name.replace(".mat", ".h5"))
    label = _parse_label(mat_name)

    logger.info("Processing: %s", mat_name)
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
                cpp, f_axis, alpha_axis = compute_cpp_pair(
                    ch0,
                    ch1,
                    segment_samples=segment_samples,
                    segment_hop_samples=segment_hop_samples,
                    fam_merge=fam_merge,
                    f_bins=f_bins,
                    alpha_bins=alpha_bins,
                    device=device,
                    pair_chunk_size=pair_chunk_size,
                )
                h5f["cpp"][sample_idx] = cpp
                h5f["labels"][sample_idx] = label

            h5f.create_dataset("f_axis", data=f_axis)
            h5f.create_dataset("alpha_axis", data=alpha_axis)

    logger.info("Done: %s -> %d CPP samples", mat_name, num_samples)
    return mat_name, num_samples


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the .mat to CPP .h5 conversion script."""

    parser = argparse.ArgumentParser(description="Convert DroneRFa .mat IQ files to CPP/FAM .h5 files")
    parser.add_argument("--data-dir", type=str, default=_default_data_dir(),
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
    parser.add_argument("--device", type=str, default="cpu",
                        help='FAM compute device: "cpu", "cuda", "cuda:0", or "mps"')
    parser.add_argument("--pair-chunk-size", type=int, default=8192,
                        help="Number of (k, l) channel pairs per torch batch")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Process at most this many .mat files")
    parser.add_argument("--max-samples-per-file", type=int, default=None,
                        help="Process at most this many samples from each .mat file")
    return parser.parse_args()


def main() -> None:
    """Run batch conversion from DroneRFa .mat files to CPP .h5 files."""

    args = parse_args()
    data_dir = os.path.expanduser(args.data_dir)
    output_dir = os.path.expanduser(args.output_dir) if args.output_dir else os.path.join(data_dir, "cpp_h5")
    os.makedirs(output_dir, exist_ok=True)

    mat_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".mat"))
    if args.max_files is not None:
        mat_files = mat_files[:args.max_files]

    logger.info("Found %d .mat files in %s", len(mat_files), data_dir)
    logger.info("Output directory: %s", output_dir)

    total_samples = 0
    for mat_file in tqdm(mat_files, desc="Processing .mat files"):
        _, num_samples = process_one_mat(
            os.path.join(data_dir, mat_file),
            output_dir,
            sample_length=args.sample_length,
            segment_samples=args.segment_samples,
            segment_hop_samples=args.segment_hop_samples,
            fam_merge=args.fam_merge,
            f_bins=args.f_bins,
            alpha_bins=args.alpha_bins,
            device=args.device,
            pair_chunk_size=args.pair_chunk_size,
            max_samples_per_file=args.max_samples_per_file,
        )
        total_samples += num_samples

    logger.info("All done. %d files -> %d CPP samples", len(mat_files), total_samples)


if __name__ == "__main__":
    main()

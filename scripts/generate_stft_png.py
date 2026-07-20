"""Generate STFT PNGs from pre-computed .h5 files.

Usage:
  python scripts/generate_stft_png.py --h5-dir ~/Desktop/dataset/droneRFa/stft_h5
  python scripts/generate_stft_png.py --h5-dir ~/Desktop/dataset/droneRFa/stft_h5 --save-root outputs/figures/stft_png
  python scripts/generate_stft_png.py --h5-dir ~/Desktop/dataset/droneRFa/stft_h5 --max-files 1 --max-samples-per-file 1

Reads .h5 files produced by precompute_stft_h5.py and generates PNG images.
Each PNG contains the STFT from the RF channel selected by the source filename.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from src.utils.logger import logger
from src.utils.cli import log_current_command
from src.utils.feature_specs import get_feature_spec
from src.visualization.h5_png_export import (
    limited_sample_count,
    list_h5_files,
    process_h5_samples,
    sample_save_path,
)

FS = 100e6
SAMPLE_LENGTH = 10_000_000


def _pooled_axis(values, output_bins):
    """按预计算时的自适应平均池化分组生成坐标。"""
    if len(values) % output_bins != 0:
        return np.linspace(values[0], values[-1], output_bins)
    return values.reshape(output_bins, -1).mean(axis=1)


def _default_h5_dir() -> str:
    return get_feature_spec("stft").default_data_dir


def _default_save_root(h5_dir: str) -> str:
    return str(Path(h5_dir).parent / "stft_png")


def plot_single_channel(
    stft_sample, save_path, sample_idx, rf_channel, label=None, *, sample_length=SAMPLE_LENGTH,
    n_fft=2048, hop_length=1024, source_time_bins=None,
):
    """Plot a single-channel STFT and save as PNG.

    stft_sample: (1, 1024, 1024) float32, z-score normalized STFT.
    """
    n_freqs, n_times = stft_sample.shape[1], stft_sample.shape[2]
    source_freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, 1.0 / FS))
    source_time_bins = source_time_bins or sample_length // hop_length + 1
    source_times = np.arange(source_time_bins, dtype=np.float64) * hop_length / FS
    freqs = _pooled_axis(source_freqs, n_freqs)
    times = _pooled_axis(source_times, n_times)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5), constrained_layout=True)
    im = ax.pcolormesh(
        freqs / 1e6, times * 1e3, stft_sample[0].T,
        shading="auto", cmap="jet",
    )
    title = f"RF{rf_channel} — Sample {sample_idx}"
    if label is not None:
        title += f" (label={label})"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Frequency (MHz)", fontsize=10)
    ax.set_ylabel("Time (ms)", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Normalized Power (z-score)", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _task_generator(h5f, name_no_extension, drone_code, save_root, max_samples_per_file):
    """Yield (stft_slice, idx, label, save_path) one sample at a time."""
    stft = h5f["stft"]
    labels = h5f["labels"]
    rf_channels = h5f["rf_channel"]
    stft_metadata = {
        "sample_length": int(h5f.attrs.get("sample_length", SAMPLE_LENGTH)),
        "n_fft": int(h5f.attrs.get("n_fft", 2048)),
        "hop_length": int(h5f.attrs.get("hop_length", 1024)),
        "source_time_bins": int(h5f.attrs.get("source_time_bins", 0)) or None,
    }
    num_samples = limited_sample_count(stft.shape[0], max_samples_per_file)

    for i in range(num_samples):
        rf_channel = int(rf_channels[i])
        save_path = sample_save_path(save_root, drone_code, name_no_extension, i, rf_channel)
        yield (stft[i], i, rf_channel, int(labels[i]), save_path, stft_metadata)


def _process_sample(args):
    """Worker: plot and save a single stft sample."""
    stft_slice, idx, rf_channel, label, save_path, stft_metadata = args
    try:
        plot_single_channel(
            stft_slice, save_path, idx, rf_channel, label, **stft_metadata
        )
    except Exception as e:
        logger.error(f"Sample {idx}: {e}")


def process_one_h5(h5_path, save_root, *, max_samples_per_file=None, num_workers=None):
    """Read a .h5 file and generate single-channel PNGs for all samples."""
    process_h5_samples(
        h5_path,
        save_root,
        feature_key="stft",
        build_tasks=_task_generator,
        process_sample=_process_sample,
        max_samples_per_file=max_samples_per_file,
        num_workers=num_workers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate STFT PNGs from pre-computed STFT .h5 files")
    parser.add_argument("--h5-dir", type=str, default=_default_h5_dir(),
                        help="Directory containing STFT .h5 files")
    parser.add_argument("--save-root", type=str, default=None,
                        help="Directory for output PNG files (default: sibling stft_png directory next to --h5-dir)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Process at most this many .h5 files")
    parser.add_argument("--max-samples-per-file", type=int, default=None,
                        help="Process at most this many samples from each .h5 file")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of PNG worker processes per .h5 file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_current_command(logger)
    h5_dir = os.path.expanduser(args.h5_dir)
    if args.save_root is None:
        save_root = _default_save_root(h5_dir)
    else:
        save_root = os.path.expanduser(args.save_root)
    os.makedirs(save_root, exist_ok=True)

    h5_files = list_h5_files(h5_dir, args.max_files)

    logger.info(f"Found {len(h5_files)} .h5 files")
    logger.info(f"Output directory: {save_root}")

    for h5_path in tqdm.tqdm(h5_files, desc="Processing .h5 files"):
        process_one_h5(
            h5_path,
            save_root,
            max_samples_per_file=args.max_samples_per_file,
            num_workers=args.num_workers,
        )

    logger.info("All done!")


if __name__ == "__main__":
    main()

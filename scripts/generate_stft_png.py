"""Generate STFT PNGs from pre-computed .h5 files.

Usage:
  python scripts/generate_stft_png.py --h5-dir ~/Desktop/dataset/droneRFa/stft_h5
  python scripts/generate_stft_png.py --h5-dir ~/Desktop/dataset/droneRFa/stft_h5 --save-root outputs/figures/stft_png
  python scripts/generate_stft_png.py --h5-dir ~/Desktop/dataset/droneRFa/stft_h5 --max-files 1 --max-samples-per-file 1

Reads .h5 files produced by precompute_stft_h5.py and generates PNG images.
Each PNG contains two STFTs (Channel 0 and Channel 1).
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
from src.utils.feature_specs import get_feature_spec
from src.utils.paths import figures_dir
from src.visualization.h5_png_export import (
    limited_sample_count,
    list_h5_files,
    process_h5_samples,
    sample_save_path,
)

FS = 100e6
SAMPLE_LENGTH = 1_000_000
SAMPLE_DURATION = SAMPLE_LENGTH / FS


def _default_h5_dir() -> str:
    return get_feature_spec("stft").default_data_dir


def _default_save_root() -> str:
    return str(figures_dir() / "stft_png")


def plot_dual_channel(stft_sample, save_path, sample_idx, label=None):
    """Plot a dual-channel stft and save as PNG.

    stft_sample: (2, 1024, 1024) float32, z-score normalized STFT.
    """
    n_freqs, n_times = stft_sample.shape[1], stft_sample.shape[2]
    freqs = np.fft.fftshift(np.fft.fftfreq(n_freqs, 1.0 / FS))
    times = np.linspace(0, SAMPLE_DURATION, n_times)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    for ax, ch, ch_name in [(ax0, 0, "Channel 0"), (ax1, 1, "Channel 1")]:
        im = ax.pcolormesh(
            freqs / 1e6, times * 1e3, stft_sample[ch].T,
            shading="auto", cmap="jet",
        )
        title = f"{ch_name} — Sample {sample_idx}"
        if label is not None:
            title += f" (label={label})"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Frequency (MHz)", fontsize=10)
        ax.set_ylabel("Time (ms)", fontsize=10)

    cbar = fig.colorbar(im, ax=[ax0, ax1], shrink=0.92)
    cbar.set_label("Normalized Power (z-score)", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _task_generator(h5f, name_no_extension, drone_code, save_root, max_samples_per_file):
    """Yield (stft_slice, idx, label, save_path) one sample at a time."""
    stft = h5f["stft"]
    labels = h5f["labels"]
    num_samples = limited_sample_count(stft.shape[0], max_samples_per_file)

    for i in range(num_samples):
        save_path = sample_save_path(save_root, drone_code, name_no_extension, i)
        yield (stft[i], i, int(labels[i]), save_path)


def _process_sample(args):
    """Worker: plot and save a single stft sample."""
    stft_slice, idx, label, save_path = args
    try:
        plot_dual_channel(stft_slice, save_path, idx, label)
    except Exception as e:
        logger.error(f"Sample {idx}: {e}")


def process_one_h5(h5_path, save_root, *, max_samples_per_file=None, num_workers=None):
    """Read a .h5 file and generate dual-channel PNGs for all samples."""
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
                        help="Directory for output PNG files (default: outputs/figures/stft_png)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Process at most this many .h5 files")
    parser.add_argument("--max-samples-per-file", type=int, default=None,
                        help="Process at most this many samples from each .h5 file")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of PNG worker processes per .h5 file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_dir = os.path.expanduser(args.h5_dir)
    if args.save_root is None:
        save_root = _default_save_root()
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

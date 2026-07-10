"""Generate CPP/FAM PNGs from pre-computed CPP .h5 files.

Usage:
  python scripts/generate_cpp_png.py --h5-dir ~/Desktop/dataset/droneRFa/cpp_h5
  python scripts/generate_cpp_png.py --h5-dir ~/Desktop/dataset/droneRFa/cpp_h5 --save-root outputs/figures/cpp_png
  python scripts/generate_cpp_png.py --h5-dir ~/Desktop/dataset/droneRFa/cpp_h5 --max-files 1 --max-samples-per-file 1
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from src.utils.cli import log_current_command
from src.utils.feature_specs import get_feature_spec
from src.utils.logger import logger
from src.visualization.h5_png_export import (
    limited_sample_count,
    list_h5_files,
    process_h5_samples,
    sample_save_path,
)


def _default_h5_dir() -> str:
    return get_feature_spec("cpp").default_data_dir


def _default_save_root(h5_dir: str) -> str:
    return str(Path(h5_dir).parent / "cpp_png")


def plot_single_channel_cpp(
    cpp_sample: np.ndarray,
    f_axis: np.ndarray,
    alpha_axis: np.ndarray,
    save_path: str,
    sample_idx: int,
    rf_channel: int,
    label: int | None = None,
) -> None:
    """Plot a single-channel CPP/FAM sample and save as PNG."""

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    im = ax.imshow(
        cpp_sample[0],
        origin="lower",
        aspect="auto",
        extent=[f_axis[0], f_axis[-1], alpha_axis[0], alpha_axis[-1]],
        cmap="jet",
    )
    title = f"RF{rf_channel} CPP - Sample {sample_idx}"
    if label is not None:
        title += f" (label={label})"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("f (cycles/sample)", fontsize=10)
    ax.set_ylabel("alpha (cycles/sample)", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("normalized |SCF|", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _task_generator(h5f, name_no_extension, drone_code, save_root, max_samples_per_file):
    cpp = h5f["cpp"]
    labels = h5f["labels"]
    f_axis = h5f["f_axis"][:]
    alpha_axis = h5f["alpha_axis"][:]
    rf_channel = int(h5f.attrs["rf_channel"])
    num_samples = limited_sample_count(cpp.shape[0], max_samples_per_file)

    for i in range(num_samples):
        save_path = sample_save_path(save_root, drone_code, name_no_extension, i)
        yield (cpp[i], f_axis, alpha_axis, i, rf_channel, int(labels[i]), save_path)


def _process_sample(args) -> None:
    cpp_slice, f_axis, alpha_axis, idx, rf_channel, label, save_path = args
    try:
        plot_single_channel_cpp(cpp_slice, f_axis, alpha_axis, save_path, idx, rf_channel, label)
    except Exception as e:
        logger.error("Sample %d: %s", idx, e)


def process_one_h5(
    h5_path: str,
    save_root: str,
    *,
    max_samples_per_file: int | None = None,
    num_workers: int | None = None,
) -> None:
    """Read a CPP .h5 file and generate single-channel CPP PNGs."""
    process_h5_samples(
        h5_path,
        save_root,
        feature_key="cpp",
        build_tasks=_task_generator,
        process_sample=_process_sample,
        max_samples_per_file=max_samples_per_file,
        num_workers=num_workers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CPP/FAM PNGs from pre-computed CPP .h5 files")
    parser.add_argument("--h5-dir", type=str, default=_default_h5_dir(),
                        help="Directory containing CPP .h5 files")
    parser.add_argument("--save-root", type=str, default=None,
                        help="Directory for output PNG files (default: sibling cpp_png directory next to --h5-dir)")
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

    logger.info("Found %d .h5 files in %s", len(h5_files), h5_dir)
    logger.info("Output directory: %s", save_root)

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

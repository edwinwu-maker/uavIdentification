"""Generate CPP/FAM PNGs from pre-computed CPP .h5 files.

Usage:
  python src/scripts/generate_cpp_png.py --h5-dir ~/Desktop/dataset/droneRFa/cpp_h5
  python src/scripts/generate_cpp_png.py --h5-dir ~/Desktop/dataset/droneRFa/cpp_h5 --save-root ~/Desktop/dataset/droneRFa/cpp_picture
  python src/scripts/generate_cpp_png.py --h5-dir ~/Desktop/dataset/droneRFa/cpp_h5 --max-files 1 --max-samples-per-file 1
"""

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from src.utils.logger import logger


def _default_h5_dir() -> str:
    if os.name == "nt":
        return "E:/dataSet/DroneRFa/cpp_h5"
    if sys.platform == "darwin":
        return os.path.expanduser("~/Desktop/dataset/droneRFa/cpp_h5")
    return "/mnt/data/wurixin/DroneRFa/cpp_h5"


def plot_dual_channel_cpp(
    cpp_sample: np.ndarray,
    f_axis: np.ndarray,
    alpha_axis: np.ndarray,
    save_path: str,
    sample_idx: int,
    label: int | None = None,
) -> None:
    """Plot a dual-channel CPP/FAM sample and save as PNG."""

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

    for ax, ch, ch_name in [(ax0, 0, "RF0"), (ax1, 1, "RF1")]:
        im = ax.imshow(
            cpp_sample[ch],
            origin="lower",
            aspect="auto",
            extent=[f_axis[0], f_axis[-1], alpha_axis[0], alpha_axis[-1]],
            cmap="jet",
        )
        title = f"{ch_name} CPP - Sample {sample_idx}"
        if label is not None:
            title += f" (label={label})"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("f (cycles/sample)", fontsize=10)
        ax.set_ylabel("alpha (cycles/sample)", fontsize=10)

    cbar = fig.colorbar(im, ax=[ax0, ax1], shrink=0.92)
    cbar.set_label("normalized |SCF|", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _task_generator(h5f, name_no_extension, drone_code, save_root, max_samples_per_file):
    cpp = h5f["cpp"]
    labels = h5f["labels"]
    f_axis = h5f["f_axis"][:]
    alpha_axis = h5f["alpha_axis"][:]
    save_sub_dir = os.path.join(save_root, drone_code)
    num_samples = cpp.shape[0]
    if max_samples_per_file is not None:
        num_samples = min(num_samples, max_samples_per_file)

    for i in range(num_samples):
        png_name = f"{name_no_extension}_sample_{i:04d}.png"
        save_path = os.path.join(save_sub_dir, png_name)
        yield (cpp[i], f_axis, alpha_axis, i, int(labels[i]), save_path)


def _process_sample(args) -> None:
    cpp_slice, f_axis, alpha_axis, idx, label, save_path = args
    try:
        plot_dual_channel_cpp(cpp_slice, f_axis, alpha_axis, save_path, idx, label)
    except Exception as e:
        logger.error("Sample %d: %s", idx, e)


def process_one_h5(
    h5_path: str,
    save_root: str,
    *,
    max_samples_per_file: int | None = None,
    num_workers: int | None = None,
) -> None:
    """Read a CPP .h5 file and generate dual-channel CPP PNGs."""

    h5_name = os.path.basename(h5_path)
    name_no_extension = os.path.splitext(h5_name)[0]
    drone_code = name_no_extension.split("_")[0]

    logger.info("Processing: %s", h5_name)
    with h5py.File(h5_path, "r") as h5f:
        num_samples = h5f["cpp"].shape[0]
        if max_samples_per_file is not None:
            num_samples = min(num_samples, max_samples_per_file)

        if num_workers is None:
            num_workers = min(2, max(1, multiprocessing.cpu_count() // 2))
        with multiprocessing.Pool(num_workers) as pool:
            tasks = _task_generator(
                h5f,
                name_no_extension,
                drone_code,
                save_root,
                max_samples_per_file,
            )
            list(tqdm.tqdm(
                pool.imap_unordered(_process_sample, tasks, chunksize=1),
                total=num_samples,
                desc=f"  {h5_name}",
            ))

    logger.info("Done: %s", h5_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CPP/FAM PNGs from pre-computed CPP .h5 files")
    parser.add_argument("--h5-dir", type=str, default=_default_h5_dir(),
                        help="Directory containing CPP .h5 files")
    parser.add_argument("--save-root", type=str, default=None,
                        help="Directory for output PNG files (default: sibling cpp_picture directory)")
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
        save_root = os.path.join(os.path.dirname(h5_dir), "cpp_picture")
    else:
        save_root = os.path.expanduser(args.save_root)
    os.makedirs(save_root, exist_ok=True)

    h5_files = [
        os.path.join(h5_dir, f)
        for f in sorted(os.listdir(h5_dir))
        if f.endswith(".h5")
    ]
    if args.max_files is not None:
        h5_files = h5_files[:args.max_files]

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

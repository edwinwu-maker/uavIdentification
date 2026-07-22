"""Prepare blinded reviews for direct BYOL STFT cleaning.

Usage:
  python scripts/prepare_stft_byol_review.py --data-dir <clean-STFT-H5> --work-dir outputs/stft_byol_cleaning --review-count 600
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.stft_byol_data import review_counts, scan_clean_stft_h5, select_reviews, split_files, write_csv
from src.utils.cli import log_current_command
from src.utils.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare blinded BYOL STFT reviews")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="outputs/stft_byol_cleaning")
    parser.add_argument("--review-count", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_current_command(logger)
    samples, file_counts = scan_clean_stft_h5(args.data_dir)
    targets = review_counts(args.review_count)
    file_roles = split_files(file_counts, targets)
    selections = select_reviews(samples, file_roles, targets)
    work_dir = Path(args.work_dir).expanduser()
    for filename in ("review.csv", "review_lookup.csv", "split_manifest.csv"):
        if (work_dir / filename).exists():
            raise FileExistsError(f"Refusing to overwrite {work_dir / filename}")
    image_dir = work_dir / "review_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    selected_counts = Counter(samples[int(row["sample_index"])].source_file for row in selections)
    split_rows = [{"source_file": name, "review_role": file_roles[name],
                   "input_row_count": count, "review_row_count": selected_counts[name]}
                  for name, count in sorted(file_counts.items())]
    split_path = work_dir / "split_manifest.csv"
    write_csv(split_path, split_rows)

    review_rows, lookup_rows, handles = [], [], {}
    try:
        for order, selection in enumerate(selections, 1):
            review_id = f"R{order:04d}"
            sample_index = int(selection["sample_index"])
            sample = samples[sample_index]
            if sample.path not in handles:
                handles[sample.path] = h5py.File(sample.path, "r")
            matrix = handles[sample.path]["stft"][sample.row_idx, 0]
            relative = Path("review_images") / f"{review_id}.png"
            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            image = ax.imshow(matrix.T, origin="lower", aspect="auto", cmap="jet", vmin=-3, vmax=3)
            ax.set_title(review_id)
            ax.set_xlabel("Frequency bin")
            ax.set_ylabel("Time bin")
            fig.colorbar(image, ax=ax, label="Normalized power (z-score)")
            fig.savefig(work_dir / relative, dpi=120)
            plt.close(fig)
            review_rows.append({"review_id": review_id, "review_role": selection["review_role"],
                                "image_path": relative.as_posix(), "manual_label": ""})
            lookup_rows.append({
                "review_id": review_id, "review_role": selection["review_role"],
                "sample_index": sample_index, "source_file": sample.source_file,
                "source_row_idx": sample.row_idx, "original_label": sample.label,
                "rf_channel": sample.rf_channel, "source_sample_idx": sample.source_sample_idx,
                "sampling_weight": selection["sampling_weight"],
            })
    finally:
        for handle in handles.values():
            handle.close()
    write_csv(work_dir / "review.csv", review_rows)
    lookup_path = work_dir / "review_lookup.csv"
    write_csv(lookup_path, lookup_rows)
    os.chmod(lookup_path, 0o444)
    os.chmod(split_path, 0o444)
    logger.info("Prepared %d reviews in %s: %s", len(selections), work_dir, targets)


if __name__ == "__main__":
    main()

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

from src.data.stft_byol_data import (
    group_samples_by_block, review_counts, sample_block_id, scan_clean_stft_h5,
    select_reviews, split_blocks, write_csv,
)
from src.utils.cli import log_current_command
from src.utils.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare blinded BYOL STFT reviews")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="outputs/stft_byol_cleaning")
    parser.add_argument("--review-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-background-count", type=int, default=10)
    return parser.parse_args()


ARTIFACT_NAMES = ("review.csv", "review_lookup.csv", "split_manifest.csv", "review_images")


def _render_artifacts(
    work_dir: Path,
    samples,
    block_roles,
    selections: list[dict[str, object]],
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    image_dir = work_dir / "review_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    groups = group_samples_by_block(samples)
    selected_counts = Counter(
        sample_block_id(samples[int(row["sample_index"])]) for row in selections
    )
    split_rows = []
    for block_id, member_indices in sorted(groups.items()):
        labels = {samples[index].label for index in member_indices}
        if len(labels) != 1:
            raise ValueError(f"Block must contain exactly one original label: {block_id}")
        split_rows.append({
            "source_file": block_id[0],
            "source_sample_idx": block_id[1],
            "original_label": next(iter(labels)),
            "review_role": block_roles[block_id],
            "input_row_count": len(member_indices),
            "review_row_count": selected_counts[block_id],
        })
    split_path = work_dir / "split_manifest.csv"
    write_csv(split_path, split_rows)

    review_rows, lookup_rows = [], []
    handles = {}
    try:
        for order, selection in enumerate(selections, 1):
            review_id = f"R{order:04d}"
            sample_index = int(selection["sample_index"])
            sample = samples[sample_index]
            if block_roles[sample_block_id(sample)] != selection["review_role"]:
                raise ValueError(f"Review role does not match block role for {review_id}")
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
            review_rows.append({
                "review_id": review_id,
                "review_role": selection["review_role"],
                "image_path": relative.as_posix(),
                "manual_label": "",
            })
            lookup_rows.append({
                "review_id": review_id, "review_role": selection["review_role"],
                "sample_index": sample_index, "source_file": sample.source_file,
                "source_row_idx": sample.row_idx, "original_label": sample.label,
                "rf_channel": sample.rf_channel, "source_sample_idx": sample.source_sample_idx,
                "sampling_weight": selection["sampling_weight"],
                "sampling_stratum": selection["sampling_stratum"],
                "sampling_population_count": selection["sampling_population_count"],
                "sampling_selected_count": selection["sampling_selected_count"],
            })
    finally:
        for handle in handles.values():
            handle.close()
    write_csv(work_dir / "review.csv", review_rows)
    lookup_path = work_dir / "review_lookup.csv"
    write_csv(lookup_path, lookup_rows)
    os.chmod(lookup_path, 0o444)
    os.chmod(split_path, 0o444)


def main() -> None:
    args = parse_args()
    log_current_command(logger)
    samples = scan_clean_stft_h5(args.data_dir)
    targets = review_counts(args.review_count)
    work_dir = Path(args.work_dir).expanduser()
    for filename in ARTIFACT_NAMES:
        if (work_dir / filename).exists():
            raise FileExistsError(f"Refusing to overwrite {work_dir / filename}")

    block_roles = split_blocks(samples, targets, args.seed)
    selections = select_reviews(
        samples, block_roles, targets, args.seed, args.eval_background_count,
    )
    _render_artifacts(work_dir, samples, block_roles, selections)
    logger.info("Prepared %d reviews in %s: %s", len(selections), work_dir, targets)


if __name__ == "__main__":
    main()

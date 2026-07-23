"""Prepare blinded reviews for direct BYOL STFT cleaning.

Usage:
  python scripts/prepare_stft_byol_review.py --data-dir <clean-STFT-H5> --work-dir outputs/stft_byol_cleaning --review-count 600
  python scripts/prepare_stft_byol_review.py --data-dir <clean-STFT-H5> --work-dir outputs/stft_byol_cleaning --rebuild-eval
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.stft_byol_data import (
    MANUAL_LABELS, file_labels_from_samples, read_csv, review_counts, scan_clean_stft_h5,
    select_reviews, split_files, write_csv,
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
    parser.add_argument(
        "--rebuild-eval", action="store_true",
        help="Preserve completed train reviews and rebuild calibration/audit in the existing work directory",
    )
    return parser.parse_args()


ARTIFACT_NAMES = ("review.csv", "review_lookup.csv", "split_manifest.csv", "review_images")


def _render_artifacts(
    work_dir: Path,
    samples,
    file_counts: dict[str, int],
    file_labels: dict[str, int],
    file_roles: dict[str, str],
    selections: list[dict[str, object]],
    preserved: tuple[list[dict[str, str]], list[dict[str, str]]] | None = None,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    image_dir = work_dir / "review_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    selected_counts = Counter(samples[int(row["sample_index"])].source_file for row in selections)
    review_rows, lookup_rows = preserved or ([], [])
    selected_counts.update(
        row["source_file"] for row in lookup_rows if row["review_role"] == "train"
    )
    split_rows = [{
        "source_file": name,
        "original_label": file_labels.get(name, ""),
        "review_role": file_roles[name],
        "input_row_count": count,
        "review_row_count": selected_counts[name],
    } for name, count in sorted(file_counts.items())]
    split_path = work_dir / "split_manifest.csv"
    write_csv(split_path, split_rows)

    handles = {}
    try:
        for order, selection in enumerate(selections, len(review_rows) + 1):
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


def _load_preserved_train(
    work_dir: Path, samples, file_counts: dict[str, int], targets: dict[str, int],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    for name in ARTIFACT_NAMES:
        if not (work_dir / name).exists():
            raise FileNotFoundError(f"Required existing review artifact does not exist: {work_dir / name}")
    reviews = read_csv(work_dir / "review.csv")
    lookups = read_csv(work_dir / "review_lookup.csv")
    manifests = read_csv(work_dir / "split_manifest.csv")
    if len(manifests) != len(file_counts) or {row["source_file"] for row in manifests} != set(file_counts):
        raise ValueError("Existing split_manifest.csv does not match input files")
    for row in manifests:
        if row["review_role"] not in ("train", "calibration", "audit"):
            raise ValueError(f"Invalid existing review role: {row['review_role']}")
        if int(row["input_row_count"]) != file_counts[row["source_file"]]:
            raise ValueError(f"Input row count changed for {row['source_file']}")
    fixed_roles = {
        row["source_file"]: "train" for row in manifests if row["review_role"] == "train"
    }
    review_ids = [row["review_id"] for row in reviews]
    if len(set(review_ids)) != len(review_ids):
        raise ValueError("Existing review.csv contains duplicate review_id")
    lookup_by_id = {row["review_id"]: row for row in lookups}
    if len(lookup_by_id) != len(lookups):
        raise ValueError("Existing review_lookup.csv contains duplicate review_id")

    train_reviews, train_lookups = [], []
    for review in reviews:
        if review["review_role"] != "train":
            continue
        review_id = review["review_id"]
        if review_id not in lookup_by_id:
            raise ValueError(f"Missing lookup for preserved train review {review_id}")
        lookup = lookup_by_id[review_id]
        if lookup["review_role"] != "train":
            raise ValueError(f"Lookup role mismatch for preserved train review {review_id}")
        label = review.get("manual_label", "").strip().lower()
        if label not in MANUAL_LABELS:
            raise ValueError(f"Invalid preserved train manual_label for {review_id}: {label}")
        index = int(lookup["sample_index"])
        if not 0 <= index < len(samples):
            raise ValueError(f"Invalid preserved train sample_index for {review_id}")
        sample = samples[index]
        actual = (
            lookup["source_file"], int(lookup["source_row_idx"]), int(lookup["original_label"]),
            int(lookup["rf_channel"]), int(lookup["source_sample_idx"]),
        )
        expected = (
            sample.source_file, sample.row_idx, sample.label,
            sample.rf_channel, sample.source_sample_idx,
        )
        if actual != expected or sample.source_file not in fixed_roles:
            raise ValueError(f"Preserved train lookup does not match source H5 for {review_id}")
        source_image = work_dir / review["image_path"]
        if not source_image.is_file():
            raise FileNotFoundError(f"Preserved train image does not exist: {source_image}")
        train_reviews.append({
            "review_id": review_id,
            "review_role": "train",
            "image_path": review["image_path"],
            "manual_label": label,
        })
        train_lookups.append({
            "review_id": review_id,
            "review_role": "train",
            "sample_index": index,
            "source_file": sample.source_file,
            "source_row_idx": sample.row_idx,
            "original_label": sample.label,
            "rf_channel": sample.rf_channel,
            "source_sample_idx": sample.source_sample_idx,
            "sampling_weight": 1.0,
            "sampling_stratum": "preserved_train",
            "sampling_population_count": 1,
            "sampling_selected_count": 1,
        })
    if len(train_reviews) != targets["train"]:
        raise ValueError(
            f"Expected {targets['train']} preserved train reviews, found {len(train_reviews)}"
        )
    if [row["review_id"] for row in train_reviews] != [f"R{index:04d}" for index in range(1, targets["train"] + 1)]:
        raise ValueError("Preserved train review IDs must be contiguous from R0001")
    return train_reviews, train_lookups, fixed_roles


def _replace_eval_artifacts(work_dir: Path, stage_dir: Path) -> Path:
    backup = work_dir / f"review_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if backup.exists():
        raise FileExistsError(f"Review backup already exists: {backup}")
    backup.mkdir()
    moved_old, moved_new = [], []
    try:
        for name in ARTIFACT_NAMES:
            os.replace(work_dir / name, backup / name)
            moved_old.append(name)
        for name in ARTIFACT_NAMES:
            os.replace(stage_dir / name, work_dir / name)
            moved_new.append(name)
    except Exception:
        for name in reversed(moved_new):
            os.replace(work_dir / name, stage_dir / name)
        for name in reversed(moved_old):
            os.replace(backup / name, work_dir / name)
        backup.rmdir()
        raise
    return backup


def main() -> None:
    args = parse_args()
    log_current_command(logger)
    samples, file_counts = scan_clean_stft_h5(args.data_dir)
    file_labels = file_labels_from_samples(samples)
    targets = review_counts(args.review_count)
    work_dir = Path(args.work_dir).expanduser()

    if not args.rebuild_eval:
        for filename in ARTIFACT_NAMES:
            if (work_dir / filename).exists():
                raise FileExistsError(f"Refusing to overwrite {work_dir / filename}")
        file_roles = split_files(
            file_counts, targets, args.seed, file_labels=file_labels,
        )
        selections = select_reviews(
            samples, file_roles, targets, args.seed, args.eval_background_count,
        )
        _render_artifacts(
            work_dir, samples, file_counts, file_labels, file_roles, selections,
        )
        logger.info("Prepared %d reviews in %s: %s", len(selections), work_dir, targets)
        return

    train_reviews, train_lookups, fixed_roles = _load_preserved_train(
        work_dir, samples, file_counts, targets,
    )
    file_roles = split_files(
        file_counts, targets, args.seed, file_labels=file_labels, fixed_roles=fixed_roles,
    )
    selections = [
        row for row in select_reviews(
            samples, file_roles, targets, args.seed, args.eval_background_count,
        )
        if row["review_role"] != "train"
    ]
    stage_path = Path(tempfile.mkdtemp(prefix=f".{work_dir.name}_rebuild_", dir=work_dir.parent))
    try:
        for review in train_reviews:
            source = work_dir / review["image_path"]
            target = stage_path / review["image_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        _render_artifacts(
            stage_path, samples, file_counts, file_labels, file_roles, selections,
            (train_reviews, train_lookups),
        )
        backup = _replace_eval_artifacts(work_dir, stage_path)
    finally:
        shutil.rmtree(stage_path, ignore_errors=True)
    logger.info(
        "Rebuilt calibration/audit in %s; preserved train and backed up old artifacts to %s",
        work_dir, backup,
    )


if __name__ == "__main__":
    main()

"""Export rows retained by the calibrated three-seed BYOL ensemble.

Usage:
  python scripts/export_byol_clean_stft_h5.py --data-dir <DEC-filtered-STFT-H5> --work-dir outputs/stft_byol_cleaning --output-dir <cleaned-H5-dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.stft_byol_data import checkpoint_sha256, export_clean_h5, read_csv, write_csv, write_json
from src.utils.cli import log_current_command
from src.utils.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BYOL-cleaned STFT H5 files")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="outputs/stft_byol_cleaning")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_current_command(logger)
    work_dir = Path(args.work_dir).expanduser()
    prediction_path = work_dir / "byol_predictions.csv"
    report_path = work_dir / "byol_cleaning_report.json"
    for path in (prediction_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required BYOL artifact does not exist: {path}")
    with report_path.open(encoding="utf-8") as file_obj:
        report = json.load(file_obj)
    weighted_audit = report.get("ensemble_weighted_audit_metrics", {})
    if float(weighted_audit.get("recall", 0.0)) < 0.95:
        raise ValueError(
            "BYOL audit rejected: weighted video recall must be at least 0.95"
        )
    group_metrics = report.get("ensemble_audit_group_metrics", {})
    for original_label in (15, 16):
        relevant = [
            values for key, values in group_metrics.items()
            if key.startswith(f"original_label={original_label}|")
            and int(values.get("actual_positive", 0)) > 0
        ]
        if not relevant:
            raise ValueError(
                f"BYOL audit rejected: original_label={original_label} has no reviewed positives"
            )
        if any(float(values["recall"]) < 1.0 for values in relevant):
            raise ValueError(
                f"BYOL audit rejected: original_label={original_label} "
                "must retain every reviewed video sample"
            )
    hashes = []
    for checkpoint in report["checkpoints"]:
        path = work_dir / checkpoint["path"]
        actual = checkpoint_sha256(path)
        if actual != checkpoint["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {path}")
        hashes.append(actual)
    manifest, counts = export_clean_h5(
        args.data_dir, args.output_dir, read_csv(prediction_path),
        float(report["threshold"]), hashes,
    )
    write_csv(work_dir / "byol_cleaning_manifest.csv", manifest)
    report["decision_counts"] = counts
    report["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
    write_json(report_path, report)
    logger.info("Exported BYOL-cleaned H5 to %s: %s", args.output_dir, counts)


if __name__ == "__main__":
    main()

"""Use reviewed DCEC clusters to export video-positive clean STFT H5 files.

Usage:
  python scripts/export_clean_stft_h5.py --data-dir ... --work-dir outputs/stft_deep_cluster --output-dir ...
  python scripts/export_clean_stft_h5.py --data-dir ... --work-dir ... --review-csv path/to/review.csv --output-dir ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.data.stft_cleaning_export import (
    calibrate_video_threshold,
    checkpoint_sha256,
    evaluate_audit_reviews,
    export_clean_h5_files,
    map_cluster_semantics,
    read_csv,
    validate_and_join_reviews,
    write_json,
    write_manifest,
)
from src.utils.cli import log_current_command
from src.utils.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reviewed video-positive STFT rows to new H5 files")
    parser.add_argument("--data-dir", required=True, help="Original clean STFT H5 directory")
    parser.add_argument("--work-dir", default="outputs/stft_deep_cluster")
    parser.add_argument("--review-csv", default=None, help="Completed review CSV (default: <work-dir>/review.csv)")
    parser.add_argument("--output-dir", required=True, help="New directory for cleaned H5 files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_current_command(logger)
    work_dir = Path(args.work_dir).expanduser()
    review_path = Path(args.review_csv).expanduser() if args.review_csv else work_dir / "review.csv"
    lookup_path = work_dir / "review_lookup.csv"
    assignments_path = work_dir / "assignments.csv"
    checkpoint_path = work_dir / "stft_dcec.pt"
    for path in (review_path, lookup_path, assignments_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required clustering artifact does not exist: {path}")

    review_rows = read_csv(review_path)
    lookup_rows = read_csv(lookup_path)
    assignment_rows = read_csv(assignments_path)
    joined_reviews = validate_and_join_reviews(review_rows, lookup_rows)
    if not assignment_rows:
        raise ValueError("assignments.csv is empty")
    q_cluster_ids = sorted(
        int(name.removeprefix("q_"))
        for name in assignment_rows[0]
        if name.startswith("q_")
    )
    if q_cluster_ids != list(range(len(q_cluster_ids))):
        raise ValueError("assignments.csv has invalid q_<cluster_id> columns")
    cluster_count = len(q_cluster_ids)
    assignments = {int(row["sample_index"]): row for row in assignment_rows}
    if len(assignments) != len(assignment_rows):
        raise ValueError("assignments.csv contains duplicate sample_index values")
    for sample_index, row in assignments.items():
        probabilities = np.asarray([float(row[f"q_{cluster_id}"]) for cluster_id in q_cluster_ids])
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-4)
            or int(row["cluster_id"]) != int(probabilities.argmax())
        ):
            raise ValueError(f"Invalid cluster probabilities for sample_index={sample_index}")
    for row in joined_reviews:
        sample_index = int(row["sample_index"])
        assignment = assignments.get(sample_index)
        if assignment is None:
            raise ValueError(f"Review references unknown sample_index={sample_index}")
        if (
            int(row["cluster_id"]) != int(assignment["cluster_id"])
            or row["source_file"] != assignment["source_file"]
            or int(row["source_row_idx"]) != int(assignment["source_row_idx"])
        ):
            raise ValueError(f"Review lookup does not match assignments for sample_index={sample_index}")

    semantics, cluster_stats = map_cluster_semantics(joined_reviews, cluster_count)
    video_clusters = {cluster_id for cluster_id, value in semantics.items() if value == "video"}
    threshold, calibration_metrics = calibrate_video_threshold(
        joined_reviews, assignments, video_clusters
    )
    audit_metrics = evaluate_audit_reviews(
        joined_reviews, assignments, video_clusters, threshold
    )
    manifest, decision_counts = export_clean_h5_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        assignment_rows=assignment_rows,
        semantics=semantics,
        threshold=threshold,
        checkpoint_hash=checkpoint_sha256(checkpoint_path),
    )
    write_manifest(work_dir / "cleaning_manifest.csv", manifest)
    total_decisions = sum(decision_counts.values())
    report = {
        "cluster_semantics": {str(key): value for key, value in semantics.items()},
        "cluster_review_stats": {str(key): value for key, value in cluster_stats.items()},
        "video_threshold": threshold,
        "calibration_metrics": calibration_metrics,
        "audit_metrics": audit_metrics,
        "decision_counts": decision_counts,
        "decision_rates": {
            key: value / total_decisions for key, value in decision_counts.items()
        },
        "retention_rate": decision_counts.get("video", 0) / total_decisions,
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
    }
    write_json(work_dir / "cleaning_report.json", report)
    logger.info("Cleaned H5 files saved to %s", args.output_dir)
    logger.info("Decision counts: %s", decision_counts)
    logger.info("Independent audit metrics: %s", audit_metrics)


if __name__ == "__main__":
    main()

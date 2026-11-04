"""根据 HDBSCAN 复核结果，仅导出融合判定为信号的 UAV 样本。

Usage:
  python scripts/export_clean_stft_h5.py --data-dir ... --work-dir outputs/stft_hdbscan --output-dir ...
  python scripts/export_clean_stft_h5.py --data-dir ... --work-dir ... --review-csv path/to/hdbscan_review.csv --output-dir ...
  ~/Desktop/venv/bin/python scripts/export_clean_stft_h5.py --data-dir ... --work-dir ... --output-dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.stft_cleaning_export import write_json, write_manifest
from src.utils.cli import log_current_command
from src.utils.files import compute_file_sha256, read_csv_records
from src.utils.logger import logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reviewed CAE-HDBSCAN signal rows"
    )
    parser.add_argument("--data-dir", required=True, help="Original clean STFT H5 directory")
    parser.add_argument("--work-dir", default="outputs/stft_hdbscan")
    parser.add_argument("--review-csv", default=None, help="Completed review CSV (default: hdbscan_review.csv in work-dir)")
    parser.add_argument("--output-dir", required=True, help="New directory for cleaned H5 files")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    log_current_command(logger)

    run_hdbscan_export(args)


def run_hdbscan_export(args):
    """只读取分析产物，先完成人工验收，再创建新的 H5。"""
    from collections import Counter
    from src.data.stft_cleaning_export import prepare_hdbscan_cleaning, export_hdbscan_h5_files

    work_dir = Path(args.work_dir).expanduser()
    paths = {
        "assignments": work_dir / "hdbscan_assignments.csv",
        "review": Path(args.review_csv).expanduser() if args.review_csv else work_dir / "hdbscan_review.csv",
        "lookup": work_dir / "hdbscan_review_lookup.csv",
        "training_report": work_dir / "training_report.json",
        "model": work_dir / "stft_autoencoder.pt",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required HDBSCAN artifact does not exist: {path}")
    training = json.loads(paths["training_report"].read_text(encoding="utf-8"))
    rows, review_metrics = prepare_hdbscan_cleaning(
        read_csv_records(paths["assignments"]), read_csv_records(paths["review"]),
        read_csv_records(paths["lookup"]), training,
    )
    counts = dict(Counter(row["decision"] for row in rows))
    report = {
        "analysis_mode": training["analysis_mode"],
        "status": review_metrics["status"],
        "review_metrics": review_metrics,
        "analysis_parameters": training["hdbscan"]["parameters"],
        "input_sha256": {name: compute_file_sha256(path) for name, path in paths.items()},
        "decision_counts": counts,
        "uav_signal_retention_rate": counts.get("signal_retained", 0) / len(rows),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
    }
    report_path = work_dir / "hdbscan_cleaning_report.json"
    if review_metrics["status"] == "rejected":
        write_json(report_path, report)
        raise ValueError(f"HDBSCAN review rejected: {review_metrics['rejection_reasons']}; see {report_path}")
    manifest, counts = export_hdbscan_h5_files(
        data_dir=args.data_dir, output_dir=args.output_dir, decision_rows=rows,
        training_report=training, model_hash=report["input_sha256"]["model"],
    )
    report["status"] = "exported"
    report["decision_counts"] = counts
    report["retention_rate"] = (counts.get("signal_retained", 0) + counts.get("t0000_passthrough", 0)) / sum(counts.values())
    write_manifest(work_dir / "hdbscan_cleaning_manifest.csv", manifest)
    write_json(report_path, report)
    logger.info("HDBSCAN cleaned H5 files saved to %s | decisions: %s", args.output_dir, counts)


if __name__ == "__main__":
    main()

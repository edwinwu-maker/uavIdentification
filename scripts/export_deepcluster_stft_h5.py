"""按人工确认结果导出 DeepCluster keep/quarantine H5，drop 仅记录到清单。

Usage:
  python scripts/export_deepcluster_stft_h5.py --data-dir ... --work-dir outputs/stft_deepcluster --output-dir ...
  ~/Desktop/venv/bin/python scripts/export_deepcluster_stft_h5.py --data-dir ... --work-dir ... --output-dir ...

填写 cluster_review.csv 的 decision/confirmed/mixed，或 sample_review.csv 的
manual_label/confirmed（video→keep、other→drop、uncertain→quarantine）；audit.csv 的 manual_label 仅用于独立评价，不改变任何决定。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.deepcluster_export import (
    compute_deepcluster_audit_report,
    export_reviewed_deepcluster_h5,
    resolve_deepcluster_decisions,
)
from src.data.deepcluster_visualization import read_deepcluster_tsne, write_deepcluster_decision_plot
from src.utils.cli import log_current_command
from src.utils.files import compute_file_sha256, read_csv_records, write_json, write_manifest
from src.utils.logger import logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export manually reviewed DeepCluster STFT samples")
    parser.add_argument("--data-dir", required=True, help="Original clean STFT H5 directory")
    parser.add_argument("--work-dir", default="outputs/stft_deepcluster")
    parser.add_argument("--output-dir", required=True, help="New output root containing keep/ and quarantine/")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    log_current_command(logger)
    work_dir = Path(args.work_dir).expanduser()
    paths = {name: work_dir / filename for name, filename in (
        ("assignments", "deepcluster_assignments.csv"), ("cluster_review", "cluster_review.csv"),
        ("sample_review", "sample_review.csv"), ("audit", "audit.csv"),
        ("training_report", "training_report.json"), ("model", "deepcluster.pt"),
    )}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required DeepCluster artifact does not exist: {path}")
    training_report = json.loads(paths["training_report"].read_text(encoding="utf-8"))
    decisions = resolve_deepcluster_decisions(
        read_csv_records(paths["assignments"]), read_csv_records(paths["cluster_review"]),
        read_csv_records(paths["sample_review"]),
    )
    audit = compute_deepcluster_audit_report(decisions, read_csv_records(paths["audit"]))
    # 先核对坐标身份，避免写出 H5 后才发现可视化属于其他运行。
    coordinates = read_deepcluster_tsne(work_dir, decisions)
    if coordinates is not None:
        paths["tsne"] = work_dir / "deepcluster_tsne.csv"
    else:
        logger.info("No DeepCluster t-SNE artifact; exporting without a decision plot")
    hashes = {name: compute_file_sha256(path) for name, path in paths.items()}
    manifest, statistics = export_reviewed_deepcluster_h5(
        data_dir=args.data_dir, output_dir=args.output_dir, decision_rows=decisions,
        model_hash=hashes["model"], training_report=training_report,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    write_manifest(output_dir / "deepcluster_cleaning_manifest.csv", manifest)
    tsne = ({"status": "unavailable", "reason": "missing_training_coordinates", "files": []}
            if coordinates is None else write_deepcluster_decision_plot(output_dir, coordinates, decisions))
    if coordinates is not None:
        tsne["projection"] = training_report.get("tsne", {})
    write_json(output_dir / "deepcluster_cleaning_report.json", {
        "analysis_mode": "deepcluster", "status": "exported", "input_sha256": hashes,
        "dataset_fingerprint": training_report["dataset_fingerprint"],
        "output_dir": str(output_dir), "audit": audit, "tsne": tsne, **statistics,
    })
    logger.info("DeepCluster H5 saved to %s | decisions: %s", output_dir, statistics["decision_counts"])


if __name__ == "__main__":
    main()

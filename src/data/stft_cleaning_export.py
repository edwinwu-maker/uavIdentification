"""校验 HDBSCAN 人工复核结果并导出信号 H5。"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from src.data.drone_rfa_io import EXCLUDED_CLASS_CODES

# 允许的人工复核标签：background=背景，signal=信号，uncertain=无法判定。
MANUAL_LABELS = frozenset(("background", "signal", "uncertain"))


def _copy_row_dataset(
    source: h5py.Dataset,
    destination_file: h5py.File,
    name: str,
    source_rows: list[int],
    *,
    batch_size: int = 16,
) -> None:
    shape = (len(source_rows), *source.shape[1:])
    kwargs = {}
    if source.chunks is not None and source_rows:
        kwargs["chunks"] = (min(source.chunks[0], len(source_rows)), *source.chunks[1:])
    if source.compression is not None and source_rows:
        kwargs["compression"] = source.compression
        kwargs["compression_opts"] = source.compression_opts
    destination = destination_file.create_dataset(name, shape=shape, dtype=source.dtype, **kwargs)
    for key, value in source.attrs.items():
        destination.attrs[key] = value
    for start in range(0, len(source_rows), batch_size):
        rows = source_rows[start : start + batch_size]
        destination[start : start + len(rows)] = source[rows]


def _export_decisions(source_root, destination_root, source_paths, by_file,
                      assignment_fields, decisions, model_hash, *, analysis_report):
    """全量校验源行后，按 HDBSCAN 判定原子写入各输出文件。"""
    manifest = []
    source_names = {path.name for path in source_paths}
    unknown_files = sorted(set(by_file) - source_names)
    if unknown_files:
        raise FileNotFoundError(
            f"Source H5 listed in assignments is missing: {source_root / unknown_files[0]}"
        )
    targets = [destination_root / path.name for path in source_paths]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing cleaned H5: {existing[0]}")

    # 写入任何输出前完成全量源行校验，避免后续文件错误留下部分导出结果。
    for source_path in source_paths:
        filename = source_path.name
        rows = by_file.get(filename, [])
        with h5py.File(source_path, "r") as source:
            source_count = source["stft"].shape[0]
            labels = {int(value) for value in source["labels"][:]}
            if len(labels) != 1:
                raise ValueError(f"Each H5 file must contain exactly one label: {source_path}")
            label = next(iter(labels))
            if label == 0:
                if rows:
                    raise ValueError(f"T0000 must not have HDBSCAN assignments: {filename}")
                for source_row in range(source_count):
                    passthrough = {field: "" for field in assignment_fields}
                    passthrough.update({
                        "sample_index": "",
                        "source_file": filename,
                        "source_row_idx": source_row,
                        "original_label": 0,
                        "rf_channel": int(source["rf_channel"][source_row]),
                        "source_sample_idx": int(source["source_sample_idx"][source_row]),
                        "decision": "t0000_passthrough",
                    })
                    by_file[filename].append(passthrough)
                    decisions["t0000_passthrough"] += 1
                continue
            source_rows = [int(row["source_row_idx"]) for row in rows]
            if len(source_rows) != len(set(source_rows)):
                raise ValueError(f"assignments.csv contains duplicate source rows for {filename}")
            if set(source_rows) != set(range(source_count)):
                raise ValueError(f"Assignments must cover every UAV row in {filename}")
            for row, source_row in zip(rows, source_rows):
                if int(source["labels"][source_row]) != int(row["original_label"]):
                    raise ValueError(
                        f"Assignment label does not match source H5: {filename} row {source_row}"
                    )
                for field in ("rf_channel", "source_sample_idx"):
                    if int(source[field][source_row]) != int(row[field]):
                        raise ValueError(f"Assignment {field} does not match source H5: {filename} row {source_row}")
    destination_root.mkdir(parents=True, exist_ok=True)

    for filename in sorted(source_names):
        rows = sorted(by_file[filename], key=lambda row: int(row["source_row_idx"]))
        source_path = source_root / filename
        selected = [row for row in rows if row["decision"] not in ("background_removed", "uncertain_removed")]
        source_rows = [int(row["source_row_idx"]) for row in selected]
        output_path = destination_root / filename
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        try:
            with h5py.File(source_path, "r") as source, h5py.File(temporary_path, "w") as output:
                count = source["stft"].shape[0]
                for key, value in source.attrs.items():
                    output.attrs[key] = value
                output.attrs["cleaning_checkpoint_sha256"] = model_hash
                output.attrs["cleaning_source_file"] = filename
                output.attrs["cleaning_method"] = "CAE-HDBSCAN-signal-filter"
                output.attrs["cleaning_hdbscan_cluster_count"] = analysis_report["hdbscan"]["cluster_count"]
                output.attrs["cleaning_analysis_parameters"] = json.dumps(analysis_report["hdbscan"]["parameters"], sort_keys=True)
                output.attrs["cleaning_input_size"] = 256
                output.attrs["cleaning_clip_min"] = -5.0
                output.attrs["cleaning_clip_max"] = 5.0
                output.attrs["cleaning_scale_divisor"] = 5.0
                for name, dataset in source.items():
                    if dataset.shape[:1] == (count,):
                        _copy_row_dataset(dataset, output, name, source_rows)
                    else:
                        source.copy(name, output)
                if int(source["labels"][0]) != 0:
                    for name, field, dtype in (
                        ("hdbscan_source_row_idx", "source_row_idx", np.int64),
                        ("hdbscan_label", "hdbscan_label", np.int32),
                        ("hdbscan_membership_strength", "membership_strength", np.float64),
                        ("hdbscan_outlier_score", "outlier_score", np.float64),
                    ):
                        output.create_dataset(name, data=np.asarray([row[field] for row in selected], dtype=dtype))
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        output_index_by_source = {source_row: index for index, source_row in enumerate(source_rows)}
        for row in rows:
            source_row = int(row["source_row_idx"])
            manifest.append({
                **row,
                "output_file": (
                    filename if row["decision"] not in ("background_removed", "uncertain_removed") else ""
                ),
                "output_row_idx": output_index_by_source.get(source_row, ""),
            })
    return manifest, dict(decisions)


def write_manifest(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty cleaning manifest")
    fieldnames = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    with Path(path).open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, allow_nan=False)


def prepare_hdbscan_cleaning(assignment_rows, review_rows, lookup_rows, training_report):
    """关联已填写复核表，按固定融合规则评估；不拟合聚类或覆盖人工样本。"""
    from src.data.stft_analysis import StftAnalysisConfig, fuse_structure

    if training_report.get("analysis_mode") != "cae-hdbscan":
        raise ValueError("training_report.json must describe cae-hdbscan analysis")
    parameters = training_report["hdbscan"]["parameters"]
    config = StftAnalysisConfig(**{name: parameters[name] for name in StftAnalysisConfig.__dataclass_fields__})
    config.validate()
    if not assignment_rows or len(assignment_rows) != training_report["sample_count"]:
        raise ValueError("HDBSCAN assignments must match training sample_count")
    assignments = {}
    for row in assignment_rows:
        index = int(row["sample_index"])
        if index in assignments or index < 0:
            raise ValueError("Duplicate or negative HDBSCAN sample_index")
        if row["source_file"] != Path(row["source_file"]).name:
            raise ValueError("source_file must be a basename")
        if row["structure_label"] not in MANUAL_LABELS or row["fusion_label"] not in MANUAL_LABELS:
            raise ValueError(f"Invalid analysis label for sample_index={index}")
        if int(row["hdbscan_label"]) < -1:
            raise ValueError(f"Invalid HDBSCAN label for sample_index={index}")
        fused = fuse_structure(row, int(row["hdbscan_label"]), float(row["membership_strength"]),
                               float(row["outlier_score"]), config)
        if (fused["fusion_label"] != row["fusion_label"]
                or fused["fusion_reason"] != row["fusion_reason"]
                or str(fused["density_boundary"]).lower() != row["density_boundary"].lower()):
            raise ValueError(f"Inconsistent HDBSCAN fusion for sample_index={index}")
        assignments[index] = row
    if set(assignments) != set(range(len(assignments))):
        raise ValueError("HDBSCAN sample_index must cover 0..N-1")
    lookup = {row["review_id"]: row for row in lookup_rows}
    if not lookup or len(lookup) != len(lookup_rows) or len(review_rows) != len(lookup):
        raise ValueError("HDBSCAN review and lookup must contain the same unique rows")
    labels = {}
    seen_ids = set()
    for review in review_rows:
        review_id = review["review_id"]
        label = review.get("manual_label", "").strip().lower()
        if review_id not in lookup or review_id in seen_ids or label not in MANUAL_LABELS:
            raise ValueError(f"Invalid or incomplete HDBSCAN review: {review_id}")
        seen_ids.add(review_id)
        located = lookup[review_id]
        index = int(located["sample_index"])
        if index in labels or index not in assignments:
            raise ValueError(f"Unknown or duplicate review sample: {review_id}")
        for field, value in assignments[index].items():
            other = located.get(field)
            # CSV 原生非有限分数保留为字符串，不参与概率归一化。
            if other != value:
                raise ValueError(f"HDBSCAN lookup mismatch: {review_id} field {field}")
        labels[index] = label
    rows = []
    for index, row in assignments.items():
        label = row["fusion_label"]
        rows.append({**row, "manual_label": labels.get(index, ""),
                     "decision_reason": "fusion_label=" + label,
                     "decision": "signal_retained" if label == "signal" else label + "_removed"})
    definite = [row for row in rows if row["manual_label"] in ("signal", "background")]
    signals = [row for row in definite if row["manual_label"] == "signal"]
    backgrounds = [row for row in definite if row["manual_label"] == "background"]
    mistakes = [int(row["sample_index"]) for row in signals if row["fusion_label"] != "signal"]
    removed = [row for row in definite if row["fusion_label"] != "signal"]
    true_removed = sum(row["fusion_label"] != "signal" for row in backgrounds)
    reasons = []
    if mistakes:
        reasons.append("reviewed_signal_would_be_removed")
    if any(row["fusion_label"] != "signal" for row in rows) and not signals:
        reasons.append("no_reviewed_signal_to_evaluate")
    return rows, {
        "review_type": "non_independent_review",
        "status": "rejected" if reasons else "approved",
        "rejection_reasons": reasons,
        "misremoved_signal_indices": mistakes,
        "review_count": len(labels),
        "definite_count": len(definite),
        "manual_uncertain_count": len(labels) - len(definite),
        "signal_retention": (len(signals) - len(mistakes)) / len(signals) if signals else None,
        "background_precision": true_removed / len(removed) if removed else None,
        "background_recall": true_removed / len(backgrounds) if backgrounds else None,
        "confusion_matrix": [[len(signals) - len(mistakes), len(mistakes)],
                             [len(backgrounds) - true_removed, true_removed]],
    }


def export_hdbscan_h5_files(*, data_dir, output_dir, decision_rows, training_report, model_hash):
    """仅保留融合信号，T0000 透传；校验后按行复制。"""
    source_root = Path(data_dir).expanduser().resolve()
    destination_root = Path(output_dir).expanduser().resolve()
    if source_root == destination_root:
        raise ValueError("output-dir must differ from data-dir")
    source_paths = [path for path in sorted(source_root.glob("*.h5"))
                    if path.stem.split("_")[0] not in EXCLUDED_CLASS_CODES]
    if not source_paths:
        raise ValueError(f"No H5 files found in {source_root}")
    by_file = defaultdict(list)
    for row in decision_rows:
        by_file[row["source_file"]].append(row)
    return _export_decisions(
        source_root, destination_root, source_paths, by_file, list(decision_rows[0]),
        Counter(row["decision"] for row in decision_rows), model_hash,
        analysis_report=training_report,
    )

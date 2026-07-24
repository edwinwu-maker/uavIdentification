"""Map reviewed DCEC clusters to background decisions and export filtered H5 files."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

MANUAL_LABELS = frozenset(("background", "signal", "uncertain"))
REVIEW_ROLES = frozenset(("mapping", "calibration", "audit"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def validate_and_join_reviews(
    review_rows: list[dict[str, str]],
    lookup_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    lookup = {row["review_id"]: row for row in lookup_rows}
    if len(lookup) != len(lookup_rows):
        raise ValueError("review_lookup.csv contains duplicate review_id values")
    if len(review_rows) != len(lookup_rows):
        raise ValueError("review.csv and review_lookup.csv must contain the same rows")

    joined: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in review_rows:
        review_id = row.get("review_id", "")
        if review_id in seen or review_id not in lookup:
            raise ValueError(f"Unknown or duplicate review_id: {review_id}")
        seen.add(review_id)
        role = row.get("review_role", "")
        label = row.get("manual_label", "").strip().lower()
        if role not in REVIEW_ROLES:
            raise ValueError(f"Unknown review_role for {review_id}: {role}")
        if label not in MANUAL_LABELS:
            raise ValueError(
                f"manual_label for {review_id} must be background, signal, or uncertain"
            )
        lookup_row = lookup[review_id]
        if lookup_row.get("review_role") != role:
            raise ValueError(f"review_role was changed for {review_id}")
        joined.append({**lookup_row, "review_role": role, "manual_label": label})
    return joined


def map_cluster_semantics(
    joined_reviews: list[dict[str, str]],
    cluster_count: int,
) -> tuple[dict[int, str], dict[int, dict[str, float | int | str | None]]]:
    mapping_rows = [row for row in joined_reviews if row["review_role"] == "mapping"]
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for row in mapping_rows:
        by_cluster[int(row["cluster_id"])].append(row["manual_label"])

    semantics: dict[int, str] = {}
    stats: dict[int, dict[str, float | int | str | None]] = {}
    for cluster_id in range(cluster_count):
        counts = Counter(by_cluster.get(cluster_id, []))
        definite_count = counts["background"] + counts["signal"]
        background_fraction = counts["background"] / definite_count if definite_count else None
        if definite_count >= 5 and background_fraction is not None and background_fraction >= 0.8:
            semantic = "background"
        elif definite_count >= 5 and background_fraction is not None and background_fraction <= 0.2:
            semantic = "signal"
        else:
            semantic = "mixed"
        semantics[cluster_id] = semantic
        stats[cluster_id] = {
            "background": counts["background"],
            "signal": counts["signal"],
            "uncertain": counts["uncertain"],
            "definite_count": definite_count,
            "background_fraction": background_fraction,
            "semantic": semantic,
        }
    if "background" not in semantics.values():
        raise ValueError("No cluster satisfies the reviewed background-cluster rule")
    return semantics, stats


def _background_score(assignment: dict[str, str], background_clusters: set[int]) -> float:
    return sum(float(assignment[f"q_{cluster_id}"]) for cluster_id in background_clusters)


def _predict_background(
    assignment: dict[str, str],
    background_clusters: set[int],
    threshold: float,
) -> bool:
    return (
        int(assignment["cluster_id"]) in background_clusters
        and _background_score(assignment, background_clusters) >= threshold
    )


def calibrate_background_threshold(
    joined_reviews: list[dict[str, str]],
    assignments: dict[int, dict[str, str]],
    background_clusters: set[int],
) -> tuple[float, dict[str, float | int]]:
    rows = [
        row for row in joined_reviews
        if row["review_role"] == "calibration" and row["manual_label"] != "uncertain"
    ]
    labels = {row["manual_label"] for row in rows}
    if labels != {"background", "signal"}:
        raise ValueError("Calibration reviews must contain both background and signal samples")
    y_true = np.array([row["manual_label"] == "background" for row in rows], dtype=np.int64)
    best: tuple[float, float] | None = None
    best_metrics: dict[str, float | int] | None = None
    for threshold in np.linspace(0.50, 1.00, 51):
        y_pred = np.array([
            _predict_background(
                assignments[int(row["sample_index"])], background_clusters, float(threshold)
            )
            for row in rows
        ], dtype=np.int64)
        false_background = int(np.sum((y_true == 0) & (y_pred == 1)))
        true_background = int(np.sum((y_true == 1) & (y_pred == 1)))
        if false_background or not true_background:
            continue
        background_recall = float(recall_score(y_true, y_pred, zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        key = (background_recall, float(threshold))
        if best is None or key > best:
            best = key
            best_metrics = {
                "sample_count": len(rows),
                "threshold": float(threshold),
                "background_precision": precision,
                "background_recall": background_recall,
                "signal_retention": 1.0,
            }
    if best is None or best_metrics is None:
        raise ValueError(
            "No useful background threshold avoids removing reviewed signal samples"
        )
    return best[1], best_metrics


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def evaluate_background_audit(
    joined_reviews: list[dict[str, str]],
    assignments: dict[int, dict[str, str]],
    background_clusters: set[int],
    threshold: float,
) -> dict[str, object]:
    all_audit = [row for row in joined_reviews if row["review_role"] == "audit"]
    rows = [row for row in all_audit if row["manual_label"] != "uncertain"]
    if not rows:
        raise ValueError("Audit reviews contain no definite labels")
    labels = {row["manual_label"] for row in rows}
    if labels != {"background", "signal"}:
        raise ValueError("Audit reviews must contain both background and signal samples")
    y_true = np.array([row["manual_label"] == "background" for row in rows], dtype=np.int64)
    y_pred = np.array([
        _predict_background(
            assignments[int(row["sample_index"])], background_clusters, threshold
        )
        for row in rows
    ], dtype=np.int64)
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    predicted_positive = int(np.sum(y_pred == 1))
    actual_background = int(np.sum(y_true == 1))
    actual_signal = int(np.sum(y_true == 0))
    retained_signal = int(np.sum((y_true == 0) & (y_pred == 0)))
    balanced = (
        float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) == 2 else None
    )
    return {
        "review_count": len(all_audit),
        "definite_count": len(rows),
        "manual_uncertain_count": len(all_audit) - len(rows),
        "background_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "background_precision_wilson_95": _wilson_interval(
            true_positive, predicted_positive
        ),
        "background_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "background_recall_wilson_95": _wilson_interval(
            true_positive, actual_background
        ),
        "signal_retention": retained_signal / actual_signal,
        "signal_retention_wilson_95": _wilson_interval(retained_signal, actual_signal),
        "balanced_accuracy": balanced,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=(0, 1)).tolist(),
    }


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def export_clean_h5_files(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    assignment_rows: list[dict[str, str]],
    semantics: dict[int, str],
    threshold: float,
    checkpoint_hash: str,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    source_root = Path(data_dir).expanduser().resolve()
    destination_root = Path(output_dir).expanduser().resolve()
    if source_root == destination_root:
        raise ValueError("output-dir must differ from data-dir")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")

    source_paths = sorted(source_root.glob("*.h5"))
    if not source_paths:
        raise ValueError(f"No H5 files found in {source_root}")
    background_clusters = {
        cluster_id for cluster_id, value in semantics.items() if value == "background"
    }
    if not background_clusters:
        raise ValueError("Cluster semantics contain no background cluster")
    assignment_fields = list(assignment_rows[0]) if assignment_rows else []
    by_file: dict[str, list[dict[str, object]]] = defaultdict(list)
    manifest: list[dict[str, object]] = []
    decisions = Counter()
    for row in assignment_rows:
        score = _background_score(row, background_clusters)
        cluster_id = int(row["cluster_id"])
        if row["source_file"] != Path(row["source_file"]).name:
            raise ValueError(f"source_file must be a basename: {row['source_file']}")
        if _predict_background(row, background_clusters, threshold):
            decision = "background_removed"
        elif semantics[cluster_id] == "signal":
            decision = "signal_retained"
        else:
            decision = "uncertain_retained"
        enriched = {**row, "background_score": score, "decision": decision}
        by_file[row["source_file"]].append(enriched)
        decisions[decision] += 1

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
                    raise ValueError(f"T0000 must not have DCEC assignments: {filename}")
                for source_row in range(source_count):
                    passthrough = {field: "" for field in assignment_fields}
                    passthrough.update({
                        "sample_index": "",
                        "source_file": filename,
                        "source_row_idx": source_row,
                        "original_label": 0,
                        "rf_channel": int(source["rf_channel"][source_row]),
                        "source_sample_idx": int(source["source_sample_idx"][source_row]),
                        "cluster_id": "",
                        "cluster_confidence": "",
                        "background_score": "",
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
    destination_root.mkdir(parents=True, exist_ok=True)

    for filename in sorted(source_names):
        rows = sorted(by_file[filename], key=lambda row: int(row["source_row_idx"]))
        source_path = source_root / filename
        selected = [row for row in rows if row["decision"] != "background_removed"]
        source_rows = [int(row["source_row_idx"]) for row in selected]
        output_path = destination_root / filename
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        try:
            with h5py.File(source_path, "r") as source, h5py.File(temporary_path, "w") as output:
                count = source["stft"].shape[0]
                for key, value in source.attrs.items():
                    output.attrs[key] = value
                output.attrs["cleaning_method"] = "DCEC-background-filter"
                output.attrs["cleaning_selected_k"] = len(semantics)
                output.attrs["cleaning_background_threshold"] = threshold
                output.attrs["cleaning_checkpoint_sha256"] = checkpoint_hash
                output.attrs["cleaning_source_file"] = filename
                output.attrs["cleaning_cluster_semantics"] = json.dumps(semantics, sort_keys=True)
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
                    output.create_dataset(
                        "dec_source_row_idx", data=np.asarray(source_rows, dtype=np.int64)
                    )
                    output.create_dataset(
                        "dec_cluster_id",
                        data=np.asarray([int(row["cluster_id"]) for row in selected], dtype=np.int32),
                    )
                    output.create_dataset(
                        "dec_cluster_confidence",
                        data=np.asarray(
                            [float(row["cluster_confidence"]) for row in selected],
                            dtype=np.float32,
                        ),
                    )
                    output.create_dataset(
                        "dec_background_score",
                        data=np.asarray(
                            [float(row["background_score"]) for row in selected],
                            dtype=np.float32,
                        ),
                    )
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
                    filename if row["decision"] != "background_removed" else ""
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

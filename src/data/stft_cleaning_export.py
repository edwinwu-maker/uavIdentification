"""Map reviewed DCEC clusters to video decisions and export cleaned H5 files."""

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
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from src.data.stft_clustering import REQUIRED_SAMPLE_DATASETS

MANUAL_LABELS = frozenset(("video", "non_video", "uncertain"))
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
                f"manual_label for {review_id} must be video, non_video, or uncertain"
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
        definite_count = counts["video"] + counts["non_video"]
        video_fraction = counts["video"] / definite_count if definite_count else None
        if definite_count >= 5 and video_fraction is not None and video_fraction >= 0.8:
            semantic = "video"
        elif definite_count >= 5 and video_fraction is not None and video_fraction <= 0.2:
            semantic = "non_video"
        else:
            semantic = "mixed"
        semantics[cluster_id] = semantic
        stats[cluster_id] = {
            "video": counts["video"],
            "non_video": counts["non_video"],
            "uncertain": counts["uncertain"],
            "definite_count": definite_count,
            "video_fraction": video_fraction,
            "semantic": semantic,
        }
    if "video" not in semantics.values():
        raise ValueError("No cluster satisfies the reviewed video-cluster rule")
    return semantics, stats


def _video_score(assignment: dict[str, str], video_clusters: set[int]) -> float:
    return sum(float(assignment[f"q_{cluster_id}"]) for cluster_id in video_clusters)


def _predict_video(
    assignment: dict[str, str],
    video_clusters: set[int],
    threshold: float,
) -> bool:
    return (
        int(assignment["cluster_id"]) in video_clusters
        and _video_score(assignment, video_clusters) >= threshold
    )


def calibrate_video_threshold(
    joined_reviews: list[dict[str, str]],
    assignments: dict[int, dict[str, str]],
    video_clusters: set[int],
) -> tuple[float, dict[str, float | int]]:
    rows = [
        row for row in joined_reviews
        if row["review_role"] == "calibration" and row["manual_label"] != "uncertain"
    ]
    if not rows or not any(row["manual_label"] == "video" for row in rows):
        raise ValueError("Calibration reviews must contain at least one definite video sample")
    y_true = np.array([row["manual_label"] == "video" for row in rows], dtype=np.int64)
    best: tuple[float, float, float] | None = None
    best_metrics: dict[str, float | int] | None = None
    for threshold in np.linspace(0.50, 0.99, 50):
        y_pred = np.array([
            _predict_video(assignments[int(row["sample_index"])], video_clusters, float(threshold))
            for row in rows
        ], dtype=np.int64)
        f05 = float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        key = (f05, precision, float(threshold))
        if best is None or key > best:
            best = key
            best_metrics = {
                "sample_count": len(rows),
                "threshold": float(threshold),
                "precision": precision,
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "f0_5": f05,
            }
    assert best is not None and best_metrics is not None
    return best[2], best_metrics


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


def evaluate_audit_reviews(
    joined_reviews: list[dict[str, str]],
    assignments: dict[int, dict[str, str]],
    video_clusters: set[int],
    threshold: float,
) -> dict[str, object]:
    all_audit = [row for row in joined_reviews if row["review_role"] == "audit"]
    rows = [row for row in all_audit if row["manual_label"] != "uncertain"]
    if not rows:
        raise ValueError("Audit reviews contain no definite labels")
    y_true = np.array([row["manual_label"] == "video" for row in rows], dtype=np.int64)
    y_pred = np.array([
        _predict_video(assignments[int(row["sample_index"])], video_clusters, threshold)
        for row in rows
    ], dtype=np.int64)
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    predicted_positive = int(np.sum(y_pred == 1))
    actual_positive = int(np.sum(y_true == 1))
    balanced = (
        float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) == 2 else None
    )
    return {
        "review_count": len(all_audit),
        "definite_count": len(rows),
        "manual_uncertain_count": len(all_audit) - len(rows),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "precision_wilson_95": _wilson_interval(true_positive, predicted_positive),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "recall_wilson_95": _wilson_interval(true_positive, actual_positive),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f0_5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
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

    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    video_clusters = {cluster_id for cluster_id, value in semantics.items() if value == "video"}
    manifest: list[dict[str, object]] = []
    decisions = Counter()
    for row in assignment_rows:
        score = _video_score(row, video_clusters)
        cluster_id = int(row["cluster_id"])
        if row["source_file"] != Path(row["source_file"]).name:
            raise ValueError(f"source_file must be a basename: {row['source_file']}")
        if cluster_id in video_clusters and score >= threshold:
            decision = "video"
        elif semantics[cluster_id] == "non_video" and float(row["cluster_confidence"]) >= threshold:
            decision = "non_video"
        else:
            decision = "uncertain"
        enriched = {**row, "video_score": score, "decision": decision}
        by_file[row["source_file"]].append(enriched)
        decisions[decision] += 1

    targets = [destination_root / filename for filename in sorted(by_file)]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing cleaned H5: {existing[0]}")

    # 写入任何输出前完成全量源行校验，避免后续文件错误留下部分导出结果。
    for filename, rows in by_file.items():
        source_path = source_root / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Source H5 listed in assignments is missing: {source_path}")
        source_rows = [int(row["source_row_idx"]) for row in rows]
        if len(source_rows) != len(set(source_rows)):
            raise ValueError(f"assignments.csv contains duplicate source rows for {filename}")
        with h5py.File(source_path, "r") as source:
            source_count = source["stft"].shape[0]
            for row, source_row in zip(rows, source_rows):
                if not 0 <= source_row < source_count:
                    raise ValueError(f"Invalid source_row_idx for {filename}: {source_row}")
                if int(source["labels"][source_row]) != int(row["original_label"]):
                    raise ValueError(
                        f"Assignment label does not match source H5: {filename} row {source_row}"
                    )
    destination_root.mkdir(parents=True, exist_ok=True)

    for filename in sorted(by_file):
        rows = sorted(by_file[filename], key=lambda row: int(row["source_row_idx"]))
        source_path = source_root / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Source H5 listed in assignments is missing: {source_path}")
        selected = [row for row in rows if row["decision"] == "video"]
        source_rows = [int(row["source_row_idx"]) for row in selected]
        output_path = destination_root / filename
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        try:
            with h5py.File(source_path, "r") as source, h5py.File(temporary_path, "w") as output:
                for key, value in source.attrs.items():
                    output.attrs[key] = value
                output.attrs["cleaning_method"] = "DCEC"
                output.attrs["cleaning_selected_k"] = len(semantics)
                output.attrs["cleaning_video_threshold"] = threshold
                output.attrs["cleaning_checkpoint_sha256"] = checkpoint_hash
                output.attrs["cleaning_source_file"] = filename
                output.attrs["cleaning_cluster_semantics"] = json.dumps(semantics, sort_keys=True)
                output.attrs["cleaning_input_size"] = 256
                output.attrs["cleaning_clip_min"] = -5.0
                output.attrs["cleaning_clip_max"] = 5.0
                output.attrs["cleaning_scale_divisor"] = 5.0
                for name in REQUIRED_SAMPLE_DATASETS:
                    _copy_row_dataset(source[name], output, name, source_rows)
                output.create_dataset("source_row_idx", data=np.asarray(source_rows, dtype=np.int64))
                output.create_dataset(
                    "cluster_id",
                    data=np.asarray([int(row["cluster_id"]) for row in selected], dtype=np.int32),
                )
                output.create_dataset(
                    "cluster_confidence",
                    data=np.asarray([float(row["cluster_confidence"]) for row in selected], dtype=np.float32),
                )
                output.create_dataset(
                    "video_score",
                    data=np.asarray([float(row["video_score"]) for row in selected], dtype=np.float32),
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
                "output_file": filename if row["decision"] == "video" else "",
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

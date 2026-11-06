"""按人工确认处置导出 DeepCluster 样本；独立审核只计算指标。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from src.data.deepcluster_schema import DECISIONS, MANUAL_LABELS as REVIEW_MANUAL_LABELS, SOURCE_IDENTITY_FIELDS
from src.data.drone_rfa_io import EXCLUDED_CLASS_CODES, RESEARCH_LABELS, parse_label
from src.utils.files import compute_file_sha256

MANUAL_LABELS = frozenset(REVIEW_MANUAL_LABELS)
IDENTITY_FIELDS = (*SOURCE_IDENTITY_FIELDS, "cluster_id")


def _boolean(value, field: str) -> bool:
    text = str(value).strip().lower()
    if text not in ("", "true", "false", "1", "0"):
        raise ValueError(f"Invalid {field}: {value}")
    return text in ("true", "1")


def _action(row: dict) -> str:
    action = str(row.get("decision", "")).strip().lower()
    if action and action not in DECISIONS:
        raise ValueError(f"Invalid decision: {action}")
    return action


def _match_review(row: dict, assignments: dict[int, dict]) -> int:
    index = int(row["sample_index"])
    if index not in assignments:
        raise ValueError(f"Unknown review sample_index: {index}")
    for field in IDENTITY_FIELDS:
        actual = row.get(field)
        expected = assignments[index][field]
        if actual is None or (str(actual) != str(expected) if field == "source_file" else int(actual) != int(expected)):
            raise ValueError(f"Review identity mismatch: sample_index={index}, field={field}")
    label = str(row.get("manual_label", "")).strip().lower()
    if label and label not in MANUAL_LABELS:
        raise ValueError(f"Invalid manual_label: {label}")
    return index


def prepare_deepcluster_cleaning(assignment_rows, cluster_review_rows, sample_review_rows) -> list[dict]:
    """未确认默认隔离；混合簇禁止整簇处置，已确认单样本处置优先。"""
    assignments = {}
    identities = set()
    cluster_counts = Counter()
    for original in assignment_rows:
        row = dict(original)
        index = int(row["sample_index"])
        if index < 0 or index in assignments:
            raise ValueError("Duplicate or negative sample_index")
        filename = row["source_file"]
        if not filename or filename != Path(filename).name or filename in (".", ".."):
            raise ValueError("source_file must be a basename")
        for field in IDENTITY_FIELDS[1:]:
            row[field] = int(row[field])
        if (row["original_label"] not in RESEARCH_LABELS or row["original_label"] == 0
                or row["rf_channel"] not in (0, 1) or row["source_row_idx"] < 0
                or row["source_sample_idx"] < 0 or row["cluster_id"] < 0):
            raise ValueError(f"Invalid assignment identity: sample_index={index}")
        identity = (filename, row["source_row_idx"])
        if identity in identities:
            raise ValueError(f"Duplicate source row: {identity}")
        identities.add(identity)
        row["sample_index"] = index
        assignments[index] = row
        cluster_counts[row["cluster_id"]] += 1
    if not assignments or set(assignments) != set(range(len(assignments))):
        raise ValueError("sample_index must cover 0..N-1")

    cluster_actions = {}
    for row in cluster_review_rows:
        cluster = int(row["cluster_id"])
        if cluster not in cluster_counts or cluster in cluster_actions:
            raise ValueError(f"Unknown or duplicate reviewed cluster: {cluster}")
        if int(row["cluster_size"]) != cluster_counts[cluster]:
            raise ValueError(f"Cluster size mismatch: {cluster}")
        action = _action(row)
        confirmed = _boolean(row.get("confirmed", ""), "confirmed")
        mixed = _boolean(row.get("mixed", ""), "mixed")
        if mixed:
            cluster_actions[cluster] = ("quarantine", "mixed_cluster")
        elif confirmed and action:
            cluster_actions[cluster] = (action, "confirmed_cluster")
        else:
            cluster_actions[cluster] = ("quarantine", "unconfirmed_cluster")

    sample_reviews = {}
    for row in sample_review_rows:
        index = _match_review(row, assignments)
        if index in sample_reviews:
            raise ValueError(f"Duplicate sample review: {index}")
        action = _action(row)
        confirmed = _boolean(row.get("confirmed", ""), "confirmed")
        sample_reviews[index] = (row, action if confirmed else "")

    rows = []
    for index in range(len(assignments)):
        row = assignments[index]
        action, reason = cluster_actions.get(row["cluster_id"], ("quarantine", "unreviewed_cluster"))
        review, override = sample_reviews.get(index, ({}, ""))
        if override:
            action, reason = override, "confirmed_sample_override"
        rows.append({**row, "manual_label": str(review.get("manual_label", "")).strip().lower(),
                     "sample_reviewed": index in sample_reviews, "decision": action, "decision_reason": reason})
    return rows


def _audit_rates(rows: list[dict]) -> dict:
    videos = [row for row in rows if row["audit_label"] == "video"]
    negatives = [row for row in rows if row["audit_label"] in ("background", "control_only", "interference")]
    definite_drops = [row for row in videos + negatives if row["decision"] == "drop"]
    return {
        "video_retention": sum(row["decision"] == "keep" for row in videos) / len(videos) if videos else None,
        "video_false_drop_rate": sum(row["decision"] == "drop" for row in videos) / len(videos) if videos else None,
        "video_quarantine_rate": sum(row["decision"] == "quarantine" for row in videos) / len(videos) if videos else None,
        "nonvideo_drop_recall": sum(row["decision"] == "drop" for row in negatives) / len(negatives) if negatives else None,
        "drop_precision": sum(row["audit_label"] != "video" for row in definite_drops) / len(definite_drops) if definite_drops else None,
        "quarantine_rate": sum(row["decision"] == "quarantine" for row in rows) / len(rows) if rows else None,
    }


def evaluate_deepcluster_audit(decision_rows, audit_rows, *, bootstrap_samples: int = 1000, seed: int = 42) -> dict:
    """审核标签不影响决定；按源文件重采样，避免将相关切片当成独立观测。"""
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    assignments = {int(row["sample_index"]): row for row in decision_rows}
    reviewed = []
    seen = set()
    for row in audit_rows:
        index = _match_review(row, assignments)
        if index in seen or assignments[index].get("sample_reviewed", False):
            raise ValueError(f"Duplicate audit or development/audit overlap: {index}")
        seen.add(index)
        label = str(row.get("manual_label", "")).strip().lower()
        if label:
            reviewed.append({**assignments[index], "audit_label": label})
    rates = _audit_rates(reviewed)
    groups = defaultdict(list)
    for row in reviewed:
        groups[row["source_file"]].append(row)
    intervals = {key: None for key in rates}
    if len(groups) >= 2 and bootstrap_samples:
        files = sorted(groups)
        rng = np.random.default_rng(seed)
        values = defaultdict(list)
        for _ in range(bootstrap_samples):
            sample = [row for index in rng.integers(len(files), size=len(files)) for row in groups[files[index]]]
            for key, value in _audit_rates(sample).items():
                if value is not None:
                    values[key].append(value)
        for key, samples in values.items():
            intervals[key] = np.quantile(samples, (0.025, 0.975)).tolist()
    return {
        "review_type": "held_out_sample_audit", "audit_count": len(audit_rows),
        "labeled_count": len(reviewed), "source_file_count": len(groups),
        "label_counts": dict(Counter(row["audit_label"] for row in reviewed)),
        "label_decisions": {label: dict(Counter(row["decision"] for row in reviewed if row["audit_label"] == label))
                            for label in sorted(MANUAL_LABELS)},
        **rates, "file_bootstrap_ci95": intervals, "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "false_dropped_video_indices": [row["sample_index"] for row in reviewed if row["audit_label"] == "video" and row["decision"] == "drop"],
        "quarantined_video_indices": [row["sample_index"] for row in reviewed if row["audit_label"] == "video" and row["decision"] == "quarantine"],
    }


def _validate_export_sources(source_root, source_paths, by_file, decision_rows, training_report):
    """核对训练指纹和全部源行，并补齐背景的直通处置。"""
    selected_files = set(by_file)
    if training_report is not None:
        payload = training_report["fingerprint_payload"]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != training_report["dataset_fingerprint"]:
            raise ValueError("Training dataset fingerprint is inconsistent")
        expected_files = {row["source_file"]: row["sha256"] for row in payload["files"]}
        if len(expected_files) != len(payload["files"]) or set(expected_files) != selected_files:
            raise ValueError("Assignments do not match training source files")
        identity_fields = IDENTITY_FIELDS[:-1]
        expected_samples = [tuple(str(row[field]) for field in identity_fields) for row in payload["samples"]]
        actual_samples = [tuple(str(row[field]) for field in identity_fields)
                          for row in sorted(decision_rows, key=lambda row: int(row["sample_index"]))]
        if expected_samples != actual_samples:
            raise ValueError("Assignments do not match training sample identities")
        for filename, expected_hash in expected_files.items():
            if compute_file_sha256(source_root / filename) != expected_hash:
                raise ValueError(f"Source H5 content changed since training: {filename}")

    # 所有源文件校验通过后才写目标，避免身份错误留下部分导出。
    for path in source_paths:
        with h5py.File(path, "r") as source:
            count = source["stft"].shape[0]
            label = parse_label(path.name)
            for field in ("labels", "rf_channel", "source_sample_idx"):
                if source[field].shape != (count,):
                    raise ValueError(f"Invalid source dataset {field}: {path}")
            if set(int(value) for value in source["labels"][:]) - {label}:
                raise ValueError(f"Source label mismatch: {path}")
            rows = by_file[path.name]
            if label == 0:
                if rows:
                    raise ValueError("T0000 must not appear in assignments")
                for index in range(count):
                    rows.append({"sample_index": "", "source_file": path.name, "source_row_idx": index,
                                 "original_label": 0, "rf_channel": int(source["rf_channel"][index]),
                                 "source_sample_idx": int(source["source_sample_idx"][index]), "cluster_id": "",
                                 "decision": "keep", "decision_reason": "t0000_passthrough"})
                continue
            indices = [int(row["source_row_idx"]) for row in rows]
            if len(indices) != count or set(indices) != set(range(count)):
                raise ValueError(f"Assignments must cover every UAV source row once: {path.name}")
            for row in rows:
                index = int(row["source_row_idx"])
                for field, source_field in (("original_label", "labels"), ("rf_channel", "rf_channel"),
                                            ("source_sample_idx", "source_sample_idx")):
                    if int(row[field]) != int(source[source_field][index]):
                        raise ValueError(f"Source identity mismatch: {path.name}, row={index}, field={field}")


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


def _write_h5_partition(source_path, output_path, source_rows, *, action, model_hash):
    """按选中源行复制一个分区，保留原有属性与非逐行数据。"""
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "x") as output:
        count = source["stft"].shape[0]
        for key, value in source.attrs.items():
            output.attrs[key] = value
        output.attrs["deepcluster_cleaning_method"] = "reviewed_deepcluster"
        output.attrs["deepcluster_cleaning_decision"] = action
        output.attrs["deepcluster_checkpoint_sha256"] = model_hash
        for name, dataset in source.items():
            if isinstance(dataset, h5py.Dataset) and dataset.shape[:1] == (count,):
                _copy_row_dataset(dataset, output, name, source_rows)
            else:
                source.copy(name, output)


def _summarize_manifest(manifest, source_paths, excluded_files):
    """汇总互斥处置清单，保留空源文件与距离编码的统计口径。"""
    per_class = {}
    for label in sorted({int(row["original_label"]) for row in manifest}):
        rows = [row for row in manifest if int(row["original_label"]) == label]
        per_class[str(label)] = {
            "sample_counts": dict(Counter(row["decision"] for row in rows)),
            "source_file_counts": {action: len({row["source_file"] for row in rows if row["decision"] == action})
                                   for action in sorted(DECISIONS)},
            "source_file_count": len({row["source_file"] for row in rows}),
        }
    grouped = {"per_source_file": defaultdict(Counter), "per_rf_channel": defaultdict(Counter),
               "per_distance": defaultdict(Counter)}
    for path in source_paths:
        grouped["per_source_file"][path.name] = Counter()
    for row in manifest:
        # 只解析明确、唯一的距离编码；缺失或有歧义的文件归入 unknown。
        distances = [part for part in Path(row["source_file"]).stem.split("_") if part in ("D00", "D01", "D10")]
        distance = distances[0] if len(distances) == 1 else "unknown"
        for name, key in (("per_source_file", row["source_file"]), ("per_rf_channel", str(row["rf_channel"])),
                          ("per_distance", distance)):
            grouped[name][key][row["decision"]] += 1
    breakdowns = {
        name: {key: {"sample_count": sum(counts.values()),
                     "decision_counts": {action: counts[action] for action in sorted(DECISIONS)},
                     "keep_rate": counts["keep"] / sum(counts.values()) if counts else None}
               for key, counts in sorted(groups.items())}
        for name, groups in grouped.items()
    }
    return {"decision_counts": dict(Counter(row["decision"] for row in manifest)),
            "source_file_count": len(source_paths), "excluded_source_files": excluded_files,
            "per_class": per_class, **breakdowns}


def export_deepcluster_h5_files(*, data_dir, output_dir, decision_rows, model_hash: str = "", training_report=None):
    """全量核对身份后创建 keep/quarantine H5；drop 仅进入互斥清单。"""
    source_root = Path(data_dir).expanduser().resolve()
    destination_root = Path(output_dir).expanduser().resolve()
    if destination_root == source_root or source_root in destination_root.parents or destination_root in source_root.parents:
        raise ValueError("output-dir and data-dir must be separate directories")
    if destination_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {destination_root}")
    all_source_paths = [path for path in sorted(source_root.glob("*.h5"))
                        if path.stem.split("_")[0] not in EXCLUDED_CLASS_CODES]
    if not all_source_paths:
        raise ValueError(f"No source H5 found in {source_root}")
    by_file = defaultdict(list)
    for original in decision_rows:
        row = dict(original)
        if row["decision"] not in DECISIONS:
            raise ValueError(f"Invalid decision: {row['decision']}")
        by_file[row["source_file"]].append(row)
    selected_files = set(by_file)
    unknown = selected_files - {path.name for path in all_source_paths}
    if unknown:
        raise ValueError(f"Assignments reference missing source files: {sorted(unknown)}")
    source_paths = [path for path in all_source_paths if path.name in selected_files or path.stem.split("_")[0] == "T0000"]
    excluded_files = [path.name for path in all_source_paths if path not in source_paths]
    _validate_export_sources(source_root, source_paths, by_file, decision_rows, training_report)

    destination_root.mkdir(parents=True)
    for action in ("keep", "quarantine"):
        (destination_root / action).mkdir()
    manifest = []
    for path in source_paths:
        rows = sorted(by_file[path.name], key=lambda row: int(row["source_row_idx"]))
        locations = {}
        for action in ("keep", "quarantine"):
            selected = [row for row in rows if row["decision"] == action]
            indices = [int(row["source_row_idx"]) for row in selected]
            output_path = destination_root / action / path.name
            _write_h5_partition(path, output_path, indices, action=action, model_hash=model_hash)
            for output_index, row in enumerate(selected):
                locations[int(row["source_row_idx"])] = (f"{action}/{path.name}", output_index)
        for row in rows:
            output_file, output_index = locations.get(int(row["source_row_idx"]), ("", ""))
            manifest.append({**row, "output_file": output_file, "output_row_idx": output_index})
    fields = list(dict.fromkeys(field for row in manifest for field in row))
    manifest = [{field: row.get(field, "") for field in fields} for row in manifest]
    return manifest, _summarize_manifest(manifest, source_paths, excluded_files)

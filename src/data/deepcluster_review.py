"""DeepCluster 的样本身份、内容指纹与独立盲审材料。"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np

from src.data.stft_clustering import _save_cluster_review_image

IDENTITY_FIELDS = ("source_file", "source_row_idx", "original_label", "rf_channel", "source_sample_idx")
MANUAL_LABELS = ("video", "background", "control_only", "interference", "weak_or_mixed", "uncertain")
PREPROCESSING = {"input_size": 256, "clip_range": [-5.0, 5.0], "scale_divisor": 5.0}


def sample_identity(sample):
    return {"source_file": Path(sample.path).name, "source_row_idx": sample.row_idx,
            "original_label": sample.label, "rf_channel": sample.rf_channel,
            "source_sample_idx": sample.source_sample_idx}


def _identity(row):
    return tuple(str(row[field]) for field in IDENTITY_FIELDS)


def dataset_fingerprint(samples, logger=None):
    """顺序读取源 H5 内容；路径移动不改变指纹，内容改变则禁用审核复用。"""
    files = []
    for filename in sorted({sample.path for sample in samples}):
        if logger is not None:
            logger.info("Fingerprinting source H5: %s", filename)
        digest = hashlib.sha256()
        with open(filename, "rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        files.append({"source_file": Path(filename).name, "sha256": digest.hexdigest()})
    payload = {"files": files, "samples": [sample_identity(sample) for sample in samples],
               "preprocessing": PREPROCESSING}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _previous_reviews(work_dir, fingerprint, *, required=False):
    report_path = work_dir / "training_report.json"
    if not report_path.exists():
        if required:
            raise FileNotFoundError(f"Missing review training report: {report_path}")
        return [], []
    with report_path.open(encoding="utf-8") as source:
        report = json.load(source)
    if report.get("dataset_fingerprint") != fingerprint:
        if required:
            raise ValueError("Cannot reuse reviews from a different dataset fingerprint")
        return [], []
    return _read_csv(work_dir / "sample_review.csv"), _read_csv(work_dir / "audit.csv")


def write_deepcluster_review(work_dir, samples, result, *, fingerprint, seed=42,
                             audit_per_class=30, reuse_review_dir=None):
    """导出簇开发样本及固定独立审计集；簇编号重排后必须重新确认整簇处置。"""
    work_dir = Path(work_dir)
    review_dir = Path(reuse_review_dir).expanduser() if reuse_review_dir is not None else work_dir
    previous_samples, previous_audit = _previous_reviews(
        review_dir, fingerprint, required=reuse_review_dir is not None)
    identities = {_identity(sample_identity(sample)): index for index, sample in enumerate(samples)}
    old_samples = {_identity(row): row for row in previous_samples}
    old_audit = {_identity(row): row for row in previous_audit}
    if len(old_samples) != len(previous_samples) or len(old_audit) != len(previous_audit):
        raise ValueError("Previous reviews contain duplicate sample identities")
    if any(key not in identities for key in (*old_samples, *old_audit)):
        raise ValueError("Previous review identities do not match the dataset fingerprint")
    if set(old_samples) & set(old_audit):
        raise ValueError("Previous development and audit samples overlap")
    for row in (*previous_samples, *previous_audit):
        if row.get("manual_label", "") not in ("", *MANUAL_LABELS):
            raise ValueError("Invalid previous manual_label")

    assignments = []
    labels = np.asarray(result.assignments)
    for index, sample in enumerate(samples):
        assignments.append({"sample_index": index, **sample_identity(sample), "source_path": sample.path,
                            "cluster_id": int(labels[index]),
                            "centroid_distance": float(result.distances[index])})

    # 首轮先选开发样本，余下按类别随机盲抽；后续固定审计身份并排除出开发集。
    audit_indices = [identities[key] for key in old_audit]
    reserved = set(audit_indices)
    rng = np.random.default_rng(seed)
    selections = []
    for cluster_id in sorted(set(labels.tolist())):
        candidates = np.array([int(index) for index in np.flatnonzero(labels == cluster_id)
                               if int(index) not in reserved], dtype=np.int64)
        available = set(candidates.tolist())
        for role, order in (("center", candidates[np.argsort(result.distances[candidates], kind="stable")]),
                            ("boundary", candidates[np.argsort(-result.distances[candidates], kind="stable")])):
            # 边界使用到所属中心的远距离样本，作为人工审查的困难样本。
            chosen = [int(index) for index in order if int(index) in available][:4]
            selections.extend((index, role) for index in chosen)
            available.difference_update(chosen)
        random_indices = rng.permutation(sorted(available))[:4]
        selections.extend((int(index), "random") for index in random_indices)
    development = {index for index, _ in selections}
    if not previous_audit:
        for label in sorted({sample.label for sample in samples}):
            candidates = [index for index, sample in enumerate(samples)
                          if sample.label == label and index not in development]
            audit_indices.extend(int(index) for index in rng.permutation(candidates)[:audit_per_class])
    # 跨目录对照也重新生成旧样本图片，避免复用的路径指向不存在的目标文件。
    for key, previous in old_samples.items():
        index = identities[key]
        if index not in development and (previous.get("decision", "") or previous.get("manual_label", "")):
            selections.append((index, "previous_review"))

    review_fields = ("review_id", "image_path", "review_role", *assignments[0], "manual_label")
    development_rows, audit_rows = [], []
    image_dir = work_dir / "deepcluster_review_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handles = {}
        for index, role in [*selections, *((index, "audit") for index in audit_indices)]:
            assignment = assignments[index]
            key = _identity(assignment)
            review_id = "D" + hashlib.sha256(json.dumps(key).encode("utf-8")).hexdigest()[:16]
            image_path = Path("deepcluster_review_images") / f"{review_id}.png"
            sample = samples[index]
            if sample.path not in handles:
                handles[sample.path] = stack.enter_context(h5py.File(sample.path, "r"))
            _save_cluster_review_image(handles[sample.path]["stft"][sample.row_idx, 0],
                                       review_id, work_dir / image_path)
            previous = (old_audit if role == "audit" else old_samples).get(key, {})
            row = {"review_id": review_id, "image_path": image_path.as_posix(), "review_role": role,
                   **assignment, "manual_label": previous.get("manual_label", "")}
            if role == "audit":
                audit_rows.append(row)
            else:
                row.update({"decision": previous.get("decision", ""),
                            "confirmed": previous.get("confirmed", "false")})
                development_rows.append(row)
    counts = Counter(int(label) for label in labels)
    clusters = [{"cluster_id": cluster_id, "cluster_size": counts[cluster_id],
                 "decision": "quarantine", "confirmed": "false", "mixed": "false", "review_note": ""}
                for cluster_id in sorted(counts)]
    _write_csv(work_dir / "deepcluster_assignments.csv", assignments, assignments[0])
    _write_csv(work_dir / "cluster_review.csv", clusters, clusters[0])
    _write_csv(work_dir / "sample_review.csv", development_rows, (*review_fields, "decision", "confirmed"))
    _write_csv(work_dir / "audit.csv", audit_rows, review_fields)
    return {"development_count": len(development_rows), "audit_count": len(audit_rows),
            "audit_per_class": dict(Counter(row["original_label"] for row in audit_rows)),
            "manual_labels": list(MANUAL_LABELS), "audit_reused": bool(previous_audit),
            "boundary_selection": "largest_assigned_centroid_distance"}

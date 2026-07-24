"""Data preparation and export helpers for post-DEC BYOL STFT cleaning."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import Dataset

from src.data.stft_clustering import REQUIRED_SAMPLE_DATASETS, preprocess_stft

ROLES = ("train", "calibration", "audit")
MANUAL_LABELS = frozenset(("video_present", "no_video", "uncertain"))
BLOCK_SAMPLE_LENGTH = 10_000_000
BYOL_INPUT_SIZE = 512
FULL_SPLIT_POLICY = "full_three_role"
SPARSE_SPLIT_POLICY = "train_only_sparse"
BlockId = tuple[str, int]


@dataclass(frozen=True)
class ByolSample:
    path: str
    row_idx: int
    label: int
    rf_channel: int
    source_sample_idx: int
    dec_source_row_idx: int | None = None

    @property
    def source_file(self) -> str:
        return Path(self.path).name


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def sample_block_id(sample: ByolSample) -> BlockId:
    return sample.source_file, sample.source_sample_idx


def group_samples_by_block(samples: list[ByolSample]) -> dict[BlockId, list[int]]:
    groups: dict[BlockId, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample_block_id(sample)].append(index)
    return dict(groups)


def scan_clean_stft_h5(data_dir: str | Path) -> list[ByolSample]:
    """Validate DEC-filtered H5 files and index only T0001-T10000 rows."""

    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"STFT H5 directory does not exist: {root}")
    paths = sorted(root.glob("*.h5"))
    if not paths:
        raise ValueError(f"No H5 files found in {root}")
    samples: list[ByolSample] = []
    for path in paths:
        with h5py.File(path, "r") as h5f:
            missing = [name for name in REQUIRED_SAMPLE_DATASETS if name not in h5f]
            if missing:
                raise ValueError(f"Missing datasets {missing} in {path}")
            shape = h5f["stft"].shape
            if len(shape) != 4 or shape[1:] != (1, 1024, 1024):
                raise ValueError(f"STFT must have shape (N, 1, 1024, 1024), got {shape} in {path}")
            count = shape[0]
            for name in REQUIRED_SAMPLE_DATASETS[1:]:
                if h5f[name].shape[:1] != (count,):
                    raise ValueError(f"Dataset '{name}' has invalid length in {path}")
            if _text(h5f.attrs.get("noise_profile", "")) != "clean":
                raise ValueError(f"Only noise_profile=clean is supported: {path}")
            if int(h5f.attrs.get("sample_length", -1)) != BLOCK_SAMPLE_LENGTH:
                raise ValueError(
                    f"BYOL requires sample_length={BLOCK_SAMPLE_LENGTH}: {path}"
                )
            labels = np.asarray(h5f["labels"][:], dtype=np.int64)
            clean = np.asarray(h5f["is_clean"][:], dtype=bool)
            channels = np.asarray(h5f["rf_channel"][:], dtype=np.int64)
            source_indices = np.asarray(h5f["source_sample_idx"][:], dtype=np.int64)
            if not clean.all():
                raise ValueError(f"All included rows must be clean: {path}")
            if np.any((labels < 0) | (labels > 16)):
                raise ValueError(f"Invalid DroneRFa labels in {path}")
            if count == 0:
                continue
            if len(np.unique(labels)) != 1:
                raise ValueError(f"Each H5 file must contain exactly one original label: {path}")
            if np.any((channels < 0) | (channels > 1)):
                raise ValueError(f"Invalid rf_channel values in {path}")
            if np.any(source_indices < 0):
                raise ValueError(f"source_sample_idx must be non-negative: {path}")
            row_keys = list(zip(source_indices.tolist(), channels.tolist()))
            if len(set(row_keys)) != len(row_keys):
                raise ValueError(
                    "Duplicate (source_sample_idx, rf_channel) rows in "
                    f"{path}"
                )
            if int(labels[0]) == 0:
                continue
            cleaning_method = _text(h5f.attrs.get("cleaning_method", ""))
            if "DCEC-background-filter" not in cleaning_method:
                raise ValueError(f"BYOL requires DCEC-background-filter input: {path}")
            if "dec_source_row_idx" not in h5f:
                raise ValueError(f"DEC source-row lineage is missing: {path}")
            dec_source_rows = np.asarray(h5f["dec_source_row_idx"][:], dtype=np.int64)
            if dec_source_rows.shape != (count,) or np.any(dec_source_rows < 0):
                raise ValueError(f"Invalid dec_source_row_idx in {path}")
            samples.extend(
                ByolSample(
                    str(path.resolve()), row_idx, int(labels[row_idx]),
                    int(channels[row_idx]), int(source_indices[row_idx]),
                    int(dec_source_rows[row_idx]),
                )
                for row_idx in range(count)
            )
    if not samples:
        raise ValueError(f"Clean STFT input contains no rows: {root}")
    return samples


def review_counts(total: int) -> dict[str, int]:
    if total < 6:
        raise ValueError("review-count must be at least 6")
    train = total * 2 // 3
    calibration = (total - train) // 2
    return {"train": train, "calibration": calibration, "audit": total - train - calibration}


def split_blocks(
    samples: list[ByolSample],
    targets: dict[str, int],
    seed: int = 42,
) -> dict[BlockId, str]:
    if set(targets) != set(ROLES) or any(targets[role] <= 0 for role in ROLES):
        raise ValueError("targets must contain positive train, calibration, and audit counts")

    blocks_by_file: dict[str, list[BlockId]] = defaultdict(list)
    for block_id in group_samples_by_block(samples):
        blocks_by_file[block_id[0]].append(block_id)

    result: dict[BlockId, str] = {}
    for source_file in sorted(blocks_by_file):
        block_ids = sorted(blocks_by_file[source_file], key=lambda value: value[1])
        if len(block_ids) < len(ROLES):
            # 稀疏文件无法形成三路互斥划分，全部用于训练并从评估中排除。
            for block_id in block_ids:
                result[block_id] = "train"
            continue

        # 文件名参与稳定种子，避免不同文件获得完全相同的块排列。
        digest = hashlib.sha256(f"{seed}:{source_file}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        block_ids = [block_ids[index] for index in rng.permutation(len(block_ids))]

        assigned = Counter({role: 1 for role in ROLES})
        for _ in range(len(block_ids) - len(ROLES)):
            role = min(
                ROLES,
                key=lambda candidate: (
                    assigned[candidate] / targets[candidate],
                    assigned[candidate],
                    ROLES.index(candidate),
                ),
            )
            assigned[role] += 1

        offset = 0
        for role in ROLES:
            for block_id in block_ids[offset:offset + assigned[role]]:
                result[block_id] = role
            offset += assigned[role]
    return result


def summarize_split_coverage(
    samples: list[ByolSample],
    block_roles: dict[BlockId, str],
) -> dict[str, object]:
    """汇总稀疏文件覆盖范围，供训练日志和清洗报告审计。"""

    groups = group_samples_by_block(samples)
    blocks_by_file = Counter(source_file for source_file, _source_sample_idx in groups)
    samples_by_file = Counter(sample.source_file for sample in samples)
    roles_by_file: dict[str, set[str]] = defaultdict(set)
    for block_id in groups:
        roles_by_file[block_id[0]].add(block_roles[block_id])
    labels_by_file: dict[str, int] = {}
    for sample in samples:
        previous = labels_by_file.setdefault(sample.source_file, sample.label)
        if previous != sample.label:
            raise ValueError(f"Source file contains multiple labels: {sample.source_file}")

    sparse_files = []
    for source_file, block_count in sorted(blocks_by_file.items()):
        if block_count < len(ROLES):
            if roles_by_file[source_file] != {"train"}:
                raise ValueError(
                    f"Sparse source file must be train-only: {source_file}"
                )
            sparse_files.append({
                "source_file": source_file,
                "original_label": labels_by_file[source_file],
                "block_count": block_count,
                "sample_count": samples_by_file[source_file],
            })

    return {
        "full_three_role_source_file_count": len(blocks_by_file) - len(sparse_files),
        "train_only_sparse_source_file_count": len(sparse_files),
        "train_only_sparse_source_files": sparse_files,
    }


def select_reviews(
    samples: list[ByolSample],
    block_roles: dict[BlockId, str],
    targets: dict[str, int],
    seed: int = 42,
) -> list[dict[str, object]]:
    """Stratify post-DEC UAV rows by original label/RF and retain sampling weights."""

    rng = np.random.default_rng(seed)
    selected_rows: list[dict[str, object]] = []
    sample_file_counts = Counter(sample.source_file for sample in samples)
    for role in ROLES:
        if role != "train":
            target = targets[role]
            by_stratum: dict[tuple[int, int], list[int]] = defaultdict(list)
            for index, sample in enumerate(samples):
                if sample.label != 0 and block_roles[sample_block_id(sample)] == role:
                    by_stratum[(sample.label, sample.rf_channel)].append(index)
            strata = tuple(sorted(by_stratum))
            if not strata:
                raise ValueError(f"{role} contains no UAV review candidates")
            if target < len(strata):
                raise ValueError(
                    f"Evaluation target is too small to cover all {len(strata)} "
                    f"available original_label/rf_channel strata in {role}"
                )
            base, extra = divmod(target, len(strata))
            quotas = {stratum: base for stratum in strata}
            for stratum_index in rng.permutation(len(strata))[:extra]:
                quotas[strata[int(stratum_index)]] += 1

            for stratum, quota in quotas.items():
                population = by_stratum[stratum]
                if len(population) < quota:
                    raise ValueError(
                        f"Not enough candidates for {stratum} in {role}: "
                        f"need {quota}, found {len(population)}"
                    )
                chosen = rng.choice(population, size=quota, replace=False)
                weight = len(population) / quota
                selected_rows.extend({
                    "review_role": role,
                    "sample_index": int(index),
                    "sampling_weight": weight,
                    "sampling_stratum": (
                        f"original_label={stratum[0]}|rf_channel={stratum[1]}"
                    ),
                    "sampling_population_count": len(population),
                    "sampling_selected_count": quota,
                } for index in chosen)
            continue

        groups: dict[tuple[int, int, int, str], list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            if sample.label != 0 and block_roles[sample_block_id(sample)] == role:
                file_count = sample_file_counts[sample.source_file]
                position_band = min(3, sample.row_idx * 4 // max(1, file_count))
                groups[(sample.label, sample.rf_channel, position_band, sample.source_file)].append(index)
        for indices in groups.values():
            rng.shuffle(indices)
        keys = sorted(groups)
        rng.shuffle(keys)
        chosen: list[tuple[tuple[int, int, int, str], int]] = []
        while len(chosen) < targets[role] and keys:
            remaining = []
            for key in keys:
                if groups[key] and len(chosen) < targets[role]:
                    chosen.append((key, groups[key].pop()))
                if groups[key]:
                    remaining.append(key)
            keys = remaining
        if len(chosen) != targets[role]:
            raise ValueError(f"Not enough {role} review candidates")
        chosen_counts = Counter(key for key, _index in chosen)
        population_counts = Counter(
            (sample.label, sample.rf_channel, min(
                3, sample.row_idx * 4 // max(1, sample_file_counts[sample.source_file])
            ), sample.source_file)
            for sample in samples if block_roles[sample_block_id(sample)] == role
        )
        for key, index in chosen:
            selected_rows.append({
                "review_role": role,
                "sample_index": index,
                "sampling_weight": population_counts[key] / chosen_counts[key],
                "sampling_stratum": "|".join(map(str, key)),
                "sampling_population_count": population_counts[key],
                "sampling_selected_count": chosen_counts[key],
            })
    return selected_rows


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def load_block_roles(
    manifest_path: str | Path,
    samples: list[ByolSample],
) -> dict[BlockId, str]:
    rows = read_csv(manifest_path)
    required = {
        "source_file", "source_sample_idx", "original_label", "review_role",
        "input_row_count", "review_row_count",
    }
    policy_fields = {"source_block_count", "split_policy"}
    if not rows:
        raise ValueError("split_manifest.csv is empty")
    if not required.issubset(rows[0]):
        if "source_sample_idx" not in rows[0]:
            raise ValueError(
                "Legacy file-level split_manifest.csv is not supported; "
                "rebuild reviews with 10M block-level splitting"
            )
        raise ValueError(f"split_manifest.csv must contain columns: {sorted(required)}")
    present_policy_fields = policy_fields.intersection(rows[0])
    if present_policy_fields and present_policy_fields != policy_fields:
        raise ValueError(
            "split_manifest.csv must contain both source_block_count and split_policy"
        )

    groups = group_samples_by_block(samples)
    block_counts_by_file = Counter(
        source_file for source_file, _source_sample_idx in groups
    )
    has_policy_fields = policy_fields.issubset(rows[0])
    roles: dict[BlockId, str] = {}
    for row in rows:
        block_id = (row["source_file"], int(row["source_sample_idx"]))
        if block_id in roles:
            raise ValueError(f"Duplicate block_id in split_manifest.csv: {block_id}")
        if block_id not in groups:
            raise ValueError(f"Unknown block_id in split_manifest.csv: {block_id}")
        role = row["review_role"]
        if role not in ROLES:
            raise ValueError(f"Invalid review role for {block_id}: {role}")
        member_indices = groups[block_id]
        labels = {samples[index].label for index in member_indices}
        if labels != {int(row["original_label"])}:
            raise ValueError(f"Original label changed for block_id {block_id}")
        input_count = int(row["input_row_count"])
        review_count = int(row["review_row_count"])
        if input_count != len(member_indices):
            raise ValueError(f"Input row count changed for block_id {block_id}")
        if not 0 <= review_count <= input_count:
            raise ValueError(f"Invalid review row count for block_id {block_id}")
        block_count = block_counts_by_file[block_id[0]]
        expected_policy = (
            SPARSE_SPLIT_POLICY
            if block_count < len(ROLES)
            else FULL_SPLIT_POLICY
        )
        if has_policy_fields:
            if int(row["source_block_count"]) != block_count:
                raise ValueError(f"Source block count changed for block_id {block_id}")
            if row["split_policy"] != expected_policy:
                raise ValueError(f"Invalid split policy for block_id {block_id}")
        roles[block_id] = role

    missing = sorted(set(groups) - set(roles))
    if missing:
        raise ValueError(f"split_manifest.csv is missing block_id {missing[0]}")
    roles_by_file: dict[str, set[str]] = defaultdict(set)
    for (source_file, _source_sample_idx), role in roles.items():
        roles_by_file[source_file].add(role)
    for source_file, file_roles in sorted(roles_by_file.items()):
        block_count = block_counts_by_file[source_file]
        if block_count < len(ROLES):
            # 旧 manifest 没有显式稀疏策略，不能静默放宽其完整性约束。
            if not has_policy_fields:
                raise ValueError(
                    "Sparse source files require a sparse-aware split manifest; "
                    "rebuild reviews"
                )
            if file_roles != {"train"}:
                raise ValueError(
                    f"Sparse source file must be train-only: {source_file}"
                )
            continue
        if file_roles != set(ROLES):
            missing_roles = sorted(set(ROLES) - file_roles)
            raise ValueError(
                f"Source file {source_file} is missing block roles {missing_roles}"
            )
    return roles


def join_reviews(review_path: Path, lookup_path: Path, samples: list[ByolSample]) -> list[dict[str, str]]:
    reviews, lookups = read_csv(review_path), read_csv(lookup_path)
    by_id = {row["review_id"]: row for row in lookups}
    if len(by_id) != len(lookups) or len(reviews) != len(lookups):
        raise ValueError("Review and lookup rows must match one-to-one")
    if lookups and "dec_source_row_idx" not in lookups[0]:
        raise ValueError(
            "Legacy BYOL reviews lack DEC lineage; rebuild them from DEC-filtered H5"
        )
    joined, seen = [], set()
    for review in reviews:
        review_id = review.get("review_id", "")
        if review_id in seen or review_id not in by_id:
            raise ValueError(f"Unknown or duplicate review_id: {review_id}")
        seen.add(review_id)
        lookup = by_id[review_id]
        if review.get("review_role") != lookup["review_role"]:
            raise ValueError(f"review_role was changed for {review_id}")
        label = review.get("manual_label", "").strip().lower()
        if label not in MANUAL_LABELS:
            raise ValueError(f"Invalid manual_label for {review_id}: {label}")
        index = int(lookup["sample_index"])
        if not 0 <= index < len(samples):
            raise ValueError(f"Invalid sample_index for {review_id}")
        sample = samples[index]
        sampling_weight = float(lookup["sampling_weight"])
        if not math.isfinite(sampling_weight) or sampling_weight <= 0:
            raise ValueError(f"Invalid sampling_weight for {review_id}")
        actual = (
            lookup["source_file"],
            int(lookup["source_row_idx"]),
            int(lookup["original_label"]),
            int(lookup["rf_channel"]),
            int(lookup["source_sample_idx"]),
            int(lookup["dec_source_row_idx"]),
        )
        expected = (
            sample.source_file,
            sample.row_idx,
            sample.label,
            sample.rf_channel,
            sample.source_sample_idx,
            sample.dec_source_row_idx,
        )
        if actual != expected:
            raise ValueError(f"Lookup does not match source H5 for {review_id}")
        joined.append({**lookup, "manual_label": label})
    return joined


def validate_label_counts(rows: list[dict[str, str]]) -> None:
    minimum = {"train": 40, "calibration": 20, "audit": 20}
    for role in ROLES:
        counts = Counter(row["manual_label"] for row in rows if row["review_role"] == role)
        for label in ("video_present", "no_video"):
            if counts[label] < minimum[role]:
                raise ValueError(f"{role} requires {minimum[role]} '{label}' labels; found {counts[label]}")


class ByolStftDataset(Dataset):
    def __init__(self, samples: list[ByolSample], indices: list[int] | None = None):
        self.samples = samples
        self.indices = indices if indices is not None else list(range(len(samples)))
        self.files: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        index = self.indices[item]
        sample = self.samples[index]
        if sample.path not in self.files:
            self.files[sample.path] = h5py.File(sample.path, "r", rdcc_nbytes=64 * 1024 * 1024)
        value = torch.from_numpy(self.files[sample.path]["stft"][sample.row_idx])
        return preprocess_stft(value, BYOL_INPUT_SIZE), index

    def __getstate__(self):
        state = self.__dict__.copy()
        state["files"] = {}
        return state

    def close(self) -> None:
        for file_obj in self.files.values():
            file_obj.close()
        self.files.clear()


def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minimum_recall: float = 0.98,
    sample_weight: np.ndarray | None = None,
    groups: list[tuple[int, int]] | None = None,
):
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.shape != probabilities.shape or set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Calibration requires both classes and matching probabilities")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.shape != y_true.shape or not np.isfinite(sample_weight).all():
            raise ValueError("Calibration weights must be finite and match labels")
        if np.any(sample_weight <= 0):
            raise ValueError("Calibration weights must be positive")
    if groups is not None and len(groups) != len(y_true):
        raise ValueError("Calibration groups must match labels")
    best = None
    for threshold in np.unique(np.concatenate(([0.0], probabilities, [1.0]))):
        predicted = probabilities >= threshold
        recall = recall_score(y_true, predicted, sample_weight=sample_weight, zero_division=0)
        if recall + 1e-12 < minimum_recall:
            continue
        if groups is not None:
            failed_group = any(
                not predicted[
                    (y_true == 1)
                    & np.asarray([value == group for value in groups], dtype=bool)
                ].any()
                for group in set(groups)
                if np.any(
                    (y_true == 1)
                    & np.asarray([value == group for value in groups], dtype=bool)
                )
            )
            if failed_group:
                continue
        key = (
            precision_score(y_true, predicted, sample_weight=sample_weight, zero_division=0),
            -float(threshold),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold))
    if best is None:
        raise ValueError("No threshold satisfies the recall constraint")
    return best[1]


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z, p = 1.959963984540054, successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def metrics(y_true, y_pred, weights=None) -> dict[str, object]:
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=(0, 1), sample_weight=weights)
    tn, fp, fn, tp = matrix.ravel()
    result = {
        "sample_count": int(len(y_true)),
        "precision": float(precision_score(y_true, y_pred, sample_weight=weights, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, sample_weight=weights, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, sample_weight=weights, zero_division=0)),
        "no_video_removal_rate": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": matrix.tolist(),
    }
    if weights is None:
        result["precision_wilson_95"] = _wilson(int(tp), int(tp + fp))
        result["recall_wilson_95"] = _wilson(int(tp), int(tp + fn))
        result["no_video_removal_wilson_95"] = _wilson(int(tn), int(tn + fp))
    return result


def metrics_by_group(y_true, y_pred, groups, weights=None) -> dict[str, dict[str, object]]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(groups) != len(y_true):
        raise ValueError("Metric groups must match labels")
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
    result = {}
    for group in sorted(set(groups)):
        mask = np.asarray([value == group for value in groups], dtype=bool)
        key = f"original_label={group[0]}|rf_channel={group[1]}"
        values = metrics(
            y_true[mask],
            y_pred[mask],
            None if weights is None else weights[mask],
        )
        values["actual_positive"] = int(y_true[mask].sum())
        result[key] = values
    return result


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: dict[str, object]) -> None:
    with Path(path).open("w", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, ensure_ascii=False, indent=2, allow_nan=False)


def _copy_rows(source: h5py.Dataset, output: h5py.File, name: str, rows: list[int]) -> None:
    target = output.create_dataset(name, shape=(len(rows), *source.shape[1:]), dtype=source.dtype)
    for key, value in source.attrs.items():
        target.attrs[key] = value
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        target[start:start + len(batch)] = source[batch]


def export_clean_h5(data_dir, output_dir, prediction_rows, threshold, checkpoint_hashes):
    source_root, target_root = Path(data_dir).resolve(), Path(output_dir).resolve()
    if source_root == target_root:
        raise ValueError("output-dir must differ from data-dir")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and within [0, 1]")
    paths = sorted(source_root.glob("*.h5"))
    existing = [target_root / path.name for path in paths if (target_root / path.name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite {existing[0]}")
    by_key = {(row["source_file"], int(row["source_row_idx"])): row for row in prediction_rows}
    expected = set()
    counts_by_file = {}
    background_files = set()
    for path in paths:
        with h5py.File(path, "r") as source:
            count = source["stft"].shape[0]
            counts_by_file[path.name] = count
            labels = np.asarray(source["labels"][:], dtype=np.int64)
            if count and len(np.unique(labels)) != 1:
                raise ValueError(f"Each H5 file must contain exactly one label: {path}")
            if count and int(labels[0]) == 0:
                background_files.add(path.name)
            else:
                expected.update((path.name, index) for index in range(count))
    if len(by_key) != len(prediction_rows) or set(by_key) != expected:
        raise ValueError("Predictions must match all non-background input rows one-to-one")
    target_root.mkdir(parents=True, exist_ok=True)
    manifest, decisions = [], Counter()
    prediction_fields = list(prediction_rows[0]) if prediction_rows else []
    for path in paths:
        count = counts_by_file[path.name]
        is_background = path.name in background_files
        if is_background:
            rows = []
            with h5py.File(path, "r") as background_source:
                for source_row in range(count):
                    row = {field: "" for field in prediction_fields}
                    row.update({
                        "sample_index": "",
                        "source_file": path.name,
                        "source_row_idx": source_row,
                        "original_label": 0,
                        "rf_channel": int(background_source["rf_channel"][source_row]),
                        "source_sample_idx": int(
                            background_source["source_sample_idx"][source_row]
                        ),
                        "dec_source_row_idx": "",
                        "decision": "t0000_passthrough",
                    })
                    rows.append(row)
            selected = list(range(count))
        else:
            rows = [by_key[(path.name, index)] for index in range(count)]
            selected = [
                int(row["source_row_idx"])
                for row in rows
                if float(row["ensemble_probability"]) >= threshold
            ]
        output_positions = {source_row: output_row for output_row, source_row in enumerate(selected)}
        temporary, target = target_root / f".{path.name}.tmp", target_root / path.name
        try:
            with h5py.File(path, "r") as source, h5py.File(temporary, "w") as output:
                count = source["stft"].shape[0]
                for key, value in source.attrs.items():
                    output.attrs[key] = value
                upstream = _text(source.attrs.get("cleaning_method", "")).strip()
                output.attrs["cleaning_method"] = (
                    f"{upstream}+BYOL-video-filter" if upstream else "BYOL-video-filter"
                )
                output.attrs["cleaning_input_size"] = 512
                output.attrs["cleaning_video_threshold"] = threshold
                output.attrs["cleaning_checkpoint_sha256"] = json.dumps(checkpoint_hashes)
                output.attrs["cleaning_positive_definition"] = "video_present"
                for name, dataset in source.items():
                    if dataset.shape[:1] == (count,):
                        _copy_rows(dataset, output, name, selected)
                    else:
                        source.copy(name, output)
                if not is_background:
                    output.create_dataset("video_probability", data=np.asarray([
                        float(rows[index]["ensemble_probability"]) for index in selected
                    ], dtype=np.float32))
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        for row in rows:
            source_row = int(row["source_row_idx"])
            kept = source_row in output_positions
            decision = (
                "t0000_passthrough"
                if is_background
                else "video_present" if kept else "no_video"
            )
            decisions[decision] += 1
            manifest.append({**row, "decision": decision,
                             "output_file": path.name if kept else "",
                             "output_row_idx": output_positions.get(source_row, "")})
    return manifest, dict(decisions)

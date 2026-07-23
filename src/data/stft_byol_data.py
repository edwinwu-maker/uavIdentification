"""Data preparation and export helpers for direct BYOL STFT cleaning."""

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


@dataclass(frozen=True)
class ByolSample:
    path: str
    row_idx: int
    label: int
    rf_channel: int
    source_sample_idx: int

    @property
    def source_file(self) -> str:
        return Path(self.path).name


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def scan_clean_stft_h5(data_dir: str | Path) -> tuple[list[ByolSample], dict[str, int]]:
    """Validate and index all background and UAV rows in clean STFT H5 files."""

    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"STFT H5 directory does not exist: {root}")
    paths = sorted(root.glob("*.h5"))
    if not paths:
        raise ValueError(f"No H5 files found in {root}")
    samples: list[ByolSample] = []
    file_counts: dict[str, int] = {}
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
            labels = np.asarray(h5f["labels"][:], dtype=np.int64)
            clean = np.asarray(h5f["is_clean"][:], dtype=bool)
            channels = np.asarray(h5f["rf_channel"][:], dtype=np.int64)
            source_indices = np.asarray(h5f["source_sample_idx"][:], dtype=np.int64)
            if not clean.all():
                raise ValueError(f"All included rows must be clean: {path}")
            if np.any((labels < 0) | (labels > 16)):
                raise ValueError(f"Invalid DroneRFa labels in {path}")
            if np.any((channels < 0) | (channels > 1)):
                raise ValueError(f"Invalid rf_channel values in {path}")
            file_counts[path.name] = count
            samples.extend(
                ByolSample(
                    str(path.resolve()), row_idx, int(labels[row_idx]),
                    int(channels[row_idx]), int(source_indices[row_idx]),
                )
                for row_idx in range(count)
            )
    if not samples:
        raise ValueError(f"Clean STFT input contains no rows: {root}")
    return samples, file_counts


def review_counts(total: int) -> dict[str, int]:
    if total < 6:
        raise ValueError("review-count must be at least 6")
    train = total * 2 // 3
    calibration = (total - train) // 2
    return {"train": train, "calibration": calibration, "audit": total - train - calibration}


def split_files(file_counts: dict[str, int], targets: dict[str, int], seed: int = 42) -> dict[str, str]:
    nonempty = [name for name, count in file_counts.items() if count]
    if len(nonempty) < 3:
        raise ValueError("At least three non-empty H5 files are required")
    rng = np.random.default_rng(seed)
    tie = {name: float(rng.random()) for name in nonempty}
    files = sorted(nonempty, key=lambda name: (-file_counts[name], tie[name]))
    suffix = [0] * (len(files) + 1)
    for index in range(len(files) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + file_counts[files[index]]

    def ordered_roles(index: int, train: int, calibration: int, audit: int) -> list[str]:
        values = {"train": train, "calibration": calibration, "audit": audit}
        return sorted(
            ROLES,
            key=lambda role: (targets[role] - values[role]) / targets[role],
            reverse=True,
        )

    # 每个文件原先占一层 Python 递归，文件较多时会超过递归深度；显式栈保持相同的 DFS 顺序。
    initial = (0, 0, 0, 0)
    stack: list[tuple[tuple[int, int, int, int], list[str], int]] = [
        (initial, ordered_roles(*initial), 0),
    ]
    sequence: list[str] = []
    failed: set[tuple[int, int, int, int]] = set()
    while stack:
        state, roles, next_role = stack[-1]
        index, train, calibration, audit = state
        values = {"train": train, "calibration": calibration, "audit": audit}
        if index == len(files):
            if all(values[role] >= targets[role] for role in ROLES):
                break
            failed.add(state)
            stack.pop()
            if sequence:
                sequence.pop()
            continue
        if suffix[index] < sum(max(0, targets[role] - values[role]) for role in ROLES):
            failed.add(state)
            stack.pop()
            if sequence:
                sequence.pop()
            continue
        if next_role == len(roles):
            failed.add(state)
            stack.pop()
            if sequence:
                sequence.pop()
            continue

        role = roles[next_role]
        stack[-1] = (state, roles, next_role + 1)
        updated = dict(values)
        updated[role] = min(targets[role], updated[role] + file_counts[files[index]])
        child = (index + 1, updated["train"], updated["calibration"], updated["audit"])
        if child in failed:
            continue
        sequence.append(role)
        stack.append((child, ordered_roles(*child), 0))

    if not stack:
        raise ValueError("Cannot create file-disjoint review pools with available files")
    result = dict(zip(files, sequence))
    result.update({name: "train" for name, count in file_counts.items() if count == 0})
    return result


def select_reviews(
    samples: list[ByolSample], file_roles: dict[str, str], targets: dict[str, int], seed: int = 42,
) -> list[dict[str, object]]:
    """Stratify by original label/RF and retain inverse inclusion weights."""

    rng = np.random.default_rng(seed)
    selected_rows: list[dict[str, object]] = []
    sample_file_counts = Counter(sample.source_file for sample in samples)
    for role in ROLES:
        groups: dict[tuple[int, int, int, str], list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            if file_roles[sample.source_file] == role:
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
            (sample.label, sample.rf_channel) for sample in samples
            if file_roles[sample.source_file] == role
        )
        for key, index in chosen:
            selected_rows.append({
                "review_role": role,
                "sample_index": index,
                "sampling_weight": population_counts[key] / chosen_counts[key],
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


def join_reviews(review_path: Path, lookup_path: Path, samples: list[ByolSample]) -> list[dict[str, str]]:
    reviews, lookups = read_csv(review_path), read_csv(lookup_path)
    by_id = {row["review_id"]: row for row in lookups}
    if len(by_id) != len(lookups) or len(reviews) != len(lookups):
        raise ValueError("Review and lookup rows must match one-to-one")
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
        actual = (lookup["source_file"], int(lookup["source_row_idx"]), int(lookup["original_label"]),
                  int(lookup["rf_channel"]), int(lookup["source_sample_idx"]))
        expected = (sample.source_file, sample.row_idx, sample.label, sample.rf_channel, sample.source_sample_idx)
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
    def __init__(self, samples: list[ByolSample], indices: list[int] | None = None, input_size: int = 512):
        self.samples = samples
        self.indices = indices if indices is not None else list(range(len(samples)))
        self.input_size = input_size
        self.files: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        index = self.indices[item]
        sample = self.samples[index]
        if sample.path not in self.files:
            self.files[sample.path] = h5py.File(sample.path, "r", rdcc_nbytes=64 * 1024 * 1024)
        value = torch.from_numpy(self.files[sample.path]["stft"][sample.row_idx])
        return preprocess_stft(value, self.input_size), index

    def __getstate__(self):
        state = self.__dict__.copy()
        state["files"] = {}
        return state

    def close(self) -> None:
        for file_obj in self.files.values():
            file_obj.close()
        self.files.clear()


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, minimum_recall: float = 0.95):
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.shape != probabilities.shape or set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Calibration requires both classes and matching probabilities")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    best = None
    for threshold in np.unique(np.concatenate(([0.0], probabilities, [1.0]))):
        predicted = probabilities >= threshold
        recall = recall_score(y_true, predicted, zero_division=0)
        if recall + 1e-12 < minimum_recall:
            continue
        key = (precision_score(y_true, predicted, zero_division=0), float(threshold))
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
    for path in paths:
        with h5py.File(path, "r") as source:
            counts_by_file[path.name] = source["stft"].shape[0]
            expected.update((path.name, index) for index in range(source["stft"].shape[0]))
    if len(by_key) != len(prediction_rows) or set(by_key) != expected:
        raise ValueError("Predictions must match all input rows one-to-one")
    target_root.mkdir(parents=True, exist_ok=True)
    manifest, decisions = [], Counter()
    for path in paths:
        rows = [by_key[(path.name, index)] for index in range(counts_by_file[path.name])]
        selected = [int(row["source_row_idx"]) for row in rows if float(row["ensemble_probability"]) >= threshold]
        output_positions = {source_row: output_row for output_row, source_row in enumerate(selected)}
        temporary, target = target_root / f".{path.name}.tmp", target_root / path.name
        try:
            with h5py.File(path, "r") as source, h5py.File(temporary, "w") as output:
                count = source["stft"].shape[0]
                for key, value in source.attrs.items():
                    output.attrs[key] = value
                output.attrs["cleaning_method"] = "BYOL-ensemble"
                output.attrs["cleaning_input_size"] = 512
                output.attrs["cleaning_video_threshold"] = threshold
                output.attrs["cleaning_checkpoint_sha256"] = json.dumps(checkpoint_hashes)
                output.attrs["cleaning_positive_definition"] = "video_present (including video+WiFi)"
                for name, dataset in source.items():
                    if dataset.shape[:1] == (count,):
                        _copy_rows(dataset, output, name, selected)
                    else:
                        source.copy(name, output)
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
            decisions["retained" if kept else "removed"] += 1
            manifest.append({**row, "decision": "video_present" if kept else "no_video",
                             "output_file": path.name if kept else "",
                             "output_row_idx": output_positions.get(source_row, "")})
    return manifest, dict(decisions)

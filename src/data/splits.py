import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

__all__ = ["load_test_file_ids", "load_test_subset", "prepare_training_split"]

SPLIT_NAMES = ("train", "val", "test")


def _file_id(path: str) -> str:
    """使用不含扩展名的文件名，让原始 .mat 与预计算 .h5 共用 manifest。"""

    return Path(path).stem


def _file_split_counts(
    file_count: int,
    *,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int, int]:
    if file_count < 3:
        raise ValueError("File-level split requires at least 3 files per class")

    train_count = max(1, int(file_count * train_ratio))
    val_count = max(1, int(file_count * val_ratio))
    if train_count + val_count >= file_count:
        train_count = file_count - 2
        val_count = 1
    test_count = file_count - train_count - val_count
    return train_count, val_count, test_count


def _build_file_split(
    file_labels: dict[str, int],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for file_id, label in file_labels.items():
        grouped[label].append(file_id)

    split_by_file: dict[str, str] = {}
    for label in sorted(grouped):
        file_ids = sorted(grouped[label])
        generator = torch.Generator().manual_seed(seed + label)
        order = torch.randperm(len(file_ids), generator=generator).tolist()
        file_ids = [file_ids[index] for index in order]
        train_count, val_count, _ = _file_split_counts(
            len(file_ids),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        for index, file_id in enumerate(file_ids):
            if index < train_count:
                split_name = "train"
            elif index < train_count + val_count:
                split_name = "val"
            else:
                split_name = "test"
            split_by_file[file_id] = split_name
    return split_by_file


def _select_files_per_class(
    file_labels: dict[str, int],
    *,
    files_per_class: int | None,
    seed: int,
) -> dict[str, int]:
    """按类别和固定 seed 随机选择文件；None 表示使用全部文件。"""

    if files_per_class is None:
        return dict(file_labels)
    if files_per_class <= 0:
        raise ValueError("files_per_class must be positive")

    grouped: dict[int, list[str]] = defaultdict(list)
    for file_id, label in file_labels.items():
        grouped[label].append(file_id)

    selected: dict[str, int] = {}
    for label in sorted(grouped):
        file_ids = sorted(grouped[label])
        if len(file_ids) < files_per_class:
            raise ValueError(
                f"Class {label} has {len(file_ids)} files, "
                f"but files_per_class={files_per_class}"
            )
        generator = torch.Generator().manual_seed(seed + label)
        order = torch.randperm(len(file_ids), generator=generator).tolist()
        for index in order[:files_per_class]:
            selected[file_ids[index]] = label
    return selected


def _save_file_split_manifest(
    manifest_path: str | os.PathLike[str],
    file_labels: dict[str, int],
    split_by_file: dict[str, str],
) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["file_id", "label", "split"])
        writer.writeheader()
        for file_id in sorted(file_labels):
            writer.writerow({
                "file_id": file_id,
                "label": file_labels[file_id],
                "split": split_by_file[file_id],
            })


def _load_file_split_manifest(
    manifest_path: str | os.PathLike[str],
    file_labels: dict[str, int],
    *,
    files_per_class: int | None,
    include_labels: set[int] | None = None,
) -> dict[str, str]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")

    split_by_file: dict[str, str] = {}
    manifest_labels: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        required_columns = {"file_id", "label", "split"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "Split manifest must contain columns: file_id, label, split"
            )
        for row in reader:
            file_id = row["file_id"]
            split_name = row["split"]
            if split_name not in SPLIT_NAMES:
                raise ValueError(f"Unknown split '{split_name}' for file {file_id}")
            if file_id in split_by_file:
                raise ValueError(f"Duplicate file_id in split manifest: {file_id}")
            split_by_file[file_id] = split_name
            manifest_labels[file_id] = int(row["label"])

    if not split_by_file:
        raise ValueError(f"Split manifest is empty: {path}")

    missing = sorted(set(split_by_file) - set(file_labels))
    if missing:
        raise ValueError(
            "Split manifest contains files missing from the dataset: "
            f"{missing[:5]}"
        )
    mismatched = sorted(
        file_id for file_id, label in manifest_labels.items()
        if file_labels[file_id] != label
    )
    if mismatched:
        raise ValueError(f"Split manifest labels do not match the dataset: {mismatched[:5]}")

    dataset_labels = set(file_labels.values())
    if include_labels is not None:
        dataset_labels &= include_labels
        split_by_file = {
            file_id: split_name
            for file_id, split_name in split_by_file.items()
            if manifest_labels[file_id] in include_labels
        }
        manifest_labels = {
            file_id: label
            for file_id, label in manifest_labels.items()
            if label in include_labels
        }

    manifest_label_set = set(manifest_labels.values())
    if manifest_label_set != dataset_labels:
        missing_labels = sorted(dataset_labels - manifest_label_set)
        extra_labels = sorted(manifest_label_set - dataset_labels)
        raise ValueError(
            "Split manifest class labels do not match the dataset: "
            f"missing={missing_labels}, extra={extra_labels}"
        )

    grouped_splits: dict[int, set[str]] = defaultdict(set)
    for file_id, split_name in split_by_file.items():
        grouped_splits[manifest_labels[file_id]].add(split_name)
    incomplete = {
        label: sorted(set(SPLIT_NAMES) - grouped_splits[label])
        for label in sorted(dataset_labels)
        if grouped_splits[label] != set(SPLIT_NAMES)
    }
    if incomplete:
        raise ValueError(
            "Each class in the split manifest must contain train, val, and test files: "
            f"{incomplete}"
        )

    if files_per_class is not None:
        grouped_counts = Counter(manifest_labels.values())
        invalid = {
            label: grouped_counts.get(label, 0)
            for label in sorted(set(file_labels.values()))
            if grouped_counts.get(label, 0) != files_per_class
        }
        if invalid:
            raise ValueError(
                "Split manifest file counts do not match files_per_class="
                f"{files_per_class}: {invalid}"
            )
    return split_by_file


def _collect_file_labels(path_labels: list[tuple[str, int]]) -> dict[str, int]:
    file_labels: dict[str, int] = {}
    for path, label in path_labels:
        file_id = _file_id(path)
        previous_label = file_labels.setdefault(file_id, label)
        if previous_label != label:
            raise ValueError(f"File has multiple labels: {file_id}")
    return file_labels


def _indices_by_split(
    path_labels: list[tuple[str, int]],
    split_by_file: dict[str, str],
) -> dict[str, list[int]]:
    indices = {name: [] for name in SPLIT_NAMES}
    for index, (path, _label) in enumerate(path_labels):
        split_name = split_by_file.get(_file_id(path))
        if split_name is not None:
            indices[split_name].append(index)
    return indices


def prepare_training_split(
    dataset: Dataset,
    *,
    manifest_path: str | os.PathLike[str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    files_per_class: int | None = None,
) -> tuple[Subset, Subset, Subset]:
    """训练专用：按 H5 源文件创建或复用 CSV manifest。"""

    path_labels = [(entry[0], int(entry[2])) for entry in dataset.index]
    file_labels = _collect_file_labels(path_labels)
    if Path(manifest_path).is_file():
        split_by_file = _load_file_split_manifest(
            manifest_path,
            file_labels,
            files_per_class=files_per_class,
        )
    else:
        selected_file_labels = _select_files_per_class(
            file_labels,
            files_per_class=files_per_class,
            seed=seed,
        )
        split_by_file = _build_file_split(
            selected_file_labels,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        _save_file_split_manifest(manifest_path, selected_file_labels, split_by_file)

    indices = _indices_by_split(path_labels, split_by_file)
    return (
        Subset(dataset, indices["train"]),
        Subset(dataset, indices["val"]),
        Subset(dataset, indices["test"]),
    )


def load_test_file_ids(
    path_labels: list[tuple[str, int]],
    *,
    manifest_path: str | os.PathLike[str],
    include_labels: list[int] | tuple[int, ...] | set[int] | None = None,
) -> set[str]:
    """严格读取已有 manifest，并返回指定类别的测试文件 ID。"""

    file_labels = _collect_file_labels(path_labels)
    split_by_file = _load_file_split_manifest(
        manifest_path,
        file_labels,
        files_per_class=None,
        include_labels=None if include_labels is None else set(include_labels),
    )
    return {
        file_id for file_id, split_name in split_by_file.items()
        if split_name == "test"
    }


def load_test_subset(
    dataset: Dataset,
    *,
    manifest_path: str | os.PathLike[str],
) -> Subset:
    """严格按已有 manifest 返回预计算特征测试子集。"""

    path_labels = [(entry[0], int(entry[2])) for entry in dataset.index]
    test_file_ids = load_test_file_ids(
        path_labels,
        manifest_path=manifest_path,
    )
    test_indices = [
        index for index, (path, _label) in enumerate(path_labels)
        if _file_id(path) in test_file_ids
    ]
    if not test_indices:
        raise ValueError("No test samples available from the split manifest")
    return Subset(dataset, test_indices)

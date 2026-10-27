import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

__all__ = ["load_test_file_ids", "load_test_subset", "prepare_training_split"]

SPLIT_NAMES = ("train", "val", "test")


def _file_split_counts(
    file_count: int,
    *,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int, int]:
    """计算单个类别的训练、验证和测试文件数量。

    输入：
        file_count: 该类别的文件总数，至少为 3。
        train_ratio: 训练文件数的计算比例。
        val_ratio: 验证文件数的计算比例，剩余文件分配给测试集。
    输出：
        (train_count, val_count, test_count) 整数元组，总和为 file_count。
        按比例取整并调整数量，确保三个子集各至少有一个文件。
    """
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
    """按类别随机划分文件，固定 seed 时可复现。

    输入：
        file_labels: {文件 ID: 整数标签}，每个类别至少包含 3 个文件。
        train_ratio: 每类训练文件数的计算比例。
        val_ratio: 每类验证文件数的计算比例。
        seed: 随机种子，每类使用 seed + label 生成打乱顺序。
    输出：
        {文件 ID: 子集名称} 字典，名称为 "train"、"val" 或 "test"。
        每个文件只属于一个子集，每个类别在三个子集中均有文件。
    """
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
    """按类别和固定 seed 随机选择文件。

    输入：
        file_labels: {文件 ID: 整数标签}。
        files_per_class: 每类选取的正整数文件数，None 表示使用全部文件。
        seed: 随机种子，每类使用 seed + label 生成抽取顺序。
    输出：
        所选文件的 {文件 ID: 整数标签} 字典，不修改输入字典。
        指定数量时，任一类别文件不足会抛出 ValueError。
    """

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
    """将文件级划分写入 CSV manifest。

    输入：
        manifest_path: CSV 输出路径，接受字符串或 os.PathLike。
        file_labels: 待写入的 {文件 ID: 整数标签}。
        split_by_file: {文件 ID: 子集名称}，须包含 file_labels 中的全部 ID。
    输出：
        返回 None；按文件 ID 排序写入 file_id、label、split 三列。
        自动创建父目录，目标文件已存在时覆盖。
    """
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
    """读取并校验已有 CSV manifest，可按类别筛选划分记录。

    输入：
        manifest_path: 已存在的 CSV 路径，接受字符串或 os.PathLike。
        file_labels: 当前数据集的 {文件 ID: 整数标签}，用于校验文件和标签。
        files_per_class: 每类文件数的校验值，None 表示不校验数量。
        include_labels: 要保留的整数标签集合，None 表示不筛选类别。
    输出：
        校验并筛选后的 {文件 ID: 子集名称} 字典。
        CSV 必须包含 file_id、label、split 列，所选每类须覆盖三个子集。
        路径不存在时抛出 FileNotFoundError，记录不符合校验要求时抛出 ValueError。
    """
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
    """从样本路径与标签中收集文件级标签。

    输入：
        path_labels: 按样本排列的 (文件路径, 整数标签) 列表，同文件可重复出现。
    输出：
        {文件 ID: 整数标签} 字典，ID 为不含最后一个扩展名的文件名。
        同 ID 的记录合并；标签不一致时抛出 ValueError。
    """
    file_labels: dict[str, int] = {}
    for path, label in path_labels:
        file_id = Path(path).stem
        previous_label = file_labels.setdefault(file_id, label)
        if previous_label != label:
            raise ValueError(f"File has multiple labels: {file_id}")
    return file_labels


def _indices_by_split(
    path_labels: list[tuple[str, int]],
    split_by_file: dict[str, str],
) -> dict[str, list[int]]:
    """按文件级划分收集样本索引。

    输入：
        path_labels: 按数据集索引顺序排列的 (文件路径, 整数标签) 列表。
        split_by_file: {文件 ID: 子集名称}，名称为 "train"、"val" 或 "test"。
    输出：
        以三个子集名称为键、样本索引列表为值的字典，各列表保留原始索引顺序。
        同一文件的样本归入同一子集，未出现在划分字典中的文件被跳过。
    """
    indices = {name: [] for name in SPLIT_NAMES}
    for index, (path, _label) in enumerate(path_labels):
        split_name = split_by_file.get(Path(path).stem)
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
    """训练专用：按 H5 源文件创建或复用 CSV manifest。

    输入：
        dataset: 带 index 属性的数据集，每项 entry[0] 为路径、entry[2] 为标签。
        manifest_path: CSV 路径；已存在时校验并复用，不存在时创建。
        train_ratio: 新建划分时每类训练文件的比例，默认 0.6。
        val_ratio: 新建划分时每类验证文件的比例，默认 0.2。
        seed: 新建划分及抽取文件时的随机种子，默认 42。
        files_per_class: 新建时每类抽取的文件数，复用时校验每类文件数。
            None 表示新建时使用全部文件、复用时不校验数量。
    输出：
        按 (训练集, 验证集, 测试集) 顺序返回三个 torch.utils.data.Subset。
        子集引用原始 dataset，同一源文件的样本不会跨子集。
    """

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
    """严格读取已有 manifest，并返回指定类别的测试文件 ID。

    输入：
        path_labels: 当前数据集的 (文件路径, 整数标签) 列表，同文件可重复出现。
        manifest_path: 已存在的 CSV manifest 路径，接受字符串或 os.PathLike。
        include_labels: 要保留的标签列表、元组或集合，None 表示不筛选类别。
    输出：
        测试文件 ID 集合 set[str]，ID 为不含最后一个扩展名的文件名。
        原始 .mat 与同名预计算 .h5 可匹配相同 ID；本函数不创建或修改 manifest。
    """

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
    """严格按已有 manifest 返回预计算特征测试子集。

    输入：
        dataset: 带 index 属性的数据集，每项 entry[0] 为路径、entry[2] 为标签。
        manifest_path: 已存在的 CSV manifest 路径，接受字符串或 os.PathLike。
    输出：
        引用原始 dataset 的 torch.utils.data.Subset，按原始索引顺序包含测试样本。
        不创建或修改 manifest；没有可用测试样本时抛出 ValueError。
    """

    path_labels = [(entry[0], int(entry[2])) for entry in dataset.index]
    test_file_ids = load_test_file_ids(
        path_labels,
        manifest_path=manifest_path,
    )
    test_indices = [
        index for index, (path, _label) in enumerate(path_labels)
        if Path(path).stem in test_file_ids
    ]
    if not test_indices:
        raise ValueError("No test samples available from the split manifest")
    return Subset(dataset, test_indices)

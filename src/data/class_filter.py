"""Utilities for training on a selected subset of the original classes."""

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset


@dataclass(frozen=True)
class ClassMapping:
    """原始标签与连续模型标签之间的映射。"""

    original_labels: tuple[int, ...]

    @property
    def num_classes(self) -> int:
        return len(self.original_labels)

    def to_model(self, original_label: int) -> int:
        try:
            return self.original_labels.index(int(original_label))
        except ValueError as exc:
            raise ValueError(f"Original label is excluded: {original_label}") from exc

    def to_original(self, model_label: int) -> int:
        try:
            return self.original_labels[int(model_label)]
        except IndexError as exc:
            raise ValueError(f"Unknown model label: {model_label}") from exc


def build_class_mapping(num_classes: int, excluded_labels: list[int] | tuple[int, ...] | None) -> ClassMapping:
    excluded = set(excluded_labels or ())
    invalid = sorted(label for label in excluded if label < 0 or label >= num_classes)
    if invalid:
        raise ValueError(f"Excluded labels are outside [0, {num_classes - 1}]: {invalid}")
    original_labels = tuple(label for label in range(num_classes) if label not in excluded)
    if len(original_labels) < 2:
        raise ValueError("At least two classes must remain after exclusion")
    return ClassMapping(original_labels=original_labels)


def add_exclusion_suffix(path: str | Path, excluded_labels) -> str:
    """为屏蔽类别实验生成独立输出名；未屏蔽时保持原路径。"""

    labels = sorted(set(excluded_labels or ()))
    if not labels:
        return str(path)
    output_path = Path(path)
    suffix = "_exclude_" + "_".join(str(label) for label in labels)
    return str(output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}"))


class RemappedSubset(Dataset):
    """过滤 Subset 中的类别，并把保留标签映射为连续编号。"""

    def __init__(self, subset, mapping: ClassMapping) -> None:
        self.subset = subset
        self.mapping = mapping
        allowed = set(mapping.original_labels)
        self.indices = [
            index
            for index in subset.indices
            if int(subset.dataset.index[index][2]) in allowed
        ]
        if not self.indices:
            raise ValueError("No samples remain after class exclusion")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        feature, original_label = self.subset.dataset[self.indices[index]]
        return feature, original_label.new_tensor(self.mapping.to_model(original_label.item()))

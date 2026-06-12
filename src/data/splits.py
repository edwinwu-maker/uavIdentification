import torch
from torch.utils.data import Dataset, Subset, random_split


def split_lengths(
    total_size: int,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[int, int, int]:
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    return train_size, val_size, test_size


def split_dataset(
    dataset: Dataset,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    return random_split(
        dataset,
        split_lengths(len(dataset), train_ratio=train_ratio, val_ratio=val_ratio),
        generator=torch.Generator().manual_seed(seed),
    )

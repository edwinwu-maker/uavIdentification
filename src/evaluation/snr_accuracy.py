from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.drone_rfa_io import count_iq_samples, parse_label
from src.data.splits import split_dataset


@dataclass(frozen=True)
class SampleRecord:
    path: str
    sample_idx: int
    label: int


class H5FileCache:
    """缓存并统一关闭 HDF5 文件句柄。"""

    def __init__(self) -> None:
        self._files: dict[str, h5py.File] = {}

    def get(self, path: str) -> h5py.File:
        if path not in self._files:
            self._files[path] = h5py.File(path, "r", rdcc_nbytes=64 * 1024 * 1024)
        return self._files[path]

    def close(self) -> None:
        for file_obj in self._files.values():
            file_obj.close()
        self._files.clear()


def build_sample_index(
    data_dir: str,
    *,
    sample_length: int,
    max_files: int | None = None,
    max_samples_per_file: int | None = None,
) -> list[SampleRecord]:
    """扫描原始 .mat 文件，生成稳定的样本索引。"""

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Raw data directory does not exist: {data_dir}")

    sample_index: list[SampleRecord] = []
    mat_files = sorted(fname for fname in os.listdir(data_dir) if fname.endswith(".mat"))
    if max_files is not None:
        mat_files = mat_files[:max_files]

    for fname in mat_files:
        path = os.path.join(data_dir, fname)
        label = parse_label(fname)
        with h5py.File(path, "r") as src:
            num_samples = count_iq_samples(
                src,
                sample_length=sample_length,
                max_samples=max_samples_per_file,
            )
        for sample_idx in range(num_samples):
            sample_index.append(SampleRecord(path=path, sample_idx=sample_idx, label=label))

    return sample_index


def prepare_test_records(
    sample_index: list[SampleRecord],
    *,
    data_dir: str,
    max_samples: int | None,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[SampleRecord], int, int]:
    """按现有训练划分规则获取测试样本，并返回 train/val 计数用于日志。"""

    if not sample_index:
        raise ValueError(f"No .mat samples found in {data_dir}")

    if len(sample_index) < 3 or (train_ratio <= 0.0 and val_ratio <= 0.0):
        train_count = 0
        val_count = 0
        test_records = list(sample_index)
    else:
        train_ds, val_ds, test_ds = split_dataset(
            sample_index,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_count = len(train_ds)
        val_count = len(val_ds)
        test_records = [test_ds[i] for i in range(len(test_ds))]

    if max_samples is not None:
        test_records = test_records[:max_samples]
    if not test_records:
        raise ValueError("No test samples available after splitting and filtering")

    return test_records, train_count, val_count


def add_awgn_for_snr(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """按目标 SNR 给复数 IQ 添加 AWGN。"""

    iq = np.asarray(iq, dtype=np.complex64)
    signal_power = float(np.mean(np.abs(iq) ** 2))
    if signal_power <= 0.0:
        return iq.copy()

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape)
    )
    return (iq + noise).astype(np.complex64, copy=False)


def make_record_loader(records: list[SampleRecord], *, batch_size: int) -> DataLoader:
    """只把 DataLoader 用作样本记录批迭代器，避免 HDF5/RNG 进入 worker。"""

    return DataLoader(
        records,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )


def save_snr_accuracy_csv(rows: list[dict[str, object]], output_csv: str | Path) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["snr_db", "num_samples", "num_correct", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def save_snr_accuracy_plot(
    rows: list[dict[str, object]],
    output_png: str | Path,
    *,
    title: str,
) -> None:
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snrs = [int(row["snr_db"]) for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(snrs, accuracies, marker="o", linewidth=2)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

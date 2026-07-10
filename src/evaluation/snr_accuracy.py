from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.drone_rfa_io import (
    LABEL_MAPPING,
    count_iq_samples,
    group_mat_files_by_class,
    parse_label,
    rf_channel_for_file,
)
from src.data.splits import split_records_by_file


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
    files_per_class: int | None = None,
    max_samples_per_file: int | None = None,
) -> list[SampleRecord]:
    """扫描原始 .mat 文件，生成稳定的样本索引。"""

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Raw data directory does not exist: {data_dir}")

    sample_index: list[SampleRecord] = []
    mat_files = sorted(fname for fname in os.listdir(data_dir) if fname.endswith(".mat"))
    if files_per_class is not None:
        if files_per_class <= 0:
            raise ValueError("files_per_class must be positive")
        grouped = group_mat_files_by_class(mat_files)
        mat_files = []
        for class_code in LABEL_MAPPING:
            mat_files.extend(grouped[class_code][:files_per_class])
    elif max_files is not None:
        mat_files = mat_files[:max_files]

    for fname in mat_files:
        path = os.path.join(data_dir, fname)
        label = parse_label(fname)
        rf_channel = rf_channel_for_file(fname)
        with h5py.File(path, "r") as src:
            num_samples = count_iq_samples(
                src,
                rf_channel=rf_channel,
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
    split_manifest: str | Path,
    files_per_class: int | None = None,
) -> tuple[list[SampleRecord], int, int]:
    """按现有训练划分规则获取测试样本，并返回 train/val 计数用于日志。"""

    if not sample_index:
        raise ValueError(f"No .mat samples found in {data_dir}")

    train_records, val_records, test_records = split_records_by_file(
        sample_index,
        manifest_path=split_manifest,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        files_per_class=files_per_class,
    )
    train_count = len(train_records)
    val_count = len(val_records)

    if max_samples is not None:
        test_records = test_records[:max_samples]
    if not test_records:
        raise ValueError("No test samples available after splitting and filtering")

    return test_records, train_count, val_count


def save_prediction_diagnostics(
    rows: list[dict[str, object]],
    *,
    predictions_csv: str | Path | None,
    per_file_csv: str | Path | None,
) -> None:
    """保存逐样本预测及按 SNR/源文件聚合的诊断结果。"""

    if predictions_csv is not None:
        path = Path(predictions_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["snr_db", "source_file", "sample_idx", "true_label", "pred_label", "correct"]
        with path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if per_file_csv is None:
        return

    grouped: dict[tuple[float, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (float(row["snr_db"]), str(row["source_file"]), int(row["true_label"]))
        grouped[key].append(row)

    output_rows = []
    for (snr_db, source_file, true_label), file_rows in sorted(grouped.items()):
        correct = sum(int(row["correct"]) for row in file_rows)
        wrong_predictions = Counter(
            int(row["pred_label"]) for row in file_rows if not int(row["correct"])
        )
        if wrong_predictions:
            top_wrong_label, top_wrong_count = wrong_predictions.most_common(1)[0]
        else:
            top_wrong_label, top_wrong_count = "", 0
        output_rows.append({
            "snr_db": snr_db,
            "source_file": source_file,
            "true_label": true_label,
            "num_samples": len(file_rows),
            "num_correct": correct,
            "accuracy": correct / len(file_rows),
            "top_wrong_label": top_wrong_label,
            "top_wrong_count": top_wrong_count,
        })

    path = Path(per_file_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "snr_db", "source_file", "true_label", "num_samples", "num_correct",
        "accuracy", "top_wrong_label", "top_wrong_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


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


def format_snr_for_filename(snr_db: float) -> str:
    """把 SNR 数值转换成稳定、安全的文件名片段。"""

    snr = float(snr_db)
    if snr < 0:
        prefix = "m"
    elif snr > 0:
        prefix = "p"
    else:
        prefix = ""

    value = abs(snr)
    if value.is_integer():
        value_text = str(int(value))
    else:
        value_text = f"{value:g}".replace(".", "p")
    return f"{prefix}{value_text}"


def save_snr_confusion_matrix(
    cm: np.ndarray,
    output_npy: str | Path,
    output_png: str | Path,
    *,
    title: str,
) -> None:
    """保存单个 SNR 点的混淆矩阵数组和图片。"""

    npy_path = Path(output_npy)
    png_path = Path(output_png)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, cm)

    from src.training.metrics import save_confusion_matrix_image

    save_confusion_matrix_image(cm, png_path, title=title)


def save_snr_accuracy_plot(
    rows: list[dict[str, object]],
    output_png: str | Path,
    *,
    title: str,
) -> None:
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snrs = [float(row["snr_db"]) for row in rows]
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

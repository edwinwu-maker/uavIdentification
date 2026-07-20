from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_prediction_diagnostics(
    rows: list[dict[str, object]],
    *,
    predictions_csv: str | Path | None,
    per_file_csv: str | Path | None,
) -> None:
    """保存逐样本预测及按 SNR/源文件/通道聚合的诊断结果。"""

    if predictions_csv is not None:
        path = Path(predictions_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "snr_db", "source_file", "sample_idx", "rf_channel", "true_label", "pred_label",
        ]
        if rows and "original_true_label" in rows[0]:
            fieldnames.extend(["original_true_label", "original_pred_label"])
        fieldnames.append("correct")
        with path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if per_file_csv is None:
        return

    grouped: dict[tuple[float, str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            float(row["snr_db"]), str(row["source_file"]),
            int(row["rf_channel"]), int(row["true_label"]),
        )
        grouped[key].append(row)

    output_rows = []
    for (snr_db, source_file, rf_channel, true_label), file_rows in sorted(grouped.items()):
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
            "rf_channel": rf_channel,
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
        "snr_db", "source_file", "rf_channel", "true_label", "num_samples", "num_correct",
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
    class_labels=None,
) -> None:
    """保存单个 SNR 点的混淆矩阵数组和图片。"""

    npy_path = Path(output_npy)
    png_path = Path(output_png)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, cm)

    from src.training.metrics import save_confusion_matrix_image

    save_confusion_matrix_image(cm, png_path, title=title, class_labels=class_labels)


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

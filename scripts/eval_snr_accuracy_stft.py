"""Evaluate STFT ResNet robustness under synthetic SNR levels.

Usage:
  python scripts/eval_snr_accuracy_stft.py
  python scripts/eval_snr_accuracy_stft.py --data-dir E:/dataSet/DroneRFa
  python scripts/eval_snr_accuracy_stft.py --model-path outputs/checkpoints/best_stft_model.pth
  python scripts/eval_snr_accuracy_stft.py --snrs -10 0 10 --max-samples 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.drone_rfa_io import count_iq_samples, default_raw_data_dir, parse_label, read_iq_batch
from src.data.splits import split_dataset
from src.models.resnet import DroneRFaResNet18
from src.training.checkpoint import load_checkpoint
from src.preprocess.stft import compute_stft
from src.utils.device import default_device
from src.utils.logger import logger
from src.utils.paths import checkpoint_dir, figures_dir, metrics_dir

DEFAULT_SNRS = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
DEFAULT_SAMPLE_LENGTH = 1_000_000
DEFAULT_BATCH_SIZE = 8
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42
DEFAULT_N_FFT = 1024
DEFAULT_WIN_LENGTH = 1024
DEFAULT_SPEC_TIME_BINS = 1024
DEFAULT_MODEL_NAME = "best_stft_model.pth"
DEFAULT_OUTPUT_CSV = metrics_dir() / "stft_snr_accuracy.csv"
DEFAULT_OUTPUT_PNG = figures_dir() / "stft_snr_accuracy.png"

# (frozen=True)对象创建初始化之后，不能修改、新增、删除任何实例属性
@dataclass(frozen=True)
class SampleRecord:
    path: str       # 样本归属 .mat 文件
    sample_idx: int # .mat 文件里的第几个样本
    label: int      # 类别标签


class H5FileCache:
    """ 缓存并统一关闭 HDF5 文件句柄 """
    def __init__(self) -> None:
        # 初始化一个字典 self._files，用来保存已经打开过的 h5py.File
        self._files: dict[str, h5py.File] = {}

    def get(self, path: str) -> h5py.File:
        # 如果某个 .mat 文件还没打开，就用 h5py.File 打开并缓存起来；如果已经打开过，直接复用
        if path not in self._files:
            self._files[path] = h5py.File(path, "r", rdcc_nbytes=64 * 1024 * 1024)
        return self._files[path]

    def close(self) -> None:
        # 把缓存里的所有文件都关闭，然后清空字典
        for file_obj in self._files.values():
            file_obj.close()
        self._files.clear()


def build_sample_index(
    data_dir: str,
    *,
    sample_length: int = DEFAULT_SAMPLE_LENGTH,
    max_files: int | None = None,
    max_samples_per_file: int | None = None,
) -> list[SampleRecord]:
    """Scan raw .mat files and build a stable sample index."""

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Raw data directory does not exist: {data_dir}")

    sample_index: list[SampleRecord] = []
    mat_files = sorted(fname for fname in os.listdir(data_dir) if fname.endswith(".mat"))

    # 如果传了 max_files，就只取前 N 个文件(一般测试用)
    if max_files is not None:
        mat_files = mat_files[:max_files]

    for fname in mat_files:
        path = os.path.join(data_dir, fname)
        label = parse_label(fname)
        with h5py.File(path, "r") as src:
            # 打开文件，用 count_iq_samples() 计算这个文件里有多少个可用 IQ 样本
            num_samples = count_iq_samples(
                src,
                sample_length=sample_length,
                max_samples=max_samples_per_file,
            )
        for sample_idx in range(num_samples):
            sample_index.append(SampleRecord(path=path, sample_idx=sample_idx, label=label))

    return sample_index


def add_awgn_for_snr(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN to one IQ sample at the requested SNR."""

    iq = np.asarray(iq, dtype=np.complex64)
    signal_power = float(np.mean(np.abs(iq) ** 2))
    if signal_power <= 0.0:
        return iq.copy()

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape)
    )
    return (iq + noise).astype(np.complex64, copy=False)


def _load_iq_batch(
    file_cache: H5FileCache,
    record: SampleRecord,
    *,
    sample_length: int,
) -> np.ndarray:
    """从缓存里的 HDF5 文件中读取某一条样本，并把它转换成单个复数 IQ 样本(np.complex64)返回"""
    src = file_cache.get(record.path)
    # iq形状通常是 (1, 2, sample_length)
    iq = read_iq_batch(
        src,
        sample_length=sample_length,
        start_idx=record.sample_idx,
        end_idx=record.sample_idx + 1,
    )
    # 这里取第 0 个样本，返回(2, sample_length)
    return iq[0]


def _save_csv(rows: list[dict[str, object]], output_csv: str | Path) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["snr_db", "num_samples", "num_correct", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(rows: list[dict[str, object]], output_png: str | Path) -> None:
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snrs = [int(row["snr_db"]) for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(snrs, accuracies, marker="o", linewidth=2)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("STFT SNR-Accuracy Curve")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate_snr_accuracy(
    *,
    data_dir: str,
    model_path: str,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    output_png: str | Path = DEFAULT_OUTPUT_PNG,
    device: str = default_device(),
    batch_size: int = DEFAULT_BATCH_SIZE,
    snrs: list[int] | tuple[int, ...] = DEFAULT_SNRS,
    sample_length: int = DEFAULT_SAMPLE_LENGTH,
    max_files: int | None = None,
    max_samples_per_file: int | None = None,
    max_samples: int | None = None,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> list[dict[str, object]]:
    """Run SNR-wise evaluation and save CSV/PNG outputs."""

    # 扫描原始 .mat 文件，按 sample_length 切成list[SampleRecord]
    sample_index = build_sample_index(
        data_dir,
        sample_length=sample_length,
        max_files=max_files,
        max_samples_per_file=max_samples_per_file,
    )
    if len(sample_index) < 3 or (train_ratio <= 0.0 and val_ratio <= 0.0):
        train_ds, val_ds = [], []
        test_records = list(sample_index)
    else:
        train_ds, val_ds, test_ds = split_dataset(
            sample_index,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        test_records = [test_ds[i] for i in range(len(test_ds))]
    if max_samples is not None:
        test_records = test_records[:max_samples]

    logger.info(
        "Loaded %d samples from %d files; split train=%d val=%d test=%d",
        len(sample_index),
        len(set(record.path for record in sample_index)),
        len(train_ds),
        len(val_ds),
        len(test_records),
    )

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)

    model = DroneRFaResNet18(num_classes=25).to(torch_device)
    logger.info("Loading model from %s", model_path)
    state_dict = load_checkpoint(model_path, map_location=torch_device)
    model.load_state_dict(state_dict)
    model.eval()

    file_cache = H5FileCache()
    try:
        rows: list[dict[str, object]] = []
        for snr_db in tqdm(snrs, desc="SNR", unit="snr"):
            rng = np.random.default_rng(seed)
            num_correct = 0     # 统计这个 SNR 下预测正确的样本数量
            num_samples = 0     # 统计这个 SNR 下测试的总样本数量

            # 把 test_records 按 batch_size 切成批
            test_loader = DataLoader(
                test_records,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch: batch, 
            )

            with torch.inference_mode():
                # batch_records 是一个 SampleRecord 的列表，长度不超过 batch_size
                for batch_records in tqdm(
                    test_loader,
                    desc=f"SNR {snr_db} dB",
                    unit="batch",
                    leave=False,
                ):
                    iq_batch = []
                    labels = []
                    for record in batch_records:
                        iq = _load_iq_batch(file_cache, record, sample_length=sample_length)
                        noisy_iq = add_awgn_for_snr(iq, snr_db, rng)
                        iq_batch.append(noisy_iq)
                        labels.append(record.label)

                    stft_batch = compute_stft(
                        np.stack(iq_batch, axis=0),
                        device=device,
                        n_fft=DEFAULT_N_FFT,
                        win_length=DEFAULT_WIN_LENGTH,
                        spec_time_bins=DEFAULT_SPEC_TIME_BINS,
                    )
                    logits = model(stft_batch.to(torch_device))
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    labels_np = np.asarray(labels, dtype=np.int64)
                    num_correct += int(np.sum(preds == labels_np))
                    num_samples += len(labels_np)

            accuracy = float(num_correct / num_samples) if num_samples else 0.0
            rows.append(
                {
                    "snr_db": int(snr_db),
                    "num_samples": int(num_samples),
                    "num_correct": int(num_correct),
                    "accuracy": accuracy,
                }
            )
            logger.info("SNR %s dB: %d/%d = %.4f", snr_db, num_correct, num_samples, accuracy)

        _save_csv(rows, output_csv)
        _save_plot(rows, output_png)
        logger.info("Saved CSV to %s", output_csv)
        logger.info("Saved plot to %s", output_png)
        return rows
    finally:
        file_cache.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate STFT SNR-accuracy curve on raw DroneRFa IQ files")
    parser.add_argument("--data-dir", type=str, default=default_raw_data_dir(),
                        help="Directory containing raw .mat files")
    parser.add_argument("--model-path", type=str, default=str(checkpoint_dir() / DEFAULT_MODEL_NAME),
                        help="Path to the trained STFT checkpoint")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV),
                        help="CSV path for SNR results")
    parser.add_argument("--output-png", type=str, default=str(DEFAULT_OUTPUT_PNG),
                        help="PNG path for SNR curve")
    parser.add_argument("--device", type=str, default=default_device(),
                        help="Device, e.g. 'cuda:0', 'mps', or 'cpu'")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--snrs", type=int, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--sample-length", type=int, default=DEFAULT_SAMPLE_LENGTH)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-samples-per-file", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main() -> None:
    args = parse_args()
    evaluate_snr_accuracy(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_csv=args.output_csv,
        output_png=args.output_png,
        device=args.device,
        batch_size=args.batch_size,
        snrs=args.snrs,
        sample_length=args.sample_length,
        max_files=args.max_files,
        max_samples_per_file=args.max_samples_per_file,
        max_samples=args.max_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

"""Evaluate CPP ResNet robustness under synthetic SNR levels.

Usage:
  # CLI — quick evaluation with defaults
  python scripts/eval_snr_accuracy_cpp.py --split-manifest outputs/splits/full.csv
  python scripts/eval_snr_accuracy_cpp.py --data-dir E:/dataSet/DroneRFa --snrs -10 0 10 --split-manifest outputs/splits/full.csv
  python scripts/eval_snr_accuracy_cpp.py --files-per-class 12 \
    --split-manifest outputs/splits/select12.csv \
    --predictions-csv outputs/metrics/select12_cpp_predictions.csv \
    --per-file-csv outputs/metrics/select12_cpp_per_file.csv

  # API — import and call from other scripts
  from scripts.eval_snr_accuracy_cpp import evaluate_snr_accuracy
  rows = evaluate_snr_accuracy(
      data_dir="E:/dataSet/DroneRFa",
      model_path="outputs/checkpoints/best_cpp_model.pth",
      split_manifest="outputs/splits/full.csv",
      snrs=[0, 10, 20],
      max_samples=50,
  )
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.drone_rfa_io import default_raw_data_dir, read_iq_batch
from src.evaluation.snr_accuracy import (
    H5FileCache,
    SampleRecord,
    add_awgn_for_snr,
    build_sample_index,
    format_snr_for_filename,
    make_record_loader,
    prepare_test_records,
    save_snr_accuracy_csv,
    save_snr_accuracy_plot,
    save_snr_confusion_matrix,
    save_prediction_diagnostics,
)
from src.models.resnet import NUM_CLASSES, build_model
from src.preprocess.cpp import compute_cpp, normalize_cpp
from src.preprocess.rf_segmentation import segment_predominant_rf
from src.training.checkpoint import load_checkpoint
from src.utils.cli import log_current_command
from src.utils.device import default_device
from src.utils.logger import logger
from src.utils.paths import checkpoint_dir, figures_dir, metrics_dir

DEFAULT_SNRS = [-15, -10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]
DEFAULT_SAMPLE_LENGTH = 1_000_000
DEFAULT_SEGMENT_SAMPLES = 262_144
DEFAULT_SEGMENT_HOP_SAMPLES = None
DEFAULT_FAM_MERGE = "mean"
DEFAULT_F_BINS = 257
DEFAULT_ALPHA_BINS = 257
DEFAULT_PAIR_CHUNK_SIZE = 8192
DEFAULT_BATCH_SIZE = 1
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42
DEFAULT_MODEL_NAME = "best_cpp_model.pth"
DEFAULT_OUTPUT_CSV = metrics_dir() / "cpp_snr_accuracy.csv"
DEFAULT_OUTPUT_PNG = figures_dir() / "cpp_snr_accuracy.png"
DEFAULT_OUTPUT_CM_PREFIX = "cpp_snr_confusion_matrix"
DEFAULT_RF_FRAME_LEN = 10_000
DEFAULT_RF_TARGET_LEN = 100_000


def _load_dual_iq_sample(
    file_cache: H5FileCache,
    record: SampleRecord,
    *,
    sample_length: int,
) -> np.ndarray:
    src = file_cache.get(record.path)
    iq_batch = read_iq_batch(
        src,
        sample_length=sample_length,
        start_idx=record.sample_idx,
        end_idx=record.sample_idx + 1,
    )
    return iq_batch[0]


def evaluate_snr_accuracy(
    *,
    data_dir: str,
    model_path: str,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    output_png: str | Path = DEFAULT_OUTPUT_PNG,
    output_cm_prefix: str = DEFAULT_OUTPUT_CM_PREFIX,
    device: str = default_device(),
    model_name: str = "resnet18-small-stem",
    batch_size: int = DEFAULT_BATCH_SIZE,
    snrs: list[float] | tuple[float, ...] = DEFAULT_SNRS,
    sample_length: int = DEFAULT_SAMPLE_LENGTH,
    segment_samples: int = DEFAULT_SEGMENT_SAMPLES,
    segment_hop_samples: int | None = DEFAULT_SEGMENT_HOP_SAMPLES,
    fam_merge: str = DEFAULT_FAM_MERGE,
    f_bins: int = DEFAULT_F_BINS,
    alpha_bins: int = DEFAULT_ALPHA_BINS,
    pair_chunk_size: int = DEFAULT_PAIR_CHUNK_SIZE,
    fam_nfft: int = 256,
    fam_hop: int = 256,
    cpp_normalization: str = "log-zscore-sample",
    max_files: int | None = None,
    files_per_class: int | None = None,
    max_samples_per_file: int | None = None,
    max_samples: int | None = None,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    split_manifest: str | Path,
    predictions_csv: str | Path | None = None,
    per_file_csv: str | Path | None = None,
    use_rf_segmentation: bool = False,
    rf_frame_len: int = DEFAULT_RF_FRAME_LEN,
    rf_target_len: int | None = DEFAULT_RF_TARGET_LEN,
    rf_top_k: int | None = None,
) -> list[dict[str, object]]:
    """Run SNR-wise CPP evaluation and save CSV/PNG outputs."""

    sample_index = build_sample_index(
        data_dir,
        sample_length=sample_length,
        max_files=max_files,
        files_per_class=None,
        max_samples_per_file=max_samples_per_file,
    )

    test_records, train_count, val_count = prepare_test_records(
        sample_index,
        data_dir=data_dir,
        max_samples=max_samples,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_manifest=split_manifest,
        files_per_class=files_per_class,
    )

    logger.info(
        "Loaded %d samples from %d files; split train=%d val=%d test=%d",
        len(sample_index),
        len(set(record.path for record in sample_index)),
        train_count,
        val_count,
        len(test_records),
    )

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)

    model = build_model(model_name, num_classes=NUM_CLASSES).to(torch_device)
    logger.info("Loading model from %s", model_path)
    state_dict = load_checkpoint(model_path, map_location=torch_device)
    model.load_state_dict(state_dict)
    model.eval()

    file_cache = H5FileCache()
    try:
        rows: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        for snr_db in tqdm(snrs, desc="SNR", unit="snr"):
            rng = np.random.default_rng(seed)
            num_correct = 0
            num_samples = 0
            all_labels: list[int] = []
            all_preds: list[int] = []

            with torch.inference_mode():
                test_loader = make_record_loader(test_records, batch_size=batch_size)
                for batch_records in tqdm(
                    test_loader,
                    total=len(test_loader),
                    desc=f"SNR {snr_db} dB",
                    unit="batch",
                    leave=False,
                ):
                    cpp_batch = []
                    labels = []
                    for record in batch_records:
                        iq = _load_dual_iq_sample(file_cache, record, sample_length=sample_length)
                        noisy_iq = add_awgn_for_snr(iq, snr_db, rng)
                        noisy_iq = torch.as_tensor(noisy_iq, dtype=torch.complex64, device=device)
                        ch0 = noisy_iq[0]
                        ch1 = noisy_iq[1]
                        if use_rf_segmentation:
                            ch0, _, _ = segment_predominant_rf(
                                ch0,
                                frame_len=rf_frame_len,
                                target_len=rf_target_len,
                                top_k=rf_top_k,
                                device=device,
                            )
                            ch1, _, _ = segment_predominant_rf(
                                ch1,
                                frame_len=rf_frame_len,
                                target_len=rf_target_len,
                                top_k=rf_top_k,
                                device=device,
                            )
                        cpp, _, _ = compute_cpp(
                            ch0,
                            ch1,
                            segment_samples=segment_samples,
                            segment_hop_samples=segment_hop_samples,
                            fam_merge=fam_merge,
                            f_bins=f_bins,
                            alpha_bins=alpha_bins,
                            device=device,
                            pair_chunk_size=pair_chunk_size,
                            fam_nfft=fam_nfft,
                            fam_hop=fam_hop,
                        )
                        cpp_batch.append(normalize_cpp(cpp, mode=cpp_normalization))
                        labels.append(record.label)

                    inputs = torch.stack(cpp_batch, dim=0).to(torch_device)
                    logits = model(inputs)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    labels_np = np.asarray(labels, dtype=np.int64)
                    num_correct += int(np.sum(preds == labels_np))
                    num_samples += len(labels_np)
                    all_labels.extend(labels_np.tolist())
                    all_preds.extend(preds.tolist())
                    for record, label, pred in zip(batch_records, labels_np, preds):
                        prediction_rows.append({
                            "snr_db": float(snr_db),
                            "source_file": Path(record.path).name,
                            "sample_idx": int(record.sample_idx),
                            "true_label": int(label),
                            "pred_label": int(pred),
                            "correct": int(label == pred),
                        })

            accuracy = float(num_correct / num_samples) if num_samples else 0.0
            snr_name = format_snr_for_filename(float(snr_db))
            cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
            cm_npy = metrics_dir() / f"{output_cm_prefix}_snr_{snr_name}.npy"
            cm_png = figures_dir() / f"{output_cm_prefix}_snr_{snr_name}.png"
            save_snr_confusion_matrix(
                cm,
                cm_npy,
                cm_png,
                title=f"CPP Confusion Matrix ({snr_db:g} dB)",
            )
            rows.append(
                {
                    "snr_db": float(snr_db),
                    "num_samples": int(num_samples),
                    "num_correct": int(num_correct),
                    "accuracy": accuracy,
                }
            )
            logger.info("SNR %s dB: %d/%d = %.4f", snr_db, num_correct, num_samples, accuracy)
            logger.info("SNR %s dB confusion matrix saved to %s and %s", snr_db, cm_npy, cm_png)

        save_snr_accuracy_csv(rows, output_csv)
        save_snr_accuracy_plot(rows, output_png, title="CPP SNR-Accuracy Curve")
        save_prediction_diagnostics(
            prediction_rows,
            predictions_csv=predictions_csv,
            per_file_csv=per_file_csv,
        )
        logger.info("Saved CSV to %s", output_csv)
        logger.info("Saved plot to %s", output_png)
        return rows
    finally:
        file_cache.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CPP SNR-accuracy curve on raw DroneRFa IQ files")
    parser.add_argument("--data-dir", type=str, default=default_raw_data_dir(),
                        help="Directory containing raw .mat files")
    parser.add_argument("--model-path", type=str, default=str(checkpoint_dir() / DEFAULT_MODEL_NAME),
                        help="Path to the trained CPP checkpoint")
    parser.add_argument("--model", type=str, default="resnet18-small-stem", choices=("resnet18", "resnet18-small-stem"),
                        help="Model architecture used by the CPP checkpoint")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV),
                        help="CSV path for SNR results")
    parser.add_argument("--output-png", type=str, default=str(DEFAULT_OUTPUT_PNG),
                        help="PNG path for SNR curve")
    parser.add_argument("--output-cm-prefix", type=str, default=DEFAULT_OUTPUT_CM_PREFIX,
                        help="Filename prefix for per-SNR confusion matrices in outputs/metrics and outputs/figures")
    parser.add_argument("--device", type=str, default=default_device(),
                        help="Device, e.g. 'cuda:0', 'mps', or 'cpu'")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--snrs", type=float, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--sample-length", type=int, default=DEFAULT_SAMPLE_LENGTH)
    parser.add_argument("--segment-samples", type=int, default=DEFAULT_SEGMENT_SAMPLES)
    parser.add_argument("--segment-hop-samples", type=int, default=DEFAULT_SEGMENT_HOP_SAMPLES)
    parser.add_argument("--fam-merge", type=str, default=DEFAULT_FAM_MERGE, choices=("mean", "max"))
    parser.add_argument("--f-bins", type=int, default=DEFAULT_F_BINS)
    parser.add_argument("--alpha-bins", type=int, default=DEFAULT_ALPHA_BINS)
    parser.add_argument("--pair-chunk-size", type=int, default=DEFAULT_PAIR_CHUNK_SIZE)
    parser.add_argument("--fam-nfft", type=int, default=256)
    parser.add_argument("--fam-hop", type=int, default=256)
    parser.add_argument("--cpp-normalization", choices=("max", "log-zscore-sample"),
                        default="log-zscore-sample")
    parser.add_argument("--use-rf-segmentation", action="store_true",
                        help="Apply ST-ESER predominant segment selection after AWGN and before CPP")
    parser.add_argument("--rf-frame-len", type=int, default=DEFAULT_RF_FRAME_LEN)
    parser.add_argument("--rf-target-len", type=int, default=DEFAULT_RF_TARGET_LEN)
    parser.add_argument("--rf-top-k", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--files-per-class", type=int, default=None,
                        help="Select this many files per class when creating the manifest, or validate an existing manifest")
    parser.add_argument("--max-samples-per-file", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-manifest", type=str, required=True,
                        help="Shared CSV manifest for file-level train/val/test splitting")
    parser.add_argument("--predictions-csv", type=str, default=None,
                        help="Optional CSV path for per-sample predictions")
    parser.add_argument("--per-file-csv", type=str, default=None,
                        help="Optional CSV path for per-SNR, per-file metrics")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main() -> None:
    args = parse_args()
    log_current_command(logger)
    evaluate_snr_accuracy(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_csv=args.output_csv,
        output_png=args.output_png,
        output_cm_prefix=args.output_cm_prefix,
        device=args.device,
        model_name=args.model,
        batch_size=args.batch_size,
        snrs=args.snrs,
        sample_length=args.sample_length,
        segment_samples=args.segment_samples,
        segment_hop_samples=args.segment_hop_samples,
        fam_merge=args.fam_merge,
        f_bins=args.f_bins,
        alpha_bins=args.alpha_bins,
        pair_chunk_size=args.pair_chunk_size,
        fam_nfft=args.fam_nfft,
        fam_hop=args.fam_hop,
        cpp_normalization=args.cpp_normalization,
        max_files=args.max_files,
        files_per_class=args.files_per_class,
        max_samples_per_file=args.max_samples_per_file,
        max_samples=args.max_samples,
        seed=args.seed,
        split_manifest=args.split_manifest,
        predictions_csv=args.predictions_csv,
        per_file_csv=args.per_file_csv,
        use_rf_segmentation=args.use_rf_segmentation,
        rf_frame_len=args.rf_frame_len,
        rf_target_len=args.rf_target_len,
        rf_top_k=args.rf_top_k,
    )


if __name__ == "__main__":
    main()

"""Evaluate CPP ResNet robustness under synthetic SNR levels.

Usage:
  # CLI — quick evaluation with defaults
  python scripts/eval_snr_accuracy_cpp.py --split-manifest outputs/splits/full.csv
  python scripts/eval_snr_accuracy_cpp.py --data-dir E:/dataSet/DroneRFa --snrs -10 0 10 --split-manifest outputs/splits/full.csv
  python scripts/eval_snr_accuracy_cpp.py --split-manifest outputs/splits/select12.csv \
    --predictions-csv outputs/metrics/select12_cpp_predictions.csv \
    --per-file-csv outputs/metrics/select12_cpp_per_file.csv

  # API — import and call from other scripts
  from scripts.eval_snr_accuracy_cpp import evaluate_snr_accuracy
  rows = evaluate_snr_accuracy(
      data_dir="E:/dataSet/DroneRFa",
      model_path="outputs/checkpoints/best_cpp_17class_model.pth",
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

from src.data.drone_rfa_io import (
    H5FileHandleCache,
    build_sample_index,
    default_raw_data_dir,
    list_mat_file_labels,
    load_iq_sample,
)
from src.data.splits import load_test_file_ids
from src.data.class_filter import add_exclusion_suffix, build_class_mapping
from src.evaluation.snr_accuracy import (
    add_awgn_for_snr,
    format_snr_for_filename,
    make_record_loader,
    save_snr_accuracy_csv,
    save_snr_accuracy_plot,
    save_snr_confusion_matrix,
    save_prediction_diagnostics,
)
from src.models.resnet import NUM_CLASSES, build_model
from src.preprocess.cpp import compute_cpp, normalize_cpp
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
DEFAULT_SEED = 42
DEFAULT_MODEL_NAME = "best_cpp_17class_model.pth"
DEFAULT_OUTPUT_CSV = metrics_dir() / "cpp_17class_snr_accuracy.csv"
DEFAULT_OUTPUT_PNG = figures_dir() / "cpp_17class_snr_accuracy.png"
DEFAULT_OUTPUT_CM_PREFIX = "cpp_17class_snr_confusion_matrix"


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
    max_samples_per_file: int | None = None,
    max_samples: int | None = None,
    seed: int = DEFAULT_SEED,
    split_manifest: str | Path,
    predictions_csv: str | Path | None = None,
    per_file_csv: str | Path | None = None,
    exclude_labels: list[int] | tuple[int, ...] | None = None,
) -> list[dict[str, object]]:
    """Run SNR-wise CPP evaluation and save CSV/PNG outputs."""

    if not Path(split_manifest).is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {split_manifest}")
    mapping = build_class_mapping(NUM_CLASSES, exclude_labels)
    allowed_labels = set(mapping.original_labels)
    path_labels = list_mat_file_labels(data_dir)
    test_file_ids = load_test_file_ids(
        path_labels,
        manifest_path=split_manifest,
        include_labels=allowed_labels,
    )
    test_records = build_sample_index(
        data_dir,
        sample_length=sample_length,
        max_samples_per_file=max_samples_per_file,
        file_ids=test_file_ids,
    )
    test_records = [record for record in test_records if record.label in allowed_labels]
    if max_samples is not None:
        test_records = test_records[:max_samples]
    if not test_records:
        raise ValueError("No test samples available after manifest filtering")

    logger.info(
        "Loaded test split from %s: %d files, %d samples",
        split_manifest,
        len(test_file_ids),
        len(test_records),
    )

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)

    logger.info("Class mapping (model -> original): %s", mapping.original_labels)
    model = build_model(model_name, num_classes=mapping.num_classes).to(torch_device)
    logger.info("Loading model from %s", model_path)
    state_dict = load_checkpoint(model_path, map_location=torch_device)
    model.load_state_dict(state_dict)
    model.eval()

    file_cache = H5FileHandleCache()
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
                        iq = load_iq_sample(
                            file_cache.get(record.path),
                            record,
                            sample_length=sample_length,
                        )
                        noisy_iq = add_awgn_for_snr(iq, snr_db, rng)
                        noisy_iq = torch.as_tensor(noisy_iq, dtype=torch.complex64, device=device)
                        signal = noisy_iq[0]
                        cpp, _, _ = compute_cpp(
                            signal,
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
                        cpp_batch.append(normalize_cpp(cpp))
                        labels.append(mapping.to_model(record.label))

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
                            "rf_channel": int(record.rf_channel),
                            "true_label": int(label),
                            "pred_label": int(pred),
                            "original_true_label": int(mapping.to_original(label)),
                            "original_pred_label": int(mapping.to_original(pred)),
                            "correct": int(label == pred),
                        })

            accuracy = float(num_correct / num_samples) if num_samples else 0.0
            snr_name = format_snr_for_filename(float(snr_db))
            cm = confusion_matrix(all_labels, all_preds, labels=range(mapping.num_classes))
            cm_npy = metrics_dir() / f"{output_cm_prefix}_snr_{snr_name}.npy"
            cm_png = figures_dir() / f"{output_cm_prefix}_snr_{snr_name}.png"
            save_snr_confusion_matrix(
                cm,
                cm_npy,
                cm_png,
                title=f"CPP Confusion Matrix ({snr_db:g} dB)",
                class_labels=mapping.original_labels,
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
    parser.add_argument("--max-samples-per-file", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-manifest", type=str, required=True,
                        help="Existing CSV manifest generated by scripts/train.py")
    parser.add_argument("--predictions-csv", type=str, default=None,
                        help="Optional CSV path for per-sample predictions")
    parser.add_argument("--per-file-csv", type=str, default=None,
                        help="Optional CSV path for per-SNR, per-file metrics")
    parser.add_argument("--exclude-labels", type=int, nargs="*", default=None,
                        help="Original class labels excluded during training")
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.exclude_labels:
        if args.model_path == str(checkpoint_dir() / DEFAULT_MODEL_NAME):
            args.model_path = add_exclusion_suffix(args.model_path, args.exclude_labels)
        if args.output_csv == str(DEFAULT_OUTPUT_CSV):
            args.output_csv = add_exclusion_suffix(args.output_csv, args.exclude_labels)
        if args.output_png == str(DEFAULT_OUTPUT_PNG):
            args.output_png = add_exclusion_suffix(args.output_png, args.exclude_labels)
        if args.output_cm_prefix == DEFAULT_OUTPUT_CM_PREFIX:
            args.output_cm_prefix = add_exclusion_suffix(args.output_cm_prefix, args.exclude_labels)
    return args


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
        max_samples_per_file=args.max_samples_per_file,
        max_samples=args.max_samples,
        seed=args.seed,
        split_manifest=args.split_manifest,
        predictions_csv=args.predictions_csv,
        per_file_csv=args.per_file_csv,
        exclude_labels=args.exclude_labels,
    )


if __name__ == "__main__":
    main()

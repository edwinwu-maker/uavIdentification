"""Train a three-seed BYOL ensemble for direct STFT video cleaning.

Usage:
  python scripts/train_stft_byol_cleaner.py --data-dir <clean-STFT-H5> --work-dir outputs/stft_byol_cleaning --seeds 42 43 44 --input-size 512 --batch-size 8 --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.stft_byol_data import (
    ByolStftDataset, checkpoint_sha256, choose_threshold, join_reviews, metrics,
    read_csv, scan_clean_stft_h5, validate_label_counts, write_csv, write_json,
)
from src.models.stft_byol_model import StftByol
from src.training.stft_byol_training import (
    BalancedSampler, augment, ema_momentum, representation_stats, set_seed,
)
from src.utils.cli import log_current_command
from src.utils.device import default_device
from src.utils.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a direct BYOL STFT cleaner")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="outputs/stft_byol_cleaning")
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--input-size", type=int, default=512, choices=(512,))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def _load_roles(path: Path, file_counts: dict[str, int]) -> dict[str, str]:
    rows = read_csv(path)
    if len(rows) != len(file_counts) or {row["source_file"] for row in rows} != set(file_counts):
        raise ValueError("split_manifest.csv does not match input files")
    roles = {}
    for row in rows:
        if row["review_role"] not in ("train", "calibration", "audit"):
            raise ValueError(f"Invalid review role: {row['review_role']}")
        if int(row["input_row_count"]) != file_counts[row["source_file"]]:
            raise ValueError(f"Input row count changed for {row['source_file']}")
        roles[row["source_file"]] = row["review_role"]
    return roles


def _pretrain(model, loader, device, epochs):
    optimizer = AdamW(model.online_parameters(), lr=3e-4, weight_decay=1e-4)
    total_steps = max(1, epochs * len(loader))
    global_step, history = 0, []
    for epoch in range(epochs):
        model.train()
        losses, last_embeddings = [], None
        for inputs, _indices in loader:
            if len(inputs) < 2:
                continue
            inputs = inputs.to(device)
            loss, embeddings = model.byol_loss(augment(inputs), augment(inputs))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_target(ema_momentum(global_step, total_steps))
            global_step += 1
            losses.append(float(loss.detach()))
            if last_embeddings is None or len(inputs) == loader.batch_size:
                last_embeddings = embeddings
        if not losses:
            raise ValueError("BYOL produced no usable batches")
        assert last_embeddings is not None
        stats = representation_stats(last_embeddings)
        epoch_report = {
            "loss": float(np.mean(losses)),
            **stats,
        }
        history.append(epoch_report)
        logger.info("BYOL epoch %d/%d: %s", epoch + 1, epochs, epoch_report)
    final = history[-1]
    if final["embedding_std"] < 0.01 or final["effective_rank"] < 2.0:
        raise RuntimeError(f"BYOL representation collapse detected: {final}")
    return history


def _finetune(model, loader, labels_by_index, device, epochs):
    optimizer = AdamW([*model.online_encoder.parameters(), *model.classifier.parameters()],
                      lr=1e-4, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for inputs, indices in loader:
            labels = torch.tensor([labels_by_index[int(index)] for index in indices],
                                  dtype=torch.long, device=device)
            loss = F.cross_entropy(model.classify(augment(inputs.to(device))), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        logger.info("Classifier epoch %d/%d loss=%.6f", epoch + 1, epochs, history[-1])
    return history


@torch.no_grad()
def _infer(model, loader, device, count):
    model.eval()
    probabilities = np.empty(count, dtype=np.float64)
    for inputs, indices in loader:
        probability = torch.softmax(model.classify(inputs.to(device)), dim=1)[:, 1].cpu().numpy()
        probabilities[np.asarray(indices)] = probability
    return probabilities


def _labeled_arrays(rows, probabilities):
    y_true = np.asarray([row["manual_label"] == "video_present" for row in rows], dtype=np.int64)
    values = np.asarray([probabilities[int(row["sample_index"])] for row in rows])
    weights = np.asarray([float(row["sampling_weight"]) for row in rows])
    return y_true, values, weights


def main() -> None:
    args = parse_args()
    log_current_command(logger)
    if args.batch_size < 2 or args.num_workers < 0:
        raise ValueError("batch-size must be >= 2 and num-workers must be non-negative")
    if args.pretrain_epochs <= 0 or args.finetune_epochs <= 0:
        raise ValueError("Epoch counts must be positive")
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise ValueError("Exactly three distinct seeds are required")
    device, work_dir = torch.device(args.device), Path(args.work_dir).expanduser()
    required = [work_dir / name for name in ("review.csv", "review_lookup.csv", "split_manifest.csv")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required review artifact does not exist: {path}")
    samples, file_counts = scan_clean_stft_h5(args.data_dir)
    roles = _load_roles(work_dir / "split_manifest.csv", file_counts)
    reviews = join_reviews(work_dir / "review.csv", work_dir / "review_lookup.csv", samples)
    for row in reviews:
        if roles[row["source_file"]] != row["review_role"]:
            raise ValueError(f"Split role mismatch for {row['review_id']}")
    validate_label_counts(reviews)
    train_pool = [index for index, sample in enumerate(samples) if roles[sample.source_file] == "train"]
    train_reviews = [row for row in reviews
                     if row["review_role"] == "train" and row["manual_label"] != "uncertain"]
    train_indices = [int(row["sample_index"]) for row in train_reviews]
    labels_by_index = {int(row["sample_index"]): int(row["manual_label"] == "video_present")
                       for row in train_reviews}
    train_labels = [labels_by_index[index] for index in train_indices]
    calibration = [row for row in reviews
                   if row["review_role"] == "calibration" and row["manual_label"] != "uncertain"]
    audit = [row for row in reviews
             if row["review_role"] == "audit" and row["manual_label"] != "uncertain"]
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    all_probabilities, checkpoints, seed_reports = [], [], {}

    for seed in args.seeds:
        set_seed(seed)
        pretrain_data = ByolStftDataset(samples, train_pool, args.input_size)
        fine_data = ByolStftDataset(samples, train_indices, args.input_size)
        inference_data = ByolStftDataset(samples, input_size=args.input_size)
        pretrain_loader = DataLoader(pretrain_data, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
        fine_loader = DataLoader(fine_data, batch_size=args.batch_size,
                                 sampler=BalancedSampler(train_labels, seed), **loader_kwargs)
        inference_loader = DataLoader(inference_data, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        try:
            model = StftByol().to(device)
            pretrain_history = _pretrain(model, pretrain_loader, device, args.pretrain_epochs)
            finetune_history = _finetune(model, fine_loader, labels_by_index, device, args.finetune_epochs)
            probabilities = _infer(model, inference_loader, device, len(samples))
        finally:
            pretrain_data.close(); fine_data.close(); inference_data.close()
        calibration_true, calibration_prob, calibration_weights = _labeled_arrays(
            calibration, probabilities,
        )
        seed_threshold = choose_threshold(
            calibration_true, calibration_prob, sample_weight=calibration_weights,
        )
        audit_true, audit_prob, audit_weights = _labeled_arrays(audit, probabilities)
        seed_reports[str(seed)] = {
            "threshold": seed_threshold,
            "audit_metrics": metrics(audit_true, audit_prob >= seed_threshold),
            "weighted_audit_metrics": metrics(audit_true, audit_prob >= seed_threshold, audit_weights),
            "pretrain_history": pretrain_history,
            "finetune_history": finetune_history,
        }
        checkpoint = work_dir / f"stft_byol_seed{seed}.pt"
        torch.save({"model_state_dict": model.state_dict(), "seed": seed, "input_size": 512,
                    "positive_class": "video_present"}, checkpoint)
        checkpoints.append({"seed": seed, "path": checkpoint.name, "sha256": checkpoint_sha256(checkpoint)})
        all_probabilities.append(probabilities)

    ensemble = np.mean(np.stack(all_probabilities), axis=0)
    calibration_true, calibration_prob, calibration_weights = _labeled_arrays(calibration, ensemble)
    threshold = choose_threshold(
        calibration_true, calibration_prob, sample_weight=calibration_weights,
    )
    calibration_metrics = metrics(calibration_true, calibration_prob >= threshold)
    weighted_calibration = metrics(
        calibration_true, calibration_prob >= threshold, calibration_weights,
    )
    audit_true, audit_prob, audit_weights = _labeled_arrays(audit, ensemble)
    audit_metrics = metrics(audit_true, audit_prob >= threshold)
    weighted_audit = metrics(audit_true, audit_prob >= threshold, audit_weights)
    metric_names = ("precision", "recall", "f1", "no_video_removal_rate")
    seed_metric_summary = {
        name: {
            "mean": float(np.mean([seed_reports[str(seed)]["audit_metrics"][name] for seed in args.seeds])),
            "std": float(np.std([seed_reports[str(seed)]["audit_metrics"][name] for seed in args.seeds])),
        }
        for name in metric_names
    }
    manual = {int(row["sample_index"]): row["manual_label"] for row in reviews}
    prediction_rows = []
    for index, sample in enumerate(samples):
        row = {"sample_index": index, "source_file": sample.source_file,
               "source_row_idx": sample.row_idx, "original_label": sample.label,
               "rf_channel": sample.rf_channel, "source_sample_idx": sample.source_sample_idx,
               "review_role": roles[sample.source_file], "manual_label": manual.get(index, "")}
        row.update({f"probability_seed_{seed}": float(values[index])
                    for seed, values in zip(args.seeds, all_probabilities)})
        row["ensemble_probability"] = float(ensemble[index])
        row["decision"] = "video_present" if ensemble[index] >= threshold else "no_video"
        prediction_rows.append(row)
    write_csv(work_dir / "byol_predictions.csv", prediction_rows)
    report = {
        "input_size": 512, "seeds": args.seeds, "threshold": threshold,
        "minimum_calibration_recall": 0.95, "positive_definition": "video_present; video+WiFi is positive",
        "retention_rate": float(np.mean(ensemble >= threshold)), "checkpoints": checkpoints,
        "ensemble_calibration_metrics": calibration_metrics,
        "ensemble_weighted_calibration_metrics": weighted_calibration,
        "ensemble_audit_metrics": audit_metrics,
        "ensemble_weighted_audit_metrics": weighted_audit,
        "seed_reports": seed_reports, "seed_metric_summary": seed_metric_summary,
    }
    write_json(work_dir / "byol_cleaning_report.json", report)
    logger.info("BYOL ensemble threshold %.8f; audit=%s", threshold, audit_metrics)


if __name__ == "__main__":
    main()

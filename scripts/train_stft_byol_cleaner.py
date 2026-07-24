"""Train a three-seed BYOL ensemble on DCEC background-filtered STFT data.

Usage:
  python scripts/train_stft_byol_cleaner.py --data-dir <DEC-filtered-STFT-H5> --work-dir outputs/stft_byol_cleaning --seeds 42 43 44 --batch-size 8 --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.stft_byol_data import (
    BYOL_INPUT_SIZE, ROLES, ByolStftDataset, checkpoint_sha256, choose_threshold,
    join_reviews, load_block_roles, metrics, metrics_by_group, sample_block_id,
    scan_clean_stft_h5, summarize_split_coverage, validate_label_counts,
    write_csv, write_json,
)
from src.models.stft_byol_model import StftByol
from src.training.stft_byol_training import (
    BalancedSampler, augment, ema_momentum, representation_stats, set_seed,
)
from src.utils.cli import log_current_command
from src.utils.device import default_device
from src.utils.logger import logger

REPRESENTATION_SAMPLE_LIMIT = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a post-DEC BYOL STFT cleaner")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="outputs/stft_byol_cleaning")
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def _batch_progress(loader, description):
    return tqdm(
        loader,
        total=len(loader),
        desc=description,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=None,
    )


def _format_metrics(values):
    names = ("precision", "recall", "f1", "no_video_removal_rate")
    return " | ".join(
        f"{name}={values[name]:.4f}" if values[name] is not None else f"{name}=n/a"
        for name in names
    )


def _log_dataset_summary(samples, roles, reviews):
    source_files = {sample.source_file for sample in samples}
    block_counts = Counter(roles.values())
    sample_counts = Counter(roles[sample_block_id(sample)] for sample in samples)
    coverage = summarize_split_coverage(samples, roles)
    logger.info(
        "Dataset | source_files=%d | blocks=%d | samples=%d",
        len(source_files), len(roles), len(samples),
    )
    logger.info(
        "Split | %s",
        " | ".join(
            f"{role}: blocks={block_counts[role]}, samples={sample_counts[role]}"
            for role in ROLES
        ),
    )
    for role in ROLES:
        labels = Counter(
            row["manual_label"] for row in reviews if row["review_role"] == role
        )
        logger.info(
            "Reviews %s | video_present=%d | no_video=%d | uncertain=%d",
            role,
            labels["video_present"],
            labels["no_video"],
            labels["uncertain"],
        )
    sparse_count = int(coverage["train_only_sparse_source_file_count"])
    if sparse_count:
        logger.warning(
            "Split coverage | train_only_sparse_files=%d; "
            "calibration/audit metrics do not cover these source files",
            sparse_count,
        )
    return coverage


def _reuse_checkpoint(model, path, *, seed, device):
    if not path.is_file():
        return False
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    expected = {
        "seed": seed,
        "input_size": BYOL_INPUT_SIZE,
        "positive_class": "video_present",
        "training_stage": "post_dec_video_filter",
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise ValueError(
                f"Checkpoint {name} mismatch for {path}: "
                f"expected {value!r}, got {checkpoint.get(name)!r}"
            )
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint model_state_dict is missing: {path}")
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint model weights are incompatible: {path}") from exc
    return True


def _collapse_signals(stats):
    signals = []
    if stats["embedding_std"] < 0.01:
        signals.append("embedding_std")
    if stats["mean_cosine_similarity"] > 0.99:
        signals.append("mean_cosine_similarity")
    if stats["effective_rank"] < 2.0:
        signals.append("effective_rank")
    return signals


def _pretrain(model, loader, device, epochs, *, seed):
    optimizer = AdamW(model.online_parameters(), lr=3e-4, weight_decay=1e-4)
    total_steps = max(1, epochs * len(loader))
    global_step, history, collapse_streak = 0, [], 0
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        model.train()
        losses, loss_sum, diagnostic_features = [], 0.0, []
        diagnostic_count = 0
        description = f"Seed {seed} | BYOL {epoch + 1}/{epochs}"
        with _batch_progress(loader, description) as progress:
            for inputs, _indices in progress:
                if len(inputs) < 2:
                    continue
                inputs = inputs.to(device)
                loss, embeddings = model.byol_loss(augment(inputs), augment(inputs))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                momentum = ema_momentum(global_step, total_steps)
                model.update_target(momentum)
                global_step += 1
                loss_value = float(loss.detach())
                losses.append(loss_value)
                loss_sum += loss_value
                if diagnostic_count < REPRESENTATION_SAMPLE_LIMIT:
                    remaining = REPRESENTATION_SAMPLE_LIMIT - diagnostic_count
                    selected = embeddings[:remaining].detach().cpu()
                    diagnostic_features.append(selected)
                    diagnostic_count += len(selected)
                progress.set_postfix(
                    loss=f"{loss_sum / len(losses):.6f}",
                    ema=f"{momentum:.6f}",
                    refresh=False,
                )
        if not losses:
            raise ValueError("BYOL produced no usable batches")
        stats = representation_stats(torch.cat(diagnostic_features))
        epoch_report = {
            "loss": float(np.mean(losses)),
            **stats,
        }
        history.append(epoch_report)
        logger.info(
            "Seed %d | BYOL epoch %d/%d | loss=%.6f | embedding_std=%.6f | "
            "mean_cosine_similarity=%.6f | effective_rank=%.3f | elapsed=%.1fs",
            seed, epoch + 1, epochs, epoch_report["loss"],
            epoch_report["embedding_std"], epoch_report["mean_cosine_similarity"],
            epoch_report["effective_rank"], time.perf_counter() - epoch_started,
        )
        signals = _collapse_signals(epoch_report)
        if signals:
            logger.warning(
                "Seed %d | BYOL epoch %d/%d | collapse warning: %s",
                seed, epoch + 1, epochs, ", ".join(signals),
            )
        collapse_streak = collapse_streak + 1 if len(signals) >= 2 else 0
        if collapse_streak >= 3:
            raise RuntimeError(
                "BYOL representation collapse detected for three consecutive epochs: "
                f"{epoch_report}"
            )
    return history


def _finetune(model, loader, labels_by_index, device, epochs, *, seed):
    optimizer = AdamW([*model.online_encoder.parameters(), *model.classifier.parameters()],
                      lr=1e-4, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        model.train()
        losses, loss_sum = [], 0.0
        description = f"Seed {seed} | classifier {epoch + 1}/{epochs}"
        with _batch_progress(loader, description) as progress:
            for inputs, indices in progress:
                labels = torch.tensor([labels_by_index[int(index)] for index in indices],
                                      dtype=torch.long, device=device)
                loss = F.cross_entropy(model.classify(augment(inputs.to(device))), labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach())
                losses.append(loss_value)
                loss_sum += loss_value
                progress.set_postfix(
                    loss=f"{loss_sum / len(losses):.6f}",
                    refresh=False,
                )
        history.append(float(np.mean(losses)))
        logger.info(
            "Seed %d | classifier epoch %d/%d | loss=%.6f | elapsed=%.1fs",
            seed, epoch + 1, epochs, history[-1], time.perf_counter() - epoch_started,
        )
    return history


@torch.no_grad()
def _infer(model, loader, device, count, *, seed):
    model.eval()
    probabilities = np.empty(count, dtype=np.float64)
    processed = 0
    with _batch_progress(loader, f"Seed {seed} | inference") as progress:
        for inputs, indices in progress:
            probability = torch.softmax(model.classify(inputs.to(device)), dim=1)[:, 1].cpu().numpy()
            probabilities[np.asarray(indices)] = probability
            processed += len(indices)
            progress.set_postfix(samples=f"{processed}/{count}", refresh=False)
    return probabilities


def _labeled_arrays(rows, probabilities):
    y_true = np.asarray([row["manual_label"] == "video_present" for row in rows], dtype=np.int64)
    values = np.asarray([probabilities[int(row["sample_index"])] for row in rows])
    weights = np.asarray([float(row["sampling_weight"]) for row in rows])
    return y_true, values, weights


def _review_groups(rows):
    return [
        (int(row["original_label"]), int(row["rf_channel"]))
        for row in rows
    ]


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    log_current_command(logger)
    if args.batch_size < 2 or args.num_workers < 0:
        raise ValueError("batch-size must be >= 2 and num-workers must be non-negative")
    if args.pretrain_epochs <= 0 or args.finetune_epochs <= 0:
        raise ValueError("Epoch counts must be positive")
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise ValueError("Exactly three distinct seeds are required")
    device, work_dir = torch.device(args.device), Path(args.work_dir).expanduser()
    logger.info(
        "Training config | device=%s | seeds=%s | input_size=%d | batch_size=%d | "
        "num_workers=%d | pretrain_epochs=%d | finetune_epochs=%d",
        device, list(args.seeds), BYOL_INPUT_SIZE, args.batch_size,
        args.num_workers, args.pretrain_epochs, args.finetune_epochs,
    )
    logger.info(
        "Paths | data_dir=%s | work_dir=%s",
        Path(args.data_dir).expanduser(), work_dir,
    )
    required = [work_dir / name for name in ("review.csv", "review_lookup.csv", "split_manifest.csv")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required review artifact does not exist: {path}")
    logger.info("Loading and validating STFT samples and review artifacts")
    samples = scan_clean_stft_h5(args.data_dir)
    roles = load_block_roles(work_dir / "split_manifest.csv", samples)
    reviews = join_reviews(work_dir / "review.csv", work_dir / "review_lookup.csv", samples)
    for row in reviews:
        block_id = (row["source_file"], int(row["source_sample_idx"]))
        if roles[block_id] != row["review_role"]:
            raise ValueError(f"Split role mismatch for {row['review_id']}")
    validate_label_counts(reviews)
    split_coverage = _log_dataset_summary(samples, roles, reviews)
    train_pool = [
        index for index, sample in enumerate(samples)
        if roles[sample_block_id(sample)] == "train"
    ]
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

    for seed_index, seed in enumerate(args.seeds, 1):
        seed_started = time.perf_counter()
        set_seed(seed)
        checkpoint = work_dir / f"stft_byol_seed{seed}.pt"
        pretrain_data = ByolStftDataset(samples, train_pool)
        fine_data = ByolStftDataset(samples, train_indices)
        inference_data = ByolStftDataset(samples)
        pretrain_loader = DataLoader(pretrain_data, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
        fine_loader = DataLoader(fine_data, batch_size=args.batch_size,
                                 sampler=BalancedSampler(train_labels, seed), **loader_kwargs)
        inference_loader = DataLoader(inference_data, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        logger.info(
            "Seed %d (%d/%d) started | pretrain_samples=%d, batches=%d | "
            "finetune_samples=%d, balanced_batches=%d | inference_samples=%d, batches=%d",
            seed, seed_index, len(args.seeds),
            len(pretrain_data), len(pretrain_loader),
            len(fine_data), len(fine_loader),
            len(inference_data), len(inference_loader),
        )
        try:
            model = StftByol().to(device)
            checkpoint_reused = _reuse_checkpoint(
                model, checkpoint, seed=seed, device=device,
            )
            if checkpoint_reused:
                pretrain_history, finetune_history = [], []
                logger.info(
                    "Seed %d | checkpoint reused; pretraining and fine-tuning skipped | path=%s",
                    seed, checkpoint,
                )
            else:
                logger.info("Seed %d | checkpoint not found; training from scratch", seed)
                phase_started = time.perf_counter()
                logger.info("Seed %d | BYOL pretraining started", seed)
                pretrain_history = _pretrain(
                    model, pretrain_loader, device, args.pretrain_epochs, seed=seed,
                )
                logger.info(
                    "Seed %d | BYOL pretraining completed | final_loss=%.6f | elapsed=%.1fs",
                    seed, pretrain_history[-1]["loss"], time.perf_counter() - phase_started,
                )

                phase_started = time.perf_counter()
                logger.info("Seed %d | classifier fine-tuning started", seed)
                finetune_history = _finetune(
                    model, fine_loader, labels_by_index, device, args.finetune_epochs,
                    seed=seed,
                )
                logger.info(
                    "Seed %d | classifier fine-tuning completed | final_loss=%.6f | elapsed=%.1fs",
                    seed, finetune_history[-1], time.perf_counter() - phase_started,
                )

            phase_started = time.perf_counter()
            logger.info("Seed %d | full-dataset inference started", seed)
            probabilities = _infer(
                model, inference_loader, device, len(samples), seed=seed,
            )
            logger.info(
                "Seed %d | full-dataset inference completed | elapsed=%.1fs",
                seed, time.perf_counter() - phase_started,
            )
        finally:
            pretrain_data.close(); fine_data.close(); inference_data.close()
        calibration_true, calibration_prob, calibration_weights = _labeled_arrays(
            calibration, probabilities,
        )
        seed_threshold = choose_threshold(
            calibration_true,
            calibration_prob,
            sample_weight=calibration_weights,
            groups=_review_groups(calibration),
        )
        audit_true, audit_prob, audit_weights = _labeled_arrays(audit, probabilities)
        seed_audit_metrics = metrics(audit_true, audit_prob >= seed_threshold)
        seed_weighted_audit = metrics(
            audit_true, audit_prob >= seed_threshold, audit_weights,
        )
        seed_reports[str(seed)] = {
            "threshold": seed_threshold,
            "audit_metrics": seed_audit_metrics,
            "weighted_audit_metrics": seed_weighted_audit,
            "pretrain_history": pretrain_history,
            "finetune_history": finetune_history,
            "checkpoint_reused": checkpoint_reused,
        }
        if not checkpoint_reused:
            torch.save({"model_state_dict": model.state_dict(), "seed": seed,
                        "input_size": BYOL_INPUT_SIZE,
                        "positive_class": "video_present",
                        "training_stage": "post_dec_video_filter"}, checkpoint)
        checkpoint_hash = checkpoint_sha256(checkpoint)
        checkpoints.append({"seed": seed, "path": checkpoint.name, "sha256": checkpoint_hash})
        all_probabilities.append(probabilities)
        logger.info(
            "Seed %d | threshold=%.8f | audit: %s",
            seed, seed_threshold, _format_metrics(seed_audit_metrics),
        )
        logger.info(
            "Seed %d | weighted audit: %s",
            seed, _format_metrics(seed_weighted_audit),
        )
        logger.info(
            "Seed %d (%d/%d) completed | checkpoint=%s | sha256=%s | elapsed=%.1fs",
            seed, seed_index, len(args.seeds), checkpoint, checkpoint_hash,
            time.perf_counter() - seed_started,
        )

    stacked_probabilities = np.stack(all_probabilities)
    mean_probabilities = np.mean(stacked_probabilities, axis=0)
    ensemble = np.median(stacked_probabilities, axis=0)
    calibration_true, calibration_prob, calibration_weights = _labeled_arrays(calibration, ensemble)
    threshold = choose_threshold(
        calibration_true,
        calibration_prob,
        sample_weight=calibration_weights,
        groups=_review_groups(calibration),
    )
    calibration_metrics = metrics(calibration_true, calibration_prob >= threshold)
    weighted_calibration = metrics(
        calibration_true, calibration_prob >= threshold, calibration_weights,
    )
    audit_true, audit_prob, audit_weights = _labeled_arrays(audit, ensemble)
    audit_metrics = metrics(audit_true, audit_prob >= threshold)
    weighted_audit = metrics(audit_true, audit_prob >= threshold, audit_weights)
    audit_group_metrics = metrics_by_group(
        audit_true,
        audit_prob >= threshold,
        _review_groups(audit),
    )
    weighted_audit_group_metrics = metrics_by_group(
        audit_true,
        audit_prob >= threshold,
        _review_groups(audit),
        audit_weights,
    )
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
               "dec_source_row_idx": sample.dec_source_row_idx,
               "review_role": roles[sample_block_id(sample)], "manual_label": manual.get(index, "")}
        row.update({f"probability_seed_{seed}": float(values[index])
                    for seed, values in zip(args.seeds, all_probabilities)})
        row["mean_probability"] = float(mean_probabilities[index])
        row["ensemble_probability"] = float(ensemble[index])
        row["decision"] = "video_present" if ensemble[index] >= threshold else "no_video"
        prediction_rows.append(row)
    prediction_path = work_dir / "byol_predictions.csv"
    report_path = work_dir / "byol_cleaning_report.json"
    write_csv(prediction_path, prediction_rows)
    report = {
        "input_size": BYOL_INPUT_SIZE, "seeds": args.seeds, "threshold": threshold,
        "ensemble_method": "median",
        "minimum_calibration_recall": 0.98,
        "positive_definition": "video_present",
        "retention_rate": float(np.mean(ensemble >= threshold)), "checkpoints": checkpoints,
        "ensemble_calibration_metrics": calibration_metrics,
        "ensemble_weighted_calibration_metrics": weighted_calibration,
        "ensemble_audit_metrics": audit_metrics,
        "ensemble_weighted_audit_metrics": weighted_audit,
        "ensemble_audit_group_metrics": audit_group_metrics,
        "ensemble_weighted_audit_group_metrics": weighted_audit_group_metrics,
        "split_coverage": split_coverage,
        "seed_reports": seed_reports, "seed_metric_summary": seed_metric_summary,
    }
    write_json(report_path, report)
    logger.info(
        "Ensemble | threshold=%.8f | retention_rate=%.4f",
        threshold, report["retention_rate"],
    )
    logger.info("Ensemble calibration | %s", _format_metrics(calibration_metrics))
    logger.info("Ensemble weighted calibration | %s", _format_metrics(weighted_calibration))
    logger.info("Ensemble audit | %s", _format_metrics(audit_metrics))
    logger.info("Ensemble weighted audit | %s", _format_metrics(weighted_audit))
    logger.info(
        "Training completed | predictions=%s | report=%s | checkpoints=%s | elapsed=%.1fs",
        prediction_path, report_path, work_dir, time.perf_counter() - run_started,
    )


if __name__ == "__main__":
    main()

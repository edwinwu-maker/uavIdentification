"""Train DCEC on clean STFT H5 files and create a blinded review set.

Usage:
  python scripts/train_stft_deep_cluster.py --data-dir ... --work-dir outputs/stft_deep_cluster
  python scripts/train_stft_deep_cluster.py --data-dir ... --work-dir ... --device cuda:0 --batch-size 32
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from src.data.stft_clustering import (
    CleanStftClusteringDataset,
    StftClusterSample,
    scan_clean_stft_h5,
    source_filename,
)
from src.models.stft_deep_cluster import StftAutoencoder, StftDcec
from src.training.stft_deep_cluster import (
    encode_dataset,
    infer_dcec,
    pretrain_autoencoder,
    select_cluster_count,
    set_random_seed,
    train_dcec,
)
from src.utils.cli import log_current_command
from src.utils.device import default_device
from src.utils.logger import logger

SEED = 42
REVIEW_ROLES = (("mapping", 150), ("calibration", 50), ("audit", 50))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCEC and create blinded STFT review samples")
    parser.add_argument("--data-dir", required=True, help="Directory containing clean STFT .h5 files")
    parser.add_argument("--work-dir", default="outputs/stft_deep_cluster")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--dcec-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def _allocation(total: int, cluster_count: int) -> list[int]:
    base, remainder = divmod(total, cluster_count)
    return [base + (cluster_id < remainder) for cluster_id in range(cluster_count)]


def _take_diverse(
    ordered_indices: list[int],
    count: int,
    samples: list[StftClusterSample],
    used_indices: set[int],
    *,
    forbidden_paths: set[str] | None = None,
) -> list[int]:
    if count == 0:
        return []
    forbidden_paths = forbidden_paths or set()
    candidates = [
        index for index in ordered_indices
        if index not in used_indices and samples[index].path not in forbidden_paths
    ]
    selected: list[int] = []
    selected_paths: set[str] = set()
    for index in candidates:
        path = samples[index].path
        if path not in selected_paths:
            selected.append(index)
            selected_paths.add(path)
            if len(selected) == count:
                return selected
    for index in candidates:
        if index not in selected:
            selected.append(index)
            if len(selected) == count:
                return selected
    raise ValueError(f"Not enough distinct review candidates: requested {count}, found {len(selected)}")


def _spread_order(indices: np.ndarray, values: np.ndarray, front_count: int) -> list[int]:
    ordered = indices[np.argsort(values[indices])]
    if len(ordered) <= 1 or front_count <= 0:
        return ordered.tolist()
    positions = np.linspace(
        0, len(ordered) - 1, min(front_count, len(ordered))
    ).round().astype(int)
    front = ordered[np.unique(positions)].tolist()
    front_set = set(front)
    return [*front, *(index for index in ordered.tolist() if index not in front_set)]


def select_review_indices(
    samples: list[StftClusterSample],
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    centers: np.ndarray,
) -> list[tuple[str, int]]:
    """Select mapping/calibration/audit rows; audit source files stay disjoint."""

    required_count = sum(count for _role, count in REVIEW_ROLES)
    if len(samples) < required_count:
        raise ValueError(f"At least {required_count} samples are required for blinded review")
    labels = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    cluster_count = probabilities.shape[1]
    used_indices: set[int] = set()
    selected_by_role: dict[str, list[int]] = {role: [] for role, _count in REVIEW_ROLES}

    # 先预留 audit 文件，保证语义映射/阈值校准不会接触这些文件的人工标签。
    audit_alloc = _allocation(dict(REVIEW_ROLES)["audit"], cluster_count)
    for cluster_id, count in enumerate(audit_alloc):
        cluster_indices = np.flatnonzero(labels == cluster_id)
        indices_by_path: dict[str, list[int]] = {}
        for index in cluster_indices:
            indices_by_path.setdefault(samples[int(index)].path, []).append(int(index))
        if len(indices_by_path) < 2:
            raise ValueError(
                f"Cluster {cluster_id} must span at least two source files for file-disjoint audit"
            )
        path_groups = sorted(indices_by_path.values(), key=lambda group: (-len(group), samples[group[0]].path))
        reserved: list[int] = []
        for group in path_groups[:-1]:
            reserved.extend(group)
            if len(reserved) >= count:
                break
        if len(reserved) < count:
            raise ValueError(f"Cluster {cluster_id} cannot provide {count} file-disjoint audit rows")
        order = _spread_order(np.asarray(reserved, dtype=np.int64), confidences, count)
        chosen = order[:count]
        selected_by_role["audit"].extend(chosen)
        used_indices.update(chosen)
    audit_paths = {samples[index].path for index in selected_by_role["audit"]}

    mapping_alloc = _allocation(dict(REVIEW_ROLES)["mapping"], cluster_count)
    for cluster_id, count in enumerate(mapping_alloc):
        cluster_indices = np.flatnonzero(labels == cluster_id)
        distances = np.linalg.norm(embeddings - centers[cluster_id], axis=1)
        ordered = cluster_indices[np.argsort(distances[cluster_indices])]
        central_count = (count + 1) // 2
        central = _take_diverse(
            ordered.tolist(), central_count, samples, used_indices, forbidden_paths=audit_paths
        )
        used_indices.update(central)
        peripheral_order = ordered[::-1].tolist()
        peripheral = _take_diverse(
            peripheral_order,
            count - central_count,
            samples,
            used_indices,
            forbidden_paths=audit_paths,
        )
        used_indices.update(peripheral)
        selected_by_role["mapping"].extend([*central, *peripheral])

    calibration_count = dict(REVIEW_ROLES)["calibration"]
    available = np.array(
        [
            index for index in range(len(samples))
            if index not in used_indices and samples[index].path not in audit_paths
        ],
        dtype=np.int64,
    )
    calibration_order = _spread_order(available, confidences, calibration_count)
    calibration = _take_diverse(calibration_order, calibration_count, samples, used_indices)
    selected_by_role["calibration"] = calibration

    return [
        (role, index)
        for role, _count in REVIEW_ROLES
        for index in selected_by_role[role]
    ]


def write_assignments(
    path: Path,
    samples: list[StftClusterSample],
    probabilities: np.ndarray,
) -> None:
    probability_fields = [f"q_{cluster_id}" for cluster_id in range(probabilities.shape[1])]
    fieldnames = [
        "sample_index", "source_file", "source_row_idx", "original_label",
        "rf_channel", "source_sample_idx", "cluster_id", "cluster_confidence",
        *probability_fields,
    ]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample in enumerate(samples):
            row = {
                "sample_index": index,
                "source_file": source_filename(sample),
                "source_row_idx": sample.row_idx,
                "original_label": sample.label,
                "rf_channel": sample.rf_channel,
                "source_sample_idx": sample.source_sample_idx,
                "cluster_id": int(probabilities[index].argmax()),
                "cluster_confidence": float(probabilities[index].max()),
            }
            row.update({name: float(value) for name, value in zip(probability_fields, probabilities[index])})
            writer.writerow(row)


def write_review_artifacts(
    work_dir: Path,
    selections: list[tuple[str, int]],
    samples: list[StftClusterSample],
    probabilities: np.ndarray,
) -> None:
    image_dir = work_dir / "review_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    review_path = work_dir / "review.csv"
    lookup_path = work_dir / "review_lookup.csv"
    handles: dict[str, h5py.File] = {}
    try:
        with review_path.open("w", newline="", encoding="utf-8") as review_file, lookup_path.open(
            "w", newline="", encoding="utf-8"
        ) as lookup_file:
            review_writer = csv.DictWriter(
                review_file,
                fieldnames=("review_id", "review_role", "image_path", "manual_label"),
            )
            lookup_writer = csv.DictWriter(
                lookup_file,
                fieldnames=(
                    "review_id", "review_role", "sample_index", "source_file",
                    "source_row_idx", "cluster_id",
                ),
            )
            review_writer.writeheader()
            lookup_writer.writeheader()
            for order, (role, sample_index) in enumerate(selections, start=1):
                review_id = f"R{order:04d}"
                sample = samples[sample_index]
                if sample.path not in handles:
                    handles[sample.path] = h5py.File(sample.path, "r")
                stft = handles[sample.path]["stft"][sample.row_idx, 0]
                relative_image = Path("review_images") / f"{review_id}.png"
                fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
                image = ax.imshow(
                    stft.T,
                    origin="lower",
                    aspect="auto",
                    cmap="jet",
                    vmin=-3.0,
                    vmax=3.0,
                )
                ax.set_title(review_id)
                ax.set_xlabel("Frequency bin")
                ax.set_ylabel("Time bin")
                fig.colorbar(image, ax=ax, label="Normalized power (z-score)")
                fig.savefig(work_dir / relative_image, dpi=120)
                plt.close(fig)
                review_writer.writerow({
                    "review_id": review_id,
                    "review_role": role,
                    "image_path": relative_image.as_posix(),
                    "manual_label": "",
                })
                lookup_writer.writerow({
                    "review_id": review_id,
                    "review_role": role,
                    "sample_index": sample_index,
                    "source_file": source_filename(sample),
                    "source_row_idx": sample.row_idx,
                    "cluster_id": int(probabilities[sample_index].argmax()),
                })
    finally:
        for h5f in handles.values():
            h5f.close()


def main() -> None:
    args = parse_args()
    log_current_command(logger)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers must be non-negative")
    if args.pretrain_epochs <= 0 or args.dcec_epochs <= 0 or args.patience <= 0:
        raise ValueError("epoch and patience values must be positive")
    set_random_seed(SEED)
    device = torch.device(args.device)
    work_dir = Path(args.work_dir).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)

    samples = scan_clean_stft_h5(args.data_dir)
    logger.info("Loaded %d clean UAV STFT samples", len(samples))
    dataset = CleanStftClusteringDataset(samples)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(dataset, shuffle=True, **loader_kwargs)
    inference_loader = DataLoader(dataset, shuffle=False, **loader_kwargs)
    try:
        autoencoder = StftAutoencoder()
        pretrain_history = pretrain_autoencoder(
            autoencoder,
            train_loader,
            device=device,
            epochs=args.pretrain_epochs,
            learning_rate=1e-3,
            patience=args.patience,
            logger=logger,
        )
        embeddings = encode_dataset(autoencoder, inference_loader, device=device)
        selected_k, candidate_reports = select_cluster_count(embeddings, seed=SEED)
        logger.info("Selected K=%d", selected_k)
        initial_kmeans = KMeans(n_clusters=selected_k, n_init=20, random_state=SEED).fit(embeddings)
        model = StftDcec(autoencoder, selected_k).to(device)
        with torch.no_grad():
            model.clustering.centers.copy_(
                torch.from_numpy(initial_kmeans.cluster_centers_).to(device=device, dtype=torch.float32)
            )
        dcec_history = train_dcec(
            model,
            train_loader,
            device=device,
            epochs=args.dcec_epochs,
            learning_rate=1e-4,
            logger=logger,
        )
        final_embeddings, probabilities = infer_dcec(model, inference_loader, device=device)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "selected_k": selected_k,
            "latent_dim": autoencoder.latent_dim,
            "input_size": 256,
            "clip_range": (-5.0, 5.0),
            "scale_divisor": 5.0,
            "seed": SEED,
        }
        torch.save(checkpoint, work_dir / "stft_dcec.pt")
        write_assignments(work_dir / "assignments.csv", samples, probabilities)
        selections = select_review_indices(
            samples,
            final_embeddings,
            probabilities,
            model.clustering.centers.detach().cpu().numpy(),
        )
        write_review_artifacts(work_dir, selections, samples, probabilities)
        report = {
            "sample_count": len(samples),
            "selected_k": selected_k,
            "final_cluster_counts": np.bincount(
                probabilities.argmax(axis=1), minlength=selected_k
            ).tolist(),
            "cluster_candidates": [asdict(item) for item in candidate_reports],
            "pretrain_history": pretrain_history,
            "dcec_history": dcec_history,
            "review_counts": dict(REVIEW_ROLES),
        }
        with (work_dir / "training_report.json").open("w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)
        logger.info("Checkpoint, assignments, and blinded review set saved to %s", work_dir)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()

"""Training helpers for STFT convolutional deep embedded clustering."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.stft_deep_cluster import StftAutoencoder, StftDcec, target_distribution


@dataclass(frozen=True)
class ClusterCandidate:
    num_clusters: int
    silhouette: float
    stability_ari: float
    min_cluster_share: float
    score: float
    accepted: bool


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretrain_autoencoder(
    model: StftAutoencoder,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
    logger=None,
) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.to(device)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for inputs, _indices in tqdm(loader, desc=f"CAE epoch {epoch}", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            optimizer.zero_grad()
            reconstruction, _latent = model(inputs)
            loss = F.smooth_l1_loss(reconstruction, inputs)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * inputs.shape[0]
            sample_count += inputs.shape[0]
        epoch_loss = loss_sum / sample_count
        history.append(epoch_loss)
        if logger is not None:
            logger.info("CAE epoch %d | reconstruction_loss: %.6f", epoch, epoch_loss)
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def encode_dataset(
    model: StftAutoencoder,
    loader: DataLoader,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    embeddings = np.empty((len(loader.dataset), model.latent_dim), dtype=np.float32)
    for inputs, indices in tqdm(loader, desc="Encoding STFT", leave=False):
        latent = model.encode(inputs.to(device, non_blocking=True)).cpu().numpy()
        embeddings[indices.numpy()] = latent
    return embeddings


def select_cluster_count(
    embeddings: np.ndarray,
    *,
    candidates: tuple[int, ...] = (4, 6, 8, 10),
    sample_size: int = 5000,
    seed: int = 42,
) -> tuple[int, list[ClusterCandidate]]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if len(embeddings) < min(candidates):
        raise ValueError("Not enough samples for the requested cluster candidates")

    generator = np.random.default_rng(seed)
    selected_indices = generator.choice(
        len(embeddings), size=min(sample_size, len(embeddings)), replace=False
    )
    selected = embeddings[selected_indices]
    reports: list[ClusterCandidate] = []
    seeds = (seed, seed + 1, seed + 2)
    for num_clusters in candidates:
        if len(selected) <= num_clusters:
            continue
        label_runs = [
            KMeans(n_clusters=num_clusters, n_init=10, random_state=current_seed).fit_predict(selected)
            for current_seed in seeds
        ]
        min_share = min(
            float(np.bincount(labels, minlength=num_clusters).min() / len(selected))
            for labels in label_runs
        )
        silhouette = float(
            silhouette_score(
                selected,
                label_runs[0],
                sample_size=min(2000, len(selected)),
                random_state=seed,
            )
        )
        pairwise_ari = [
            adjusted_rand_score(label_runs[left], label_runs[right])
            for left in range(len(label_runs))
            for right in range(left + 1, len(label_runs))
        ]
        stability = float(np.mean(pairwise_ari))
        accepted = min_share >= 0.01
        score = 0.5 * (silhouette + stability)
        reports.append(
            ClusterCandidate(
                num_clusters=num_clusters,
                silhouette=silhouette,
                stability_ari=stability,
                min_cluster_share=min_share,
                score=score,
                accepted=accepted,
            )
        )

    accepted_reports = [report for report in reports if report.accepted]
    if not accepted_reports:
        raise ValueError("All cluster candidates contain a cluster smaller than 1%")
    best_score = max(report.score for report in accepted_reports)
    near_best = [report for report in accepted_reports if best_score - report.score < 0.01]
    selected_k = min(report.num_clusters for report in near_best)
    return selected_k, reports


@torch.no_grad()
def infer_dcec(
    model: StftDcec,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    latent_dim = model.autoencoder.latent_dim
    cluster_count = model.clustering.centers.shape[0]
    embeddings = np.empty((len(loader.dataset), latent_dim), dtype=np.float32)
    probabilities = np.empty((len(loader.dataset), cluster_count), dtype=np.float32)
    for inputs, indices in tqdm(loader, desc="DCEC inference", leave=False):
        _reconstruction, latent, soft = model(inputs.to(device, non_blocking=True))
        embeddings[indices.numpy()] = latent.cpu().numpy()
        probabilities[indices.numpy()] = soft.cpu().numpy()
    return embeddings, probabilities


def train_dcec(
    model: StftDcec,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    reconstruction_weight: float = 1.0,
    clustering_weight: float = 0.1,
    logger=None,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    _embeddings, probabilities = infer_dcec(model, loader, device=device)
    previous_labels = probabilities.argmax(axis=1)
    stable_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        target = target_distribution(torch.from_numpy(probabilities)).to(device)
        model.train()
        reconstruction_sum = 0.0
        clustering_sum = 0.0
        sample_count = 0
        for inputs, indices in tqdm(loader, desc=f"DCEC epoch {epoch}", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            batch_target = target[indices.to(device)]
            optimizer.zero_grad()
            reconstruction, _latent, soft = model(inputs)
            reconstruction_loss = F.smooth_l1_loss(reconstruction, inputs)
            clustering_loss = F.kl_div(
                soft.clamp_min(1e-12).log(), batch_target, reduction="batchmean"
            )
            loss = reconstruction_weight * reconstruction_loss + clustering_weight * clustering_loss
            loss.backward()
            optimizer.step()
            batch_size = inputs.shape[0]
            reconstruction_sum += reconstruction_loss.item() * batch_size
            clustering_sum += clustering_loss.item() * batch_size
            sample_count += batch_size

        _embeddings, probabilities = infer_dcec(model, loader, device=device)
        labels = probabilities.argmax(axis=1)
        changed = float(np.mean(labels != previous_labels))
        row = {
            "epoch": float(epoch),
            "reconstruction_loss": reconstruction_sum / sample_count,
            "clustering_loss": clustering_sum / sample_count,
            "assignment_change": changed,
        }
        history.append(row)
        if logger is not None:
            logger.info(
                "DCEC epoch %d | reconstruction_loss: %.6f | clustering_loss: %.6f | assignment_change: %.6f",
                epoch,
                row["reconstruction_loss"],
                row["clustering_loss"],
                changed,
            )
        stable_epochs = stable_epochs + 1 if changed < 0.001 else 0
        previous_labels = labels
        if stable_epochs >= 3:
            break
    return history

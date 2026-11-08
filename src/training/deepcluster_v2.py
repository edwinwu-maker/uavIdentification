"""面向小规模 STFT 数据的 DeepCluster-v2：双视图、固定球面中心与余弦调度。

参考 facebookresearch/swav/main_deepclusterv2.py；用每轮全量确定性编码
代替分布式 memory bank，保留单个 prototype head，不引入 LARC 或多卡依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from src.models import StftResNet18
from src.training.deepcluster import DeepClusterResult, encode_dataset
from src.training.stft_views import make_stft_views


@dataclass
class SphericalKMeans:
    """保存单位中心和单位空间平方欧氏距离总和。"""

    cluster_centers_: np.ndarray
    inertia_: float


def fit_spherical_features(
    embeddings: np.ndarray,
    *,
    n_clusters: int = 32,
    seed: int = 42,
    max_iter: int = 100,
    n_init: int = 10,
) -> tuple[SphericalKMeans, np.ndarray, np.ndarray, np.ndarray]:
    """L2 特征上按最大余弦相似度分配，更新后再次归一化中心。"""
    features = np.asarray(embeddings, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] == 0 or not np.isfinite(features).all():
        raise ValueError("DeepCluster-v2 embeddings must be a finite non-empty two-dimensional array")
    if not 2 <= n_clusters <= len(features) or max_iter < 1 or n_init < 1:
        raise ValueError("DeepCluster-v2 requires 2 <= n_clusters <= sample count and positive iterations")
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("DeepCluster-v2 embeddings contain zero-norm features")
    features = features / norms[:, None]
    if np.var(features.astype(np.float64), axis=0).max() <= np.finfo(np.float32).eps:
        raise ValueError("DeepCluster-v2 embeddings have collapsed or have negligible variance")

    generator = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = features[generator.choice(len(features), n_clusters, replace=False)].copy()
        previous_assignments = None
        for _ in range(max_iter):
            assignments = (features @ centers.T).argmax(axis=1)
            sums = np.zeros_like(centers)
            np.add.at(sums, assignments, features)
            center_norms = np.linalg.norm(sums, axis=1)
            valid = center_norms > 1e-12
            repaired = not valid.all()
            centers[valid] = sums[valid] / center_norms[valid, None]
            # 空簇或均值相互抵消时，用已有中心表示最差的样本补充方向。
            for cluster in np.flatnonzero(~valid):
                similarities = (features @ centers[valid].T).max(axis=1) if valid.any() else np.zeros(len(features))
                centers[cluster] = features[int(similarities.argmin())]
                valid[cluster] = True
            if not repaired and previous_assignments is not None and np.array_equal(assignments, previous_assignments):
                break
            previous_assignments = assignments

        assignments = (features @ centers.T).argmax(axis=1).astype(np.int64)
        if np.any(np.bincount(assignments, minlength=n_clusters) == 0):
            continue
        distances = np.linalg.norm(features - centers[assignments], axis=1)
        inertia = float(np.square(distances, dtype=np.float64).sum())
        if best is None or inertia < best[0].inertia_:
            best = (SphericalKMeans(centers.copy(), inertia), features, assignments, distances)
    if best is None:
        raise ValueError("DeepCluster-v2 produced empty clusters; reduce n_clusters or inspect feature collapse")
    return best


def _learning_rate_at_step(
    step: int, *, total_steps: int, warmup_steps: int, learning_rate: float, min_learning_rate: float,
) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / max(decay_steps - 1, 1)
    return min_learning_rate + 0.5 * (learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress))


def _shuffled_loader(inference_loader: DataLoader, seed: int) -> DataLoader:
    return DataLoader(
        inference_loader.dataset,
        batch_size=inference_loader.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
        num_workers=inference_loader.num_workers,
        pin_memory=inference_loader.pin_memory,
        collate_fn=inference_loader.collate_fn,
        worker_init_fn=inference_loader.worker_init_fn,
    )


@torch.no_grad()
def _prime_batch_norm(model, loader: DataLoader, *, device: torch.device, seed: int) -> None:
    """首轮聚类前估计 projector BN 统计；只更新 buffers，不更新参数。"""
    model.train()
    generator = torch.Generator().manual_seed(seed)
    for inputs, _ in tqdm(loader, desc="Initializing DeepCluster-v2 BN", leave=False):
        # 双视图合并后至少有两个样本，尾批 B=1 也能更新 BatchNorm1d。
        views = make_stft_views(inputs, generator)
        model.encode(torch.cat(views, dim=0).to(device, non_blocking=True))
    model.eval()


def _effective_rank(features: np.ndarray) -> float:
    singular_values = np.linalg.svd(features - features.mean(axis=0), compute_uv=False)
    variances = np.square(singular_values.astype(np.float64))
    probabilities = variances[variances > 0] / variances.sum()
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def train_deepcluster_v2(
    model: StftResNet18,
    inference_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int = 20,
    n_clusters: int = 32,
    learning_rate: float = 3e-4,
    temperature: float = 0.1,
    warmup_epochs: int = 5,
    min_learning_rate: float = 1e-6,
    seed: int = 42,
    logger=None,
) -> DeepClusterResult:
    """固定本轮单位中心，监督同一样本的两个 STFT 视图；每轮完整遍历一次数据。

    canonical 输入用于每轮全量聚类及最终导出，不采用官方的大规模增强 memory bank。
    inference_loader 必须按全局索引顺序覆盖全部样本，且预处理确定。
    """
    if epochs < 1 or warmup_epochs < 0:
        raise ValueError("DeepCluster-v2 epochs must be positive and warmup_epochs non-negative")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("DeepCluster-v2 learning_rate and temperature must be finite and positive")
    if not np.isfinite(min_learning_rate) or not 0 <= min_learning_rate <= learning_rate:
        raise ValueError("DeepCluster-v2 min_learning_rate must be within 0..learning_rate")
    if inference_loader.batch_size is None or inference_loader.drop_last:
        raise ValueError("DeepCluster-v2 inference loader requires batch_size and drop_last=False")
    if not isinstance(inference_loader.sampler, SequentialSampler) or len(inference_loader.sampler) != len(inference_loader.dataset):
        raise ValueError("DeepCluster-v2 inference loader requires a complete SequentialSampler")
    sample_count = len(inference_loader.dataset)
    if not 2 <= n_clusters <= sample_count:
        raise ValueError("DeepCluster-v2 requires 2 <= n_clusters <= sample count")
    model.to(device)
    _prime_batch_norm(model, _shuffled_loader(inference_loader, seed), device=device, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    steps_per_epoch = len(inference_loader)
    total_steps = epochs * steps_per_epoch
    # 短冒烟运行也保留至少一轮余弦阶段，避免所有 step 都在 warmup。
    warmup_steps = min(warmup_epochs, epochs - 1) * steps_per_epoch
    history: list[dict] = []
    previous_assignments = None

    for epoch in range(1, epochs + 1):
        embeddings = encode_dataset(model, inference_loader, device=device)
        kmeans, features, assignments, _ = fit_spherical_features(
            embeddings, n_clusters=n_clusters, seed=seed + epoch - 1,
        )
        counts = np.bincount(assignments, minlength=n_clusters)
        # 中心只是固定张量，不加入优化器，避免退回随机可训练分类头。
        centers = torch.from_numpy(kmeans.cluster_centers_).to(device)
        targets = torch.from_numpy(assignments)
        train_loader = _shuffled_loader(inference_loader, seed + epoch)
        view_generator = torch.Generator().manual_seed(seed + epoch)
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        trained_count = 0
        learning_rates = []
        for batch_index, (inputs, indices) in enumerate(tqdm(train_loader, desc=f"DeepCluster-v2 epoch {epoch}", leave=False)):
            lr = _learning_rate_at_step(
                (epoch - 1) * steps_per_epoch + batch_index,
                total_steps=total_steps, warmup_steps=warmup_steps,
                learning_rate=learning_rate, min_learning_rate=min_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            learning_rates.append(lr)
            views = make_stft_views(inputs, view_generator)
            batch = torch.cat(views, dim=0).to(device, non_blocking=True)
            labels = targets[indices.to(device="cpu", dtype=torch.long)].repeat(2).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            projected = F.normalize(model.encode(batch), dim=1)
            logits = F.linear(projected, centers) / temperature
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("DeepCluster-v2 training produced non-finite loss")
            loss.backward()
            optimizer.step()
            loss_sum.add_(loss.detach() * inputs.shape[0])
            correct.add_((logits.detach().argmax(dim=1) == labels).sum())
            trained_count += inputs.shape[0]

        row = {
            "epoch": epoch,
            "loss": (loss_sum / trained_count).item(),
            "pseudo_label_accuracy": (correct / (2 * trained_count)).item(),
            "learning_rate_start": learning_rates[0],
            "learning_rate_end": learning_rates[-1],
            "inertia": kmeans.inertia_,
            "cluster_counts": counts.tolist(),
            "largest_cluster_fraction": float(counts.max() / sample_count),
            "effective_clusters": float(np.exp(-(counts / sample_count * np.log(counts / sample_count)).sum())),
            "feature_std_mean": float(features.std(axis=0).mean()),
            "effective_rank": _effective_rank(features),
            "reassignment_nmi": None if previous_assignments is None else float(
                normalized_mutual_info_score(previous_assignments, assignments)
            ),
        }
        history.append(row)
        previous_assignments = assignments
        if logger is not None:
            logger.info(
                "DeepCluster-v2 epoch %d | loss: %.6f | pseudo_acc: %.3f | lr: %.3g | largest_cluster: %.3f | NMI: %s | rank: %.2f",
                epoch, row["loss"], row["pseudo_label_accuracy"], row["learning_rate_end"],
                row["largest_cluster_fraction"], row["reassignment_nmi"], row["effective_rank"],
            )

    embeddings = encode_dataset(model, inference_loader, device=device)
    kmeans, features, assignments, distances = fit_spherical_features(
        embeddings, n_clusters=n_clusters, seed=seed + epochs,
    )
    model.zero_grad(set_to_none=True)
    return DeepClusterResult(model, None, kmeans, embeddings, features, assignments, distances, history)

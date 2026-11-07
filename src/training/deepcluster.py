"""以聚类伪标签迭代训练 STFT 编码器，保留独立的最终聚类结果。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm

from src.models import StftCNN, StftResNet18


@dataclass
class DeepClusterResult:
    model: StftCNN | StftResNet18
    pca: PCA
    kmeans: KMeans
    embeddings: np.ndarray
    features: np.ndarray
    assignments: np.ndarray
    distances: np.ndarray
    history: list[dict]


@torch.inference_mode()
def encode_dataset(
    model: StftCNN | StftResNet18,
    loader: DataLoader,
    *,
    device: torch.device,
) -> np.ndarray:
    """提取 CNN 特征，按完整数据集的全局索引写回；调用后保持 eval 模式。"""
    model.eval()
    embeddings = torch.empty(
        (len(loader.dataset), model.latent_dim), device=device, dtype=torch.float32,
    )
    for inputs, indices in tqdm(loader, desc="Encoding STFT", leave=False):
        latent = model.encode(inputs.to(device, non_blocking=True))
        indices = indices.to(device=device, dtype=torch.long, non_blocking=True)
        embeddings.index_copy_(0, indices, latent.to(dtype=torch.float32))
    # 一次性回传，避免逐批同步。
    return embeddings.cpu().numpy()


def fit_deepcluster_features(
    embeddings: np.ndarray,
    *,
    n_clusters: int = 32,
    pca_dim: int = 64,
    seed: int = 42,
) -> tuple[PCA, KMeans, np.ndarray, np.ndarray, np.ndarray]:
    """拟合 PCA 白化、L2 归一化与 KMeans；距离为每个样本到所属中心的欧氏距离。"""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("DeepCluster embeddings must be a finite two-dimensional array")
    sample_count, feature_dim = embeddings.shape
    if not 2 <= n_clusters <= sample_count:
        raise ValueError("DeepCluster requires 2 <= n_clusters <= sample count")
    if not 1 <= pca_dim <= 64 or feature_dim == 0:
        raise ValueError("DeepCluster requires 1 <= pca_dim <= 64 and non-empty features")
    scale = max(float(np.mean(np.square(embeddings, dtype=np.float64))), 1.0)
    if np.var(embeddings.astype(np.float64), axis=0).max() <= np.finfo(np.float32).eps * scale:
        raise ValueError("DeepCluster embeddings have collapsed or have negligible variance")

    pca = PCA(
        n_components=min(pca_dim, sample_count - 1, feature_dim),
        whiten=True,
        svd_solver="full",
        random_state=seed,
    ).fit(embeddings)
    variances = pca.explained_variance_
    if not np.isfinite(variances).all() or variances[0] <= np.finfo(np.float32).eps * scale:
        raise ValueError("DeepCluster embeddings have collapsed or have negligible variance")

    # 去掉接近零方差的方向，避免白化放大数值噪声；保存的 PCA 可直接用于后续 transform。
    effective_dim = int(np.count_nonzero(variances > variances[0] * 1e-6))
    pca.components_ = pca.components_[:effective_dim]
    pca.explained_variance_ = pca.explained_variance_[:effective_dim]
    pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:effective_dim]
    pca.singular_values_ = pca.singular_values_[:effective_dim]
    pca.n_components_ = effective_dim
    pca.n_components = effective_dim
    features = pca.transform(embeddings).astype(np.float32, copy=False)
    norms = np.linalg.norm(features, axis=1)
    if not np.isfinite(features).all() or np.any(norms <= 1e-12):
        raise ValueError("DeepCluster whitening produced non-finite or zero-norm features")
    features /= norms[:, None]

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    assignments = kmeans.fit_predict(features).astype(np.int64)
    counts = np.bincount(assignments, minlength=n_clusters)
    if np.any(counts == 0):
        raise ValueError("DeepCluster produced empty clusters; reduce n_clusters or inspect feature collapse")
    distances = np.linalg.norm(features - kmeans.cluster_centers_[assignments], axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("DeepCluster produced non-finite centroid distances")
    return pca, kmeans, features, assignments, distances


def train_deepcluster(
    model: StftCNN | StftResNet18,
    inference_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int = 20,
    n_clusters: int = 32,
    pca_dim: int = 64,
    learning_rate: float = 1e-4,
    seed: int = 42,
    logger=None,
) -> DeepClusterResult:
    """原地训练随机初始化的 CNN，返回最终权重对应的特征和聚类。

    inference_loader 须使用 SequentialSampler，数据集返回 (STFT, 全局索引) 并使用确定性预处理。
    每轮用全部样本聚类，按簇均衡抽取 N 次训练，不使用重建预训练。
    """
    if epochs <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("DeepCluster epochs and learning_rate must be positive")
    if inference_loader.batch_size is None or inference_loader.drop_last:
        raise ValueError("DeepCluster inference loader requires batch_size and drop_last=False")
    if not isinstance(inference_loader.sampler, SequentialSampler) or len(inference_loader.sampler) != len(inference_loader.dataset):
        raise ValueError("DeepCluster inference loader requires a complete SequentialSampler")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict] = []
    previous_assignments = None
    sample_count = len(inference_loader.dataset)

    for epoch in range(1, epochs + 1):
        embeddings = encode_dataset(model, inference_loader, device=device)
        pca, kmeans, features, assignments, _ = fit_deepcluster_features(
            embeddings, n_clusters=n_clusters, pca_dim=pca_dim, seed=seed + epoch - 1,
        )
        counts = np.bincount(assignments, minlength=n_clusters)
        targets = torch.from_numpy(assignments)
        # 簇内样本等权、簇间总权重相同；每轮抽 N 次，允许重复以覆盖小簇。
        generator = torch.Generator().manual_seed(seed + epoch)
        sampler = WeightedRandomSampler(
            torch.from_numpy(1.0 / counts[assignments]), sample_count,
            replacement=True, generator=generator,
        )
        train_loader = DataLoader(
            inference_loader.dataset,
            batch_size=inference_loader.batch_size,
            sampler=sampler,
            num_workers=inference_loader.num_workers,
            pin_memory=inference_loader.pin_memory,
            collate_fn=inference_loader.collate_fn,
            worker_init_fn=inference_loader.worker_init_fn,
        )

        # 本轮簇编号不继承上轮语义，重新初始化分类头及其优化器。
        head = nn.Linear(model.latent_dim, n_clusters)
        head_generator = torch.Generator().manual_seed(seed + epoch)
        with torch.no_grad():
            head.weight.normal_(0, 0.01, generator=head_generator)
            head.bias.zero_()
        head.to(device)
        head_optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
        model.train()
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        trained_count = 0
        for inputs, indices in tqdm(train_loader, desc=f"DeepCluster epoch {epoch}", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = targets[indices.to(device="cpu", dtype=torch.long)].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            head_optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(model.encode(inputs)), labels)
            if not torch.isfinite(loss):
                raise ValueError("DeepCluster training produced non-finite loss")
            loss.backward()
            optimizer.step()
            head_optimizer.step()
            loss_sum.add_(loss.detach() * inputs.shape[0])
            trained_count += inputs.shape[0]

        epoch_loss = (loss_sum / trained_count).item()
        row = {
            "epoch": epoch,
            "loss": epoch_loss,
            "pca_components": pca.n_components_,
            "inertia": float(kmeans.inertia_),
            "cluster_counts": counts.tolist(),
            "largest_cluster_fraction": float(counts.max() / sample_count),
            "feature_std_mean": float(embeddings.std(axis=0).mean()),
            "reassignment_nmi": None if previous_assignments is None else float(
                normalized_mutual_info_score(previous_assignments, assignments)
            ),
        }
        history.append(row)
        previous_assignments = assignments
        if logger is not None:
            logger.info(
                "DeepCluster epoch %d | loss: %.6f | largest_cluster: %.3f | NMI: %s",
                epoch, epoch_loss, row["largest_cluster_fraction"], row["reassignment_nmi"],
            )

    # 最后一轮更新后重新提取和拟合，避免导出前一轮权重对应的旧伪标签。
    embeddings = encode_dataset(model, inference_loader, device=device)
    pca, kmeans, features, assignments, distances = fit_deepcluster_features(
        embeddings, n_clusters=n_clusters, pca_dim=pca_dim, seed=seed + epochs,
    )
    model.zero_grad(set_to_none=True)
    return DeepClusterResult(model, pca, kmeans, embeddings, features, assignments, distances, history)

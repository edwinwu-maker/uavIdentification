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

from src.models import StftCNN


@dataclass
class DeepClusterResult:
    """保存训练后的编码器、特征变换、聚类结果及训练历史。"""

    model: StftCNN            # 训练后的 CNN 编码器
    pca: PCA | None           # 拟合后的 PCA 白化变换，关闭时为 None
    kmeans: KMeans            # 拟合后的 KMeans 聚类器
    embeddings: np.ndarray    # 原始 CNN 特征，形状 (N, latent_dim)
    features: np.ndarray      # 可选 PCA 白化后经 L2 归一化的聚类特征，形状 (N, D)
    assignments: np.ndarray   # 各样本所属簇编号，形状 (N,)
    distances: np.ndarray     # 到所属簇中心的欧氏距离，形状 (N,)
    history: list[dict]       # 各轮训练损失与聚类统计


@torch.inference_mode()
def _encode_dataset(
    model: StftCNN,
    loader: DataLoader,
    *,
    device: torch.device,
) -> np.ndarray:
    """提取 CNN 特征，按完整数据集的全局索引写回；调用后保持 eval 模式。"""
    model.eval()

    # 预分配矩阵 (样本数, 特征维度)
    embeddings = torch.empty(
        (len(loader.dataset), model.latent_dim), device=device, dtype=torch.float32,
    )

    # input(B, C, 256, 256), indices(B)
    for inputs, indices in tqdm(loader, desc="Encoding STFT", leave=False):
        latent = model(inputs.to(device, non_blocking=True))
        indices = indices.to(device=device, dtype=torch.long, non_blocking=True)
        embeddings.index_copy_(0, indices, latent.to(dtype=torch.float32))

    # 一次性回传，避免逐批同步。
    return embeddings.cpu().numpy()


def _validate_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, float]:
    """校验两种聚类路径共用的输入，返回 float32 特征及方差判断尺度。"""

    # 统一为 float32，确保输入是无 NaN/Inf 的二维特征矩阵。
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("DeepCluster embeddings must be a finite two-dimensional array")

    # 聚类至少需要两个样本，并拒绝空特征。
    sample_count, feature_dim = embeddings.shape
    if sample_count < 2:
        raise ValueError("DeepCluster requires at least two samples")
    if feature_dim == 0:
        raise ValueError("DeepCluster requires non-empty features")

    # 按特征尺度设定方差阈值，检测特征塌缩或变化过小。
    scale = max(float(np.mean(np.square(embeddings, dtype=np.float64))), 1.0)
    if np.var(embeddings.astype(np.float64), axis=0).max() <= np.finfo(np.float32).eps * scale:
        raise ValueError("DeepCluster embeddings have collapsed or have negligible variance")
    return embeddings, scale


def _fit_pca_whitening(
    embeddings: np.ndarray,
    *,
    pca_dim: int = 64,
    seed: int = 42,
) -> tuple[PCA, np.ndarray]:
    """校验特征并拟合 PCA，返回裁剪后的 PCA 和未做 L2 归一化的白化特征。"""
    embeddings, scale = _validate_embeddings(embeddings)
    if not 1 <= pca_dim <= 64:
        raise ValueError("DeepCluster requires 1 <= pca_dim <= 64")
    sample_count, feature_dim = embeddings.shape

    # 按样本数和特征维数限制 PCA 白化维数，检查主成分方差是否有效。
    pca = PCA(
        n_components=min(pca_dim, sample_count - 1, feature_dim),
        whiten=True,
        svd_solver="full",
        random_state=seed,
    ).fit(embeddings)

    variances = pca.explained_variance_ # variances：各主成分的方差，按从大到小排列。
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
    return pca, features


def _fit_l2_kmeans(
    features: np.ndarray,
    *,
    n_clusters: int = 32,
    seed: int = 42,
) -> tuple[KMeans, np.ndarray, np.ndarray, np.ndarray]:
    """对原始或已白化特征做 L2 归一化与 KMeans，返回聚类器、特征、簇标签及所属中心距离。"""
    features, _ = _validate_embeddings(features)
    if not 2 <= n_clusters <= len(features):
        raise ValueError("DeepCluster requires 2 <= n_clusters <= sample count")

    # 后续原地归一化不能改写调用方的特征。
    features = features.copy()
    # 检查聚类特征的非有限值及近零范数，再逐样本做 L2 归一化。
    norms = np.linalg.norm(features, axis=1)
    if not np.isfinite(features).all() or np.any(norms <= 1e-12):
        raise ValueError("DeepCluster preprocessing produced non-finite or zero-norm features")
    features /= norms[:, None]

    # 拟合 KMeans 并生成簇标签，检查是否存在空簇。
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    assignments = kmeans.fit_predict(features).astype(np.int64)
    counts = np.bincount(assignments, minlength=n_clusters)
    if np.any(counts == 0):
        raise ValueError("DeepCluster produced empty clusters; reduce n_clusters or inspect feature collapse")

    # 计算样本到所属簇中心的欧氏距离，确保结果无 NaN/Inf。
    distances = np.linalg.norm(features - kmeans.cluster_centers_[assignments], axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("DeepCluster produced non-finite centroid distances")
    
    return kmeans, features, assignments, distances


def train_deepcluster(
    model: StftCNN,
    inference_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int = 20,
    n_clusters: int = 32,
    use_pca: bool = True,
    pca_dim: int = 64,
    learning_rate: float = 1e-4,
    seed: int = 42,
    logger=None,
) -> DeepClusterResult:
    """原地训练随机初始化的 CNN，返回最终权重对应的特征和聚类。

    inference_loader 须使用 SequentialSampler，数据集返回 (STFT, 全局索引) 并使用确定性预处理。
    每轮用全部样本聚类，按簇均衡抽取 N 次训练，不使用重建预训练。
    use_pca=False 时直接对原始 CNN 特征做 L2 归一化，忽略 pca_dim。
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
        embeddings = _encode_dataset(model, inference_loader, device=device)
        pca, clustering_features = None, embeddings
        if use_pca:
            pca, clustering_features = _fit_pca_whitening(
                embeddings, pca_dim=pca_dim, seed=seed + epoch - 1,
            )
        kmeans, features, assignments, _ = _fit_l2_kmeans(
            clustering_features, n_clusters=n_clusters, seed=seed + epoch - 1,
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
            loss = F.cross_entropy(head(model(inputs)), labels)
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
            "pca_components": pca.n_components_ if pca is not None else None,
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
    embeddings = _encode_dataset(model, inference_loader, device=device)
    pca, clustering_features = None, embeddings
    if use_pca:
        pca, clustering_features = _fit_pca_whitening(
            embeddings, pca_dim=pca_dim, seed=seed + epochs,
        )
    kmeans, features, assignments, distances = _fit_l2_kmeans(
        clustering_features, n_clusters=n_clusters, seed=seed + epochs,
    )
    model.zero_grad(set_to_none=True)
    return DeepClusterResult(model, pca, kmeans, embeddings, features, assignments, distances, history)

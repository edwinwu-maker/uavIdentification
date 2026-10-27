"""Training helpers for STFT convolutional deep embedded clustering."""

from __future__ import annotations

import copy
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
    """保存一个候选簇数的聚类质量评估结果；实例创建后字段不可修改。"""

    num_clusters: int           # 候选簇数 K。
    silhouette: float           # 首次 KMeans 结果的轮廓系数，最多使用 2000 个样本计算。
    stability_ari: float        # 三次不同种子 KMeans 结果两两比较的平均调整兰德指数（ARI）。
    min_cluster_share: float    # 三次运行中最小的单簇样本占比，用于排除过小的簇。
    score: float                # 综合得分：0.5 * (silhouette + stability_ari)。
    accepted: bool              # 最小簇占比是否达到 1%，决定候选是否参与最终选择。


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
    """以输入 STFT 为重建目标预训练自编码器，并恢复最佳训练损失对应的权重。

    参数：
        model: 待训练的 StftAutoencoder，会被迁移到 device 并原地更新参数。
        loader: 非空数据加载器，每批返回 (inputs, indices)；inputs 为已预处理的
            (B, 1, 256, 256) 浮点张量，B 为批次大小，indices 在本函数中不参与计算。
        device: 模型和输入张量的计算设备。
        epochs: 最大训练轮数。
        learning_rate: AdamW 优化器的学习率，权重衰减固定为 1e-4。
        patience: 允许连续未改善的轮数；损失须比最佳值下降超过 1e-6 才视为改善。
        logger: 可选日志对象，需提供 info 方法；None 时不记录每轮损失日志。

    返回：
        各实际训练轮次的平均 Smooth L1 重建损失列表，按样本数加权。
        model 的参数恢复为该训练过程中最佳损失对应的状态，不写入模型文件。
    """
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
    """关闭梯度计算，将完整数据集编码为按全局样本索引排列的潜在向量。

    参数：
        model: 已放到 device 上的 StftAutoencoder；本函数设为 eval 模式并保持该模式。
        loader: 每批返回 (inputs, indices) 的数据加载器；inputs 的形状为
            (B, 1, 256, 256)，indices 为 CPU 整数张量，表示数据集中的全局样本索引。
            须完整覆盖 0～N-1，其中 N = len(loader.dataset)，不能丢弃末批；可打乱顺序。
        device: 输入张量迁移到的设备，须与模型所在设备一致。

    返回：
        形状为 (N, D) 的 float32 NumPy 数组，D = model.latent_dim；
        第 i 行保存全局索引 i 的潜在向量，结果位于 CPU 内存中。
    """
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
    """根据轮廓系数、聚类稳定性和最小簇占比选择簇数 K。

    参数：
        embeddings: 形状为 (N, D) 的 NumPy 潜在向量数组，每行对应一个样本。
        candidates: 非空候选簇数元组，默认 (4, 6, 8, 10)；跳过不小于抽样数量的 K。
        sample_size: 无放回抽样的数量上限，实际使用 min(sample_size, N) 个样本。
        seed: 抽样和轮廓系数计算的种子；KMeans 分别使用 seed、seed+1、seed+2。

    返回：
        (selected_k, reports)：选中的整数簇数，以及各已评估候选的 ClusterCandidate 列表。
        仅从最小簇占比达到 1% 的候选中选择；与最高综合得分之差小于 0.01 时取最小 K。
        输入非二维、样本不足或没有候选通过筛选时抛出 ValueError。
    """
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
    """关闭梯度计算，推理完整数据集的潜在向量和聚类软分配概率。

    参数：
        model: 已放到 device 上的 StftDcec；本函数设为 eval 模式并保持该模式。
        loader: 每批返回 (inputs, indices) 的数据加载器；inputs 的形状为
            (B, 1, 256, 256)，indices 为 CPU 整数张量，表示数据集中的全局样本索引。
            须完整覆盖 0～N-1，其中 N = len(loader.dataset)，不能丢弃末批；可打乱顺序。
        device: 输入张量迁移到的设备，须与模型所在设备一致。

    返回：
        (embeddings, probabilities)，均为 CPU 内存中的 float32 NumPy 数组：
        embeddings 的形状为 (N, D)，D 为自编码器潜在维度；probabilities 的形状为
        (N, K)，K 为簇数，第 i 行第 j 列表示样本 i 属于簇 j 的软分配概率。
        两个数组均按全局样本索引排列，不返回重建图像。
    """
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

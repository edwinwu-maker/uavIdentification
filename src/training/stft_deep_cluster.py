"""Training helpers for STFT convolutional deep embedded clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.autoencoder import StftAutoencoder
from src.models.clustering import target_distribution
from src.models.dcec import StftDcec


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
    """预训练 STFT 自编码器，恢复最佳重建损失对应的权重。

    输入：
        model: 自编码器，迁移至 device 并原地训练。
        loader: 非空，每批为 (inputs, indices)；inputs 为 (B, 1, 256, 256)
            浮点 STFT，B 为批次大小；indices 不参与预训练。
        device: 模型和输入的计算设备。
        epochs: 最大训练轮数。
        learning_rate: AdamW 学习率，权重衰减固定为 1e-4。
        patience: 连续未改善的早停轮数，改善阈值为损失下降超过 1e-6。
        logger: 可选日志对象，需提供 info 方法。

    输出：
        history: list[float]，逐轮按样本数加权的平均 Smooth L1 损失。
        model: 原地恢复最佳权重并清除梯度，保留在 device，不写文件。
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss = float("inf")
    # CPU 独立快照：避免共享参数内存，并减少显存占用。
    best_state = {name: value.detach().to(device="cpu", copy=True) for name, value in model.state_dict().items()}
    stale_epochs = 0
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        sample_count = 0
        for inputs, _ in tqdm(loader, desc=f"CAE epoch {epoch}", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            optimizer.zero_grad()
            # 输入 (B, 1, 256, 256) → 同形重建图、潜在向量 (B, latent_dim)。
            reconstruction, _ = model(inputs)
            loss = F.smooth_l1_loss(reconstruction, inputs)
            loss.backward()
            optimizer.step()
            # 按样本数累计脱离计算图的损失，每轮仅读取一次标量。
            loss_sum.add_(loss.detach().to(dtype=torch.float32) * inputs.shape[0])
            sample_count += inputs.shape[0]
        epoch_loss = (loss_sum / sample_count).item()
        history.append(epoch_loss)
        if logger is not None:
            logger.info("CAE epoch %d | reconstruction_loss: %.6f", epoch, epoch_loss)
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = {name: value.detach().to(device="cpu", copy=True) for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    # 清除末批梯度，释放显存。
    model.zero_grad(set_to_none=True)
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


@torch.inference_mode()
def infer_dcec(
    model: StftDcec,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """编码并计算聚类软分配概率，跳过解码；调用后模型保持 eval 模式。

    输入：
        model: 已位于 device 的模型；device: 推理设备。
        loader: 每批返回 (inputs, indices)，inputs 为 (B, 1, 256, 256) 浮点张量；
            indices 为 CPU 整数索引，须完整覆盖 0～N-1，可乱序，N = len(loader.dataset)。
    输出：
        (embeddings, probabilities): 按全局索引排列的 CPU float32 NumPy 数组；
        embeddings 为 (N, D) 潜在向量，probabilities 为 (N, K) 软分配概率，D 为潜在维度，K 为簇数。

    CUDA 全量缓存额外占用 N * (D + K) * 4 字节显存，结束后回传；CPU/MPS 在 CPU 收集。
    """
    model.eval()
    latent_dim = model.autoencoder.latent_dim           # 潜在向量维度 D。
    cluster_count = model.clustering.centers.shape[0]   # 聚类中心数量 K。
    sample_count = len(loader.dataset)                  # 数据集样本总数 N。

    output_device = device if device.type == "cuda" else torch.device("cpu")

    # 全量输出缓冲区：embeddings 为 (N, D) 潜在向量，probabilities 为 (N, K) 软分配概率。
    embeddings = torch.empty((sample_count, latent_dim), device=output_device, dtype=torch.float32)
    probabilities = torch.empty((sample_count, cluster_count), device=output_device, dtype=torch.float32)

    # inputs 为本批 STFT 浮点张量 (B, 1, 256, 256)；indices 为对应样本的 CPU 全局整数索引 (B,)。
    for inputs, indices in tqdm(loader, desc="DCEC inference", leave=False):
        # 输入迁移至推理设备，编码为本批潜在向量 (B, D)。
        latent = model.autoencoder.encode(inputs.to(device, non_blocking=True))
        # 根据与簇中心的 Student-t 相似度计算软分配 (B, K)，逐行归一化。
        soft = model.clustering(latent)
        output_indices = indices.to(device=output_device, dtype=torch.long, non_blocking=True)
        embeddings.index_copy_(0, output_indices, latent.to(device=output_device, dtype=torch.float32))
        probabilities.index_copy_(0, output_indices, soft.to(device=output_device, dtype=torch.float32))
    # CUDA 同步回传；CPU 直接共享内存。
    return embeddings.cpu().numpy(), probabilities.cpu().numpy()


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
    """联合优化重建与聚类损失，连续 3 轮标签变化率低于 0.001 时早停。

    输入：
        model: 待训练的 StftDcec，迁移至 device 并原地更新。
        loader: 非空，每批返回 (inputs, indices)；inputs 为 (B, 1, 256, 256) 浮点张量，
            indices 为 CPU 整数索引，须完整覆盖 0～N-1，可乱序，N = len(loader.dataset)。
        device: 训练设备。
        epochs: 最大轮数。
        learning_rate: AdamW 学习率。
        reconstruction_weight/clustering_weight: Smooth L1 重建损失与 KL 聚类损失的权重。
        logger: 可选日志对象，需提供 info 方法。
    输出：
        逐轮字典列表，含 epoch（从 1 起）、reconstruction_loss、clustering_loss、
        assignment_change（标签变化比例）；两项损失均为未乘权重的样本均值。
        model 保留末轮权重，结束时处于 eval 模式。

    概率与目标分布在 device 上使用 Torch 缓存，不保存全量潜在向量。
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    dataset_size = len(loader.dataset)
    cluster_count = model.clustering.centers.shape[0]
    probabilities = torch.empty((dataset_size, cluster_count), device=device, dtype=torch.float32)

    @torch.no_grad()    # 禁用梯度记录，避免概率刷新构建计算图。
    def update_probabilities() -> None:
        """仅编码并聚类，按全局索引刷新概率 (N, K)，保持 eval 模式。"""
        model.eval()
        for inputs, indices in tqdm(loader, desc="DCEC inference", leave=False):
            latent = model.autoencoder.encode(inputs.to(device, non_blocking=True))
            soft = model.clustering(latent)
            output_indices = indices.to(device=device, dtype=torch.long, non_blocking=True)
            probabilities.index_copy_(0, output_indices, soft.to(dtype=torch.float32))

    update_probabilities()
    previous_labels = probabilities.argmax(dim=1)
    stable_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        # 用上轮全量概率固定本轮目标，按全局索引匹配训练样本。
        target = target_distribution(probabilities)
        model.train()
        loss_sums = torch.zeros(2, device=device, dtype=torch.float32)
        sample_count = 0

        # inputs 为 STFT 浮点张量 (B, 1, 256, 256)；indices 为 CPU 全局整数索引 (B,)。
        for inputs, indices in tqdm(loader, desc=f"DCEC epoch {epoch}", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            batch_target = target[indices.to(device=device, dtype=torch.long, non_blocking=True)]
            optimizer.zero_grad()
            # 输出重建图 (B, 1, 256, 256)、潜在向量 (B, D) 和软分配 (B, K)。
            reconstruction, _latent, soft = model(inputs)
            # Smooth L1 重建误差，对全部元素取均值。
            reconstruction_loss = F.smooth_l1_loss(reconstruction, inputs)
            # KL(P || Q)：以固定目标 P 约束软分配 Q，按批次样本数归一化。
            clustering_loss = F.kl_div(
                soft.clamp_min(1e-12).log(), batch_target, reduction="batchmean"
            )
            # 加权联合重建与聚类损失。
            loss = reconstruction_weight * reconstruction_loss + clustering_weight * clustering_loss
            # 反向传播，计算模型参数梯度。
            loss.backward()
            # AdamW 更新自编码器与聚类中心。
            optimizer.step()
            batch_size = inputs.shape[0]
            # 脱离计算图并在设备上累加，避免每批读取标量触发 GPU 同步。
            loss_sums[0].add_(reconstruction_loss.detach().to(dtype=torch.float32) * batch_size)
            loss_sums[1].add_(clustering_loss.detach().to(dtype=torch.float32) * batch_size)
            sample_count += batch_size

        update_probabilities()
        labels = probabilities.argmax(dim=1)
        changed = (labels != previous_labels).sum().item() / dataset_size
        reconstruction_mean, clustering_mean = (loss_sums / sample_count).cpu().tolist()
        row = {
            "epoch": float(epoch),
            "reconstruction_loss": reconstruction_mean,
            "clustering_loss": clustering_mean,
            "assignment_change": changed,
        }
        history.append(row)

        # 可选日志：记录轮次、平均重建与聚类损失及标签变化率。
        if logger is not None:
            logger.info(
                "DCEC epoch %d | reconstruction_loss: %.6f | clustering_loss: %.6f | assignment_change: %.6f",
                epoch,
                row["reconstruction_loss"],
                row["clustering_loss"],
                changed,
            )

        previous_labels = labels    # 保存本轮标签，供下一轮比较变化率。

        # 变化率低于 0.001 则累计稳定轮数，否则清零；连续 3 轮稳定时早停。
        stable_epochs = stable_epochs + 1 if changed < 0.001 else 0
        if stable_epochs >= 3:
            break
    return history

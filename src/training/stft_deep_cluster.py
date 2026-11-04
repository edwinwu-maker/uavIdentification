"""STFT 自编码器的重建训练与潜在向量提取。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.autoencoder import StftAutoencoder


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

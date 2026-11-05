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
    """预训练 STFT 自编码器，恢复最佳重建权重。

    输入：
        model: 待原地训练的自编码器。
        loader: 非空，每批为 (inputs, indices)；inputs 为 (B, 1, 256, 256)
            浮点 STFT，B 为批次大小；indices 不参与训练。
        device: 模型与输入迁移到的计算设备。
        epochs: 最大训练轮数。
        learning_rate: AdamW 学习率，权重衰减为 1e-4。
        patience: 连续未改善的早停轮数；损失下降超过 1e-6 视为改善。
        logger: 可选日志对象，提供 info 方法。

    输出：
        返回 history（list[float]）：逐轮按样本数加权的平均 Smooth L1 损失。
        原地更新 model：恢复最佳权重、清除梯度，保留在 device。
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss = float("inf")
    # CPU 独立快照，避免权重共享内存并节省显存。
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
            # 重建图与输入同形，潜在向量不参与损失计算。
            reconstruction, _ = model(inputs)
            loss = F.smooth_l1_loss(reconstruction, inputs)
            loss.backward()
            optimizer.step()
            # 脱离计算图，按样本数累计损失；每轮仅同步一次标量。
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


@torch.inference_mode()
def encode_dataset(
    model: StftAutoencoder,
    loader: DataLoader,
    *,
    device: torch.device,
) -> np.ndarray:
    """推理编码完整数据集，按全局索引返回潜在向量。

    输入：
        model: 位于 device 的自编码器，调用后保持 eval 模式。
        loader: 每批为 (inputs, indices)，inputs 为 (B, 1, 256, 256) 浮点张量，
            indices 为 CPU 整数张量；可乱序，须完整覆盖全局索引 0～N-1。
        device: 计算与结果缓冲设备；缓冲占用 N × D × 4 字节。

    输出：
        CPU 上的 float32 NumPy 数组 (N, D)，第 i 行对应全局样本 i；
        N = len(loader.dataset)，D = model.latent_dim。
    """
    model.eval()
    embeddings = torch.empty(
        (len(loader.dataset), model.latent_dim), device=device, dtype=torch.float32
    )
    for inputs, indices in tqdm(loader, desc="Encoding STFT", leave=False):
        latent = model.encode(inputs.to(device, non_blocking=True))
        indices = indices.to(device=device, dtype=torch.long, non_blocking=True)
        embeddings.index_copy_(0, indices, latent.to(dtype=torch.float32))
    # 一次性回传，避免逐批同步。
    return embeddings.cpu().numpy()

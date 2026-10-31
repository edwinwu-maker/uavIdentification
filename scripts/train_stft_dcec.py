"""Train DCEC on clean STFT H5 files and create a blinded review set.

Usage:
  python scripts/train_stft_dcec.py --data-dir ... --work-dir outputs/stft_deep_cluster
  python scripts/train_stft_dcec.py --data-dir ... --work-dir ... --device cuda:0 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from src.data.stft_clustering import (
    REVIEW_ROLES,
    CleanStftClusteringDataset,
    scan_clean_stft_h5,
    select_cluster_review_indices,
    write_cluster_assignments,
    write_cluster_review_artifacts,
)
from src.models.autoencoder import StftAutoencoder
from src.models.dcec import StftDcec
from src.training.stft_deep_cluster import (
    encode_dataset,
    infer_dcec,
    pretrain_autoencoder,
    select_cluster_count,
    train_dcec,
)
from src.utils.cli import log_current_command
from src.utils.device import default_device
from src.utils.logger import logger
from src.utils.random import set_random_seed

SEED = 42


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


def main() -> None:
    # 校验运行参数，避免无效配置进入数据准备和训练。
    args = parse_args()
    log_current_command(logger)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers must be non-negative")
    if args.pretrain_epochs <= 0 or args.dcec_epochs <= 0 or args.patience <= 0:
        raise ValueError("epoch and patience values must be positive")

    # 固定随机种子，并初始化计算设备与输出目录。
    set_random_seed(SEED)
    device = torch.device(args.device)
    work_dir = Path(args.work_dir).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)

    # 提前检查盲审所需总数，避免完成训练后才因样本不足失败。
    samples = scan_clean_stft_h5(args.data_dir)
    logger.info("Loaded %d clean UAV STFT samples", len(samples))
    review_counts = dict(REVIEW_ROLES)
    required_count = sum(review_counts.values())
    if len(samples) < required_count:
        raise ValueError(f"At least {required_count} samples are required for blinded review")

    dataset = CleanStftClusteringDataset(samples)
    try:
        # 训练打乱样本，推理顺序读取；推理结果按全局样本索引对齐。
        loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
        }
        train_loader = DataLoader(dataset, shuffle=True, **loader_kwargs)
        inference_loader = DataLoader(dataset, shuffle=False, **loader_kwargs)

        # 预训练自编码器；训练函数会恢复重建损失最低的权重。
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

        # 用预训练潜在向量评估候选簇数 K。
        embeddings = encode_dataset(autoencoder, inference_loader, device=device)
        selected_k, candidate_reports = select_cluster_count(embeddings, seed=SEED)
        logger.info("Selected K=%d", selected_k)

        # 用 KMeans 初始化可训练簇中心，初始化赋值不记录梯度。
        initial_kmeans = KMeans(n_clusters=selected_k, n_init=20, random_state=SEED).fit(embeddings)
        model = StftDcec(autoencoder, selected_k).to(device)
        with torch.no_grad():
            model.clustering.centers.copy_(
                torch.from_numpy(initial_kmeans.cluster_centers_).to(device=device, dtype=torch.float32)
            )

        # 联合优化重建与聚类目标，再推理全量样本的潜在向量和软分配概率。
        dcec_history = train_dcec(
            model,
            train_loader,
            device=device,
            epochs=args.dcec_epochs,
            learning_rate=1e-4,
            logger=logger,
        )
        final_embeddings, probabilities = infer_dcec(model, inference_loader, device=device)

        # 保存模型权重及预处理元数据，供后续复用。
        model_artifact = {
            "model_state_dict": model.state_dict(),
            "selected_k": selected_k,
            "latent_dim": autoencoder.latent_dim,
            "input_size": 256,
            "clip_range": (-5.0, 5.0),
            "scale_divisor": 5.0,
            "seed": SEED,
        }
        torch.save(model_artifact, work_dir / "stft_dcec.pt")

        # 导出全量聚类分配，生成语义映射、阈值校准和独立审核的盲审资料。
        write_cluster_assignments(work_dir / "assignments.csv", samples, probabilities)
        selections = select_cluster_review_indices(
            samples,
            final_embeddings,
            probabilities,
            model.clustering.centers.detach().cpu().numpy(),
        )
        write_cluster_review_artifacts(work_dir, selections, samples, probabilities)

        # 汇总各簇样本量、候选 K 评估和两阶段训练历史。
        report = {
            "sample_count": len(samples),
            "selected_k": selected_k,
            "final_cluster_counts": np.bincount(
                probabilities.argmax(axis=1), minlength=selected_k
            ).tolist(),
            "cluster_candidates": [asdict(item) for item in candidate_reports],
            "pretrain_history": pretrain_history,
            "dcec_history": dcec_history,
            "review_counts": review_counts,
        }
        with (work_dir / "training_report.json").open("w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)
        logger.info("Model artifact, assignments, and blinded review set saved to %s", work_dir)
    finally:
        # 正常结束或任一步骤失败时，都关闭数据集持有的 H5 句柄。
        dataset.close()


if __name__ == "__main__":
    main()

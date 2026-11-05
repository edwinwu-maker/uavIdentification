"""训练 STFT 自编码器，使用 HDBSCAN 聚类并生成复核材料。

Usage:
  python scripts/train_stft_dcec.py --data-dir ... --work-dir outputs/stft_hdbscan
  python scripts/train_stft_dcec.py --data-dir ... --work-dir ... --device cuda:0 --batch-size 32
  ~/Desktop/venv/bin/python scripts/train_stft_dcec.py --data-dir ...
  ~/Desktop/venv/bin/python scripts/train_stft_dcec.py --data-dir ... --work-dir ... --reuse-review-dir outputs/stft_hdbscan
  ~/Desktop/venv/bin/python scripts/train_stft_dcec.py --data-dir ... --force-retrain
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.stft_clustering import (
    CleanStftClusteringDataset,
    scan_clean_stft_h5,
)
from src.models.autoencoder import StftAutoencoder
from src.training.stft_autoencoder import (
    encode_dataset,
    pretrain_autoencoder,
)
from src.utils.cli import log_current_command
from src.utils.device import default_device
from src.utils.logger import logger
from src.utils.random import set_random_seed

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STFT autoencoder and create HDBSCAN review samples")
    parser.add_argument("--data-dir", required=True, help="Directory containing clean STFT .h5 files")
    parser.add_argument("--work-dir", default="outputs/stft_hdbscan")
    parser.add_argument("--reuse-review-dir", type=Path, default=None,
                        help="Reuse review samples and labels from this directory (default: work-dir)")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--force-retrain", action="store_true",
                        help="Train from scratch and overwrite any existing autoencoder checkpoint")
    parser.add_argument("--min-cluster-size", type=int, default=30,
                        help="HDBSCAN minimum cluster size (default: 30)")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="HDBSCAN density neighborhood size (default: 10)")
    parser.add_argument("--mask-threshold", type=float, default=1.5,
                        help="Raw z-score STFT mask uses values strictly above this threshold (default: 1.5)")
    parser.add_argument("--min-region-pixels", type=int, default=64,
                        help="Minimum 8-connected region area in raw pixels (default: 64)")
    parser.add_argument("--signal-area-threshold", type=float, default=0.001,
                        help="Minimum largest-region area / full image area for signal (default: 0.001)")
    parser.add_argument("--signal-span-threshold", type=float, default=0.09,
                        help="Minimum frequency or time span / axis length for signal (default: 0.09)")
    parser.add_argument("--membership-threshold", type=float, default=0.5,
                        help="Membership strength below this value forces uncertain (default: 0.5)")
    parser.add_argument("--outlier-threshold", type=float, default=0.9,
                        help="GLOSH score above this value forces uncertain (default: 0.9)")
    return parser.parse_args()


def _load_autoencoder(path: Path, model: StftAutoencoder) -> None:
    """加载兼容权重；失败时保留文件并提示显式重训。"""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "latent_dim": model.latent_dim,
            "input_size": 256,
            "clip_range": (-5.0, 5.0),
            "scale_divisor": 5.0,
        }
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be a dictionary")
        for name, value in expected.items():
            if checkpoint.get(name) != value:
                raise ValueError(f"incompatible {name}: expected {value}, got {checkpoint.get(name)}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as exc:
        # 读取、元数据或权重校验失败均不能隐式覆盖已有文件。
        raise ValueError(
            f"Cannot reuse autoencoder checkpoint {path}: {exc}. "
            "Use --force-retrain to train from scratch and overwrite it."
        ) from exc


def main() -> None:
    # 校验运行参数，避免无效配置进入数据准备和训练。
    args = parse_args()
    log_current_command(logger)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers must be non-negative")
    if args.pretrain_epochs <= 0 or args.patience <= 0:
        raise ValueError("epoch and patience values must be positive")

    from src.data.stft_analysis import StftAnalysisConfig, load_hdbscan, write_hdbscan_analysis

    analysis_config = StftAnalysisConfig(**{
        field.name: getattr(args, field.name) for field in fields(StftAnalysisConfig)
    })
    analysis_config.validate()
    load_hdbscan()

    # 固定随机种子，并初始化计算设备与输出目录。
    set_random_seed(SEED)
    device = torch.device(args.device)
    work_dir = Path(args.work_dir).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)

    # 提前检查密度聚类所需总数，避免完成训练后才因样本不足失败。
    samples = scan_clean_stft_h5(args.data_dir)
    logger.info("Loaded %d clean UAV STFT samples", len(samples))
    analysis_config.validate(len(samples))

    dataset = CleanStftClusteringDataset(samples)
    try:
        # 训练打乱样本，推理顺序读取；推理结果按全局样本索引对齐。
        loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
        }
        inference_loader = DataLoader(dataset, shuffle=False, **loader_kwargs)

        model_path = work_dir / "stft_autoencoder.pt"
        model_reused = model_path.exists() and not args.force_retrain
        autoencoder = StftAutoencoder()
        pretrain_history = []
        if model_reused:
            _load_autoencoder(model_path, autoencoder)
            autoencoder.to(device)
            logger.info("Reusing autoencoder checkpoint: %s", model_path)
        else:
            logger.info("Training autoencoder from scratch: %s", model_path)
            train_loader = DataLoader(dataset, shuffle=True, **loader_kwargs)
            pretrain_history = pretrain_autoencoder(
                autoencoder,
                train_loader,
                device=device,
                epochs=args.pretrain_epochs,
                learning_rate=1e-3,
                patience=args.patience,
                logger=logger,
            )

        # 直接对自编码器潜在向量做密度聚类，不再训练固定簇中心。
        embeddings = encode_dataset(autoencoder, inference_loader, device=device)

        if not model_reused:
            # 保存模型权重及预处理元数据，供后续复用。
            model_artifact = {
                "model_state_dict": autoencoder.state_dict(),
                "latent_dim": autoencoder.latent_dim,
                "input_size": 256,
                "clip_range": (-5.0, 5.0),
                "scale_divisor": 5.0,
                "seed": SEED,
            }
            torch.save(model_artifact, model_path)

        # 进入 HDBSCAN 聚类及结构分析
        analysis_report = write_hdbscan_analysis(
            work_dir, samples, embeddings, analysis_config, reuse_review_dir=args.reuse_review_dir
        )
        report = {
            "analysis_mode": "cae-hdbscan",
            "sample_count": len(samples),
            "final_cluster_counts": analysis_report["cluster_counts"],
            "pretrain_history": pretrain_history,
            "model_reused": model_reused,
            "review_counts": {"analysis": analysis_report["review_count"]},
            "hdbscan": analysis_report,
        }
        with (work_dir / "training_report.json").open("w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)
        logger.info("Model artifact, assignments, and blinded review set saved to %s", work_dir)
    finally:
        # 正常结束或任一步骤失败时，都关闭数据集持有的 H5 句柄。
        dataset.close()


if __name__ == "__main__":
    main()

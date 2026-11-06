"""默认使用轻量 Conv-BN-ReLU CNN，从随机初始化训练 STFT DeepCluster，生成簇复核与独立盲审材料。

Usage:
  ~/Desktop/venv/bin/python scripts/train_stft_deepcluster.py --data-dir /path/to/clean_stft_h5
  python scripts/train_stft_deepcluster.py --data-dir ... --device cuda:0 --work-dir outputs/stft_deepcluster
  python scripts/train_stft_deepcluster.py --data-dir ... --labels 4 6 8 --n-clusters 16 --pca-dim 32
  ~/Desktop/venv/bin/python scripts/train_stft_deepcluster.py --data-dir ... --input-mode rgb --work-dir outputs/stft_deepcluster_rgb
  ~/Desktop/venv/bin/python scripts/train_stft_deepcluster.py --data-dir ... --backbone resnet18 --batch-size 8 --work-dir outputs/stft_deepcluster_resnet18
  python scripts/train_stft_deepcluster.py --data-dir ... --reuse-review-dir outputs/stft_cae_kmeans
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.deepcluster_review import dataset_fingerprint, write_deepcluster_review
from src.data.deepcluster_visualization import write_deepcluster_tsne
from src.data.stft_clustering import CleanStftClusteringDataset, scan_clean_stft_h5
from src.models.stft_cnn import StftCNN
from src.models.stft_resnet import StftResNet18
from src.training.stft_deepcluster import train_deepcluster
from src.utils.cli import log_current_command
from src.utils.device import default_device, setup_device
from src.utils.logger import logger
from src.utils.files import read_csv_records
from src.utils.random import set_random_seed

DEFAULT_LABELS = (4, 6, 8, 10, 11, 12, 13, 14, 15, 16)


def parse_args():
    parser = argparse.ArgumentParser(description="Train STFT DeepCluster and create blinded review materials")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="outputs/stft_deepcluster")
    parser.add_argument("--backbone", choices=("cnn", "resnet18"), default="cnn",
                        help="Lightweight Conv-BN-ReLU CNN (default) or STFT-adapted ResNet-18")
    parser.add_argument("--input-mode", choices=("grayscale", "rgb"), default="grayscale",
                        help="Single-channel STFT (default) or jet RGB pixels after pooling")
    parser.add_argument("--labels", nargs="+", type=int, default=list(DEFAULT_LABELS))
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-clusters", type=int, default=32)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-per-class", type=int, default=30)
    parser.add_argument("--reuse-review-dir", type=Path, default=None,
                        help="Reuse fixed audit identities and sample reviews from an identical dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    log_current_command(logger)
    for name in ("batch_size", "epochs", "n_clusters", "pca_dim", "audit_per_class"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("num-workers must be non-negative and learning-rate must be positive")
    if args.n_clusters < 2 or not 1 <= args.pca_dim <= 64:
        raise ValueError("n-clusters must be >= 2 and pca-dim must be within 1..64")
    if not set(args.labels).issubset(set(range(2, 17))):
        raise ValueError("labels must be original UAV labels within 2..16")
    set_random_seed(args.seed)
    device = setup_device(args.device, logger)
    samples = [sample for sample in scan_clean_stft_h5(args.data_dir) if sample.label in args.labels]
    if len(samples) < args.n_clusters:
        raise ValueError("Selected samples must cover n-clusters")
    logger.info("Selected %d UAV samples from labels %s", len(samples), sorted(set(args.labels)))
    fingerprint, fingerprint_payload = dataset_fingerprint(samples, logger)
    work_dir = Path(args.work_dir).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset = CleanStftClusteringDataset(samples, input_mode=args.input_mode)
    try:
        loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                         "pin_memory": device.type == "cuda"}
        inference_loader = DataLoader(dataset, shuffle=False, **loader_kwargs)
        # 每次从随机 CNN 开始，首轮直接用初始特征生成聚类伪标签。
        model_class = StftResNet18 if args.backbone == "resnet18" else StftCNN
        model_type = "stft_resnet18" if args.backbone == "resnet18" else "stft_light_cnn"
        model = model_class(in_channels=3 if args.input_mode == "rgb" else 1)
        logger.info("DeepCluster backbone: %s", model_type)
        result = train_deepcluster(model, inference_loader, device=device, epochs=args.epochs,
                                   n_clusters=args.n_clusters, pca_dim=args.pca_dim,
                                   learning_rate=args.learning_rate, seed=args.seed, logger=logger)
        logger.info("Writing new clustering results; all cluster decisions will reset to unconfirmed quarantine")
        review = write_deepcluster_review(work_dir, samples, result, fingerprint=fingerprint,
                                         seed=args.seed, audit_per_class=args.audit_per_class,
                                         reuse_review_dir=args.reuse_review_dir)
        parameters = vars(args) | {
            "reuse_review_dir": str(args.reuse_review_dir) if args.reuse_review_dir else None,
        }
        # 用纯张量保存 sklearn 推理参数，避免加载 checkpoint 时依赖 pickle 对象。
        torch.save({"model_type": model_type,
                    "input_mode": args.input_mode, "in_channels": model.in_channels,
                    "colormap": "jet" if args.input_mode == "rgb" else None,
                    "model_state_dict": result.model.state_dict(), "latent_dim": model.latent_dim,
                    "input_size": 256, "clip_range": (-5.0, 5.0), "scale_divisor": 5.0,
                    "dataset_fingerprint": fingerprint, "fingerprint_payload": fingerprint_payload,
                    "pca": {"mean": torch.from_numpy(result.pca.mean_),
                            "components": torch.from_numpy(result.pca.components_),
                            "explained_variance": torch.from_numpy(result.pca.explained_variance_),
                            "whiten": True},
                    "cluster_centers": torch.from_numpy(result.kmeans.cluster_centers_),
                    "parameters": parameters},
                   work_dir / "deepcluster.pt")
        tsne = write_deepcluster_tsne(
            work_dir, result.features, read_csv_records(work_dir / "deepcluster_assignments.csv"),
            seed=args.seed,
        )
        report = {"analysis_mode": "deepcluster", "model_type": model_type,
                  "sample_count": len(samples),
                  "dataset_fingerprint": fingerprint, "fingerprint_payload": fingerprint_payload,
                  "parameters": parameters,
                  "history": result.history, "review": review, "tsne": tsne}
        with (work_dir / "training_report.json").open("w", encoding="utf-8") as destination:
            json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
        logger.info("DeepCluster checkpoint and review materials saved to %s", work_dir)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()

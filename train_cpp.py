"""
Single-GPU training for ResNet on pre-computed CPP/FAM .h5 matrices.

Each .h5 file contains:
  /cpp    (N, 2, alpha_bins, f_bins) float32
  /labels (N,) int64

Usage:
  CUDA_VISIBLE_DEVICES=1 python src/train_cpp.py --batch-size 64
  python src/train_cpp.py --data-dir ~/Desktop/dataset/droneRFa/cpp_h5
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.cpp_dataset import CppDataset
from src.data.splits import split_dataset
from src.models.resnet import DroneRFaResNet18
from src.training.trainer import train_model
from src.utils.device import default_device
from src.utils.logger import logger

NUM_CLASSES = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
PATIENCE = 10
CHECKPOINT_NAME = "best_cpp_model.pth"


def _default_data_dir() -> str:
    if os.name == "nt":
        return "E:/dataSet/DroneRFa/cpp_h5"
    if sys.platform == "darwin":
        return os.path.expanduser("~/Desktop/dataset/droneRFa/cpp_h5")
    return "/mnt/data/wurixin/DroneRFa/cpp_h5"


def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU training on pre-computed CPP/FAM .h5 matrices")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing CPP .h5 files")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = main process only)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--device", type=str, default=default_device(),
                        help="Device, e.g. 'cuda:0', 'cuda:1', 'mps', 'cpu'")
    return parser.parse_args()


def train(args):
    if args.data_dir is None:
        args.data_dir = _default_data_dir()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    logger.info("Single-GPU mode")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    dataset = CppDataset(args.data_dir)
    try:
        logger.info("Loaded %d CPP samples from %d .h5 files in %s",
                    len(dataset), len(set(s[0] for s in dataset.index)), args.data_dir)

        train_ds, val_ds, _test_ds = split_dataset(
            dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=42,
        )
        train_size, val_size, test_size = len(train_ds), len(val_ds), len(_test_ds)
        logger.info("Split - train: %d, val: %d, test: %d", train_size, val_size, test_size)

        loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
            loader_kwargs["persistent_workers"] = True

        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

        model = DroneRFaResNet18(num_classes=NUM_CLASSES).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            checkpoint_name=CHECKPOINT_NAME,
            logger=logger,
            train_desc="Training Epochs",
            eval_desc="Evaluating",
        )
        logger.info("Training complete. Best val_acc: %.4f", result.best_val_acc)
    finally:
        dataset.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)

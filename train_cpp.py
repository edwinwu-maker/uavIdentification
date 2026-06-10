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
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.cpp_dataset import CppDataset
from src.models.resnet import DroneRFaResNet18
from src.utils.logger import logger

NUM_CLASSES = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
PATIENCE = 10
CHECKPOINT_NAME = "best_cpp_model.pth"


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_data_dir() -> str:
    if os.name == "nt":
        return "E:/dataSet/DroneRFa/cpp_h5"
    if sys.platform == "darwin":
        return os.path.expanduser("~/Desktop/dataset/droneRFa/cpp_h5")
    return "/mnt/data/wurixin/DroneRFa/cpp_h5"


def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss_sum = 0.0
    count = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in tqdm(
            dataloader, desc="Evaluating", leave=False, unit="batch",
        ):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            loss_sum += loss.item() * batch_size
            count += batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    return loss_sum / count, correct / count


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
    parser.add_argument("--device", type=str, default=_default_device(),
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

        total_size = len(dataset)
        train_size = int(total_size * TRAIN_RATIO)
        val_size = int(total_size * VAL_RATIO)
        test_size = total_size - train_size - val_size
        train_ds, val_ds, _test_ds = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
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

        checkpoint_dir = os.path.join(str(Path(__file__).resolve().parent), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_val_acc = 0.0
        patience_counter = 0

        epoch_bar = tqdm(total=args.epochs, desc="Training Epochs", unit="epoch")
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            batch_bar = tqdm(
                train_loader, total=len(train_loader),
                desc=f"Epoch {epoch}", leave=False, unit="batch",
            )
            for inputs, labels in batch_bar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_size = inputs.size(0)
                train_loss_sum += loss.item() * batch_size
                train_count += batch_size
                batch_bar.set_postfix({
                    "batch_loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                })

            batch_bar.close()
            train_loss = train_loss_sum / train_count
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            epoch_bar.update(1)
            logger.info(
                "Epoch %3d | train_loss: %.4f | val_loss: %.4f | val_acc: %.4f",
                epoch, train_loss, val_loss, val_acc,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, CHECKPOINT_NAME))
                logger.info("  -> saved best CPP model (val_acc=%.4f)", val_acc)
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        epoch_bar.close()
        logger.info("Training complete. Best val_acc: %.4f", best_val_acc)
    finally:
        dataset.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)

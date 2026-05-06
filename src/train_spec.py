"""
Train ResNet on pre-computed spectrograms (fast, no CPU STFT bottleneck).

Usage:
  python src/train_spec.py --cache E:/dataSet/DroneRFa/spectrogram_cache --gpus 0,1,2,3,4
"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.spectrogram_dataset import SpectrogramDataset
from models.resnet import DroneRFaResNet18
from utils.logger import logger

# ── Paper hyperparameters (Section 4.3) ──
NUM_CLASSES = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
PATIENCE = 10


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = (preds == labels).mean()
    return avg_loss, acc, preds, labels


def compute_metrics(preds, labels):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train on pre-computed spectrograms")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs, e.g. '0,1,2'")
    parser.add_argument("--cache", type=str, required=True,
                        help="Path to spectrogram cache directory (.npy files)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default: min(8, cpu_count))")
    return parser.parse_args()


def train(args):
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info("CUDA_VISIBLE_DEVICES set to: %s", args.gpus)

    # ── Dataset (no transform needed — spectrograms are pre-computed) ──
    dataset = SpectrogramDataset(args.cache)
    logger.info("Loaded %d pre-computed spectrograms from %s", len(dataset), args.cache)

    total = len(dataset)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    test_size = total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info("Split — train: %d, val: %d, test: %d", train_size, val_size, test_size)

    num_workers = args.num_workers if args.num_workers is not None else min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── Model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU count: %d", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            logger.info("  GPU %d: %s", i, torch.cuda.get_device_name(i))

    model = DroneRFaResNet18(num_classes=NUM_CLASSES)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info("Using DataParallel across %d GPUs", torch.cuda.device_count())
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ── Training ──
    checkpoint_dir = os.path.join(str(Path(__file__).parent.parent), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, 200):
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

            if batch_idx % 50 == 0:
                logger.info(
                    "Epoch %3d | Batch %3d | batch_loss: %.4f",
                    epoch, batch_idx, loss.item(),
                )

        train_loss /= len(train_ds)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        logger.info(
            "Epoch %3d | train_loss: %.4f | val_loss: %.4f | val_acc: %.4f",
            epoch, train_loss, val_loss, val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            logger.info("  -> saved best model (val_acc=%.4f)", val_acc)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # ── Test ──
    logger.info("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pth")))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(preds, labels)

    logger.info("=" * 55)
    logger.info("Test Results:")
    logger.info("  Accuracy:  %.4f", metrics["accuracy"])
    logger.info("  Precision: %.4f", metrics["precision"])
    logger.info("  Recall:    %.4f", metrics["recall"])
    logger.info("  F1-Score:  %.4f", metrics["f1"])
    logger.info("  Test Loss: %.4f", test_loss)
    logger.info("=" * 55)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    np.save(os.path.join(checkpoint_dir, "confusion_matrix.npy"), cm)
    logger.info("Confusion matrix saved to checkpoints/confusion_matrix.npy")


if __name__ == "__main__":
    train(parse_args())

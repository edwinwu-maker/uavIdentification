import glob
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.droneRFa_dataset import DroneRFaDataset
from data.transforms import IQToSpectrogram
from models.resnet import DroneRFaResNet18
from utils.logger import logger

# ── Paper hyperparameters (Section 4.3) ──
SAMPLE_LENGTH = 1_000_000   # 1M IQ points ≈ 10 ms
NUM_CLASSES = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
PATIENCE = 10  # early stopping patience

# ── Evaluation function ──
def evaluate(model, dataloader, criterion, device):
    model.eval()            # 推理模式（Dropout关闭、BN固定）
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():   # 禁用梯度计算，省显存、提速
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


def train():
    # ── Data paths ──
    if os.name == "nt":
        DATA_DIR = "E:/dataSet/DroneRFa"
    else:
        DATA_DIR = "/mnt/data/wurixin/DroneRFa"

    mat_files = sorted(glob.glob(f"{DATA_DIR}/*.mat"))
    logger.info("Found %d .mat files", len(mat_files))

    # ── Dataset & split ──
    dataset = DroneRFaDataset(
        mat_files,
        sample_length=SAMPLE_LENGTH,
        transform=IQToSpectrogram(),
    )
    logger.info("Total samples: %d", len(dataset))

    total = len(dataset)
    train_size = int(total * TRAIN_RATIO)       # 60% for training
    val_size = int(total * VAL_RATIO)           # 20% for validation
    test_size = total - train_size - val_size   # remaining 20% for testing
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info("Split — train: %d, val: %d, test: %d", train_size, val_size, test_size)

    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── Model, optimizer, loss ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = DroneRFaResNet18(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──
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
        # Evaluate on validation set
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        logger.info(
            "Epoch %3d | train_loss: %.4f | val_loss: %.4f | val_acc: %.4f",
            epoch, train_loss, val_loss, val_acc,
        )

        # Save best
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

    # ── Test evaluation ──
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

    # ── Confusion matrix ──
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    np.save(os.path.join(checkpoint_dir, "confusion_matrix.npy"), cm)
    logger.info("Confusion matrix saved to checkpoints/confusion_matrix.npy")


if __name__ == "__main__":
    train()

"""
Load a trained CPP model and evaluate on the test set (single GPU).

Usage:
  python src/test_cpp.py --data-dir /path/to/cpp_h5
  python src/test_cpp.py --data-dir /path/to/cpp_h5 --gpu 1
  python src/test_cpp.py --data-dir /path/to/cpp_h5 --model-path outputs/checkpoints/best_cpp_model.pth
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.data.cpp_dataset import CppDataset
from src.models.resnet import DroneRFaResNet18
from train_cpp import NUM_CLASSES, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, CHECKPOINT_NAME, _default_data_dir
from src.utils.logger import logger
from src.utils.paths import checkpoint_dir, figures_dir, metrics_dir

CONFUSION_MATRIX_NAME = "cpp_confusion_matrix.npy"
CONFUSION_MATRIX_IMAGE_NAME = "cpp_confusion_matrix.png"
__test__ = False


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_metrics(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def save_confusion_matrix_image(cm, save_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    class_labels = np.arange(cm.shape[0])
    ax.set(
        title="CPP Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="True Label",
        xticks=class_labels,
        yticks=class_labels,
    )

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss_sum = 0.0
    count = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(
            dataloader, desc="Testing", leave=False, unit="batch",
        ):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * inputs.size(0)
            count += inputs.size(0)
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    avg_loss = loss_sum / count
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = (preds == labels).mean()
    return avg_loss, acc, preds, labels


def parse_args():
    parser = argparse.ArgumentParser(description="Test on pre-computed CPP/FAM .h5 matrices (single GPU)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing CPP .h5 files")
    parser.add_argument("--model-path", type=str, default=None,
                        help=f"Path to model checkpoint (default: outputs/checkpoints/{CHECKPOINT_NAME})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = main process only)")
    parser.add_argument("--device", type=str, default=_default_device(),
                        help="Device, e.g. 'cuda:0', 'cuda:1', 'mps', 'cpu'")
    parser.add_argument("--cm-image-path", type=str, default=None,
                        help=f"Path to save confusion matrix image (default: outputs/figures/{CONFUSION_MATRIX_IMAGE_NAME})")
    return parser.parse_args()


def test(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name())

    if args.data_dir is None:
        args.data_dir = _default_data_dir()

    dataset = CppDataset(args.data_dir)
    try:
        logger.info("Loaded %d CPP samples from %d .h5 files in %s",
                    len(dataset), len(set(s[0] for s in dataset.index)), args.data_dir)

        total_size = len(dataset)
        train_size = int(total_size * TRAIN_RATIO)
        val_size = int(total_size * VAL_RATIO)
        test_size = total_size - train_size - val_size
        _, _, test_ds = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        logger.info("Test set size: %d", test_size)

        loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
            loader_kwargs["persistent_workers"] = True

        test_loader = DataLoader(test_ds, **loader_kwargs)

        model = DroneRFaResNet18(num_classes=NUM_CLASSES)
        model = model.to(device)

        if args.model_path is None:
            args.model_path = checkpoint_dir() / CHECKPOINT_NAME
        logger.info("Loading model from %s", args.model_path)
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)

        criterion = nn.CrossEntropyLoss()

        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

        metrics = compute_metrics(preds, labels)
        logger.info("=" * 55)
        logger.info("CPP Test Results:")
        logger.info("  Accuracy:  %.4f", metrics["accuracy"])
        logger.info("  Precision: %.4f", metrics["precision"])
        logger.info("  Recall:    %.4f", metrics["recall"])
        logger.info("  F1-Score:  %.4f", metrics["f1"])
        logger.info("  Test Loss: %.4f", test_loss)
        logger.info("=" * 55)

        cm = confusion_matrix(labels, preds)
        metric_path = metrics_dir()
        os.makedirs(metric_path, exist_ok=True)
        cm_path = metric_path / CONFUSION_MATRIX_NAME
        np.save(cm_path, cm)
        logger.info("Confusion matrix saved to %s", cm_path)
        if args.cm_image_path is None:
            args.cm_image_path = figures_dir() / CONFUSION_MATRIX_IMAGE_NAME
        args.cm_image_path = Path(args.cm_image_path)
        os.makedirs(args.cm_image_path.parent, exist_ok=True)
        save_confusion_matrix_image(cm, args.cm_image_path)
        logger.info("Confusion matrix image saved to %s", args.cm_image_path)
    finally:
        dataset.close()


if __name__ == "__main__":
    test(parse_args())

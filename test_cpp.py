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
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from src.data.cpp_dataset import CppDataset
from src.data.splits import split_dataset
from src.models.resnet import DroneRFaResNet18
from src.training.checkpoint import default_checkpoint_path, load_checkpoint
from src.training.evaluator import evaluate_with_predictions
from src.training.metrics import compute_metrics, save_confusion_matrix_image
from src.utils.config import expand_path, get_config_value, load_config
from train_cpp import NUM_CLASSES, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, CHECKPOINT_NAME, _default_data_dir
from src.utils.device import default_device
from src.utils.logger import logger
from src.utils.paths import figures_dir, metrics_dir

CONFUSION_MATRIX_NAME = "cpp_confusion_matrix.npy"
CONFUSION_MATRIX_IMAGE_NAME = "cpp_confusion_matrix.png"
__test__ = False


def build_parser(config=None):
    parser = argparse.ArgumentParser(description="Test on pre-computed CPP/FAM .h5 matrices (single GPU)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing CPP .h5 files")
    parser.add_argument(
        "--model-path",
        type=str,
        default=default_checkpoint_path(CHECKPOINT_NAME),
        help=f"Path to model checkpoint (default: outputs/checkpoints/{CHECKPOINT_NAME})",
    )
    parser.add_argument("--batch-size", type=int, default=get_config_value(config, "batch_size", BATCH_SIZE))
    parser.add_argument("--num-workers", type=int, default=get_config_value(config, "num_workers", 0),
                        help="DataLoader workers (0 = main process only)")
    parser.add_argument("--device", type=str, default=default_device(),
                        help="Device, e.g. 'cuda:0', 'cuda:1', 'mps', 'cpu'")
    parser.add_argument(
        "--cm-image-path",
        type=str,
        default=figures_dir() / CONFUSION_MATRIX_IMAGE_NAME,
        help=f"Path to save confusion matrix image (default: outputs/figures/{CONFUSION_MATRIX_IMAGE_NAME})",
    )
    return parser


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args(argv)
    config = load_config(config_args.config)
    parser = build_parser(config)
    args = parser.parse_args(remaining)
    args.config = config_args.config
    if args.data_dir is not None:
        args.data_dir = expand_path(args.data_dir)
    if args.model_path is not None:
        args.model_path = expand_path(args.model_path)
    args.cm_image_path = expand_path(args.cm_image_path)
    return args


def test(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name())

    if args.data_dir is None:
        args.data_dir = _default_data_dir()
    else:
        args.data_dir = expand_path(args.data_dir)

    dataset = CppDataset(args.data_dir)
    try:
        logger.info("Loaded %d CPP samples from %d .h5 files in %s",
                    len(dataset), len(set(s[0] for s in dataset.index)), args.data_dir)

        _, _, test_ds = split_dataset(
            dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=42,
        )
        test_size = len(test_ds)
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
            args.model_path = default_checkpoint_path(CHECKPOINT_NAME)
        logger.info("Loading model from %s", args.model_path)
        state_dict = load_checkpoint(args.model_path, map_location=device)
        model.load_state_dict(state_dict)

        criterion = nn.CrossEntropyLoss()

        test_loss, test_acc, preds, labels = evaluate_with_predictions(
            model, test_loader, criterion, device
        )

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
        args.cm_image_path = Path(args.cm_image_path)
        os.makedirs(args.cm_image_path.parent, exist_ok=True)
        save_confusion_matrix_image(cm, args.cm_image_path)
        logger.info("Confusion matrix image saved to %s", args.cm_image_path)
    finally:
        dataset.close()


if __name__ == "__main__":
    test(parse_args())

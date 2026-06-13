"""Evaluate ResNet on pre-computed DroneRFa features.

Usage:
  python scripts/evaluate.py --feature stft --data-dir ~/Desktop/dataset/droneRFa/stft_h5 --model-path outputs/checkpoints/best_stft_model.pth
  python scripts/evaluate.py --feature cpp --data-dir ~/Desktop/dataset/droneRFa/cpp_h5 --model-path outputs/checkpoints/best_cpp_model.pth
  python scripts/evaluate.py --config configs/stft.yaml
  python scripts/evaluate.py --config configs/cpp.yaml --device mps

Compatibility wrappers:
  python test_stft.py --data-dir ~/Desktop/dataset/droneRFa/stft_h5
  python test_cpp.py --data-dir ~/Desktop/dataset/droneRFa/cpp_h5
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import split_dataset
from src.models.resnet import DroneRFaResNet18
from src.training.checkpoint import load_checkpoint
from src.training.evaluator import evaluate_with_predictions
from src.training.metrics import compute_metrics, save_confusion_matrix_image
from src.utils.config import expand_path, get_config_value, load_config
from src.utils.device import default_device
from src.utils.feature_specs import get_feature_spec
from src.utils.logger import logger
from src.utils.paths import figures_dir, metrics_dir

NUM_CLASSES = 25
BATCH_SIZE = 64
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
SPLIT_SEED = 42


def build_parser(config=None):
    parser = argparse.ArgumentParser(description="Evaluate ResNet on pre-computed DroneRFa features")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--feature",
        type=str,
        default=get_config_value(config, "feature", "stft"),
        choices=("stft", "cpp"),
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing feature .h5 files")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=get_config_value(config, "batch_size", BATCH_SIZE))
    parser.add_argument("--num-workers", type=int, default=get_config_value(config, "num_workers", 0),
                        help="DataLoader workers (0 = main process only)")
    parser.add_argument("--device", type=str, default=default_device(),
                        help="Device, e.g. 'cuda:0', 'cuda:1', 'mps', 'cpu'")
    parser.add_argument("--cm-image-path", type=str, default=None, help="Path to save confusion matrix image")
    return parser


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args(argv)
    config = load_config(config_args.config)
    parser = build_parser(config)
    args = parser.parse_args(remaining)
    args.config = config_args.config

    spec = get_feature_spec(args.feature)
    args.data_dir = expand_path(args.data_dir) if args.data_dir is not None else spec.default_data_dir
    args.model_path = expand_path(args.model_path) if args.model_path is not None else spec.checkpoint_name
    args.cm_image_path = (
        expand_path(args.cm_image_path)
        if args.cm_image_path is not None
        else str(figures_dir() / spec.cm_image_name)
    )
    return args


def evaluate(args):
    spec = get_feature_spec(args.feature)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    dataset = spec.dataset_class(args.data_dir)
    try:
        logger.info("Loaded %d %s samples from %d .h5 files in %s",
                    len(dataset), spec.name, len(set(s[0] for s in dataset.index)), args.data_dir)

        _, _, test_ds = split_dataset(
            dataset,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            seed=SPLIT_SEED,
        )
        logger.info("Test set size: %d", len(test_ds))

        loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
            loader_kwargs["persistent_workers"] = True

        test_loader = DataLoader(test_ds, **loader_kwargs)

        model = DroneRFaResNet18(num_classes=NUM_CLASSES).to(device)
        logger.info("Loading model from %s", args.model_path)
        state_dict = load_checkpoint(args.model_path, map_location=device)
        model.load_state_dict(state_dict)

        criterion = nn.CrossEntropyLoss()
        test_loss, _test_acc, preds, labels = evaluate_with_predictions(
            model, test_loader, criterion, device
        )

        metrics = compute_metrics(preds, labels)
        logger.info("=" * 55)
        logger.info("%s Test Results:", spec.name.upper())
        logger.info("  Accuracy:  %.4f", metrics["accuracy"])
        logger.info("  Precision: %.4f", metrics["precision"])
        logger.info("  Recall:    %.4f", metrics["recall"])
        logger.info("  F1-Score:  %.4f", metrics["f1"])
        logger.info("  Test Loss: %.4f", test_loss)
        logger.info("=" * 55)

        cm = confusion_matrix(labels, preds)
        metric_path = metrics_dir()
        os.makedirs(metric_path, exist_ok=True)
        cm_path = metric_path / spec.cm_array_name
        np.save(cm_path, cm)
        logger.info("Confusion matrix saved to %s", cm_path)

        cm_image_path = Path(args.cm_image_path)
        os.makedirs(cm_image_path.parent, exist_ok=True)
        save_confusion_matrix_image(cm, cm_image_path)
        logger.info("Confusion matrix image saved to %s", cm_image_path)
        return metrics
    finally:
        dataset.close()


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()

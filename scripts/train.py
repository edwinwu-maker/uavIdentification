"""Train ResNet on pre-computed DroneRFa features.

Usage:
  python scripts/train.py --feature stft --data-dir ~/Desktop/dataset/DroneRFa_stft_h5 --batch-size 64
  python scripts/train.py --feature cpp --data-dir ~/Desktop/dataset/DroneRFa_cpp_h5 --model resnet18-small-stem --batch-size 64
  python scripts/train.py --feature cpp --data-dir ... --files-per-class 12 --split-manifest outputs/splits/random12_seed42.csv

"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import prepare_training_split
from src.data.class_filter import RemappedSubset, add_exclusion_suffix, build_class_mapping
from src.models.resnet import NUM_CLASSES, build_model
from src.training.trainer import train_model

from src.utils.cli import log_current_command
from src.utils.device import default_device
from src.utils.feature_specs import get_feature_spec
from src.utils.logger import logger

BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
PATIENCE = 10
SPLIT_SEED = 42
DEFAULT_MODEL_BY_FEATURE = {
    "stft": "resnet18",
    "cpp": "resnet18-small-stem",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Train ResNet on pre-computed DroneRFa features")
    parser.add_argument("--feature", type=str, default="stft", choices=("stft", "cpp"))
    parser.add_argument("--model", type=str, default=None, choices=("resnet18", "resnet18-small-stem"),
                        help="Model architecture (default: resnet18 for STFT, resnet18-small-stem for CPP)")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing feature .h5 files")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = main process only)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--device", type=str, default=default_device(),
                        help="Device, e.g. 'cuda:0', 'cuda:1', 'mps', 'cpu'")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Checkpoint path or name")
    parser.add_argument("--files-per-class", type=int, default=None,
                        help="Randomly select this many source files per class before file-level splitting")
    parser.add_argument("--split-manifest", type=str, required=True,
                        help="Shared CSV manifest for file-level train/val/test splitting")
    parser.add_argument("--exclude-labels", type=int, nargs="*", default=None,
                        help="Original class labels to exclude, e.g. --exclude-labels 10 11")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    spec = get_feature_spec(args.feature)
    args.model = args.model if args.model is not None else DEFAULT_MODEL_BY_FEATURE[args.feature]
    args.data_dir = os.path.expanduser(args.data_dir) if args.data_dir is not None else spec.default_data_dir
    args.checkpoint_path = (
        os.path.expanduser(args.checkpoint_path)
        if args.checkpoint_path is not None
        else add_exclusion_suffix(spec.checkpoint_name, args.exclude_labels)
    )
    args.split_manifest = os.path.expanduser(args.split_manifest)
    return args


def train(args):
    spec = get_feature_spec(args.feature)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    logger.info("Single-GPU mode")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    dataset = spec.dataset_class(args.data_dir)
    try:
        logger.info("Loaded %d %s samples from %d .h5 files in %s",
                    len(dataset), spec.name, len(set(s[0] for s in dataset.index)), args.data_dir)

        manifest_exists = Path(args.split_manifest).is_file()
        train_ds, val_ds, _test_ds = prepare_training_split(
            dataset,
            manifest_path=args.split_manifest,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            seed=SPLIT_SEED,
            files_per_class=args.files_per_class,
        )
        mapping = build_class_mapping(NUM_CLASSES, args.exclude_labels)
        train_ds = RemappedSubset(train_ds, mapping)
        val_ds = RemappedSubset(val_ds, mapping)
        _test_ds = RemappedSubset(_test_ds, mapping)
        logger.info("Class mapping (model -> original): %s", mapping.original_labels)
        logger.info(
            "Split manifest %s: %s",
            "reused" if manifest_exists else "created",
            args.split_manifest,
        )
        train_size, val_size, test_size = len(train_ds), len(val_ds), len(_test_ds)
        logger.info("Split - train: %d, val: %d, test: %d", train_size, val_size, test_size)
        split_file_counts = [
            len({dataset.index[index][0] for index in subset.indices})
            for subset in (train_ds, val_ds, _test_ds)
        ]
        logger.info(
            "Selected %d files - train: %d, val: %d, test: %d",
            sum(split_file_counts),
            *split_file_counts,
        )

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

        model = build_model(args.model, num_classes=mapping.num_classes).to(device)
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
            checkpoint_name=args.checkpoint_path,
            logger=logger,
            train_desc="Training Epochs",
            eval_desc="Evaluating",
        )
        logger.info("Training complete. Best val_acc: %.4f", result.best_val_acc)
        return result
    finally:
        dataset.close()


def main():
    args = parse_args()
    log_current_command(logger)
    train(args)


if __name__ == "__main__":
    main()

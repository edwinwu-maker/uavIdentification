# Src Layout Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the existing `src.*` import style, restore a verifiable test suite, and reduce duplicated STFT/CPP training, evaluation, preprocessing, and visualization code.

**Architecture:** Treat `src` as the canonical package namespace for this repository. Put shared feature metadata in one small module, use unified script entry points for train/evaluate, and keep legacy STFT/CPP entry scripts as thin wrappers during migration.

**Tech Stack:** Python, PyTorch, torchvision, h5py, NumPy, scikit-learn, matplotlib, pytest.

---

## Assumptions And Success Criteria

- The project will continue to use imports such as `from src.data.stft_dataset import StftDataset`.
- No `src/drone_rfa` package or `drone_rfa.*` alias will be introduced.
- Existing user-facing commands may remain available through thin wrappers while new unified scripts are added.
- Success means `~/Desktop/venv/bin/python -m pytest -q` completes collection and passes the focused tests touched by each task.

## File Structure

- Modify: `tests/*.py`
  - Convert test imports from `drone_rfa.*` to `src.*`.
  - Remove alias-only tests that contradict the chosen `src.*` direction.
- Create: `src/utils/feature_specs.py`
  - Own all STFT/CPP differences used by train/evaluate scripts.
- Create: `scripts/train.py`
  - Unified training CLI with `--feature stft|cpp`.
- Create: `scripts/evaluate.py`
  - Unified evaluation CLI with `--feature stft|cpp`.
- Modify: `train_stft.py`, `train_cpp.py`
  - Reduce to compatibility wrappers around `scripts.train`.
- Modify: `test_stft.py`, `test_cpp.py`
  - Reduce to compatibility wrappers around `scripts.evaluate`.
- Create: `src/data/drone_rfa_io.py`
  - Shared DroneRFa label parsing, default data directory, sample counting, and IQ reading helpers.
- Modify: `scripts/precompute_stft_h5.py`, `scripts/precompute_cpp_h5.py`
  - Use `src.data.drone_rfa_io` for shared IO concerns.
- Modify: `scripts/generate_stft_png.py`
  - Bring CLI behavior in line with `scripts/generate_cpp_png.py`.
- Modify: `readme.md`
  - Document canonical `src.*` direction and new unified commands.

---

### Task 1: Align Tests With `src.*`

**Files:**
- Modify: `tests/test_checkpoint.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_device.py`
- Modify: `tests/test_evaluator.py`
- Modify: `tests/test_h5_dataset.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_preprocessing.py`
- Modify: `tests/test_splits.py`
- Modify: `tests/test_trainer.py`
- Modify: `tests/test_visualization.py`
- Delete: `tests/test_package_alias.py`

- [ ] **Step 1: Replace package imports**

Replace imports like:

```python
from drone_rfa.training.evaluator import evaluate, evaluate_with_predictions
```

with:

```python
from src.training.evaluator import evaluate, evaluate_with_predictions
```

Replace monkeypatch targets like:

```python
monkeypatch.setattr("drone_rfa.preprocessing.cpp.compute_fam_grid_segmented", fake_compute_fam_grid_segmented)
```

with the actual module path used after Task 5:

```python
monkeypatch.setattr("scripts.precompute_cpp_h5.compute_fam_grid_segmented", fake_compute_fam_grid_segmented)
```

- [ ] **Step 2: Remove alias-only test**

Delete `tests/test_package_alias.py`. It verifies `drone_rfa.*` compatibility, which is explicitly out of scope.

- [ ] **Step 3: Run collection**

Run:

```bash
~/Desktop/venv/bin/python -m pytest --collect-only -q
```

Expected: collection completes without `ModuleNotFoundError: No module named 'drone_rfa'`.

- [ ] **Step 4: Commit**

```bash
git add tests
git commit -m "test: align imports with src package layout"
```

---

### Task 2: Add Feature Specifications

**Files:**
- Create: `src/utils/feature_specs.py`
- Test: `tests/test_feature_specs.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_feature_specs.py`:

```python
import os
import sys

from src.data.cpp_dataset import CppDataset
from src.data.stft_dataset import StftDataset
from src.utils.feature_specs import get_feature_spec


def test_stft_feature_spec_contains_expected_defaults():
    spec = get_feature_spec("stft")

    assert spec.name == "stft"
    assert spec.dataset_class is StftDataset
    assert spec.feature_key == "stft"
    assert spec.checkpoint_name == "best_stft_model.pth"
    assert spec.cm_array_name == "stft_confusion_matrix.npy"
    assert spec.cm_image_name == "stft_confusion_matrix.png"


def test_cpp_feature_spec_contains_expected_defaults():
    spec = get_feature_spec("cpp")

    assert spec.name == "cpp"
    assert spec.dataset_class is CppDataset
    assert spec.feature_key == "cpp"
    assert spec.checkpoint_name == "best_cpp_model.pth"
    assert spec.cm_array_name == "cpp_confusion_matrix.npy"
    assert spec.cm_image_name == "cpp_confusion_matrix.png"


def test_feature_spec_rejects_unknown_feature():
    try:
        get_feature_spec("unknown")
    except ValueError as exc:
        assert "Unsupported feature" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
~/Desktop/venv/bin/python -m pytest tests/test_feature_specs.py -q
```

Expected: FAIL because `src.utils.feature_specs` does not exist.

- [ ] **Step 3: Implement feature specs**

Create `src/utils/feature_specs.py`:

```python
import os
import sys
from dataclasses import dataclass
from typing import Type

from torch.utils.data import Dataset

from src.data.cpp_dataset import CppDataset
from src.data.stft_dataset import StftDataset


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dataset_class: Type[Dataset]
    feature_key: str
    default_data_dir: str
    checkpoint_name: str
    cm_array_name: str
    cm_image_name: str
    train_description: str
    eval_description: str


def _base_data_dir() -> str:
    if os.name == "nt":
        return "E:/dataSet/DroneRFa"
    if sys.platform == "darwin":
        return os.path.expanduser("~/Desktop/dataset/droneRFa")
    return "/mnt/data/wurixin/DroneRFa"


def _build_specs() -> dict[str, FeatureSpec]:
    base_dir = _base_data_dir()
    return {
        "stft": FeatureSpec(
            name="stft",
            dataset_class=StftDataset,
            feature_key="stft",
            default_data_dir=os.path.join(base_dir, "stft_h5"),
            checkpoint_name="best_stft_model.pth",
            cm_array_name="stft_confusion_matrix.npy",
            cm_image_name="stft_confusion_matrix.png",
            train_description="Single-GPU training on pre-computed .h5 stfts",
            eval_description="Test on pre-computed .h5 stfts",
        ),
        "cpp": FeatureSpec(
            name="cpp",
            dataset_class=CppDataset,
            feature_key="cpp",
            default_data_dir=os.path.join(base_dir, "cpp_h5"),
            checkpoint_name="best_cpp_model.pth",
            cm_array_name="cpp_confusion_matrix.npy",
            cm_image_name="cpp_confusion_matrix.png",
            train_description="Single-GPU training on pre-computed CPP/FAM .h5 matrices",
            eval_description="Test on pre-computed CPP/FAM .h5 matrices",
        ),
    }


def get_feature_spec(feature: str) -> FeatureSpec:
    specs = _build_specs()
    try:
        return specs[feature]
    except KeyError as exc:
        supported = ", ".join(sorted(specs))
        raise ValueError(f"Unsupported feature {feature!r}; choose one of: {supported}") from exc
```

- [ ] **Step 4: Run tests**

```bash
~/Desktop/venv/bin/python -m pytest tests/test_feature_specs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/feature_specs.py tests/test_feature_specs.py
git commit -m "feat: add shared feature specifications"
```

---

### Task 3: Add Unified Training CLI

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/train.py`
- Modify: `tests/test_unified_cli.py`

- [ ] **Step 1: Write or update parser tests**

Update `tests/test_unified_cli.py` training test to import `scripts.train` and expect `src.*` behavior:

```python
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_train_script_uses_feature_defaults(monkeypatch, tmp_path: Path):
    import scripts.train as train_script

    monkeypatch.setattr(
        train_script,
        "get_feature_spec",
        lambda feature: SimpleNamespace(
            name=feature,
            default_data_dir=f"/data/{feature}",
            checkpoint_name=f"best_{feature}_model.pth",
            train_description=f"train {feature}",
            dataset_class=object,
        ),
    )

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
batch_size: 32
learning_rate: 0.01
num_workers: 4
epochs: 12
patience: 3
""".strip()
    )

    args = train_script.parse_args(["--config", str(config_path), "--feature", "cpp", "--device", "cpu"])

    assert args.feature == "cpp"
    assert args.batch_size == 32
    assert args.lr == pytest.approx(0.01)
    assert args.num_workers == 4
    assert args.epochs == 12
    assert args.patience == 3
    assert args.data_dir == "/data/cpp"
    assert args.checkpoint_path == "best_cpp_model.pth"
    assert args.device == "cpu"
```

- [ ] **Step 2: Run test to verify failure**

```bash
~/Desktop/venv/bin/python -m pytest tests/test_unified_cli.py::test_train_script_uses_feature_defaults -q
```

Expected: FAIL because `scripts.train` does not exist.

- [ ] **Step 3: Implement `scripts/train.py`**

Create `scripts/__init__.py` if it does not exist.

Create `scripts/train.py` with shared logic copied from the existing train scripts and parameterized by `FeatureSpec`:

```python
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.splits import split_dataset
from src.models.resnet import DroneRFaResNet18
from src.training.trainer import train_model
from src.utils.config import expand_path, get_config_value, load_config
from src.utils.device import default_device
from src.utils.feature_specs import get_feature_spec
from src.utils.logger import logger

NUM_CLASSES = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
PATIENCE = 10
SPLIT_SEED = 42


def build_parser(config=None):
    parser = argparse.ArgumentParser(description="Train ResNet on pre-computed DroneRFa features")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--feature", choices=("stft", "cpp"), default=get_config_value(config, "feature", "stft"))
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing feature .h5 files")
    parser.add_argument("--batch-size", type=int, default=get_config_value(config, "batch_size", BATCH_SIZE))
    parser.add_argument("--lr", type=float, default=get_config_value(config, "learning_rate", LEARNING_RATE))
    parser.add_argument("--num-workers", type=int, default=get_config_value(config, "num_workers", 0))
    parser.add_argument("--epochs", type=int, default=get_config_value(config, "epochs", 200))
    parser.add_argument("--patience", type=int, default=get_config_value(config, "patience", PATIENCE))
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Checkpoint path or name")
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
    args.checkpoint_path = expand_path(args.checkpoint_path) if args.checkpoint_path is not None else spec.checkpoint_name
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
        logger.info(
            "Loaded %d %s samples from %d .h5 files in %s",
            len(dataset),
            spec.name,
            len(set(s[0] for s in dataset.index)),
            args.data_dir,
        )
        train_ds, val_ds, test_ds = split_dataset(
            dataset,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            seed=SPLIT_SEED,
        )
        logger.info("Split - train: %d, val: %d, test: %d", len(train_ds), len(val_ds), len(test_ds))

        loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
        }
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
    train(parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run focused test**

```bash
~/Desktop/venv/bin/python -m pytest tests/test_unified_cli.py::test_train_script_uses_feature_defaults -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/train.py tests/test_unified_cli.py
git commit -m "feat: add unified training cli"
```

---

### Task 4: Add Unified Evaluation CLI

**Files:**
- Create: `scripts/evaluate.py`
- Modify: `tests/test_unified_cli.py`

- [ ] **Step 1: Write parser test**

Add or keep this test in `tests/test_unified_cli.py`:

```python
def test_evaluate_script_uses_feature_defaults(monkeypatch, tmp_path: Path):
    import scripts.evaluate as evaluate_script

    monkeypatch.setattr(
        evaluate_script,
        "get_feature_spec",
        lambda feature: SimpleNamespace(
            name=feature,
            default_data_dir=f"/data/{feature}",
            checkpoint_name=f"best_{feature}_model.pth",
            cm_image_name=f"{feature}_cm.png",
            cm_array_name=f"{feature}_cm.npy",
            eval_description=f"evaluate {feature}",
            dataset_class=object,
        ),
    )

    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        """
batch_size: 64
num_workers: 2
""".strip()
    )

    args = evaluate_script.parse_args(["--config", str(config_path), "--feature", "stft", "--device", "cpu"])

    assert args.feature == "stft"
    assert args.batch_size == 64
    assert args.num_workers == 2
    assert args.data_dir == "/data/stft"
    assert args.model_path == "best_stft_model.pth"
    assert args.cm_image_path == "stft_cm.png"
    assert args.device == "cpu"
```

- [ ] **Step 2: Run test to verify failure**

```bash
~/Desktop/venv/bin/python -m pytest tests/test_unified_cli.py::test_evaluate_script_uses_feature_defaults -q
```

Expected: FAIL because `scripts.evaluate` does not exist.

- [ ] **Step 3: Implement `scripts/evaluate.py`**

Create `scripts/evaluate.py`:

```python
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

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
    parser.add_argument("--feature", choices=("stft", "cpp"), default=get_config_value(config, "feature", "stft"))
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing feature .h5 files")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=get_config_value(config, "batch_size", BATCH_SIZE))
    parser.add_argument("--num-workers", type=int, default=get_config_value(config, "num_workers", 0))
    parser.add_argument("--device", type=str, default=default_device())
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
    args.cm_image_path = expand_path(args.cm_image_path) if args.cm_image_path is not None else str(figures_dir() / spec.cm_image_name)
    return args


def evaluate(args):
    spec = get_feature_spec(args.feature)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name())

    dataset = spec.dataset_class(args.data_dir)
    try:
        logger.info(
            "Loaded %d %s samples from %d .h5 files in %s",
            len(dataset),
            spec.name,
            len(set(s[0] for s in dataset.index)),
            args.data_dir,
        )
        _, _, test_ds = split_dataset(
            dataset,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            seed=SPLIT_SEED,
        )
        logger.info("Test set size: %d", len(test_ds))

        loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
        }
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
            loader_kwargs["persistent_workers"] = True

        test_loader = DataLoader(test_ds, **loader_kwargs)
        model = DroneRFaResNet18(num_classes=NUM_CLASSES).to(device)
        logger.info("Loading model from %s", args.model_path)
        state_dict = load_checkpoint(args.model_path, map_location=device)
        model.load_state_dict(state_dict)

        criterion = nn.CrossEntropyLoss()
        test_loss, _test_acc, preds, labels = evaluate_with_predictions(model, test_loader, criterion, device)
        metrics = compute_metrics(preds, labels)

        logger.info("Accuracy: %.4f", metrics["accuracy"])
        logger.info("Precision: %.4f", metrics["precision"])
        logger.info("Recall: %.4f", metrics["recall"])
        logger.info("F1-Score: %.4f", metrics["f1"])
        logger.info("Test Loss: %.4f", test_loss)

        cm = confusion_matrix(labels, preds)
        metric_path = metrics_dir()
        os.makedirs(metric_path, exist_ok=True)
        cm_path = metric_path / spec.cm_array_name
        np.save(cm_path, cm)

        cm_image_path = Path(args.cm_image_path)
        os.makedirs(cm_image_path.parent, exist_ok=True)
        save_confusion_matrix_image(cm, cm_image_path)
        return metrics
    finally:
        dataset.close()


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run focused test**

```bash
~/Desktop/venv/bin/python -m pytest tests/test_unified_cli.py::test_evaluate_script_uses_feature_defaults -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate.py tests/test_unified_cli.py
git commit -m "feat: add unified evaluation cli"
```

---

### Task 5: Convert Legacy Train/Test Entrypoints To Wrappers

**Files:**
- Modify: `train_stft.py`
- Modify: `train_cpp.py`
- Modify: `test_stft.py`
- Modify: `test_cpp.py`

- [ ] **Step 1: Replace `train_stft.py` with wrapper**

Keep the file as a compatibility entrypoint:

```python
"""Compatibility wrapper for STFT training.

Usage:
  python train_stft.py --data-dir ~/Desktop/dataset/droneRFa/stft_h5
"""

from scripts.train import parse_args as _parse_args
from scripts.train import train


def parse_args(argv=None):
    argv = [] if argv is None else list(argv)
    if "--feature" not in argv:
        argv = ["--feature", "stft", *argv]
    return _parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
```

- [ ] **Step 2: Replace `train_cpp.py` with wrapper**

```python
"""Compatibility wrapper for CPP/FAM training.

Usage:
  python train_cpp.py --data-dir ~/Desktop/dataset/droneRFa/cpp_h5
"""

from scripts.train import parse_args as _parse_args
from scripts.train import train


def parse_args(argv=None):
    argv = [] if argv is None else list(argv)
    if "--feature" not in argv:
        argv = ["--feature", "cpp", *argv]
    return _parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
```

- [ ] **Step 3: Replace `test_stft.py` with wrapper**

```python
"""Compatibility wrapper for STFT evaluation.

Usage:
  python test_stft.py --data-dir ~/Desktop/dataset/droneRFa/stft_h5
"""

__test__ = False

from scripts.evaluate import evaluate
from scripts.evaluate import parse_args as _parse_args


def parse_args(argv=None):
    argv = [] if argv is None else list(argv)
    if "--feature" not in argv:
        argv = ["--feature", "stft", *argv]
    return _parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
```

- [ ] **Step 4: Replace `test_cpp.py` with wrapper**

```python
"""Compatibility wrapper for CPP/FAM evaluation.

Usage:
  python test_cpp.py --data-dir ~/Desktop/dataset/droneRFa/cpp_h5
"""

__test__ = False

from scripts.evaluate import evaluate
from scripts.evaluate import parse_args as _parse_args


def parse_args(argv=None):
    argv = [] if argv is None else list(argv)
    if "--feature" not in argv:
        argv = ["--feature", "cpp", *argv]
    return _parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
```

- [ ] **Step 5: Run parser checks**

```bash
~/Desktop/venv/bin/python -m pytest tests/test_unified_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add train_stft.py train_cpp.py test_stft.py test_cpp.py
git commit -m "refactor: keep legacy entrypoints as wrappers"
```

---

### Task 6: Extract Shared DroneRFa IO Helpers

**Files:**
- Create: `src/data/drone_rfa_io.py`
- Modify: `scripts/precompute_stft_h5.py`
- Modify: `scripts/precompute_cpp_h5.py`
- Test: `tests/test_drone_rfa_io.py`

- [ ] **Step 1: Write tests**

Create `tests/test_drone_rfa_io.py`:

```python
import h5py
import numpy as np

from src.data.drone_rfa_io import count_iq_samples, parse_label, read_iq_batch


def test_parse_label_reads_drone_code():
    assert parse_label("T0001_example.mat") == 1
    assert parse_label("T11000_anything.mat") == 24


def test_count_iq_samples_uses_complete_segments(tmp_path):
    path = tmp_path / "sample.mat"
    with h5py.File(path, "w") as f:
        f.create_dataset("RF0_I", data=np.zeros((1, 10), dtype=np.float32))

    with h5py.File(path, "r") as f:
        assert count_iq_samples(f, sample_length=4) == 2
        assert count_iq_samples(f, sample_length=4, max_samples=1) == 1


def test_read_iq_batch_returns_dual_complex_channels(tmp_path):
    path = tmp_path / "sample.mat"
    with h5py.File(path, "w") as f:
        f.create_dataset("RF0_I", data=np.array([[1, 2, 3, 4]], dtype=np.float32))
        f.create_dataset("RF0_Q", data=np.array([[10, 20, 30, 40]], dtype=np.float32))
        f.create_dataset("RF1_I", data=np.array([[5, 6, 7, 8]], dtype=np.float32))
        f.create_dataset("RF1_Q", data=np.array([[50, 60, 70, 80]], dtype=np.float32))

    with h5py.File(path, "r") as f:
        batch = read_iq_batch(f, sample_length=2, start_idx=1, end_idx=2)

    assert batch.shape == (1, 2, 2)
    assert batch.dtype == np.complex64
    assert np.array_equal(batch[0, 0], np.array([3 + 30j, 4 + 40j], dtype=np.complex64))
    assert np.array_equal(batch[0, 1], np.array([7 + 70j, 8 + 80j], dtype=np.complex64))
```

- [ ] **Step 2: Run test to verify failure**

```bash
~/Desktop/venv/bin/python -m pytest tests/test_drone_rfa_io.py -q
```

Expected: FAIL because `src.data.drone_rfa_io` does not exist.

- [ ] **Step 3: Implement shared IO module**

Create `src/data/drone_rfa_io.py`:

```python
import os
import sys

import h5py
import numpy as np

LABEL_MAPPING = {
    "T0000": 0, "T0001": 1, "T0010": 2, "T0011": 3,
    "T0100": 4, "T0101": 5, "T0110": 6, "T0111": 7,
    "T1000": 8, "T1001": 9, "T1010": 10, "T1011": 11,
    "T1100": 12, "T1101": 13, "T1110": 14, "T1111": 15,
    "T10000": 16, "T10001": 17, "T10010": 18, "T10011": 19,
    "T10100": 20, "T10101": 21, "T10110": 22, "T10111": 23,
    "T11000": 24,
}


def default_raw_data_dir() -> str:
    if os.name == "nt":
        return "E:/dataSet/DroneRFa"
    if sys.platform == "darwin":
        return os.path.expanduser("~/Desktop/dataset/droneRFa")
    return "/mnt/data/wurixin/DroneRFa"


def parse_label(mat_file: str) -> int:
    drone_code = os.path.basename(mat_file).split("_")[0]
    return LABEL_MAPPING[drone_code]


def count_iq_samples(src: h5py.File, *, sample_length: int, max_samples: int | None = None) -> int:
    total_points = int(src["RF0_I"].shape[1])
    num_samples = total_points // sample_length
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)
    return num_samples


def read_iq_batch(src: h5py.File, *, sample_length: int, start_idx: int, end_idx: int) -> np.ndarray:
    batch_size = end_idx - start_idx
    offset = start_idx * sample_length
    end = end_idx * sample_length

    rf0_i = src["RF0_I"][0, offset:end].reshape(batch_size, sample_length)
    rf0_q = src["RF0_Q"][0, offset:end].reshape(batch_size, sample_length)
    rf1_i = src["RF1_I"][0, offset:end].reshape(batch_size, sample_length)
    rf1_q = src["RF1_Q"][0, offset:end].reshape(batch_size, sample_length)

    iq_batch = np.empty((batch_size, 2, sample_length), dtype=np.complex64)
    iq_batch[:, 0, :].real = rf0_i
    iq_batch[:, 0, :].imag = rf0_q
    iq_batch[:, 1, :].real = rf1_i
    iq_batch[:, 1, :].imag = rf1_q
    return iq_batch
```

- [ ] **Step 4: Update preprocessing scripts**

In `scripts/precompute_stft_h5.py`, import:

```python
from src.data.drone_rfa_io import count_iq_samples, default_raw_data_dir, parse_label, read_iq_batch
```

Then remove local `LABEL_MAPPING`, `_default_data_dir`, `_parse_label`, `count_iq_samples`, and `_read_iq_batch`. Replace calls:

```python
default=_default_data_dir()
```

with:

```python
default=default_raw_data_dir()
```

Replace:

```python
label = _parse_label(mat_file)
chunk_iq = _read_iq_batch(...)
```

with:

```python
label = parse_label(mat_file)
chunk_iq = read_iq_batch(...)
```

In `scripts/precompute_cpp_h5.py`, import:

```python
from src.data.drone_rfa_io import count_iq_samples, default_raw_data_dir, parse_label
```

Then remove local `LABEL_MAPPING`, `_default_data_dir`, `_parse_label`, and `count_iq_samples`.

- [ ] **Step 5: Run tests**

```bash
~/Desktop/venv/bin/python -m pytest tests/test_drone_rfa_io.py tests/test_preprocessing.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/data/drone_rfa_io.py scripts/precompute_stft_h5.py scripts/precompute_cpp_h5.py tests/test_drone_rfa_io.py tests/test_preprocessing.py
git commit -m "refactor: share DroneRFa IO helpers"
```

---

### Task 7: Normalize STFT PNG CLI

**Files:**
- Modify: `scripts/generate_stft_png.py`
- Modify: `tests/test_visualization.py`

- [ ] **Step 1: Add parser-focused test**

Add to `tests/test_visualization.py`:

```python
def test_generate_stft_png_parse_args_accepts_limits(tmp_path: Path):
    import scripts.generate_stft_png as script

    args = script.parse_args([
        "--h5-dir", str(tmp_path / "h5"),
        "--save-root", str(tmp_path / "png"),
        "--max-files", "2",
        "--max-samples-per-file", "3",
        "--num-workers", "1",
    ])

    assert args.h5_dir == str(tmp_path / "h5")
    assert args.save_root == str(tmp_path / "png")
    assert args.max_files == 2
    assert args.max_samples_per_file == 3
    assert args.num_workers == 1
```

- [ ] **Step 2: Run test to verify failure**

```bash
~/Desktop/venv/bin/python -m pytest tests/test_visualization.py::test_generate_stft_png_parse_args_accepts_limits -q
```

Expected: FAIL because `parse_args` does not exist in `scripts.generate_stft_png`.

- [ ] **Step 3: Add CLI arguments**

In `scripts/generate_stft_png.py`, add:

```python
import argparse
```

Add:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate STFT PNGs from pre-computed STFT .h5 files")
    parser.add_argument("--h5-dir", type=str, default=_default_data_dir(), help="Directory containing STFT .h5 files")
    parser.add_argument("--save-root", type=str, default=None, help="Directory for output PNG files")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most this many .h5 files")
    parser.add_argument("--max-samples-per-file", type=int, default=None, help="Process at most this many samples from each .h5 file")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of PNG worker processes per .h5 file")
    return parser.parse_args()
```

Update `process_one_h5` to accept `max_samples_per_file` and `num_workers`, matching `scripts/generate_cpp_png.py`.

Update `main`:

```python
def main() -> None:
    args = parse_args()
    h5_dir = os.path.expanduser(args.h5_dir)
    save_root = os.path.expanduser(args.save_root) if args.save_root is not None else _default_save_root(h5_dir)
    os.makedirs(save_root, exist_ok=True)

    h5_files = [
        os.path.join(h5_dir, f)
        for f in sorted(os.listdir(h5_dir))
        if f.endswith(".h5")
    ]
    if args.max_files is not None:
        h5_files = h5_files[:args.max_files]

    logger.info("Found %d .h5 files", len(h5_files))
    for h5_path in tqdm.tqdm(h5_files, desc="Processing .h5 files"):
        process_one_h5(
            h5_path,
            save_root,
            max_samples_per_file=args.max_samples_per_file,
            num_workers=args.num_workers,
        )
    logger.info("All done!")
```

- [ ] **Step 4: Run visualization tests**

```bash
~/Desktop/venv/bin/python -m pytest tests/test_visualization.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_stft_png.py tests/test_visualization.py
git commit -m "refactor: normalize stft png cli"
```

---

### Task 8: Remove Library-Level Path Mutation

**Files:**
- Modify: `src/data/stft_dataset.py`
- Modify: `src/data/cpp_dataset.py`

- [ ] **Step 1: Remove `sys.path.insert` from library modules**

In `src/data/stft_dataset.py`, remove:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

In `src/data/cpp_dataset.py`, remove the same path mutation block.

- [ ] **Step 2: Keep direct script smoke behavior**

The `if __name__ == "__main__"` blocks can remain, but direct execution should be treated as a developer smoke check from repository root:

```bash
~/Desktop/venv/bin/python -m src.data.stft_dataset
~/Desktop/venv/bin/python -m src.data.cpp_dataset
```

- [ ] **Step 3: Run dataset tests**

```bash
~/Desktop/venv/bin/python -m pytest tests/test_h5_dataset.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/data/stft_dataset.py src/data/cpp_dataset.py
git commit -m "refactor: remove library path mutation"
```

---

### Task 9: Update README Commands

**Files:**
- Modify: `readme.md`

- [ ] **Step 1: Update training commands**

Replace separate train commands with:

```bash
python scripts/train.py --feature stft --data-dir ~/Desktop/dataset/droneRFa/stft_h5 --batch-size 64
python scripts/train.py --feature cpp --data-dir ~/Desktop/dataset/droneRFa/cpp_h5 --batch-size 64
```

Mention compatibility wrappers:

```text
Legacy wrappers `train_stft.py` and `train_cpp.py` remain available, but new automation should prefer `scripts/train.py`.
```

- [ ] **Step 2: Update evaluation commands**

Replace separate test commands with:

```bash
python scripts/evaluate.py \
  --feature stft \
  --data-dir ~/Desktop/dataset/droneRFa/stft_h5 \
  --model-path outputs/checkpoints/best_stft_model.pth

python scripts/evaluate.py \
  --feature cpp \
  --data-dir ~/Desktop/dataset/droneRFa/cpp_h5 \
  --model-path outputs/checkpoints/best_cpp_model.pth
```

- [ ] **Step 3: Document import direction**

Add:

```text
Repository modules use the existing `src.*` import namespace. Do not add a parallel `drone_rfa.*` alias unless the project later moves to a packaged `src/drone_rfa` layout.
```

- [ ] **Step 4: Commit**

```bash
git add readme.md
git commit -m "docs: document unified src-based workflow"
```

---

### Task 10: Full Verification

**Files:**
- No source files should be changed in this task.

- [ ] **Step 1: Check worktree**

```bash
git status --short --untracked-files=all
```

Expected: only intentional changes from completed tasks, or clean after commits.

- [ ] **Step 2: Run full test suite**

```bash
~/Desktop/venv/bin/python -m pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run parser smoke checks**

```bash
~/Desktop/venv/bin/python scripts/train.py --help
~/Desktop/venv/bin/python scripts/evaluate.py --help
~/Desktop/venv/bin/python scripts/generate_stft_png.py --help
~/Desktop/venv/bin/python scripts/generate_cpp_png.py --help
```

Expected: each command exits with status 0 and prints usage.

- [ ] **Step 4: Final commit if needed**

```bash
git status --short
git add readme.md tests src scripts train_stft.py train_cpp.py test_stft.py test_cpp.py
git commit -m "refactor: unify src-based DroneRFa workflow"
```

Skip this commit if all prior tasks were committed individually and the worktree is clean.

---

## Self-Review

- Spec coverage: This plan keeps `src.*` as the canonical namespace, restores tests, adds unified train/evaluate scripts, avoids `src/drone_rfa`, and reduces duplicated STFT/CPP code.
- Placeholder scan: No task contains `TBD`, vague implementation-only instructions, or missing test commands.
- Type consistency: `FeatureSpec` fields are introduced in Task 2 and reused consistently by Tasks 3 and 4.

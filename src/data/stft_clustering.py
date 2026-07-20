"""Clean STFT H5 indexing and preprocessing for deep clustering."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

REQUIRED_SAMPLE_DATASETS = (
    "stft",
    "labels",
    "is_clean",
    "snr_db",
    "source_sample_idx",
    "augmentation_variant",
    "rf_channel",
)


@dataclass(frozen=True)
class StftClusterSample:
    path: str
    row_idx: int
    label: int
    rf_channel: int
    source_sample_idx: int


def _text_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def scan_clean_stft_h5(data_dir: str) -> list[StftClusterSample]:
    """Index clean UAV STFT rows, excluding the T0000 background class."""

    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"STFT H5 directory does not exist: {root}")

    samples: list[StftClusterSample] = []
    for path in sorted(root.glob("*.h5")):
        with h5py.File(path, "r") as h5f:
            missing = [name for name in REQUIRED_SAMPLE_DATASETS if name not in h5f]
            if missing:
                raise ValueError(f"Missing datasets {missing} in {path}")
            shape = h5f["stft"].shape
            if len(shape) != 4 or shape[1:] != (1, 1024, 1024):
                raise ValueError(
                    f"STFT must have shape (N, 1, 1024, 1024), got {shape} in {path}"
                )
            count = shape[0]
            for name in REQUIRED_SAMPLE_DATASETS[1:]:
                if h5f[name].shape[:1] != (count,):
                    raise ValueError(f"Dataset '{name}' has invalid length in {path}")
            if _text_attr(h5f.attrs.get("noise_profile", "")) != "clean":
                raise ValueError(f"Only noise_profile=clean is supported: {path}")

            labels = h5f["labels"][:]
            is_clean = h5f["is_clean"][:]
            rf_channels = h5f["rf_channel"][:]
            source_indices = h5f["source_sample_idx"][:]
            if not bool(is_clean.all()):
                raise ValueError(f"All included STFT rows must be clean: {path}")
            invalid_labels = sorted({int(value) for value in labels if not 0 <= int(value) <= 16})
            if invalid_labels:
                raise ValueError(f"Invalid DroneRFa labels {invalid_labels} in {path}")
            if not all(int(value) in (0, 1) for value in rf_channels):
                raise ValueError(f"Invalid rf_channel values in {path}")

            for row_idx, label in enumerate(labels):
                label = int(label)
                if label == 0:
                    continue
                samples.append(
                    StftClusterSample(
                        path=str(path.resolve()),
                        row_idx=row_idx,
                        label=label,
                        rf_channel=int(rf_channels[row_idx]),
                        source_sample_idx=int(source_indices[row_idx]),
                    )
                )
    if not samples:
        raise ValueError(f"No clean T0001-T10000 STFT samples found in {root}")
    return samples


def preprocess_stft(stft: torch.Tensor, output_size: int = 256) -> torch.Tensor:
    """Clip, scale, and average-pool one z-score STFT for model input."""

    if stft.ndim != 3 or stft.shape[0] != 1:
        raise ValueError(f"Expected one STFT with shape (1, H, W), got {tuple(stft.shape)}")
    if not torch.isfinite(stft).all():
        raise ValueError("STFT contains non-finite values")
    stft = stft.to(dtype=torch.float32).clamp(-5.0, 5.0).div(5.0)
    return F.adaptive_avg_pool2d(stft, (output_size, output_size))


class CleanStftClusteringDataset(Dataset):
    """Lazily load clean UAV STFT rows and return stable global indices."""

    def __init__(self, samples: list[StftClusterSample], *, output_size: int = 256) -> None:
        self.samples = samples
        self.output_size = output_size
        self.files: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        h5f = self._get_file(sample.path)
        value = torch.from_numpy(h5f["stft"][sample.row_idx])
        return preprocess_stft(value, self.output_size), index

    def _get_file(self, path: str) -> h5py.File:
        if path not in self.files:
            self.files[path] = h5py.File(path, "r", rdcc_nbytes=64 * 1024 * 1024)
        return self.files[path]

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["files"] = {}
        return state

    def close(self) -> None:
        for h5f in self.files.values():
            h5f.close()
        self.files.clear()


def source_filename(sample: StftClusterSample) -> str:
    return os.path.basename(sample.path)

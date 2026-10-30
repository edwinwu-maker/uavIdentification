"""Clean STFT H5 indexing and preprocessing for deep clustering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from src.data.drone_rfa_io import EXCLUDED_CLASS_CODES, RESEARCH_LABELS
from src.preprocess.stft import clip_scale_and_pool_stft
from src.utils.files import h5_attr_to_text

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
    """不可变的单个 clean STFT 样本索引，保存位置、类别及来源信息。"""

    path: str               # 存储 STFT 特征的 H5 文件绝对路径。
    row_idx: int            # H5 中 stft 数据集的样本行索引，从 0 开始。
    label: int              # 原始 DroneRFa 整数类别标签，未重映射为模型标签。
    rf_channel: int         # 原始 IQ 数据所属的 RF 通道编号，取 0 或 1。
    source_sample_idx: int  # 所选 RF 通道内原始定长 IQ 样本编号，从 0 开始，可与 row_idx 不同。


def scan_clean_stft_h5(data_dir: str) -> list[StftClusterSample]:
    """索引 T0010-T10000 的 clean STFT，排除背景和 T0001。"""

    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"STFT H5 directory does not exist: {root}")

    samples: list[StftClusterSample] = []
    for path in sorted(root.glob("*.h5")):
        if path.stem.split("_")[0] in EXCLUDED_CLASS_CODES:
            continue
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
            if h5_attr_to_text(h5f.attrs.get("noise_profile", "")) != "clean":
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
                if label == 0 or label not in RESEARCH_LABELS:
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
        raise ValueError(f"No clean T0010-T10000 STFT samples found in {root}")
    return samples


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
        return clip_scale_and_pool_stft(value, self.output_size), index

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

"""Load pre-computed CPP/FAM samples from .h5 files.

Usage:
  python src/data/cpp_dataset.py

Default data directory:
  <DroneRFa data dir>/cpp_h5
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import torch
from torch.utils.data import Dataset

from src.utils.logger import logger


class CppDataset(Dataset):
    """Load pre-computed CPP/FAM matrices from .h5 files.

    Each .h5 file contains:
      /cpp    (N, 2, alpha_bins, f_bins) float32
      /labels (N,) int64

    The dataset scans a directory for all .h5 files and indexes every sample
    across all files, so each index maps to a single CPP slice.
    """

    def __init__(self, cache_dir: str) -> None:
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(f"CPP cache directory does not exist: {cache_dir}")

        self.cache_dir = cache_dir  # h5 directory
        self.index = []  # list of (file_path, row_idx, label)
        self.files = {}  # path -> opened h5py.File (lazy)

        for fname in sorted(os.listdir(cache_dir)):
            if fname.endswith(".h5"):
                path = os.path.join(cache_dir, fname)
                with h5py.File(path, "r") as f:
                    labels = f["labels"][:]     
                # 为每个样本建立索引，记录文件路径、行索引和标签
                for row_idx, label in enumerate(labels):
                    self.index.append((path, row_idx, int(label)))


    def __len__(self) -> int:
        # 返回总样本数，即所有文件中标签的总行数
        return len(self.index)
    

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 取一个样本：根据索引找到对应的文件路径、行索引和标签，打开文件读取CPP数据并返回
        path, row_idx, label = self.index[idx]
        f = self._get_file(path)
        cpp = f["cpp"][row_idx]  # (2, alpha_bins, f_bins) float32
        return torch.from_numpy(cpp), torch.tensor(label, dtype=torch.int64)
    

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["files"] = {}
        return state
    

    def _get_file(self, path: str) -> h5py.File:
        if path not in self.files:
            # 只在第一次用的时候打开文件，之后复用同一个打开的文件对象
            self.files[path] = h5py.File(path, "r", rdcc_nbytes=64 * 1024 * 1024)
        return self.files[path]


    def close(self) -> None:
        for f in self.files.values():
            f.close()
        self.files.clear()


if __name__ == "__main__":
    if os.name == "nt":
        h5_dir = "E:/dataSet/DroneRFa/cpp_h5"
    elif sys.platform == "darwin":
        h5_dir = os.path.expanduser("~/Desktop/dataset/droneRFa/cpp_h5")
    else:
        h5_dir = "/mnt/data/wurixin/DroneRFa/cpp_h5"
    dataset = CppDataset(h5_dir)
    logger.info(f"Loaded {len(dataset)} samples.")
    cpp, label = dataset[0]
    logger.info(f"Sample shape: {cpp.shape}, Label: {label.item()}")
    logger.info(f"Sample dtype: {cpp.dtype}, Label dtype: {label.dtype}")
    dataset.close()

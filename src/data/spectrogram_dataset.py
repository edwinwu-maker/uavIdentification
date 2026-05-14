import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.logger import logger

class SpectrogramDataset(Dataset):
    """Load pre-computed spectrograms from .h5 files.

    Each .h5 file contains:
      /stft   (N, 2, 1024, 1024) float32
      /labels (N,) int64

    The dataset scans a directory for all .h5 files and indexes every sample
    across all files, so each index maps to a single spectrogram slice.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.index = []  # list of (file_path, row_idx, label)
        self.files = {}  # path -> opened h5py.File (lazy)

        for fname in sorted(os.listdir(cache_dir)):
            if fname.endswith(".h5"):
                path = os.path.join(cache_dir, fname)
                with h5py.File(path, "r") as f:
                    labels = f["labels"][:]
                for row_idx, label in enumerate(labels):
                    self.index.append((path, row_idx, int(label)))

    def _get_file(self, path: str) -> h5py.File:
        if path not in self.files:
            self.files[path] = h5py.File(path, "r")
        return self.files[path]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, row_idx, label = self.index[idx]
        f = self._get_file(path)
        spec = f["stft"][row_idx]  # (2, 1024, 1024) float32
        return torch.from_numpy(spec.astype(np.float32)), torch.tensor(label, dtype=torch.int64)

    def close(self):
        for f in self.files.values():
            f.close()
        self.files.clear()


if __name__ == "__main__":
    if os.name == "nt":
        h5_dir = "E:/dataSet/DroneRFa"
    else:
        h5_dir = "/mnt/data/wurixin/DroneRFa"
    dataset = SpectrogramDataset(h5_dir)
    logger.info(f"Loaded {len(dataset)} samples.")
    spec, label = dataset[0]
    logger.info(f"Sample shape: {spec.shape}, Label: {label.item()}")
    logger.info(f"Sample dtype: {spec.dtype}, Label dtype: {label.dtype}")
    dataset.close()

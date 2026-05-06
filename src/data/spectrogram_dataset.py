import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """Load pre-computed spectrograms from .npy files.

    File naming: {label_id:02d}_{sample_idx:05d}.npy
    Each file: (2, 1024, 1024) float32.
    """

    def __init__(self, cache_dir: str):
        self.samples = []
        for fname in sorted(os.listdir(cache_dir)):
            if fname.endswith(".npy"):
                label = int(fname.split("_")[0])
                path = os.path.join(cache_dir, fname)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        spec = np.load(path)  # (2, 1024, 1024) float32
        return torch.from_numpy(spec), torch.tensor(label, dtype=torch.int64)

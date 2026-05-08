import os
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_MAPPING = {
    "T0000": 0, "T0001": 1, "T0010": 2, "T0011": 3,
    "T0100": 4, "T0101": 5, "T0110": 6, "T0111": 7,
    "T1000": 8, "T1001": 9, "T1010": 10, "T1011": 11,
    "T1100": 12, "T1101": 13, "T1110": 14, "T1111": 15,
    "T10000": 16, "T10001": 17, "T10010": 18, "T10011": 19,
    "T10100": 20, "T10101": 21, "T10110": 22, "T10111": 23,
    "T11000": 24,
}


class SpectrogramDataset(Dataset):
    """Load pre-computed spectrograms from .npy files.

    File naming: {base_name}_{offset:08d}.npy  (e.g. T0001_flight1_00000000.npy)
    Label parsed from drone type code (first segment before '_').
    Each file: (2, 1024, 1024) float32.
    """

    def __init__(self, cache_dir: str):
        self.samples = []
        for fname in sorted(os.listdir(cache_dir)):
            if fname.endswith(".npy"):
                drone_code = fname.split("_")[0]
                label = LABEL_MAPPING[drone_code]
                path = os.path.join(cache_dir, fname)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        spec = np.load(path)  # (2, 1024, 1024) float32
        return torch.from_numpy(spec), torch.tensor(label, dtype=torch.int64)
    
if __name__ == "__main__":
    # Test loading
    dataset = SpectrogramDataset("/mnt/data/wurixin/DroneRFa/spectrogram_cache")
    print(f"Loaded {len(dataset)} samples.")
    spec, label = dataset[0]
    print(f"Sample shape: {spec.shape}, Label: {label.item()}")

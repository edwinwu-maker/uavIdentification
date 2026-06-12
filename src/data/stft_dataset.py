"""Load pre-computed STFT spectrogram samples from .h5 files.

Usage:
  python src/data/stft_dataset.py

Default data directory:
  <DroneRFa data dir>/stft_h5
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.h5_dataset import H5FeatureDataset
from src.utils.logger import logger


class SpectrogramDataset(H5FeatureDataset):
    """Load pre-computed spectrograms from .h5 files.

    Each .h5 file contains:
      /stft   (N, 2, 1024, 1024) float32
      /labels (N,) int64

    The dataset scans a directory for all .h5 files and indexes every sample
    across all files, so each index maps to a single spectrogram slice.
    """

    def __init__(self, cache_dir: str) -> None:
        super().__init__(cache_dir, feature_key="stft", dataset_name="Spectrogram")


if __name__ == "__main__":
    if os.name == "nt":
        h5_dir = "E:/dataSet/DroneRFa/stft_h5"
    elif sys.platform == "darwin":
        h5_dir = os.path.expanduser("~/Desktop/dataset/droneRFa/stft_h5")
    else:
        h5_dir = "/mnt/data/wurixin/DroneRFa/stft_h5"
    dataset = SpectrogramDataset(h5_dir)
    logger.info(f"Loaded {len(dataset)} samples.")
    spec, label = dataset[0]
    logger.info(f"Sample shape: {spec.shape}, Label: {label.item()}")
    logger.info(f"Sample dtype: {spec.dtype}, Label dtype: {label.dtype}")
    dataset.close()

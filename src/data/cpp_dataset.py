"""Load pre-computed CPP/FAM samples from .h5 files.

Usage:
  python src/data/cpp_dataset.py

Default data directory:
  <DroneRFa data dir>/cpp_h5
"""

import os
import sys

from src.data.h5_dataset import H5FeatureDataset
from src.utils.logger import logger


class CppDataset(H5FeatureDataset):
    """Load pre-computed CPP/FAM matrices from .h5 files.

    Each .h5 file contains:
      /cpp    (N, 2, alpha_bins, f_bins) float32
      /labels (N,) int64

    The dataset scans a directory for all .h5 files and indexes every sample
    across all files, so each index maps to a single CPP slice.
    """

    def __init__(self, cache_dir: str) -> None:
        super().__init__(cache_dir, feature_key="cpp", dataset_name="CPP")


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

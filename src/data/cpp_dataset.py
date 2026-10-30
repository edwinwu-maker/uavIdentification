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
      /cpp    (N, 1, alpha_bins, f_bins) float32
      /labels (N,) int64

    The dataset scans a directory for all .h5 files and indexes every sample
    across all files, so each index maps to a single CPP slice.
    """

    def __init__(self, h5_dir: str) -> None:
        """初始化 CPP/FAM 数据集并建立跨文件样本索引。

        输入：
            h5_dir: 存放 CPP .h5 特征文件的目录路径字符串.
                每个文件需包含 /cpp, 形状为 (N, 1, alpha_bins, f_bins)
                以及形状均为 (N,) 的 /labels 和 /rf_channel; RF 通道值为 0 或 1.
        输出：
            None: 完成当前研究类别的样本索引初始化，特征数据在取样时按需读取。
            dataset[idx] 返回 (cpp, label):
                cpp: 形状为 (1, alpha_bins, f_bins) 的张量, dtype 沿用 H5 数据，
                    标准预计算数据为 torch.float32。
                label: 标量标签张量，类型为 torch.int64。
        """
        super().__init__(h5_dir, feature_key="cpp", dataset_name="CPP")


if __name__ == "__main__":
    if os.name == "nt":             # Windows
        h5_dir = "E:/dataSet/DroneRFa/cpp_h5"
    elif sys.platform == "darwin":  # Mac OS
        h5_dir = os.path.expanduser("~/Desktop/dataset/droneRFa/cpp_h5")
    else:                           # Linux
        h5_dir = "/mnt/data/wurixin/DroneRFa/cpp_h5"
    dataset = CppDataset(h5_dir)
    logger.info(f"Loaded {len(dataset)} samples.")
    cpp, label = dataset[0]
    logger.info(f"Sample shape: {cpp.shape}, Label: {label.item()}")
    logger.info(f"Sample dtype: {cpp.dtype}, Label dtype: {label.dtype}")
    dataset.close()

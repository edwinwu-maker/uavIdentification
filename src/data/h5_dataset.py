import os

import h5py
import torch
from torch.utils.data import Dataset

from src.data.drone_rfa_io import EXCLUDED_CLASS_CODES, RESEARCH_LABELS
from src.utils.files import iter_files_by_suffix


class H5FeatureDataset(Dataset):
    """Load one feature array and labels from pre-computed .h5 files."""

    def __init__(
        self,
        h5_dir: str,
        *,
        feature_key: str,
        dataset_name: str = "H5 feature",
    ) -> None:
        if not os.path.isdir(h5_dir):
            raise FileNotFoundError(f"{dataset_name} H5 directory does not exist: {h5_dir}")

        self.h5_dir = h5_dir
        self.feature_key = feature_key
        self.index = []  # list of (file_path, row_idx, label)
        self.files = {}  # path -> opened h5py.File (lazy)

        for path in iter_files_by_suffix(h5_dir, ".h5"):
            fname = os.path.basename(path)
            if os.path.splitext(fname)[0].split("_")[0] in EXCLUDED_CLASS_CODES:
                continue

            with h5py.File(path, "r") as f:
                # 确认目标特征数据集存在，仅读取形状，不加载完整特征数组。
                if feature_key not in f:
                    raise ValueError(f"{dataset_name} H5 file is missing dataset '{feature_key}': {path}")
                feature_shape = f[feature_key].shape
                # 要求特征形状为 (N, 1, H, W)，拒绝旧版双通道等不兼容格式。
                if len(feature_shape) != 4 or feature_shape[1] != 1:
                    raise ValueError(
                        f"{dataset_name} H5 feature data must have shape (N, 1, H, W), got {feature_shape} in {path}. "
                        "Re-run precomputation; legacy dual-channel caches are not supported."
                    )
                # 要求文件保存样本所属 RF 通道，缺少 rf_channel 时提示重新预计算。
                if "rf_channel" not in f:
                    raise ValueError(
                        f"{dataset_name} H5 file is missing the rf_channel dataset: {path}. "
                        "Re-run precomputation."
                    )
                labels = f["labels"][:]
                rf_channels = f["rf_channel"][:]
                # 校验 RF 通道数组与标签数组形状一致，且通道值转为整数后为 0 或 1。
                if rf_channels.shape != labels.shape or not all(
                    int(channel) in (0, 1) for channel in rf_channels
                ):
                    raise ValueError(f"{dataset_name} H5 file has invalid rf_channel values: {path}")
            # 仅为研究类别建立 (文件路径, 文件内行号, 原始整数标签) 索引，特征数据留待取样时读取。
            for row_idx, label in enumerate(labels):
                if int(label) in RESEARCH_LABELS:
                    self.index.append((path, row_idx, int(label)))
    
    def __len__(self) -> int:
        """返回已建立索引的研究类别样本总数"""
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按索引读取单个样本，首次访问源文件时惰性打开并缓存句柄。

        输入：
            idx: self.index 中的样本位置，越界时抛出 IndexError。
        输出：
            (feature, label)：特征张量形状为 (1, H, W)，dtype 沿用 H5 数据；
            标签为原始整数类别的 torch.int64 标量，不重新连续编号。
        """
        path, row_idx, label = self.index[idx]
        f = self._get_file(path)
        feature = f[self.feature_key][row_idx]
        return torch.from_numpy(feature), torch.tensor(label, dtype=torch.int64)

    def __getstate__(self) -> dict[str, object]:
        """返回序列化状态的浅拷贝，排除不可序列化的 HDF5 句柄。

        原实例的句柄保持打开；反序列化后的实例通过空缓存按需重新打开文件。
        """
        state = self.__dict__.copy()
        state["files"] = {}
        return state

    def _get_file(self, path: str) -> h5py.File:
        """按路径获取只读 HDF5 句柄，同一路径重复访问时复用缓存。

        输入：
            path: H5 文件路径字符串，作为句柄缓存的键。
        输出：
            已打开的 h5py.File；首次打开失败时异常直接向调用方传播。
            句柄由数据集持有，通过 close() 统一关闭。
        """
        if path not in self.files:
            self.files[path] = h5py.File(path, "r", rdcc_nbytes=64 * 1024 * 1024)
        return self.files[path]

    def close(self) -> None:
        """关闭全部缓存的 HDF5 句柄并清空缓存，返回 None；后续取样可重新打开文件。"""
        for f in self.files.values():
            f.close()
        self.files.clear()

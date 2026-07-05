"""CPP 特征归一化工具。"""

from __future__ import annotations

import numpy as np

SUPPORTED_CPP_NORMALIZATION_MODES = (
    "max",
    "log-zscore-sample",
    "log_zscore_sample",
    "log-zscore-dataset",
    "log_zscore_dataset",
)


def normalize_cpp(cpp: np.ndarray, *, mode: str = "max", eps: float = 1e-6) -> np.ndarray:
    """对单个 CPP 样本执行归一化，输入形状为 (C, H, W)。"""

    cpp = np.asarray(cpp, dtype=np.float32)
    if mode == "max":
        max_value = float(np.max(cpp)) if cpp.size else 0.0
        if max_value > 0:
            return (cpp / max_value).astype(np.float32)
        return cpp.astype(np.float32, copy=True)
    if mode in ("log-zscore-sample", "log_zscore_sample"):
        logged = np.log1p(cpp).astype(np.float32)
        mean = logged.mean(axis=(1, 2), keepdims=True)
        std = logged.std(axis=(1, 2), keepdims=True)
        centered = logged - mean
        denominator = np.where(std > eps, std, 1.0)
        normalized = np.where(std > eps, centered / denominator, 0.0)
        return normalized.astype(np.float32)
    if mode in ("log-zscore-dataset", "log_zscore_dataset"):
        raise NotImplementedError("log_zscore_dataset requires dataset-level statistics and is not implemented here")
    raise ValueError(f"Unsupported CPP normalization mode: {mode}")

"""特征归一化工具。"""

from __future__ import annotations

import torch


def normalize_log_zscore(
    feature_map: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """对形状为 (C, H, W) 的单个特征图执行逐通道 log1p z-score 标准化。

    输入会转换为 float32；标准差不大于 epsilon 的通道返回全零。
    """

    if not isinstance(feature_map, torch.Tensor):
        raise TypeError("feature_map must be a torch.Tensor")

    feature_map = feature_map.to(dtype=torch.float32)
    log_feature_map = torch.log1p(feature_map)
    channel_std, channel_mean = torch.std_mean(
        log_feature_map, dim=(1, 2), keepdim=True, unbiased=False
    )
    centered_feature_map = log_feature_map - channel_mean
    valid_std_mask = channel_std > epsilon
    # 低方差通道使用安全分母计算，最终统一置零，避免除零。
    safe_channel_std = torch.where(
        valid_std_mask, channel_std, torch.ones_like(channel_std)
    )
    standardized_feature_map = centered_feature_map / safe_channel_std
    return torch.where(
        valid_std_mask,
        standardized_feature_map,
        torch.zeros_like(centered_feature_map),
    )

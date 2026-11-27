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


def normalize_zscore(
    feature_map: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    fill_invalid: float = 0.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """对形状为 (C, H, W) 的单个特征图执行逐通道 z-score 标准化。

    当提供 valid_mask（形状为 (H, W) 或 (C, H, W) 的布尔张量）时，
    仅在有效区域内统计均值与标准差，并将无效区域填充为 fill_invalid；
    有效区域为空或标准差不大于 epsilon 的通道返回全零。
    """

    if not isinstance(feature_map, torch.Tensor):
        raise TypeError("feature_map must be a torch.Tensor")

    x = feature_map.to(dtype=torch.float32)
    if valid_mask is None:
        channel_std, channel_mean = torch.std_mean(
            x, dim=(1, 2), keepdim=True, unbiased=False
        )
        centered = x - channel_mean
        valid_std_mask = channel_std > epsilon
        safe_std = torch.where(valid_std_mask, channel_std, torch.ones_like(channel_std))
        standardized = centered / safe_std
        return torch.where(valid_std_mask, standardized, torch.zeros_like(centered))

    mask = valid_mask.to(device=x.device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).expand_as(x)
    elif mask.shape != x.shape:
        raise ValueError(
            f"valid_mask shape {tuple(mask.shape)} does not match feature_map shape {tuple(x.shape)}"
        )

    valid_count = mask.sum(dim=(1, 2), keepdim=True).to(dtype=x.dtype)
    has_valid = valid_count > 0
    safe_count = torch.where(has_valid, valid_count, torch.ones_like(valid_count))

    # 仅累加有效域内像素，避免主定义域外的零值拉偏均值与标准差。
    masked_x = torch.where(mask, x, torch.zeros_like(x))
    channel_mean = masked_x.sum(dim=(1, 2), keepdim=True) / safe_count
    centered = torch.where(mask, x - channel_mean, torch.zeros_like(x))
    channel_var = (centered * centered).sum(dim=(1, 2), keepdim=True) / safe_count
    channel_std = torch.sqrt(channel_var)

    valid_std_mask = has_valid & (channel_std > epsilon)
    safe_std = torch.where(valid_std_mask, channel_std, torch.ones_like(channel_std))
    standardized = torch.where(
        mask,
        centered / safe_std,
        torch.full_like(x, fill_invalid),
    )
    return torch.where(valid_std_mask, standardized, torch.zeros_like(x))

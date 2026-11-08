"""为 STFT 自监督训练生成保留完整频谱内容的双视图。"""

from __future__ import annotations

import torch


def make_stft_views(
    inputs: torch.Tensor, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 (B, C, F, T) 每个样本独立做两次不超过 5% 时间长度的循环平移。"""
    if inputs.ndim != 4:
        raise ValueError("STFT views require inputs with shape (B, C, F, T)")
    width = inputs.shape[-1]
    max_shift = width // 20
    # 固定用 CPU generator 取样，使 CPU/CUDA/MPS 都可复用同一随机种子。
    shifts = torch.randint(
        -max_shift, max_shift + 1, (2, inputs.shape[0]), generator=generator, device="cpu",
    ).to(inputs.device)
    positions = torch.arange(width, device=inputs.device).view(1, 1, 1, -1)
    views = []
    for shift in shifts:
        # 仅移动时间列，所有通道使用同一位移；不裁剪、插值或改变信号幅值。
        indices = (positions - shift.view(-1, 1, 1, 1)).remainder(width)
        views.append(torch.gather(inputs, dim=-1, index=indices.expand_as(inputs)))
    return views[0], views[1]

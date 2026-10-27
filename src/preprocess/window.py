"""PyTorch 窗函数生成工具。"""

from typing import Final

import torch


_HAMMING_WINDOWS: Final[tuple[str, ...]] = ("hamming", "hamm")
_HANN_WINDOWS: Final[tuple[str, ...]] = ("hann", "hanning")
_RECT_WINDOWS: Final[tuple[str, ...]] = ("rect", "boxcar", "rectangle")


def create_window(
    name: str,
    length: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    periodic: bool,
) -> torch.Tensor:
    """在指定设备上创建形状为 (length,)、具有指定 dtype 的窗口，不缓存结果。

    Hann/Hamming 的 periodic=True 表示周期窗，False 表示对称窗；矩形窗不受影响。
    """
    name = name.lower()
    if name in _HAMMING_WINDOWS:
        return torch.hamming_window(length, periodic=periodic, device=device, dtype=dtype)
    if name in _HANN_WINDOWS:
        return torch.hann_window(length, periodic=periodic, device=device, dtype=dtype)
    if name in _RECT_WINDOWS:
        return torch.ones(length, device=device, dtype=dtype)
    raise ValueError(f"unsupported window: {name}")

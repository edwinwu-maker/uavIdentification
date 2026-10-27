"""PyTorch 数据类型工具。"""

import torch


def get_real_dtype(dtype: torch.dtype) -> torch.dtype:
    """返回复数类型对应的实数类型，不执行张量转换。

    complex64 对应 float32，complex128 对应 float64；其他类型抛出 ValueError。
    """
    if dtype == torch.complex128:
        return torch.float64
    if dtype == torch.complex64:
        return torch.float32
    raise ValueError("dtype must be torch.complex64 or torch.complex128")

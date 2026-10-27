import multiprocessing

import torch


def default_cpu_cores() -> int:
    """取 CPU 核心数的一半并限制在 1～2，作为默认使用核数以控制并发内存占用。"""

    return min(2, max(1, multiprocessing.cpu_count() // 2))


def default_device(
    *,
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
) -> str:
    """按 CUDA、MPS、CPU 的优先级返回设备字符串。

    cuda_available 和 mps_available 为 None 时分别调用 PyTorch 检测；
    显式传入布尔值可覆盖对应检测结果，便于测试不同硬件条件。
    返回值为 "cuda:0"、"mps" 或 "cpu"。
    """
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if mps_available is None:
        mps_available = torch.backends.mps.is_available()

    if cuda_available:
        return "cuda:0"
    if mps_available:
        return "mps"
    return "cpu"


def resolve_device(device: str | torch.device) -> torch.device:
    """将指定设备解析为 torch.device，并检查 CUDA/MPS 后端是否可用。

    指定的 CUDA/MPS 不可用时抛出 RuntimeError，不自动切换设备。
    """
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    return resolved

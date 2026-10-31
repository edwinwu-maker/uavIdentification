import torch


def setup_device(device: str | torch.device, logger) -> torch.device:
    """解析并记录计算设备；CUDA 模式下同时设置进程的当前设备。

    不检查后端可用性或自动回退，解析及初始化异常直接向调用方传播。
    """
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.set_device(resolved)

    logger.info("Using device: %s", resolved, stacklevel=2)
    if resolved.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(resolved), stacklevel=2)
    return resolved


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

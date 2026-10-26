import torch


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

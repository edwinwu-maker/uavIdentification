import torch


def default_device(
    *,
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
) -> str:
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if mps_available is None:
        mps_available = torch.backends.mps.is_available()

    if cuda_available:
        return "cuda:0"
    if mps_available:
        return "mps"
    return "cpu"


def resolve_requested_device(
    device: str,
    *,
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
) -> str:
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if mps_available is None:
        mps_available = torch.backends.mps.is_available()

    if device == "cpu":
        return device
    if device == "cuda":
        return "cuda:0" if cuda_available else "cpu"
    if device.startswith("cuda"):
        return device if cuda_available else "cpu"
    if device == "mps":
        return device if mps_available else "cpu"
    return device

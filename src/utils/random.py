"""随机种子、生成器与采样工具。"""

import random
import zlib

import numpy as np
import torch


def stable_file_seed(file_id: str) -> int:
    """将文件标识转换为跨进程稳定的 32-bit seed 片段。

    输入：
        file_id: 文件标识字符串，按原样使用 UTF-8 编码，不读取文件或规范化路径。
    输出：
        范围为 0～2³²−1 的整数；相同字符串的结果稳定，CRC32 不保证不同标识结果唯一。
    """
    return zlib.crc32(file_id.encode("utf-8")) & 0xFFFFFFFF


def batch_seed(noise_seed: int, file_id: str, start_idx: int, variant_idx: int = 0) -> int:
    """按 batch 身份派生可复现的随机种子，不修改全局随机状态。

    输入：
        noise_seed: 基础随机种子。
        file_id: 文件标识字符串，使用 stable_file_seed 转换为稳定种子片段。
        start_idx: batch 的起始索引。
        variant_idx: 扩增版本索引，默认 0；非零时追加到种子分量中。
    输出：
        范围为 0～2⁶³−1 的 Python 整数；相同 batch 身份和基础种子得到相同结果。
    """
    # 吞吐优先：随机性绑定到 batch 身份，保证同一 batch 可复现。
    seed_parts = [int(noise_seed), stable_file_seed(file_id), int(start_idx)]
    # variant_idx=0 保持原有随机序列不变；扩增版本追加独立种子维度。
    if variant_idx != 0:
        seed_parts.append(int(variant_idx))
    seed_seq = np.random.SeedSequence(seed_parts)
    return int(seed_seq.generate_state(1, dtype=np.uint64)[0] & np.uint64(0x7FFFFFFFFFFFFFFF))


def make_generator(device: torch.device, seed: int) -> tuple[torch.Generator, torch.device]:
    """创建带固定种子的局部随机数生成器，不修改全局随机状态。

    输入：
        device: 期望用于生成随机数的 torch.device。
        seed: 传给生成器 manual_seed 的随机种子整数。
    输出：
        (generator, generator_device)，分别为生成器和实际生成随机数的设备。
        创建或设置种子时遇到 RuntimeError 会回退到 CPU；CPU 重试失败则向外抛出。
        使用该生成器的随机张量应在 generator_device 上创建，再按需转移到目标设备。
    """
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator, device
    except RuntimeError:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator, torch.device("cpu")


def uniform(
    shape: tuple[int, ...],
    *,
    low: float,
    high: float,
    device: torch.device,
    generator: torch.Generator,
    generator_device: torch.device,
) -> torch.Tensor:
    """使用指定生成器采样均匀分布，再将结果转移到目标设备。

    输入：
        shape: 输出张量的形状。
        low, high: 均匀分布的上下界，相等时返回固定值。
        device: 输出张量所在的目标设备。
        generator: 本次采样使用的局部随机数生成器。
        generator_device: 与 generator 匹配的采样设备，可为回退后的 CPU。
    输出：
        形状为 shape、类型为 torch.float32、位于 device 的张量。
        非空采样会推进传入生成器的状态；上下界相等时也保留原有采样调用。
    """
    values = torch.empty(shape, dtype=torch.float32, device=generator_device)
    values.uniform_(low, high, generator=generator)
    return values.to(device)


def randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
    generator_device: torch.device,
) -> torch.Tensor:
    """使用指定生成器采样标准正态分布，再将结果转移到目标设备。

    输入：
        shape: 输出张量的形状。
        device: 输出张量所在的目标设备。
        generator: 本次采样使用的局部随机数生成器。
        generator_device: 与 generator 匹配的采样设备，可为回退后的 CPU。
    输出：
        形状为 shape、类型为 torch.float32、位于 device 的张量，采样分布均值为 0、标准差为 1。
        非空采样会更新传入生成器的状态。
    """
    values = torch.randn(shape, dtype=torch.float32, device=generator_device, generator=generator)
    return values.to(device)


def set_random_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 的全局随机种子。

    参数：
        seed: 随机种子整数；CUDA 可用时也为所有 CUDA 设备设置种子。

    返回：
        None；直接更新各随机数生成器的状态。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from __future__ import annotations

import zlib

import numpy as np
import torch


def _stable_file_seed(file_id: str) -> int:
    """用文件名生成跨进程稳定的 32-bit seed 片段。"""

    return zlib.crc32(file_id.encode("utf-8")) & 0xFFFFFFFF


def _batch_seed(noise_seed: int, file_id: str, start_idx: int) -> int:
    # 吞吐优先：随机性绑定到 batch 身份，保证同一 batch 可复现。
    seed_seq = np.random.SeedSequence([int(noise_seed), _stable_file_seed(file_id), int(start_idx)])
    return int(seed_seq.generate_state(1, dtype=np.uint64)[0] & np.uint64(0x7FFFFFFFFFFFFFFF))


def _make_generator(device: torch.device, seed: int) -> tuple[torch.Generator, torch.device]:
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator, device
    except RuntimeError:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator, torch.device("cpu")


def _uniform(
    shape: tuple[int, ...],
    *,
    low: float,
    high: float,
    device: torch.device,
    generator: torch.Generator,
    generator_device: torch.device,
) -> torch.Tensor:
    values = torch.empty(shape, dtype=torch.float32, device=generator_device)
    values.uniform_(low, high, generator=generator)
    return values.to(device)


def _randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
    generator_device: torch.device,
) -> torch.Tensor:
    values = torch.randn(shape, dtype=torch.float32, device=generator_device, generator=generator)
    return values.to(device)


def add_random_snr_awgn(
    iq_batch: np.ndarray | torch.Tensor,
    *,
    file_id: str,
    start_idx: int,
    snr_min: float,
    snr_max: float,
    noise_seed: int,
    device: str = "cpu",
) -> torch.Tensor:
    """对 batch 内 IQ 批量添加随机 SNR AWGN，返回 torch.complex64 Tensor。"""

    target_device = torch.device(device)
    iq = torch.as_tensor(iq_batch, dtype=torch.complex64, device=target_device)
    generator, generator_device = _make_generator(target_device, _batch_seed(noise_seed, file_id, start_idx))

    batch_size = iq.shape[0]
    snr_db = _uniform(
        (batch_size,),
        low=snr_min,
        high=snr_max,
        device=target_device,
        generator=generator,
        generator_device=generator_device,
    )
    signal_power = torch.mean(torch.abs(iq) ** 2, dim=(1, 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = torch.sqrt(noise_power / 2.0).reshape(batch_size, 1, 1)

    real_noise = _randn(
        tuple(iq.shape),
        device=target_device,
        generator=generator,
        generator_device=generator_device,
    )
    imag_noise = _randn(
        tuple(iq.shape),
        device=target_device,
        generator=generator,
        generator_device=generator_device,
    )
    noise = torch.complex(real_noise, imag_noise) * scale
    noisy = iq + noise

    zero_power = signal_power <= 0.0
    if torch.any(zero_power):
        noisy = torch.where(zero_power.reshape(batch_size, 1, 1), iq, noisy)
    return noisy.to(torch.complex64)

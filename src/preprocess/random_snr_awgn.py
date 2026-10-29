from __future__ import annotations

import torch

from src.utils.random import batch_seed, make_generator, randn, uniform


def add_random_snr_awgn(
    iq_batch: torch.Tensor,
    *,
    file_id: str,
    start_idx: int,
    snr_min: float,
    snr_max: float,
    noise_seed: int,
    device: str = "cpu",
) -> torch.Tensor:
    """对 batch 内 IQ 批量添加随机 SNR AWGN，返回 torch.complex64 Tensor。"""

    noisy, _ = add_random_snr_awgn_with_snr(
        iq_batch,
        file_id=file_id,
        start_idx=start_idx,
        snr_min=snr_min,
        snr_max=snr_max,
        noise_seed=noise_seed,
        device=device,
    )
    return noisy


def add_random_snr_awgn_with_snr(
    iq_batch: torch.Tensor,
    *,
    file_id: str,
    start_idx: int,
    snr_min: float,
    snr_max: float,
    noise_seed: int,
    variant_idx: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """添加可复现的随机 SNR AWGN，并返回每个样本实际使用的 SNR。"""

    if not isinstance(iq_batch, torch.Tensor):
        raise TypeError("iq_batch must be a torch.Tensor")

    target_device = torch.device(device)
    iq = iq_batch.to(device=target_device, dtype=torch.complex64)
    generator, generator_device = make_generator(
        target_device,
        batch_seed(noise_seed, file_id, start_idx, variant_idx),
    )

    batch_size = iq.shape[0]
    snr_db = uniform(
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

    real_noise = randn(
        tuple(iq.shape),
        device=target_device,
        generator=generator,
        generator_device=generator_device,
    )
    imag_noise = randn(
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
    return noisy.to(torch.complex64), snr_db

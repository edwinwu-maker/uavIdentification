from __future__ import annotations

import torch

from src.utils.random import batch_seed, make_generator, randn, uniform


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
    """添加可复现的随机目标 SNR 的 AWGN，并返回加噪信号与目标 SNR。

    输入：
        iq_batch: 形状为 (B, C, L) 的 IQ 张量，处理时转为 torch.complex64。
            B 为 batch 大小，C 为通道数，L 为每个样本的采样点数。
        file_id: 文件或通道标识字符串，参与 batch 随机种子的派生。
        start_idx: batch 的起始样本索引，参与随机种子的派生。
        snr_min: 目标 SNR 的下界，单位为 dB。
        snr_max: 目标 SNR 的上界，单位为 dB；与下界相等时使用固定 SNR。
        noise_seed: 基础随机种子。
        variant_idx: 扩增版本索引，默认 0；不同版本参与派生各自的随机种子。
        device: 计算和输出设备字符串，默认 "cpu"，也可指定 "cuda:0"、"mps" 等。
    输出：
        (noisy, snr_db) 元组，两者均位于 device。
        noisy: 形状为 (B, C, L)、类型为 torch.complex64 的加噪张量。
        snr_db: 形状为 (B,)、类型为 torch.float32 的目标 SNR 张量，单位为 dB。

    每个样本独立采样一个目标 SNR，功率按通道和时间维度求均值；零功率样本保持原值。
    snr_db 是采样得到的目标值，有限长度噪声的实测 SNR 可能有偏差。
    """

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

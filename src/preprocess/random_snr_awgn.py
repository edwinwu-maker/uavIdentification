from __future__ import annotations

import zlib

import numpy as np


def _stable_file_seed(file_id: str) -> int:
    """用文件名生成跨进程稳定的 32-bit seed 片段。"""

    return zlib.crc32(file_id.encode("utf-8")) & 0xFFFFFFFF


def _sample_rng(noise_seed: int, file_id: str, sample_idx: int) -> np.random.Generator:
    # 将随机性绑定到样本身份：全局 seed 控制实验版本，文件名和样本索引区分具体样本。
    # 这样同一样本不受文件遍历顺序、batch size 或后续并行处理影响，结果仍可复现。
    seed_seq = np.random.SeedSequence([int(noise_seed), _stable_file_seed(file_id), int(sample_idx)])
    return np.random.default_rng(seed_seq)


def add_random_snr_awgn(
    iq_batch: np.ndarray,
    *,
    file_id: str,
    start_idx: int,
    snr_min: float,
    snr_max: float,
    noise_seed: int,
) -> np.ndarray:
    """对 batch 内每条 IQ 按稳定随机 SNR 加 AWGN，不改变 H5 输出结构。"""

    from src.evaluation.snr_accuracy import add_awgn_for_snr

    noisy_batch = np.empty_like(iq_batch, dtype=np.complex64)
    for batch_offset in range(iq_batch.shape[0]):
        sample_idx = start_idx + batch_offset
        rng = _sample_rng(noise_seed, file_id, sample_idx)
        snr_db = float(rng.uniform(snr_min, snr_max))
        noisy_batch[batch_offset] = add_awgn_for_snr(iq_batch[batch_offset], snr_db, rng)
    return noisy_batch

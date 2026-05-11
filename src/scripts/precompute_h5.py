"""
Convert .mat IQ files → one .h5 per file with pre-computed spectrograms.

CPU-only, multi-process. Each .mat is converted independently to a same-named
.h5 in the output directory.

Output HDF5 structure (per file):
  /stft  (N, 2, 1024, 1024) float32
  /labels        (N,) int64

Usage:
  python src/scripts/precompute_h5.py --data-dir /mnt/data/wurixin/DroneRFa
  python src/scripts/precompute_h5.py --data-dir ... --output-dir ... --num-workers 8
"""

import argparse
import concurrent.futures
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from utils.logger import logger

# ── Paper parameters (match transforms.py) ──
SAMPLE_LENGTH = 1_000_000
N_FFT = 1024
HOP_LENGTH = 512
WIN_LENGTH = 1024
SPEC_TIME_BINS = 1024

LABEL_MAPPING = {
    "T0000": 0, "T0001": 1, "T0010": 2, "T0011": 3,
    "T0100": 4, "T0101": 5, "T0110": 6, "T0111": 7,
    "T1000": 8, "T1001": 9, "T1010": 10, "T1011": 11,
    "T1100": 12, "T1101": 13, "T1110": 14, "T1111": 15,
    "T10000": 16, "T10001": 17, "T10010": 18, "T10011": 19,
    "T10100": 20, "T10101": 21, "T10110": 22, "T10111": 23,
    "T11000": 24,
}

_WINDOW: torch.Tensor | None = None


def _get_window() -> torch.Tensor:
    global _WINDOW
    if _WINDOW is None:
        _WINDOW = torch.hann_window(WIN_LENGTH)
    return _WINDOW


def compute_stft(iq_batch: np.ndarray) -> np.ndarray:
    """
    iq_batch: (B, 2, SAMPLE_LENGTH) complex64
    returns:  (B, 2, 1024, 1024) float32, z-score normalized per channel
    B: batch size
    torch.stft 支持一次性处理一整个 batch 的信号
    """
    B, C, _L = iq_batch.shape
    window = _get_window()
    results = []
    with torch.no_grad():
        for ch in range(C):
            sig = torch.from_numpy(iq_batch[:, ch, :])
            Zxx = torch.stft(
                sig, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                window=window, return_complex=True, center=True,
            )
            Zxx = torch.fft.fftshift(Zxx, dim=1)
            Zxx = Zxx[:, :, :SPEC_TIME_BINS]
            eps = torch.finfo(torch.float32).eps
            Zxx_db = 20.0 * torch.log10(Zxx.abs() + eps)
            mean = Zxx_db.mean(dim=(1, 2), keepdim=True)
            std = Zxx_db.std(dim=(1, 2), keepdim=True)
            Zxx_norm = (Zxx_db - mean) / (std + 1e-8)
            results.append(Zxx_norm)
    return torch.stack(results, dim=1).numpy().astype(np.float32)


def _convert_one_file(args):
    """
    Module-level worker for ProcessPoolExecutor.
    args: (mat_file, data_dir, output_dir, batch_size)
    Returns: (mat_file, segment_count)
    """
    mat_file, data_dir, output_dir, batch_size = args
    mat_path = os.path.join(data_dir, mat_file)
    out_path = os.path.join(output_dir, mat_file.replace(".mat", ".h5"))
    drone_code = mat_file.split("_")[0]
    label = LABEL_MAPPING[drone_code]

    with h5py.File(mat_path, "r") as src:
        total_points = int(src["RF0_I"].shape[1])
        num_samples = total_points // SAMPLE_LENGTH

    with h5py.File(out_path, "w") as h5f:
        h5f.create_dataset(
            "stft", shape=(num_samples, 2, 1024, 1024),
            chunks=(64, 2, 1024, 1024), dtype="f4",
        )
        h5f.create_dataset(
            "labels", shape=(num_samples,), chunks=(4096,), dtype="i8",
        )

        with h5py.File(mat_path, "r") as src:
            for sample_idx in range(0, num_samples, batch_size):
                batch_end = min(sample_idx + batch_size, num_samples)
                actual_batch_size = batch_end - sample_idx
                chunk_iq = np.empty((actual_batch_size, 2, SAMPLE_LENGTH), dtype=np.complex64)

                for k, i in enumerate(range(sample_idx, batch_end)):
                    offset = i * SAMPLE_LENGTH
                    end = offset + SAMPLE_LENGTH
                    ch0 = src["RF0_I"][0, offset:end] + 1j * src["RF0_Q"][0, offset:end]
                    ch1 = src["RF1_I"][0, offset:end] + 1j * src["RF1_Q"][0, offset:end]
                    chunk_iq[k, 0] = ch0
                    chunk_iq[k, 1] = ch1

                batch_stft = compute_stft(chunk_iq)
                h5f["stft"][sample_idx:sample_idx + actual_batch_size] = batch_stft
                h5f["labels"][sample_idx:sample_idx + actual_batch_size] = [label] * actual_batch_size

    return mat_file, num_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Convert .mat IQ → .h5 spectrogram files")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing .mat files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output .h5 files (default: same as data-dir)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="STFT batch size")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of worker processes (default: CPU count)")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.set_num_threads(1)

    if args.data_dir is None:
        args.data_dir = "/mnt/data/wurixin/DroneRFa" if os.name != "nt" \
            else "E:/dataSet/DroneRFa"
    if args.output_dir is None:
        args.output_dir = args.data_dir
    if args.num_workers is None:
        args.num_workers = os.cpu_count() or 4

    os.makedirs(args.output_dir, exist_ok=True)

    mat_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".mat")])
    logger.info("Found %d .mat files in %s", len(mat_files), args.data_dir)
    logger.info("Output directory: %s", args.output_dir)

    num_workers = min(args.num_workers, len(mat_files))
    logger.info("Using %d workers", num_workers)

    # 把所有 .mat 文件，打包成「多进程并行处理」的参数列表
    file_args = [
        (mf, args.data_dir, args.output_dir, args.batch_size)
        for mf in mat_files
    ]

    total_segments = 0
    # 创建多进程池，同时开启 num_workers 个 CPU 核心干活。
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_convert_one_file, fa) for fa in file_args]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures), desc="Converting",
        ):
            mf, n = future.result()
            total_segments += n

    logger.info("Done. %d files → %d spectrograms", len(mat_files), total_segments)


if __name__ == "__main__":
    main()

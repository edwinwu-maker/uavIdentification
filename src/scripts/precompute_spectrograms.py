"""
Pre-compute spectrograms from .mat IQ files using GPU torch.stft.
Multi-process: each process binds one GPU and handles its own subset of .mat files.

Usage: python src/scripts/precompute_spectrograms.py --gpus 0,1,2,3,4
"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import multiprocessing as mp

import h5py
import numpy as np
import torch
import torch.nn as nn
import tqdm

from utils.logger import logger

# Match paper parameters + transforms.py
SAMPLE_LENGTH = 1_000_000
FS = 100e6
N_FFT = 1024
HOP_LENGTH = 512  # nperseg - noverlap = 1024 - 512
WIN_LENGTH = 1024
SPEC_TIME_BINS = 1024

class STFTModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN_LENGTH))

    def forward(self, iq_batch: torch.Tensor) -> torch.Tensor:
        """
        iq_batch: (B, 2, SAMPLE_LENGTH) complex64
        returns: (B, 2, 1024, 1024) float32
        """
        B, C, L = iq_batch.shape
        results = []
        window = self.window.to(iq_batch.device)
        for ch in range(C):
            sig = iq_batch[:, ch, :]
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

        return torch.stack(results, dim=1)


def process_mat_file(mat_file, data_dir, cache_dir, stft_module, batch_size):
    """Process a single .mat file — read IQ, GPU STFT, save .npy."""
    mat_path = os.path.join(data_dir, mat_file)
    base_name = os.path.splitext(mat_file)[0]
    count = 0

    with h5py.File(mat_path, "r") as f:
        total_points = int(f["RF0_I"].shape[1])
        num_samples = total_points // SAMPLE_LENGTH

        for sample_idx in range(0, num_samples, batch_size):
            batch_end = min(sample_idx + batch_size, num_samples)
            batch_chunks = []

            for i in range(sample_idx, batch_end):
                offset = i * SAMPLE_LENGTH
                end = offset + SAMPLE_LENGTH
                ch0 = f["RF0_I"][0, offset:end] + 1j * f["RF0_Q"][0, offset:end]
                ch1 = f["RF1_I"][0, offset:end] + 1j * f["RF1_Q"][0, offset:end]
                iq = np.stack([ch0, ch1], axis=0)
                batch_chunks.append(iq)

            with torch.no_grad():
                iq_batch = torch.from_numpy(np.stack(batch_chunks))
                specs = stft_module(iq_batch).cpu()

            for j, i in enumerate(range(sample_idx, batch_end)):
                offset = i * SAMPLE_LENGTH
                save_name = f"{base_name}_{offset:08d}.npy"
                np.save(os.path.join(cache_dir, save_name), specs[j].numpy())

        count += num_samples

    return count


def worker(gpu_id, mat_file_list, data_dir, cache_dir, batch_size, result_queue):
    """One GPU, one process — bind one GPU, handle a subset of .mat files."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")
    stft_module = STFTModule().to(device)

    gpu_name = torch.cuda.get_device_name(0)
    count = 0

    for mat_file in tqdm(mat_file_list, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            count += process_mat_file(mat_file, data_dir, cache_dir, stft_module, batch_size)
        except Exception as e:
            logger.error("[GPU %d] Failed to process %s: %s", gpu_id, mat_file, e)

    msg = f"[GPU {gpu_id} | {gpu_name}] done — {count} spectrograms from {len(mat_file_list)} files"
    # Send result back to main process for logging and final tally
    result_queue.put((gpu_id, count, msg))


def main():
    parser = argparse.ArgumentParser(description="Pre-compute spectrograms")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs, e.g. '0,1,2'. "
                             "Default: single GPU without multiprocessing.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="STFT batch size per GPU step")
    args = parser.parse_args()

    if os.name == "nt":
        DATA_DIR = "E:/dataSet/DroneRFa"
        CACHE_DIR = "E:/dataSet/DroneRFa/spectrogram_cache"
    else:
        DATA_DIR = "/mnt/data/wurixin/DroneRFa"
        CACHE_DIR = "/mnt/data/wurixin/DroneRFa/spectrogram_cache"

    os.makedirs(CACHE_DIR, exist_ok=True)
    # example of mat_file: ["T0000_D00_S0000.mat", "T0000_D00_S0001.mat", ...]
    mat_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".mat")])
    logger.info("Found %d .mat files", len(mat_files))

    if args.gpus is not None:
        # ── Multi-process: one process per GPU ──
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
        logger.info("Multi-process mode: %d GPUs — %s", len(gpu_ids), gpu_ids)

        chunks = np.array_split(mat_files, len(gpu_ids))
        chunks = [list(c) for c in chunks]  # np.array -> list
        for gpu_id, chunk in zip(gpu_ids, chunks):
            logger.info("GPU %d: %d files", gpu_id, len(chunk))
        # 多进程安全队列,让子进程把结果（处理了多少张图）传回主进程
        result_queue = mp.Queue()
        processes = []  #用来保存所有创建的进程
        for gpu_id, chunk in zip(gpu_ids, chunks):
            p = mp.Process(
                target=worker,
                args=(gpu_id, chunk, DATA_DIR, CACHE_DIR,
                      args.batch_size, result_queue),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        total = 0
        while not result_queue.empty():
            gpu_id, count, msg = result_queue.get()
            logger.info(msg)
            total += count
    else:
        # ── Single GPU, no multiprocessing overhead ──
        logger.info("Single-GPU mode (no --gpus specified)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)
        stft_module = STFTModule().to(device)

        total = 0
        for mat_file in mat_files:
            try:
                n = process_mat_file(mat_file, DATA_DIR, CACHE_DIR, stft_module, args.batch_size)
                total += n
                logger.info("  %s — %d spectrograms", mat_file, n)
            except Exception as e:
                logger.error("Failed to process %s: %s", mat_file, e)

    logger.info("Done. %d total spectrograms saved to %s", total, CACHE_DIR)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

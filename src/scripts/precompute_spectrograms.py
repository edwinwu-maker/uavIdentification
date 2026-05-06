"""
Pre-compute spectrograms from .mat IQ files using GPU torch.stft.
Output: .npy files (2, 1024, 1024) float32, one per sample.

Usage: python src/scripts/precompute_spectrograms.py [--gpus 0,1]
"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.logger import logger

# Match paper parameters + transforms.py
SAMPLE_LENGTH = 1_000_000
FS = 100e6
N_FFT = 1024
HOP_LENGTH = 512  # nperseg - noverlap = 1024 - 512
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


def parse_label(filename: str) -> int:
    basename = os.path.basename(filename)
    type_code = basename.split("_")[0]
    return LABEL_MAPPING[type_code]


class STFTModule(nn.Module):
    """nn.Module wrapper for GPU STFT — required for nn.DataParallel."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN_LENGTH))

    def forward(self, iq_batch: torch.Tensor) -> torch.Tensor:
        """
        iq_batch: (B, 2, SAMPLE_LENGTH) complex64
        returns: (B, 2, 1024, 1024) float32
        B: batch size, 2: channels, SAMPLE_LENGTH: IQ points per sample
        """
        B, C, L = iq_batch.shape
        results = []
        for ch in range(C):
            sig = iq_batch[:, ch, :]  # (B, L)
            Zxx = torch.stft(
                sig, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                window=self.window, return_complex=True, center=True,
            )  # (B, N_FFT, n_frames, 2=real+imag)
            Zxx = torch.fft.fftshift(Zxx, dim=1)  # shift zero-freq to center
            Zxx = Zxx[:, :, :SPEC_TIME_BINS]  # (B, N_FFT, 1024)
            eps = torch.finfo(torch.float32).eps
            Zxx_db = 20.0 * torch.log10(Zxx.abs() + eps)  # (B, N_FFT, 1024)
            mean = Zxx_db.mean(dim=(1, 2), keepdim=True)
            std = Zxx_db.std(dim=(1, 2), keepdim=True)
            Zxx_norm = (Zxx_db - mean) / (std + 1e-8)
            results.append(Zxx_norm)

        return torch.stack(results, dim=1)  # (B, 2, 1024, 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs, e.g. '0,1'")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="STFT batch size per GPU step")
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info("CUDA_VISIBLE_DEVICES set to: %s", args.gpus)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU count: %d", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            logger.info("  GPU %d: %s", i, torch.cuda.get_device_name(i))

    stft_module = STFTModule()
    if torch.cuda.device_count() > 1:
        stft_module = nn.DataParallel(stft_module)
        logger.info("Using DataParallel across %d GPUs", torch.cuda.device_count())
    stft_module = stft_module.to(device)

    if os.name == "nt":
        DATA_DIR = "E:/dataSet/DroneRFa"
        CACHE_DIR = "E:/dataSet/DroneRFa/spectrogram_cache"
    else:
        DATA_DIR = "/mnt/data/wurixin/DroneRFa"
        CACHE_DIR = "/mnt/data/wurixin/DroneRFa/spectrogram_cache"

    os.makedirs(CACHE_DIR, exist_ok=True)

    mat_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".mat")])
    logger.info("Found %d .mat files", len(mat_files))

    total_samples = 0
    for mat_file in tqdm(mat_files, desc="Pre-computing spectrograms"):
        mat_path = os.path.join(DATA_DIR, mat_file)
        base_name = os.path.splitext(mat_file)[0]  # e.g. T0001_flight1

        try:
            with h5py.File(mat_path, "r") as f:
                total_points = int(f["RF0_I"].shape[1])
                num_samples = total_points // SAMPLE_LENGTH

                for sample_idx in range(0, num_samples, args.batch_size):
                    batch_end = min(sample_idx + args.batch_size, num_samples)
                    batch_chunks = []

                    for i in range(sample_idx, batch_end):
                        offset = i * SAMPLE_LENGTH
                        end = offset + SAMPLE_LENGTH
                        ch0 = f["RF0_I"][0, offset:end] + 1j * f["RF0_Q"][0, offset:end]
                        ch1 = f["RF1_I"][0, offset:end] + 1j * f["RF1_Q"][0, offset:end]
                        iq = np.stack([ch0, ch1], axis=0)  # (2, SAMPLE_LENGTH)
                        batch_chunks.append(iq)

                    with torch.no_grad():
                        iq_batch = torch.from_numpy(np.stack(batch_chunks))  # (B, 2, L)
                        specs = stft_module(iq_batch).cpu()

                    for j, i in enumerate(range(sample_idx, batch_end)):
                        offset = i * SAMPLE_LENGTH
                        save_name = f"{base_name}_{offset:08d}.npy"
                        np.save(os.path.join(CACHE_DIR, save_name), specs[j].numpy())

                total_samples += num_samples

        except Exception as e:
            logger.error("Failed to process %s: %s", mat_file, e)

    logger.info("Done. %d total spectrograms saved to %s", total_samples, CACHE_DIR)


if __name__ == "__main__":
    main()

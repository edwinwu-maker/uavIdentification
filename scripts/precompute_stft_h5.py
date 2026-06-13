"""
Convert .mat IQ files -> one .h5 per file with pre-computed stfts.

Each .mat is converted independently to a same-named .h5 in the output
directory.

Output HDF5 structure (per file):
  /stft  (N, 2, 1024, 1024) float32
  /labels        (N,) int64

Usage:
  python src/scripts/precompute_stft_h5.py
  python src/scripts/precompute_stft_h5.py --data-dir ~/Desktop/dataset/droneRFa
  python src/scripts/precompute_stft_h5.py --data-dir ... --output-dir ...
  python src/scripts/precompute_stft_h5.py --data-dir ... --device cuda:0
  python src/scripts/precompute_stft_h5.py --data-dir ... --device cuda:2 --batch-size 64
  python src/scripts/precompute_stft_h5.py --data-dir ... --sample-length 1000000
  python src/scripts/precompute_stft_h5.py --data-dir ... --device mps
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.data.drone_rfa_io import count_iq_samples, default_raw_data_dir, parse_label, read_iq_batch
from src.utils.logger import logger

# ── Paper parameters (match transforms.py) ──
SAMPLE_LENGTH = 1_000_000
N_FFT = 1024
WIN_LENGTH = 1024
SPEC_TIME_BINS = 1024

_WINDOW: torch.Tensor | None = None
_WINDOW_DEVICE: str | None = None


def _get_window(device: str = "cpu") -> torch.Tensor:
    global _WINDOW, _WINDOW_DEVICE
    if _WINDOW is None or _WINDOW_DEVICE != device:
        _WINDOW = torch.hann_window(WIN_LENGTH, device=device)
        _WINDOW_DEVICE = device
    return _WINDOW


def _resolve_device(device: str) -> str:
    """Normalize the requested torch device and fall back to CPU when needed."""

    if device == "cpu":
        return device
    if device == "cuda":
        device = "cuda:0"
    if device == "mps":
        if not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            return "cpu"
        return device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return "cpu"
    return device


def compute_stft(iq_batch: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    iq_batch: (B, 2, SAMPLE_LENGTH) complex64
    device:  torch device string, e.g. "cpu", "cuda:0", "mps"
    returns: (B, 2, 1024, 1024) float32, z-score normalized per channel
    B: batch size, 2: channels, 1024: freq bins, 1024: time bins
    """
    B, C, L = iq_batch.shape
    hop_length = L // (SPEC_TIME_BINS - 1)
    window = _get_window(device)
    with torch.no_grad():
        sig = torch.from_numpy(iq_batch.reshape(B * C, L)).to(device)
        # Zxx:(B * 2, 1024, 1024)
        Zxx = torch.stft(
            sig, n_fft=N_FFT, hop_length=hop_length, win_length=WIN_LENGTH,
            window=window, return_complex=True, center=True,
        )
        Zxx = torch.fft.fftshift(Zxx, dim=1)
        Zxx = Zxx[:, :, :SPEC_TIME_BINS]
        eps = torch.finfo(torch.float32).eps
        Zxx_db = 20.0 * torch.log10(Zxx.abs() + eps)
        mean = Zxx_db.mean(dim=(1, 2), keepdim=True)
        std = Zxx_db.std(dim=(1, 2), keepdim=True)
        Zxx_norm = (Zxx_db - mean) / (std + 1e-8)
        Zxx_norm = Zxx_norm.reshape(B, C, N_FFT, SPEC_TIME_BINS)
    return Zxx_norm.cpu().numpy().astype(np.float32)


def process_one_mat(
    mat_path: str,
    output_dir: str,
    *,
    sample_length: int,
    batch_size: int,
    device: str,
) -> tuple[str, int]:
    """Convert one DroneRFa .mat file into one STFT .h5 file."""

    mat_file = os.path.basename(mat_path)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    out_path = os.path.join(output_dir, mat_file.replace(".mat", ".h5"))
    label = parse_label(mat_file)

    logger.info("Processing: %s", mat_file)

    with h5py.File(mat_path, "r") as src:
        num_samples = count_iq_samples(src, sample_length=sample_length)

        with h5py.File(out_path, "w") as h5f:
            h5f.create_dataset(
                "stft", shape=(num_samples, 2, N_FFT, SPEC_TIME_BINS),
                chunks=(1, 2, N_FFT, SPEC_TIME_BINS), dtype="f4",
            )
            h5f.create_dataset(
                "labels", shape=(num_samples,), chunks=None, dtype="i8",
            )

            batch_starts = range(0, num_samples, batch_size)
            for sample_idx in tqdm(batch_starts, total=len(batch_starts), desc=f"  {mat_file}"):
                batch_end = min(sample_idx + batch_size, num_samples)
                chunk_iq = read_iq_batch(
                    src,
                    sample_length=sample_length,
                    start_idx=sample_idx,
                    end_idx=batch_end,
                )

                batch_stft = compute_stft(chunk_iq, device)
                h5f["stft"][sample_idx:batch_end] = batch_stft
                h5f["labels"][sample_idx:batch_end] = label

    logger.info("Done: %s -> %d stfts", mat_file, num_samples)
    return mat_file, num_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .mat IQ → .h5 stft files")
    parser.add_argument("--data-dir", type=str, default=default_raw_data_dir(),
                        help="Directory containing .mat files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output .h5 files (default: <data-dir>/stft_h5)")
    parser.add_argument("--sample-length", type=int, default=SAMPLE_LENGTH,
                        help="Number of IQ samples per output stft")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="STFT batch size")
    parser.add_argument("--device", type=str, default="cpu",
                        help='Torch device for STFT, e.g. "cpu", "cuda", "cuda:0", or "mps"')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = os.path.expanduser(args.data_dir)
    output_dir = os.path.expanduser(args.output_dir) if args.output_dir else os.path.join(data_dir, "stft_h5")
    device = _resolve_device(args.device)

    logger.info("Using device: %s", device)

    os.makedirs(output_dir, exist_ok=True)

    mat_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".mat"))
    logger.info("Found %d .mat files in %s", len(mat_files), data_dir)
    logger.info("Output directory: %s", output_dir)

    total_samples = 0
    for mat_file in tqdm(mat_files, desc="Processing .mat files"):
        _, num_samples = process_one_mat(
            os.path.join(data_dir, mat_file),
            output_dir,
            sample_length=args.sample_length,
            batch_size=args.batch_size,
            device=device,
        )
        total_samples += num_samples

    logger.info("All done. %d files -> %d stfts", len(mat_files), total_samples)


if __name__ == "__main__":
    main()

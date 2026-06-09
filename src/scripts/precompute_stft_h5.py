"""
Convert .mat IQ files -> one .h5 per file with pre-computed spectrograms.

Each .mat is converted independently to a same-named .h5 in the output
directory.

Output HDF5 structure (per file):
  /stft  (N, 2, 1024, 1024) float32
  /labels        (N,) int64

Usage:
  python src/scripts/precompute_h5.py
  python src/scripts/precompute_h5.py --data-dir ~/Desktop/dataset/droneRFa
  python src/scripts/precompute_h5.py --data-dir ... --output-dir ...
  python src/scripts/precompute_h5.py --data-dir ... --device cuda:0
  python src/scripts/precompute_h5.py --data-dir ... --device mps
"""

import argparse
import os
import sys
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.utils.logger import logger

# ── Paper parameters (match transforms.py) ──
SAMPLE_LENGTH = 1_000_000
N_FFT = 1024
WIN_LENGTH = 1024
SPEC_TIME_BINS = 1024
HOP_LENGTH = SAMPLE_LENGTH // (SPEC_TIME_BINS - 1)

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
_WINDOW_DEVICE: str | None = None


def _default_data_dir() -> str:
    """Return the platform-specific default directory containing DroneRFa .mat files."""

    if os.name == "nt":
        return "E:/dataSet/DroneRFa"
    if sys.platform == "darwin":
        return os.path.expanduser("~/Desktop/dataset/droneRFa")
    return "/mnt/data/wurixin/DroneRFa"


def _parse_label(mat_file: str) -> int:
    """Parse a .mat file name and return its integer drone label."""

    drone_code = os.path.basename(mat_file).split("_")[0]
    return LABEL_MAPPING[drone_code]


def _get_window(device: str = "cpu") -> torch.Tensor:
    global _WINDOW, _WINDOW_DEVICE
    if _WINDOW is None or _WINDOW_DEVICE != device:
        _WINDOW = torch.hann_window(WIN_LENGTH, device=device)
        _WINDOW_DEVICE = device
    return _WINDOW


def compute_stft(iq_batch: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    iq_batch: (B, 2, SAMPLE_LENGTH) complex64
    device:  torch device string, e.g. "cpu", "cuda:0", "mps"
    returns: (B, 2, 1024, 1024) float32, z-score normalized per channel
    """
    B, C, L = iq_batch.shape
    hop_length = L // (SPEC_TIME_BINS - 1)
    window = _get_window(device)
    results = []
    with torch.no_grad():
        for ch in range(C):
            sig = torch.from_numpy(iq_batch[:, ch, :]).to(device)
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
            results.append(Zxx_norm.cpu())
    return torch.stack(results, dim=1).numpy().astype(np.float32)


def count_iq_samples(src: h5py.File, *, sample_length: int) -> int:
    """Return the number of complete fixed-length IQ samples in one .mat file."""

    total_points = int(src["RF0_I"].shape[1])
    return total_points // sample_length


def iter_iq_pairs(
    src: h5py.File,
    *,
    sample_length: int,
    start_idx: int,
    end_idx: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield fixed-length dual-channel complex IQ samples from one .mat file."""

    for sample_idx in range(start_idx, end_idx):
        offset = sample_idx * sample_length
        end = offset + sample_length
        ch0 = src["RF0_I"][0, offset:end] + 1j * src["RF0_Q"][0, offset:end]
        ch1 = src["RF1_I"][0, offset:end] + 1j * src["RF1_Q"][0, offset:end]
        yield sample_idx, ch0, ch1


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
    label = _parse_label(mat_file)

    logger.info("Processing: %s", mat_file)

    with h5py.File(mat_path, "r") as src:
        num_samples = count_iq_samples(src, sample_length=sample_length)

        with h5py.File(out_path, "w") as h5f:
            h5f.create_dataset(
                "stft", shape=(num_samples, 2, 1024, 1024),
                chunks=(1, 2, 1024, 1024), dtype="f4",
            )
            h5f.create_dataset(
                "labels", shape=(num_samples,), chunks=None, dtype="i8",
            )

            batch_starts = range(0, num_samples, batch_size)
            for sample_idx in tqdm(batch_starts, total=len(batch_starts), desc=f"  {mat_file}"):
                batch_end = min(sample_idx + batch_size, num_samples)
                actual_batch_size = batch_end - sample_idx
                chunk_iq = np.empty((actual_batch_size, 2, sample_length), dtype=np.complex64)

                samples = iter_iq_pairs(
                    src,
                    sample_length=sample_length,
                    start_idx=sample_idx,
                    end_idx=batch_end,
                )
                for k, (_idx, ch0, ch1) in enumerate(samples):
                    chunk_iq[k, 0] = ch0
                    chunk_iq[k, 1] = ch1

                batch_stft = compute_stft(chunk_iq, device)
                h5f["stft"][sample_idx:sample_idx + actual_batch_size] = batch_stft
                h5f["labels"][sample_idx:sample_idx + actual_batch_size] = [label] * actual_batch_size

    logger.info("Done: %s -> %d spectrograms", mat_file, num_samples)
    return mat_file, num_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .mat IQ → .h5 spectrogram files")
    parser.add_argument("--data-dir", type=str, default=_default_data_dir(),
                        help="Directory containing .mat files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output .h5 files (default: <data-dir>/stft_h5)")
    parser.add_argument("--sample-length", type=int, default=SAMPLE_LENGTH,
                        help="Number of IQ samples per output spectrogram")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="STFT batch size")
    parser.add_argument("--device", type=str, default="cpu",
                        help='Torch device for STFT, e.g. "cpu", "cuda", "cuda:0", or "mps"')
    return parser.parse_args()


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

    logger.info("All done. %d files -> %d spectrograms", len(mat_files), total_samples)


if __name__ == "__main__":
    main()

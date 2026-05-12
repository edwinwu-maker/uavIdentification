"""
Convert .mat IQ files → one .h5 per file with pre-computed spectrograms.

Multi-process with GPU-accelerated STFT. Each .mat is converted independently
to a same-named .h5 in the output directory.

Output HDF5 structure (per file):
  /stft  (N, 2, 1024, 1024) float32
  /labels        (N,) int64

Usage:
  python src/scripts/precompute_h5.py --data-dir /mnt/data/wurixin/DroneRFa
  python src/scripts/precompute_h5.py --data-dir ... --output-dir ... --num-workers 8
  python src/scripts/precompute_h5.py --data-dir ... --device cuda:0 cuda:1 cuda:2 cuda:3
"""

import argparse
import concurrent.futures
import os
import sys
from collections import defaultdict
from pathlib import Path
import signal

sys.path.insert(0, str(Path(__file__).parent.parent))
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
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
_WINDOW_DEVICE: str | None = None

def cleanup_executors(executors):
    for ex in executors:
        # 取消所有未完成的任务，不等待它们完成
        ex.shutdown(wait=False, cancel_futures=True)
    children = multiprocessing.active_children()
    for child in children:
        child.terminate()

def _get_window(device: str = "cpu") -> torch.Tensor:
    global _WINDOW, _WINDOW_DEVICE
    if _WINDOW is None or _WINDOW_DEVICE != device:
        _WINDOW = torch.hann_window(WIN_LENGTH, device=device)
        _WINDOW_DEVICE = device
    return _WINDOW


def compute_stft(iq_batch: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    iq_batch: (B, 2, SAMPLE_LENGTH) complex64
    device:  torch device string, e.g. "cuda:0", "cpu"
    returns: (B, 2, 1024, 1024) float32, z-score normalized per channel
    """
    B, C, _L = iq_batch.shape
    window = _get_window(device)
    results = []
    with torch.no_grad():
        for ch in range(C):
            sig = torch.from_numpy(iq_batch[:, ch, :]).to(device)
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
            results.append(Zxx_norm.cpu())
    return torch.stack(results, dim=1).numpy().astype(np.float32)


def _convert_one_file(args):
    """
    Module-level worker for ProcessPoolExecutor.
    args: (mat_file, data_dir, output_dir, batch_size, device)
    Returns: (mat_file, segment_count)
    """
    mat_file, data_dir, output_dir, batch_size, device = args
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
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
            "labels", shape=(num_samples,), chunks=None, dtype="i8",
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

                batch_stft = compute_stft(chunk_iq, device)
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
    parser.add_argument("--device", type=str, nargs="+", default=["cuda:0"],
                        help="Torch device(s) for STFT, round-robin across workers (e.g. cuda:0 cuda:1)")
    return parser.parse_args()


def main():
    args = parse_args()

    devices = args.device
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        devices = ["cpu"]
    else:
        devices = [d if d.startswith("cuda") or d == "cpu" else f"cuda:{d}" for d in devices]
    logger.info("Using devices: %s", devices)

    torch.set_num_threads(1)

    if args.data_dir is None:
        args.data_dir = "/mnt/data/wurixin/DroneRFa" if os.name != "nt" \
            else "E:/dataSet/DroneRFa"
    if args.output_dir is None:
        args.output_dir = args.data_dir
    if args.num_workers is None:
        args.num_workers = len(devices) * 2 if devices[0] != "cpu" else os.cpu_count() or 4

    os.makedirs(args.output_dir, exist_ok=True)

    mat_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".mat")])
    logger.info("Found %d .mat files in %s", len(mat_files), args.data_dir)
    logger.info("Output directory: %s", args.output_dir)

    # round-robin: group files by assigned GPU
    gpu_files: dict[str, list[str]] = defaultdict(list)
    for i, mf in enumerate(mat_files):
        gpu_files[devices[i % len(devices)]].append(mf)

    workers_per_gpu = max(1, args.num_workers // len(devices))
    logger.info("Using %d workers across %d GPUs (%d per GPU)", args.num_workers, len(devices), workers_per_gpu)

    # 存储所有【未来任务(future) + 它所属的进程池】
    all_futures: list[tuple[concurrent.futures.Future, concurrent.futures.ProcessPoolExecutor]] = []
    # 存储所有创建出来的进程池（每张GPU对应一个独立进程池）
    executors: list[concurrent.futures.ProcessPoolExecutor] = []
    try:
        # 遍历每一张 GPU（cuda:0, cuda:1...）
        for device in devices:
            files = gpu_files.get(device, [])   # 拿到分配给这张GPU的所有.mat文件
            if not files:
                continue
            nw = min(workers_per_gpu, len(files))
            ex = concurrent.futures.ProcessPoolExecutor(max_workers=nw)
            executors.append(ex)
            logger.info("  GPU %s: %d files, %d workers", device, len(files), nw)
            for mf in files:
                fa = (mf, args.data_dir, args.output_dir, args.batch_size, device)
                fut = ex.submit(_convert_one_file, fa)
                all_futures.append((fut, ex))

        total_segments = 0
        for fut in tqdm(
            concurrent.futures.as_completed([f for f, _ in all_futures]),   # 监听所有任务，谁先做完就先处理谁
            total=len(all_futures), desc="Converting",
        ):
            mf, n = fut.result()
            total_segments += n

        for ex in executors:
            ex.shutdown()

        logger.info("Done. %d files → %d spectrograms", len(mat_files), total_segments)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user, shutting down workers...")
        cleanup_executors(executors)
        logger.info("All workers terminated. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()

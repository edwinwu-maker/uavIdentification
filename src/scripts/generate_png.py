"""Generate spectrogram PNGs from pre-computed .h5 files.

Reads .h5 files produced by precompute_h5.py and generates PNG images.
Each PNG contains two spectrograms (Channel 0 and Channel 1).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import multiprocessing
import tqdm

from utils.logger import logger
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS = 100e6
SAMPLE_LENGTH = 1_000_000
SAMPLE_DURATION = SAMPLE_LENGTH / FS


def plot_dual_channel(stft_sample, save_path, sample_idx, label=None):
    """Plot a dual-channel spectrogram and save as PNG.

    stft_sample: (2, 1024, 1024) float32, z-score normalized STFT.
    """
    n_freqs, n_times = stft_sample.shape[1], stft_sample.shape[2]
    freqs = np.fft.fftshift(np.fft.fftfreq(n_freqs, 1.0 / FS))
    times = np.linspace(0, SAMPLE_DURATION, n_times)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    for ax, ch, ch_name in [(ax0, 0, "Channel 0"), (ax1, 1, "Channel 1")]:
        im = ax.pcolormesh(
            freqs / 1e6, times * 1e3, stft_sample[ch].T,
            shading="auto", cmap="viridis",
        )
        title = f"{ch_name} — Sample {sample_idx}"
        if label is not None:
            title += f" (label={label})"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Frequency (MHz)", fontsize=10)
        ax.set_ylabel("Time (ms)", fontsize=10)

    cbar = fig.colorbar(im, ax=[ax0, ax1], shrink=0.92)
    cbar.set_label("Normalized Power (z-score)", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _task_generator(h5f, name_no_extension, drone_code, save_root):
    """ 生成器函数
     Yield (stft_slice, idx, label, save_path) one sample at a time.
    """
    stft = h5f["stft"]
    labels = h5f["labels"]
    save_sub_dir = os.path.join(save_root, drone_code)
    for i in range(stft.shape[0]):
        png_name = f"{name_no_extension}_sample_{i:04d}.png"
        save_path = os.path.join(save_sub_dir, png_name)
        yield (stft[i], i, int(labels[i]), save_path)


def _process_sample(args):
    """Worker: plot and save a single spectrogram sample."""
    stft_slice, idx, label, save_path = args
    try:
        plot_dual_channel(stft_slice, save_path, idx, label)
    except Exception as e:
        logger.error(f"Sample {idx}: {e}")


def process_one_h5(h5_path, save_root):
    """Read a .h5 file and generate dual-channel PNGs for all samples."""
    h5_name = os.path.basename(h5_path)
    name_no_extension = os.path.splitext(h5_name)[0]
    drone_code = name_no_extension.split("_")[0]

    logger.info(f"Processing: {h5_name}")

    with h5py.File(h5_path, "r") as h5f:
        num_samples = h5f["stft"].shape[0]

        num_workers = min(2, multiprocessing.cpu_count() // 2)
        with multiprocessing.Pool(num_workers) as pool:
            # 此时_task_generator函数内部一行都没跑！只是准备好，等着你要数据。
            tasks = _task_generator(h5f, name_no_extension, drone_code, save_root)
            # imap_unordered 返回的是迭代器，惰性取值，不写 list 不会真正跑任务。
            list(tqdm.tqdm(
                # chunksize=1 多进程池一次只从生成器拿 1 个任务，分给空闲进程。
                pool.imap_unordered(_process_sample, tasks, chunksize=1),
                total=num_samples,
                desc=f"  {h5_name}",
            ))

    logger.info(f"Done: {h5_name}")


if __name__ == "__main__":
    if os.name == "nt":
        DATA_DIR = "E:/dataSet/DroneRFa"
        SAVE_ROOT = "E:/dataSet/DroneRFa/picture"
    else:
        DATA_DIR = "/mnt/data/wurixin/DroneRFa"
        SAVE_ROOT = "/mnt/data/wurixin/DroneRFa/picture"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    h5_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".h5")
    ]
    logger.info(f"Found {len(h5_files)} .h5 files")

    for h5_path in tqdm.tqdm(h5_files, desc="Processing .h5 files"):
        process_one_h5(h5_path, SAVE_ROOT)

    logger.info("All done!")

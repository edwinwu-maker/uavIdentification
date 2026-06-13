import os
import sys

import h5py
import numpy as np

LABEL_MAPPING = {
    "T0000": 0, "T0001": 1, "T0010": 2, "T0011": 3,
    "T0100": 4, "T0101": 5, "T0110": 6, "T0111": 7,
    "T1000": 8, "T1001": 9, "T1010": 10, "T1011": 11,
    "T1100": 12, "T1101": 13, "T1110": 14, "T1111": 15,
    "T10000": 16, "T10001": 17, "T10010": 18, "T10011": 19,
    "T10100": 20, "T10101": 21, "T10110": 22, "T10111": 23,
    "T11000": 24,
}


def default_raw_data_dir() -> str:
    if os.name == "nt":
        return "E:/dataSet/DroneRFa"
    if sys.platform == "darwin":
        return os.path.expanduser("~/Desktop/dataset/droneRFa")
    return "/mnt/data/wurixin/DroneRFa"


def parse_label(mat_file: str) -> int:
    drone_code = os.path.basename(mat_file).split("_")[0]
    return LABEL_MAPPING[drone_code]


def count_iq_samples(src: h5py.File, *, sample_length: int, max_samples: int | None = None) -> int:
    total_points = int(src["RF0_I"].shape[1])
    num_samples = total_points // sample_length
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)
    return num_samples


def read_iq_batch(
    src: h5py.File,
    *,
    sample_length: int,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    batch_size = end_idx - start_idx
    offset = start_idx * sample_length
    end = end_idx * sample_length

    rf0_i = src["RF0_I"][0, offset:end].reshape(batch_size, sample_length)
    rf0_q = src["RF0_Q"][0, offset:end].reshape(batch_size, sample_length)
    rf1_i = src["RF1_I"][0, offset:end].reshape(batch_size, sample_length)
    rf1_q = src["RF1_Q"][0, offset:end].reshape(batch_size, sample_length)

    iq_batch = np.empty((batch_size, 2, sample_length), dtype=np.complex64)
    iq_batch[:, 0, :].real = rf0_i
    iq_batch[:, 0, :].imag = rf0_q
    iq_batch[:, 1, :].real = rf1_i
    iq_batch[:, 1, :].imag = rf1_q
    return iq_batch


def iter_iq_pairs(
    src: h5py.File,
    *,
    sample_length: int,
    num_samples: int,
):
    for sample_idx in range(num_samples):
        offset = sample_idx * sample_length
        end = offset + sample_length
        ch0 = src["RF0_I"][0, offset:end] + 1j * src["RF0_Q"][0, offset:end]
        ch1 = src["RF1_I"][0, offset:end] + 1j * src["RF1_Q"][0, offset:end]
        yield sample_idx, ch0, ch1

import os
import sys

import h5py
import numpy as np

LABEL_MAPPING = {
    "T0000": 0, "T0010": 1, "T0011": 2, "T0101": 3,
    "T0110": 4, "T0111": 5, "T1000": 6, "T1010": 7,
    "T1011": 8, "T1100": 9, "T1101": 10, "T1110": 11,
    "T1111": 12, "T10000": 13,
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
    """根据 HDF5 文件(原始.mat)里的总点数，计算能切出多少个固定长度的 IQ 样本"""
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
    """ 从 HDF5 文件(.mat 文件)里按样本区间读取 IQ 数据，并把实部/虚部重新组装成复数数组 """
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

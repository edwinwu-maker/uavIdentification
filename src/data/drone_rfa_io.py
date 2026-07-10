import os
import re
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


def rf_channel_for_file(path: str | os.PathLike[str]) -> int:
    """根据文件名中唯一的四位二进制 S 编码返回目标 RF 通道。"""

    stem = os.path.splitext(os.path.basename(os.fspath(path)))[0]
    s_tokens = [token for token in stem.split("_") if token.startswith("S")]
    if len(s_tokens) != 1 or re.fullmatch(r"S[01]{4}", s_tokens[0]) is None:
        raise ValueError(
            f"DroneRFa filename must contain exactly one four-bit binary S code: {os.path.basename(os.fspath(path))}"
        )
    signal_code = int(s_tokens[0][1:], 2)
    return 0 if signal_code <= 0b0111 else 1


def group_mat_files_by_class(mat_files: list[str]) -> dict[str, list[str]]:
    """按 DroneRFa 类别代码分组 .mat 文件，并保持每组文件名排序稳定。"""

    # 这里循环遍历的是字典的 keys："T0000", "T0010", ..., "T10000"
    grouped = {class_code: [] for class_code in LABEL_MAPPING}
    for mat_file in sorted(mat_files):
        class_code = os.path.basename(mat_file).split("_")[0]
        if class_code not in LABEL_MAPPING:
            raise ValueError(f"Unknown class code in .mat file: {class_code}")
        grouped[class_code].append(mat_file)
    return grouped


def count_iq_samples(
    src: h5py.File,
    *,
    rf_channel: int,
    sample_length: int,
    max_samples: int | None = None,
) -> int:
    """根据 HDF5 文件(原始.mat)里的总点数，计算能切出多少个固定长度的 IQ 样本"""
    if rf_channel not in (0, 1):
        raise ValueError(f"rf_channel must be 0 or 1, got {rf_channel}")
    total_points = int(src[f"RF{rf_channel}_I"].shape[1])
    num_samples = total_points // sample_length
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)
    return num_samples


def read_iq_batch(
    src: h5py.File,
    *,
    rf_channel: int,
    sample_length: int,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    """ 从 HDF5 文件(.mat 文件)里按样本区间读取 IQ 数据，并把实部/虚部重新组装成复数数组 """
    batch_size = end_idx - start_idx
    offset = start_idx * sample_length
    end = end_idx * sample_length

    if rf_channel not in (0, 1):
        raise ValueError(f"rf_channel must be 0 or 1, got {rf_channel}")
    rf_i = src[f"RF{rf_channel}_I"][0, offset:end].reshape(batch_size, sample_length)
    rf_q = src[f"RF{rf_channel}_Q"][0, offset:end].reshape(batch_size, sample_length)

    iq_batch = np.empty((batch_size, 1, sample_length), dtype=np.complex64)
    iq_batch[:, 0, :].real = rf_i
    iq_batch[:, 0, :].imag = rf_q
    return iq_batch

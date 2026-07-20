import os
import re
import sys
from dataclasses import dataclass

import h5py
import numpy as np

DRONE_CLASS_CODES = tuple(f"T{label:04b}" for label in range(17))
CONTROLLER_CLASS_CODES = tuple(f"T{label:b}" for label in range(17, 25))
DATASET_CLASS_CODES = frozenset((*DRONE_CLASS_CODES, *CONTROLLER_CLASS_CODES))
LABEL_MAPPING = {class_code: label for label, class_code in enumerate(DRONE_CLASS_CODES)}


@dataclass(frozen=True)
class SampleRecord:
    path: str
    sample_idx: int
    label: int
    rf_channel: int


class H5FileHandleCache:
    """缓存并统一关闭 HDF5 文件句柄。"""

    def __init__(self) -> None:
        self._files: dict[str, h5py.File] = {}

    def get(self, path: str) -> h5py.File:
        if path not in self._files:
            self._files[path] = h5py.File(path, "r", rdcc_nbytes=64 * 1024 * 1024)
        return self._files[path]

    def close(self) -> None:
        for file_obj in self._files.values():
            file_obj.close()
        self._files.clear()


def default_raw_data_dir() -> str:
    if os.name == "nt":
        return "E:/dataSet/DroneRFa"
    if sys.platform == "darwin":
        return os.path.expanduser("~/Desktop/dataset/droneRFa")
    return "/mnt/data/wurixin/DroneRFa"


def parse_label(mat_file: str) -> int:
    drone_code = os.path.basename(mat_file).split("_")[0]
    return LABEL_MAPPING[drone_code]


def rf_channels_for_file(path: str | os.PathLike[str]) -> tuple[int, ...]:
    """背景类使用双通道，其他类别根据四位二进制 S 编码选择单通道。"""

    stem = os.path.splitext(os.path.basename(os.fspath(path)))[0]
    class_code = stem.split("_")[0]
    if class_code == "T0000":
        return (0, 1)

    s_tokens = [token for token in stem.split("_") if token.startswith("S")]
    if len(s_tokens) != 1 or re.fullmatch(r"S[01]{4}", s_tokens[0]) is None:
        raise ValueError(
            f"DroneRFa filename must contain exactly one four-bit binary S code: {os.path.basename(os.fspath(path))}"
        )
    signal_code = int(s_tokens[0][1:], 2)
    return (0,) if signal_code <= 0b0111 else (1,)


def group_mat_files_by_class(mat_files: list[str]) -> dict[str, list[str]]:
    """按 DroneRFa 类别代码分组 .mat 文件，并保持每组文件名排序稳定。"""

    # 这里循环遍历的是字典的 keys："T0000", "T0001", ..., "T10000"
    grouped = {class_code: [] for class_code in LABEL_MAPPING}
    for mat_file in sorted(mat_files):
        class_code = os.path.basename(mat_file).split("_")[0]
        if class_code not in LABEL_MAPPING:
            raise ValueError(f"Unknown class code in .mat file: {class_code}")
        grouped[class_code].append(mat_file)
    return grouped


def select_mat_files(
    data_dir: str,
    *,
    max_files: int | None = None,
    files_per_class: int | None = None,
    include_labels: list[int] | tuple[int, ...] | set[int] | None = None,
) -> list[str]:
    """稳定选择原始 .mat 文件；按类别选择时优先于总文件数限制。"""

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Raw data directory does not exist: {data_dir}")

    allowed_labels = None
    if include_labels is not None:
        allowed_labels = set(int(label) for label in include_labels)
        valid_labels = set(LABEL_MAPPING.values())
        invalid = sorted(allowed_labels - valid_labels)
        if invalid:
            raise ValueError(f"include_labels contains unknown labels: {invalid}")

    mat_files: list[str] = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".mat"):
            continue
        class_code = os.path.basename(filename).split("_")[0]
        if class_code not in DATASET_CLASS_CODES:
            raise ValueError(f"Unknown class code in .mat file: {class_code}")
        if class_code in LABEL_MAPPING:
            mat_files.append(filename)
    if allowed_labels is not None:
        filtered_files: list[str] = []
        for filename in mat_files:
            class_code = os.path.basename(filename).split("_")[0]
            if class_code not in LABEL_MAPPING:
                raise ValueError(f"Unknown class code in .mat file: {class_code}")
            if LABEL_MAPPING[class_code] in allowed_labels:
                filtered_files.append(filename)
        mat_files = filtered_files
    if files_per_class is None:
        return mat_files[:max_files] if max_files is not None else mat_files
    if files_per_class <= 0:
        raise ValueError("files_per_class must be positive")

    grouped = group_mat_files_by_class(mat_files)
    selected: list[str] = []
    for class_code in LABEL_MAPPING:
        if allowed_labels is not None and LABEL_MAPPING[class_code] not in allowed_labels:
            continue
        selected.extend(grouped[class_code][:files_per_class])
    return selected


def list_mat_file_labels(data_dir: str) -> list[tuple[str, int]]:
    """仅扫描文件名，返回 manifest 校验所需的原始文件路径和标签。"""

    return [
        (os.path.join(data_dir, filename), parse_label(filename))
        for filename in select_mat_files(data_dir)
    ]


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


def load_iq_sample(
    src: h5py.File,
    record: SampleRecord,
    *,
    sample_length: int,
) -> np.ndarray:
    """从已打开的 HDF5 文件中读取一条复数 IQ 样本。"""

    iq_batch = read_iq_batch(
        src,
        rf_channel=record.rf_channel,
        sample_length=sample_length,
        start_idx=record.sample_idx,
        end_idx=record.sample_idx + 1,
    )
    return iq_batch[0]


def build_sample_index(
    data_dir: str,
    *,
    sample_length: int,
    max_files: int | None = None,
    files_per_class: int | None = None,
    max_samples_per_file: int | None = None,
    file_ids: set[str] | None = None,
) -> list[SampleRecord]:
    """扫描原始 .mat 文件，生成稳定的样本索引。"""

    sample_index: list[SampleRecord] = []
    mat_files = select_mat_files(
        data_dir,
        max_files=max_files,
        files_per_class=files_per_class,
    )
    if file_ids is not None:
        mat_files = [
            filename for filename in mat_files
            if os.path.splitext(filename)[0] in file_ids
        ]
    for filename in mat_files:
        path = os.path.join(data_dir, filename)
        label = parse_label(filename)
        with h5py.File(path, "r") as src:
            for rf_channel in rf_channels_for_file(filename):
                num_samples = count_iq_samples(
                    src,
                    rf_channel=rf_channel,
                    sample_length=sample_length,
                    max_samples=max_samples_per_file,
                )
                for sample_idx in range(num_samples):
                    sample_index.append(SampleRecord(
                        path=path,
                        sample_idx=sample_idx,
                        label=label,
                        rf_channel=rf_channel,
                    ))
    return sample_index

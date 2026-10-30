import os
import re
from dataclasses import dataclass

import h5py
import numpy as np

DRONE_CLASS_CODES = tuple(f"T{label:04b}" for label in range(17))
CONTROLLER_CLASS_CODES = tuple(f"T{label:b}" for label in range(17, 25))
DATASET_CLASS_CODES = frozenset((*DRONE_CLASS_CODES, *CONTROLLER_CLASS_CODES))
# T0001 经人工鉴定为基于 WiFi 的射频信号，易与 WiFi 混淆，不纳入研究。
EXCLUDED_CLASS_CODES = frozenset(("T0001",))
NUM_ORIGINAL_CLASSES = len(DRONE_CLASS_CODES)
# 保留 T 编码的原始整数标签；连续模型标签由 ClassMapping 生成。
# 部分示例：{"T0000": 0, "T0010": 2, "T0011": 3,...,"T1111": 15, "T10000": 16}。
LABEL_MAPPING = {
    class_code: label for label, class_code in enumerate(DRONE_CLASS_CODES)
    if class_code not in EXCLUDED_CLASS_CODES
}
RESEARCH_LABELS = tuple(LABEL_MAPPING.values())


@dataclass(frozen=True)
class SampleRecord:
    """不可变的单个 IQ 样本索引记录，仅保存位置和类别信息。

    sample_idx 按构建索引时的 sample_length 划分，读取时需沿用相同的样本长度。
    """

    path: str        # 原始 .mat 文件路径，文件内容为 HDF5 格式。
    sample_idx: int  # 文件内所选 RF 通道的定长样本编号，从 0 开始。
    label: int       # LABEL_MAPPING 中的原始整数类别标签。
    rf_channel: int  # RF 通道编号，取 0 或 1，用于选择对应的 I/Q 数据集。


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


def parse_label(mat_file: str) -> int:
    """从 DroneRFa 文件名中解析原始整数类别标签。

    输入：
        mat_file: .mat 文件名，文件名以 T 类别编码和下划线开头，例如 T0010_S0000.mat。
    输出：
        LABEL_MAPPING 中对应的原始整数标签，例如 T0010 返回 2；
    """
    drone_code = os.path.basename(mat_file).split("_")[0]
    return LABEL_MAPPING[drone_code]


def rf_channels_from_filename(mat_file: str | os.PathLike[str]) -> tuple[int, ...]:
    """根据文件名确定需要读取的 RF 通道，不打开文件。

    输入：
        mat_file: .mat 文件名或路径；非背景类文件名须包含一个四位二进制 S 编码，例如 T0010_S1000.mat。
    输出：
        RF 通道编号元组：T0000 背景类返回 (0, 1)，不检查 S 编码；
        其他类别中 S0000～S0111 返回 (0,)，S1000～S1111 返回 (1,)。
        非背景类的 S 编码缺失、重复或格式错误时抛出 ValueError。
    """

    stem = os.path.splitext(os.path.basename(os.fspath(mat_file)))[0]
    class_code = stem.split("_")[0]
    if class_code == "T0000":
        return (0, 1)

    s_tokens = [token for token in stem.split("_") if token.startswith("S")]
    if len(s_tokens) != 1 or re.fullmatch(r"S[01]{4}", s_tokens[0]) is None:
        raise ValueError(
            f"DroneRFa filename must contain exactly one four-bit binary S code: {os.path.basename(os.fspath(mat_file))}"
        )
    signal_code = int(s_tokens[0][1:], 2)
    return (0,) if signal_code <= 0b0111 else (1,)


def group_mat_files_by_class(mat_files: list[str]) -> dict[str, list[str]]:
    """按 DroneRFa 类别代码分组 .mat 文件，并保持每组排序稳定。

    输入：
        mat_files: .mat 文件名列表。
    输出：
        以 LABEL_MAPPING 中全部类别编码为key、文件名列表为 value 的字典。
    """

    # 保留 T0000 背景及 T0010-T10000 无人机类别。
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
    """稳定选择原始 .mat 文件；按类别选择时优先于总文件数限制。

    输入：
        data_dir: 原始数据目录，仅扫描该目录下以 .mat 结尾的名称。
        max_files: 总文件数上限; None 表示不限制，仅 files_per_class 为 None 时生效。
        files_per_class: 每类文件数上限, 须为正整数; None 表示不按类别限量。
        include_labels: 要保留的原始整数标签集合; None 表示全部研究类别。
    输出：
        筛选后的文件名列表，不含目录前缀；跳过 T0001 和遥控器类别。
        未按类别限量时按文件名字典序排列，否则按 LABEL_MAPPING 的类别顺序
        拼接各类文件名排序后的前 files_per_class 个文件。
    """

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Raw data directory does not exist: {data_dir}")

    # 规范化并校验 include_labels：None 不额外筛选，指定标签转为整数并去重，非法标签排序后报错。
    allowed_labels = None
    if include_labels is not None:
        allowed_labels = {int(label) for label in include_labels}
        invalid = sorted(allowed_labels.difference(LABEL_MAPPING.values()))
        if invalid:
            raise ValueError(f"include_labels contains unknown labels: {invalid}")

    # 按名称排序扫描 .mat 条目，并按研究类别和指定标签筛选；未知类别编码报错。
    mat_files: list[str] = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".mat"):
            continue
        class_code = filename.split("_", 1)[0]
        if class_code in LABEL_MAPPING:
            if allowed_labels is None or LABEL_MAPPING[class_code] in allowed_labels:
                mat_files.append(filename)
        elif class_code not in DATASET_CLASS_CODES:
            raise ValueError(f"Unknown class code in .mat file: {class_code}")

    # 未按类别限量时按 max_files 截取并返回（None 不限量），否则校验 files_per_class 为正数。
    if files_per_class is None:
        return mat_files[:max_files] if max_files is not None else mat_files
    if files_per_class <= 0:
        raise ValueError("files_per_class must be positive")

    # 按类别顺序收集允许类别中排序靠前的 files_per_class 个文件，此模式忽略 max_files。
    grouped = group_mat_files_by_class(mat_files)
    selected: list[str] = []
    for class_code in LABEL_MAPPING:
        if allowed_labels is not None and LABEL_MAPPING[class_code] not in allowed_labels:
            continue
        selected.extend(grouped[class_code][:files_per_class])
    return selected


def list_mat_file_labels(data_dir: str) -> list[tuple[str, int]]:
    """仅扫描文件名，返回 manifest 校验所需的原始文件路径和标签。

    输入：
        data_dir：原始 .mat 数据目录。
    输出：
        按文件名字典序排列的 (文件路径, 原始整数标签) 列表，仅包含研究类别。
        路径由 data_dir 与文件名拼接得到，不保证为绝对路径；不读取文件内容。
    """

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
    """根据所选 RF 通道的采样点数，计算完整定长 IQ 样本数.

    输入：
        src: 已打开的原始 .mat HDF5 文件.
        rf_channel: RF 通道编号，取 0 或 1.
        sample_length: 每个样本的采样点数(正整数).
        max_samples: 返回样本数上限, 非负整数; None 表示不限制.
    输出：
        完整样本数 int.
    """
    if rf_channel not in (0, 1):
        raise ValueError(f"rf_channel must be 0 or 1, got {rf_channel}")
    total_points = int(src[f"RF{rf_channel}_I"].shape[1])
    num_samples = total_points // sample_length
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)
    return num_samples


def read_iq_samples(
    h5_file: h5py.File,
    *,
    rf_channel: int,
    sample_length: int,
    start_sample_idx: int,
    end_sample_idx: int,
) -> np.ndarray:
    """按样本区间读取 .mat(HDF5 格式) 文件中的 IQ 数据，组装为复数数组。

    输入：
        h5_file: 已打开的 HDF5 文件，所选通道的 RF{rf_channel}_I 和
            RF{rf_channel}_Q 数据集形状须为 (1, 总采样点数)。
        rf_channel: RF 通道编号，取 0 或 1。
        sample_length: 每个样本的采样点数，须为正整数。
        start_sample_idx: 起始样本编号, 从 0 开始，包含该样本。
        end_sample_idx: 结束样本编号，不包含该样本；须满足
            0 <= start_sample_idx <= end_sample_idx, 且对应数据区间不超出所选通道。
    输出：
        形状为 (end_sample_idx - start_sample_idx, 1, sample_length) 的 np.complex64 数组，
        实部为 I、虚部为 Q; 中间维度表示单个已选 RF 通道。
    """
    num_samples = end_sample_idx - start_sample_idx
    start_point_idx = start_sample_idx * sample_length
    end_point_idx = end_sample_idx * sample_length

    if rf_channel not in (0, 1):
        raise ValueError(f"rf_channel must be 0 or 1, got {rf_channel}")
    i_samples = h5_file[f"RF{rf_channel}_I"][0, start_point_idx:end_point_idx].reshape(num_samples, sample_length)
    q_samples = h5_file[f"RF{rf_channel}_Q"][0, start_point_idx:end_point_idx].reshape(num_samples, sample_length)

    iq_samples = np.empty((num_samples, 1, sample_length), dtype=np.complex64)
    iq_samples[:, 0, :].real = i_samples
    iq_samples[:, 0, :].imag = q_samples
    return iq_samples


def build_sample_index(
    data_dir: str,
    *,
    sample_length: int,
    max_files: int | None = None,
    files_per_class: int | None = None,
    max_samples_per_file: int | None = None,
    file_ids: set[str] | None = None,
) -> list[SampleRecord]:
    """扫描原始 .mat 文件，生成稳定的样本索引。

    输入：
        data_dir: 原始 .mat 数据目录。
        sample_length: 每个样本的采样点数；各 RF 通道独立切分，不足一个完整样本的尾部数据不计入索引。
        max_files: 总文件数上限； None 表示不限制，仅 files_per_class 为 None 时生效。
        files_per_class: 每类文件数上限； None 表示不按类别限量。
        max_samples_per_file: 每个文件的每个所选 RF 通道的样本数上限;
            None 表示不限制，背景文件双通道分别应用此上限。
        file_ids: 要保留的文件标识集合，标识为不含路径及扩展名的文件名；
            None 表示不额外筛选，此筛选在文件数量限制之后应用。
    输出：
        SampleRecord 列表， 按选定文件顺序、RF 通道顺序和样本编号递增顺序排列；
        每条记录保存文件路径、从 0 开始的通道内样本编号、原始整数标签和 RF 通道。
        仅读取数据集形状以计算样本数，不加载 IQ 数据；读取样本时须沿用 sample_length。
    """

    sample_index: list[SampleRecord] = []
    # 稳定选择研究类别的 .mat 文件，优先应用每类文件数限制，否则应用总文件数限制。
    mat_files = select_mat_files(
        data_dir,
        max_files=max_files,
        files_per_class=files_per_class,
    )
    # 在文件数量限制后按不含扩展名的文件标识筛选；file_ids 为 None 时保留全部已选文件。
    if file_ids is not None:
        mat_files = [
            filename for filename in mat_files
            if os.path.splitext(filename)[0] in file_ids
        ]
    # 逐文件按选定 RF 通道统计完整样本数，依次记录路径、原始标签、通道及样本编号，不读取 IQ 数据。
    for filename in mat_files:
        path = os.path.join(data_dir, filename)
        label = parse_label(filename)
        with h5py.File(path, "r") as src:
            for rf_channel in rf_channels_from_filename(filename):
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

"""STFT 深度聚类的样本索引、预处理、审核选样与结果导出。"""

from __future__ import annotations

import csv
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.drone_rfa_io import EXCLUDED_CLASS_CODES, RESEARCH_LABELS
from src.preprocess.stft import clip_scale_and_pool_stft
from src.utils.files import h5_attr_to_text
from src.utils.plot import import_pyplot
from src.utils.samples import allocate_evenly, prioritize_spread_indices

REQUIRED_SAMPLE_DATASETS = (
    "stft",
    "labels",
    "is_clean",
    "snr_db",
    "source_sample_idx",
    "augmentation_variant",
    "rf_channel",
)


# 盲审角色及样本数：mapping 用于簇语义映射，calibration 用于背景阈值校准，audit 用于独立审核。
# 按此顺序共选取 250 条；audit 的来源文件与另外两类隔离。
REVIEW_ROLES = (("mapping", 150), ("calibration", 50), ("audit", 50))


@dataclass(frozen=True)
class StftClusterSample:
    """不可变的单个 clean STFT 样本索引，保存位置、类别及来源信息。"""

    path: str               # 存储 STFT 特征的 H5 文件绝对路径。
    row_idx: int            # H5 中 stft 数据集的样本行索引，从 0 开始。
    label: int              # 原始 DroneRFa 整数类别标签，未重映射为模型标签。
    rf_channel: int         # 原始 IQ 数据所属的 RF 通道编号，取 0 或 1。
    source_sample_idx: int  # 所选 RF 通道内原始定长 IQ 样本编号，从 0 开始，可与 row_idx 不同。


def _take_diverse(
    ordered_indices: list[int],
    count: int,
    samples: list[StftClusterSample],
    used_indices: set[int],
    *,
    forbidden_paths: set[str] | None = None,
) -> list[int]:
    """按候选优先级选取样本，优先覆盖不同来源文件。

    输入：
        ordered_indices: 按优先级排列且不重复的候选全局索引，均须能索引 samples。
        count: 需要选取的非负整数样本数。
        samples: 完整样本索引列表，通过样本的 path 区分来源文件。
        used_indices: 已使用的全局索引集合，其中的样本不参与选取。
        forbidden_paths: 禁止选取的来源文件路径集合； None 表示不限制来源文件。
    输出：
        包含 count 个不同全局索引的列表； count 为 0 时返回 []。
        先按候选顺序从每个来源文件选取一条，不足时再按候选顺序补选剩余样本。
        候选不足时抛出 ValueError ；不修改输入集合，调用方负责记录已选索引。
    """
    if count == 0:
        return []
    forbidden_paths = forbidden_paths or set()
    selected: list[int] = []
    selected_paths: set[str] = set()
    # 按需过滤并优先覆盖不同文件，配额满足后立即返回，避免保存完整候选列表。
    for index in ordered_indices:
        if index in used_indices:
            continue
        path = samples[index].path
        if path in forbidden_paths:
            continue
        if path not in selected_paths:
            selected.append(index)
            selected_paths.add(path)
            if len(selected) == count:
                return selected
    # 文件种类不足时按原候选顺序补选，用集合避免重复选取和列表查找。
    selected_indices = set(selected)
    for index in ordered_indices:
        if index in used_indices or index in selected_indices:
            continue
        if samples[index].path in forbidden_paths:
            continue
        selected.append(index)
        selected_indices.add(index)
        if len(selected) == count:
            return selected
    raise ValueError(f"Not enough distinct review candidates: requested {count}, found {len(selected)}")


def select_cluster_review_indices(
    samples: list[StftClusterSample],
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    centers: np.ndarray,
) -> list[tuple[str, int]]:
    """选择簇语义映射、阈值校准和独立审核样本，隔离 audit 来源文件。

    输入：
        samples: N 个样本的完整索引列表，按 path 区分来源文件。
        embeddings: 形状为 (N, D) 的潜在向量数组。
        probabilities: 形状为 (N, K) 的聚类概率数组，与样本逐行对应。
        centers: 形状为 (K, D) 的簇中心数组；输入形状及对应关系由调用方保证。
    输出：
        按 mapping、calibration、audit 顺序返回 (角色, 全局样本索引) 列表，
        配额由 REVIEW_ROLES 定义，共 250 条且样本不重复；不修改输入数据。
        先选 audit 并排除其来源文件，再选 mapping 的中心与边缘样本，
        最后从剩余候选中按置信度覆盖和文件多样性选取 calibration。
        总样本不足、簇来源文件不足或任一阶段候选不足时抛出 ValueError。
    """

    review_counts = dict(REVIEW_ROLES)
    required_count = sum(review_counts.values())
    if len(samples) < required_count:
        raise ValueError(f"At least {required_count} samples are required for blinded review")
    labels = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    cluster_count = probabilities.shape[1]
    used_indices: set[int] = set()
    selected_by_role: dict[str, list[int]] = {role: [] for role, _count in REVIEW_ROLES}

    # 先预留 audit 文件，保证语义映射/阈值校准不会接触这些文件的人工标签。
    audit_alloc = allocate_evenly(review_counts["audit"], cluster_count)
    for cluster_id, count in enumerate(audit_alloc):
        cluster_indices = np.flatnonzero(labels == cluster_id)
        indices_by_path: dict[str, list[int]] = {}
        for index in cluster_indices:
            indices_by_path.setdefault(samples[int(index)].path, []).append(int(index))
        if len(indices_by_path) < 2:
            raise ValueError(
                f"Cluster {cluster_id} must span at least two source files for file-disjoint audit"
            )
        path_groups = sorted(indices_by_path.values(), key=lambda group: (-len(group), samples[group[0]].path))
        reserved: list[int] = []
        for group in path_groups[:-1]:
            reserved.extend(group)
            if len(reserved) >= count:
                break
        if len(reserved) < count:
            raise ValueError(f"Cluster {cluster_id} cannot provide {count} file-disjoint audit rows")
        order = prioritize_spread_indices(np.asarray(reserved, dtype=np.int64), confidences, count)
        chosen = order[:count]
        selected_by_role["audit"].extend(chosen)
        used_indices.update(chosen)
    audit_paths = {samples[index].path for index in selected_by_role["audit"]}

    mapping_alloc = allocate_evenly(review_counts["mapping"], cluster_count)
    for cluster_id, count in enumerate(mapping_alloc):
        cluster_indices = np.flatnonzero(labels == cluster_id)
        distances = np.linalg.norm(embeddings - centers[cluster_id], axis=1)
        ordered = cluster_indices[np.argsort(distances[cluster_indices])]
        central_count = (count + 1) // 2
        central = _take_diverse(
            ordered.tolist(), central_count, samples, used_indices, forbidden_paths=audit_paths
        )
        used_indices.update(central)
        peripheral_order = ordered[::-1].tolist()
        peripheral = _take_diverse(
            peripheral_order,
            count - central_count,
            samples,
            used_indices,
            forbidden_paths=audit_paths,
        )
        used_indices.update(peripheral)
        selected_by_role["mapping"].extend([*central, *peripheral])

    calibration_count = review_counts["calibration"]
    available = np.array(
        [
            index for index in range(len(samples))
            if index not in used_indices and samples[index].path not in audit_paths
        ],
        dtype=np.int64,
    )
    calibration_order = prioritize_spread_indices(available, confidences, calibration_count)
    calibration = _take_diverse(calibration_order, calibration_count, samples, used_indices)
    selected_by_role["calibration"] = calibration

    return [
        (role, index)
        for role, _count in REVIEW_ROLES
        for index in selected_by_role[role]
    ]


def write_cluster_assignments(
    path: Path,
    samples: list[StftClusterSample],
    probabilities: np.ndarray,
) -> None:
    """按样本顺序逐行导出聚类分配 CSV，供后续背景清洗使用。

    输入：
        path: 输出 CSV 路径，父目录须已存在，已有文件会被覆盖。
        samples: 完整样本索引列表，其位置作为导出的 sample_index。
        probabilities: 形状为 (N, K) 的有效聚类概率数组，N = len(samples)，K >= 1，
            第 i 行对应 samples[i]；输入约束由调用方保证。
    输出：
        返回 None，写入 UTF-8 CSV，包含样本来源、簇编号、最大概率及 q_0 至 q_{K-1}。
        source_file 仅保存文件名；空样本列表只写表头，不修改输入数据。
    """
    probability_fields = [f"q_{cluster_id}" for cluster_id in range(probabilities.shape[1])]
    fieldnames = [
        "sample_index", "source_file", "source_row_idx", "original_label",
        "rf_channel", "source_sample_idx", "cluster_id", "cluster_confidence",
        *probability_fields,
    ]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample in enumerate(samples):
            row_probabilities = probabilities[index]
            row = {
                "sample_index": index,
                "source_file": Path(sample.path).name,
                "source_row_idx": sample.row_idx,
                "original_label": sample.label,
                "rf_channel": sample.rf_channel,
                "source_sample_idx": sample.source_sample_idx,
                "cluster_id": int(row_probabilities.argmax()),
                "cluster_confidence": float(row_probabilities.max()),
            }
            row.update({name: float(value) for name, value in zip(probability_fields, row_probabilities)})
            writer.writerow(row)


def _save_cluster_review_image(stft: np.ndarray, review_id: str, output_path: Path) -> None:
    """将二维 STFT 转置后保存为盲审 PNG，标题仅显示审核编号，异常时也释放 Figure。"""
    plt = import_pyplot(show=False)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    try:
        image = ax.imshow(
            stft.T,
            origin="lower",
            aspect="auto",
            cmap="jet",
            vmin=-3.0,
            vmax=3.0,
        )
        ax.set_title(review_id)
        ax.set_xlabel("Frequency bin")
        ax.set_ylabel("Time bin")
        fig.colorbar(image, ax=ax, label="Normalized power (z-score)")
        fig.savefig(output_path, dpi=120)
    finally:
        plt.close(fig)


def write_cluster_review_artifacts(
    work_dir: Path,
    selections: list[tuple[str, int]],
    samples: list[StftClusterSample],
    probabilities: np.ndarray,
) -> None:
    """导出盲审图片、人工填写表和样本查询表。

    输入：
        work_dir: 输出根目录，不存在时自动创建，已有同名产物会被覆盖。
        selections: 按导出顺序排列的 (审核角色, 全局样本索引) 列表。
        samples: 完整样本索引列表，来源 H5 的 stft 数组须可按 [row_idx, 0] 读取二维图。
        probabilities: 形状为 (N, K) 的聚类概率数组，第 i 行对应 samples[i]。
            索引及数据有效性由调用方保证。
    输出：
        返回 None；生成 review_images/Rxxxx.png、review.csv 和 review_lookup.csv。
        图片编号从 R0001 开始，人工标签留空，来源和聚类信息单独保存到查询表。
        空选取列表只生成图片目录和两份表头；不修改输入数据，退出时关闭 H5 文件。
    """
    image_dir = work_dir / "review_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    review_path = work_dir / "review.csv"
    lookup_path = work_dir / "review_lookup.csv"
    handles: dict[str, h5py.File] = {}
    with ExitStack() as file_stack:
        with review_path.open("w", newline="", encoding="utf-8") as review_file, lookup_path.open(
            "w", newline="", encoding="utf-8"
        ) as lookup_file:
            review_writer = csv.DictWriter(
                review_file,
                fieldnames=("review_id", "review_role", "image_path", "manual_label"),
            )
            lookup_writer = csv.DictWriter(
                lookup_file,
                fieldnames=(
                    "review_id", "review_role", "sample_index", "source_file",
                    "source_row_idx", "cluster_id",
                ),
            )
            review_writer.writeheader()
            lookup_writer.writeheader()
            for order, (role, sample_index) in enumerate(selections, start=1):
                review_id = f"R{order:04d}"
                sample = samples[sample_index]
                if sample.path not in handles:
                    handles[sample.path] = file_stack.enter_context(h5py.File(sample.path, "r"))
                stft = handles[sample.path]["stft"][sample.row_idx, 0]
                relative_image = Path("review_images") / f"{review_id}.png"
                _save_cluster_review_image(stft, review_id, work_dir / relative_image)
                review_writer.writerow({
                    "review_id": review_id,
                    "review_role": role,
                    "image_path": relative_image.as_posix(),
                    "manual_label": "",
                })
                lookup_writer.writerow({
                    "review_id": review_id,
                    "review_role": role,
                    "sample_index": sample_index,
                    "source_file": Path(sample.path).name,
                    "source_row_idx": sample.row_idx,
                    "cluster_id": int(probabilities[sample_index].argmax()),
                })


def scan_clean_stft_h5(data_dir: str) -> list[StftClusterSample]:
    """索引 T0010-T10000 的 clean STFT，排除背景和 T0001。"""

    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"STFT H5 directory does not exist: {root}")

    samples: list[StftClusterSample] = []
    for path in sorted(root.glob("*.h5")):
        if path.stem.split("_")[0] in EXCLUDED_CLASS_CODES:
            continue
        with h5py.File(path, "r") as h5f:
            missing = [name for name in REQUIRED_SAMPLE_DATASETS if name not in h5f]
            if missing:
                raise ValueError(f"Missing datasets {missing} in {path}")
            shape = h5f["stft"].shape
            if len(shape) != 4 or shape[1:] != (1, 1024, 1024):
                raise ValueError(
                    f"STFT must have shape (N, 1, 1024, 1024), got {shape} in {path}"
                )
            count = shape[0]
            for name in REQUIRED_SAMPLE_DATASETS[1:]:
                if h5f[name].shape[:1] != (count,):
                    raise ValueError(f"Dataset '{name}' has invalid length in {path}")
            if h5_attr_to_text(h5f.attrs.get("noise_profile", "")) != "clean":
                raise ValueError(f"Only noise_profile=clean is supported: {path}")

            labels = h5f["labels"][:]
            is_clean = h5f["is_clean"][:]
            rf_channels = h5f["rf_channel"][:]
            source_indices = h5f["source_sample_idx"][:]
            if not bool(is_clean.all()):
                raise ValueError(f"All included STFT rows must be clean: {path}")
            invalid_labels = sorted({int(value) for value in labels if not 0 <= int(value) <= 16})
            if invalid_labels:
                raise ValueError(f"Invalid DroneRFa labels {invalid_labels} in {path}")
            if not all(int(value) in (0, 1) for value in rf_channels):
                raise ValueError(f"Invalid rf_channel values in {path}")

            for row_idx, label in enumerate(labels):
                label = int(label)
                if label == 0 or label not in RESEARCH_LABELS:
                    continue
                samples.append(
                    StftClusterSample(
                        path=str(path.resolve()),
                        row_idx=row_idx,
                        label=label,
                        rf_channel=int(rf_channels[row_idx]),
                        source_sample_idx=int(source_indices[row_idx]),
                    )
                )
    if not samples:
        raise ValueError(f"No clean T0010-T10000 STFT samples found in {root}")
    return samples


class CleanStftClusteringDataset(Dataset):
    """Lazily load clean UAV STFT rows and return stable global indices."""

    def __init__(self, samples: list[StftClusterSample], *, output_size: int = 256) -> None:
        self.samples = samples
        self.output_size = output_size
        self.files: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        h5f = self._get_file(sample.path)
        value = torch.from_numpy(h5f["stft"][sample.row_idx])
        return clip_scale_and_pool_stft(value, self.output_size), index

    def _get_file(self, path: str) -> h5py.File:
        if path not in self.files:
            self.files[path] = h5py.File(path, "r", rdcc_nbytes=64 * 1024 * 1024)
        return self.files[path]

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["files"] = {}
        return state

    def close(self) -> None:
        for h5f in self.files.values():
            h5f.close()
        self.files.clear()

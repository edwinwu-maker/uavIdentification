"""原始 STFT 结构分析、HDBSCAN 融合判别和独立盲审导出。"""

from __future__ import annotations

import csv
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

from src.data.stft_clustering import StftClusterSample, _save_cluster_review_image


@dataclass(frozen=True)
class StftAnalysisConfig:
    min_cluster_size: int = 30
    min_samples: int = 10
    mask_threshold: float = 1.5
    min_region_pixels: int = 64
    signal_area_threshold: float = 0.001
    signal_span_threshold: float = 0.1
    membership_threshold: float = 0.5
    outlier_threshold: float = 0.9

    def validate(self, sample_count: int | None = None) -> None:
        if self.min_cluster_size < 2 or self.min_samples < 1:
            raise ValueError("min-cluster-size must be >= 2 and min-samples must be >= 1")
        if self.min_region_pixels < 1 or not np.isfinite(self.mask_threshold):
            raise ValueError("min-region-pixels must be positive and mask-threshold must be finite")
        for name in (
            "signal_area_threshold", "signal_span_threshold",
            "membership_threshold", "outlier_threshold",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name.replace('_', '-')} must be finite and within [0, 1]")
        # DCEC 至少需要 5 条才能评估默认最小 K=4；密度邻域还需要一个额外样本。
        required = max(5, self.min_cluster_size, self.min_samples + 1)
        if sample_count is not None and sample_count < required:
            raise ValueError(f"At least {required} samples are required for DCEC + HDBSCAN")


def load_hdbscan():
    """仅在新模式中加载独立 hdbscan 包，训练前即可报告缺失依赖。"""
    try:
        from hdbscan import HDBSCAN
    except ImportError as exc:
        raise ImportError(
            "dcec-hdbscan requires the 'hdbscan' package; install requirements.txt in your environment"
        ) from exc
    return HDBSCAN


def fit_hdbscan(embeddings: np.ndarray, config: StftAnalysisConfig):
    """直接拟合原始 z；三个原生输出的行序与输入保持一致。"""
    config.validate(len(embeddings))
    model = load_hdbscan()(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit(embeddings)
    return model.labels_, model.probabilities_, model.outlier_scores_


def analyze_stft_structure(stft: np.ndarray, config: StftAnalysisConfig) -> dict:
    """分析原始 (频率, 时间) z-score 图，不重新归一化或使用池化图。"""
    metrics = {
        "valid_region_occupancy": 0.0,
        "frequency_coverage": 0.0,
        "time_coverage": 0.0,
        "largest_region_area_share": 0.0,
        "largest_region_frequency_span": 0.0,
        "largest_region_time_span": 0.0,
    }
    if stft.ndim != 2 or not stft.size:
        raise ValueError("STFT structure analysis requires a non-empty (frequency, time) array")
    if not np.isfinite(stft).all() or np.ptp(stft) == 0:
        return {**metrics, "structure_label": "uncertain", "structure_reason": "degenerate_stft"}

    regions, _ = ndimage.label(stft > config.mask_threshold, structure=np.ones((3, 3)))
    sizes = np.bincount(regions.ravel())
    valid_ids = np.flatnonzero(sizes >= config.min_region_pixels)
    valid_ids = valid_ids[valid_ids != 0]
    if not len(valid_ids):
        return {**metrics, "structure_label": "background", "structure_reason": "no_valid_region"}
    valid_mask = np.isin(regions, valid_ids)
    largest_id = int(valid_ids[np.argmax(sizes[valid_ids])])
    frequencies, times = np.nonzero(regions == largest_id)
    metrics.update({
        "valid_region_occupancy": float(valid_mask.mean()),
        "frequency_coverage": float(valid_mask.any(axis=1).mean()),
        "time_coverage": float(valid_mask.any(axis=0).mean()),
        "largest_region_area_share": float(sizes[largest_id] / stft.size),
        "largest_region_frequency_span": float((frequencies.max() - frequencies.min() + 1) / stft.shape[0]),
        "largest_region_time_span": float((times.max() - times.min() + 1) / stft.shape[1]),
    })
    is_signal = (
        metrics["largest_region_area_share"] >= config.signal_area_threshold
        and max(metrics["largest_region_frequency_span"], metrics["largest_region_time_span"])
        >= config.signal_span_threshold
    )
    return {
        **metrics,
        "structure_label": "signal" if is_signal else "uncertain",
        "structure_reason": "signal_extent" if is_signal else "insufficient_extent",
    }


def fuse_structure(structure: dict, label: int, membership: float, outlier: float,
                   config: StftAnalysisConfig) -> dict:
    """密度异常只进入 uncertain，不据此判为背景；保留全部触发原因。"""
    reasons = []
    if label == -1:
        reasons.append("hdbscan_noise")
    if not np.isfinite(membership) or not np.isfinite(outlier):
        reasons.append("nonfinite_density_score")
    if membership < config.membership_threshold:
        reasons.append("low_membership")
    if outlier > config.outlier_threshold:
        reasons.append("high_outlier_score")
    return {
        **structure,
        "density_boundary": bool(reasons),
        "fusion_label": "uncertain" if reasons else structure["structure_label"],
        "fusion_reason": ";".join(reasons) if reasons else structure["structure_reason"],
    }


def select_hdbscan_review_indices(rows: list[dict], limit: int = 250) -> list[int]:
    """先用半数名额覆盖不确定样本，再按簇轮流选高成员强度代表，最后补齐。"""
    count = min(limit, len(rows))
    uncertain = [i for i, row in enumerate(rows) if row["fusion_label"] == "uncertain"]
    selected = uncertain[:count // 2]
    used = set(selected)
    groups = {}
    for i, row in enumerate(rows):
        if i not in used and row["hdbscan_label"] >= 0:
            groups.setdefault(row["hdbscan_label"], []).append(i)
    queues = []
    for cluster_id in sorted(groups):
        # 非有限成员强度排在末尾；相同强度按全局索引排序，确保复核可重复。
        ordered = sorted(groups[cluster_id], key=lambda i: (
            -rows[i]["membership_strength"] if np.isfinite(rows[i]["membership_strength"]) else float("inf"), i
        ))
        queues.append(deque(ordered))
    while queues and len(selected) < count:
        for queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.popleft())
        queues = [queue for queue in queues if queue]
    used = set(selected)
    for index in range(len(rows)):
        if len(selected) == count:
            break
        if index not in used:
            selected.append(index)
    return selected


def write_hdbscan_analysis(work_dir: Path, samples: list[StftClusterSample],
                          embeddings: np.ndarray, config: StftAnalysisConfig) -> dict:
    """逐条读取原始 STFT，导出独立分析资料并返回可写入训练报告的统计。"""
    labels, memberships, outliers = fit_hdbscan(embeddings, config)
    rows = []
    image_dir = work_dir / "hdbscan_review_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handles = {}
        for index, sample in enumerate(samples):
            if sample.path not in handles:
                handles[sample.path] = stack.enter_context(h5py.File(sample.path, "r"))
            stft = handles[sample.path]["stft"][sample.row_idx, 0]
            structure = analyze_stft_structure(stft, config)
            rows.append({
                "sample_index": index,
                "source_file": Path(sample.path).name,
                "source_path": sample.path,
                "source_row_idx": sample.row_idx,
                "original_label": sample.label,
                "rf_channel": sample.rf_channel,
                "source_sample_idx": sample.source_sample_idx,
                "hdbscan_label": int(labels[index]),
                "membership_strength": float(memberships[index]),
                "outlier_score": float(outliers[index]),
                **fuse_structure(structure, int(labels[index]), float(memberships[index]),
                                 float(outliers[index]), config),
            })
        with (work_dir / "hdbscan_assignments.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        selections = select_hdbscan_review_indices(rows)
        with (work_dir / "hdbscan_review.csv").open("w", newline="", encoding="utf-8") as review_file, (
            work_dir / "hdbscan_review_lookup.csv"
        ).open("w", newline="", encoding="utf-8") as lookup_file:
            review_writer = csv.DictWriter(review_file, fieldnames=("review_id", "image_path", "manual_label"))
            lookup_writer = csv.DictWriter(lookup_file, fieldnames=("review_id", *rows[0]))
            review_writer.writeheader()
            lookup_writer.writeheader()
            for order, index in enumerate(selections, start=1):
                review_id = f"H{order:04d}"
                sample = samples[index]
                relative_image = Path("hdbscan_review_images") / f"{review_id}.png"
                _save_cluster_review_image(
                    handles[sample.path]["stft"][sample.row_idx, 0], review_id, work_dir / relative_image
                )
                review_writer.writerow({"review_id": review_id, "image_path": relative_image.as_posix(), "manual_label": ""})
                lookup_writer.writerow({"review_id": review_id, **rows[index]})

    return {
        "parameters": {**asdict(config), "metric": "euclidean", "cluster_selection_method": "eom",
                       "embedding": "raw_encoder_z", "structure_axes": ["frequency", "time"]},
        "cluster_count": len(set(labels.tolist()) - {-1}),
        "cluster_counts": dict(sorted(Counter(str(int(label)) for label in labels).items())),
        "noise_ratio": float(np.mean(labels == -1)),
        "boundary_sample_count": sum(row["density_boundary"] for row in rows),
        "structure_label_counts": dict(Counter(row["structure_label"] for row in rows)),
        "fusion_label_counts": dict(Counter(row["fusion_label"] for row in rows)),
        "cluster_label_counts": {
            str(int(label)): dict(Counter(row["fusion_label"] for row in rows if row["hdbscan_label"] == label))
            for label in sorted(set(labels.tolist()))
        },
        "review_count": len(selections),
    }

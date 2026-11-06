"""DeepCluster 的样本身份、内容指纹与独立盲审材料。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from src.data.deepcluster_schema import MANUAL_LABELS, SOURCE_IDENTITY_FIELDS
from src.utils.files import compute_file_sha256, read_optional_csv_records, write_csv
from src.utils.plot import import_pyplot

if TYPE_CHECKING:
    from logging import Logger

    from src.data.stft_clustering import StftClusterSample
    from src.training.deepcluster import DeepClusterResult

IDENTITY_FIELDS = SOURCE_IDENTITY_FIELDS
PREPROCESSING = {"input_size": 256, "clip_range": [-5.0, 5.0], "scale_divisor": 5.0}

def stft_sample_to_identity(sample: StftClusterSample) -> dict[str, str | int]:
    """提取用于审核复用和数据指纹的来源身份。"""
    
    return {"source_file": Path(sample.path).name,
            "source_row_idx": sample.row_idx,
            "original_label": sample.label,
            "rf_channel": sample.rf_channel,
            "source_sample_idx": sample.source_sample_idx}

def source_identity_to_key(row: Mapping[str, object]) -> tuple[str, ...]:
    """将含五项来源身份字段的记录转为字符串元组键，用于匹配审核记录。

    输入：身份字典或含额外字段的 CSV/聚类记录；输出按 IDENTITY_FIELDS 顺序排列，忽略额外字段。
    """
    return tuple(str(row[field]) for field in IDENTITY_FIELDS)

def compute_dataset_fingerprint(
    samples: Sequence[StftClusterSample],
    logger: Logger | None = None,
) -> tuple[str, dict[str, object]]:
    """计算源文件内容与有序样本身份的联合指纹，用于核验审核复用。

    输入：有序 StftClusterSample 序列 samples，以及可选的 logging.Logger。
    输出：(SHA-256 十六进制字符串, payload 字典)；payload 含文件哈希、样本身份和预处理参数。
    """
    files = []
    for filename in sorted({sample.path for sample in samples}):
        if logger is not None:
            logger.info("Fingerprinting source H5: %s", filename)
        files.append({"source_file": Path(filename).name, "sha256": compute_file_sha256(filename)})
    payload = {"files": files, "samples": [stft_sample_to_identity(sample) for sample in samples],
               "preprocessing": PREPROCESSING}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload

def save_stft_review_image(stft: np.ndarray, review_id: str, output_path: Path) -> None:
    """将单个 STFT 样本保存为仅显示审核编号的盲审 PNG，始终释放 Figure。

    输入：二维 STFT 数组 stft（np.ndarray）、审核编号 review_id（str）、目标路径 output_path（Path）。
    输出：返回 None；将转置后的 STFT 写入指定 PNG 文件。
    """
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

def load_previous_reviews(
    work_dir: Path,
    fingerprint: str,
    *,
    required: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """按数据指纹读取可复用的开发审核和独立审计记录。

    输入：历史目录 work_dir（Path）、当前数据指纹 fingerprint（str）和校验要求 required（bool）。
    输出：(开发审核记录列表, 审计记录列表)，记录为 CSV 字符串字典；缺失 CSV 返回空列表。
    报告缺失或指纹不匹配时，required=True 抛错，否则返回两个空列表。
    """
    report_path = work_dir / "training_report.json"
    if not report_path.exists():
        if required:
            raise FileNotFoundError(f"Missing review training report: {report_path}")
        return [], []
    with report_path.open(encoding="utf-8") as source:
        report = json.load(source)
    if report.get("dataset_fingerprint") != fingerprint:
        if required:
            raise ValueError("Cannot reuse reviews from a different dataset fingerprint")
        return [], []
    return (read_optional_csv_records(work_dir / "sample_review.csv"),
            read_optional_csv_records(work_dir / "audit.csv"))

def select_review_samples(
    samples: Sequence[StftClusterSample],
    labels: np.ndarray,
    distances: np.ndarray,
    identities: Mapping[tuple[str, ...], int],
    old_samples: Mapping[tuple[str, ...], Mapping[str, str]],
    old_audit: Mapping[tuple[str, ...], Mapping[str, str]],
    *,
    seed: int,
    audit_per_class: int,
) -> tuple[list[tuple[int, str]], list[int]]:
    """选择簇中心、边界和随机开发样本，并固定或抽取独立审计样本。

    输入：有序 StftClusterSample 序列 samples、与其对齐的 (N,) 数组 labels/distances、
    身份元组到全局索引的映射 identities、以身份元组为键的历史 CSV 记录 old_samples/old_audit，
    以及随机种子 seed 和每类审计数上限 audit_per_class。
    输出：(开发选样列表 [(全局索引, 角色)], 审计全局索引列表)。
    仅计算索引；保留簇顺序、稳定排序和随机数消耗顺序。
    """
    # 首轮先选开发样本，余下按类别随机盲抽；后续固定审计身份并排除出开发集。
    audit_indices = [identities[key] for key in old_audit]
    reserved = set(audit_indices)
    rng = np.random.default_rng(seed)
    selections = []
    for cluster_id in sorted(set(labels.tolist())):
        candidates = np.array([int(index) for index in np.flatnonzero(labels == cluster_id)
                               if int(index) not in reserved], dtype=np.int64)
        available = set(candidates.tolist())
        for role, order in (("center", candidates[np.argsort(distances[candidates], kind="stable")]),
                            ("boundary", candidates[np.argsort(-distances[candidates], kind="stable")])):
            # 边界使用到所属中心的远距离样本，作为人工审查的困难样本。
            chosen = [int(index) for index in order if int(index) in available][:4]
            selections.extend((index, role) for index in chosen)
            available.difference_update(chosen)
        random_indices = rng.permutation(sorted(available))[:4]
        selections.extend((int(index), "random") for index in random_indices)
    development = {index for index, _ in selections}
    if not old_audit:
        for label in sorted({sample.label for sample in samples}):
            candidates = [index for index, sample in enumerate(samples)
                          if sample.label == label and index not in development]
            audit_indices.extend(int(index) for index in rng.permutation(candidates)[:audit_per_class])
    # 跨目录对照也重新生成旧样本图片，避免复用的路径指向不存在的目标文件。
    for key, previous in old_samples.items():
        index = identities[key]
        if index not in development and (previous.get("decision", "") or previous.get("manual_label", "")):
            selections.append((index, "previous_review"))
    return selections, audit_indices

def write_deepcluster_review(
    work_dir: str | Path,
    samples: Sequence[StftClusterSample],
    result: DeepClusterResult,
    *,
    fingerprint: str,
    seed: int = 42,
    audit_per_class: int = 30,
    reuse_review_dir: str | Path | None = None,
) -> dict[str, object]:
    """生成簇复核与独立盲审材料，保留逐样本审核并重置整簇处置确认。

    输入：工作目录 work_dir（str/Path）、有序 StftClusterSample 序列 samples、
    逐样本 assignments/distances 与 samples 对齐的 DeepClusterResult，以及数据指纹 fingerprint、
    随机种子 seed、每类审计数上限 audit_per_class 和可选复用目录 reuse_review_dir（str/Path）。
    输出：dict[str, object]，含开发/审计样本数、按类审计数、人工标签取值及复用状态；
    同时写出 assignments、cluster_review、sample_review、audit 四份 CSV 和盲审 PNG。
    """
    work_dir = Path(work_dir)
    review_dir = Path(reuse_review_dir).expanduser() if reuse_review_dir is not None else work_dir
    previous_samples, previous_audit = load_previous_reviews(
        review_dir, fingerprint, required=reuse_review_dir is not None)
    identities = {source_identity_to_key(stft_sample_to_identity(sample)): index for index, sample in enumerate(samples)}
    old_samples = {source_identity_to_key(row): row for row in previous_samples}
    old_audit = {source_identity_to_key(row): row for row in previous_audit}
    if len(old_samples) != len(previous_samples) or len(old_audit) != len(previous_audit):
        raise ValueError("Previous reviews contain duplicate sample identities")
    if any(key not in identities for key in (*old_samples, *old_audit)):
        raise ValueError("Previous review identities do not match the dataset fingerprint")
    if set(old_samples) & set(old_audit):
        raise ValueError("Previous development and audit samples overlap")
    for row in (*previous_samples, *previous_audit):
        if row.get("manual_label", "") not in ("", *MANUAL_LABELS):
            raise ValueError("Invalid previous manual_label")

    assignments = []
    labels = np.asarray(result.assignments)
    for index, sample in enumerate(samples):
        assignments.append({"sample_index": index, **stft_sample_to_identity(sample), "source_path": sample.path,
                            "cluster_id": int(labels[index]),
                            "centroid_distance": float(result.distances[index])})
    selections, audit_indices = select_review_samples(
        samples, labels, result.distances, identities, old_samples, old_audit,
        seed=seed, audit_per_class=audit_per_class)

    review_fields = ("review_id", "image_path", "review_role", *assignments[0], "manual_label")
    development_rows, audit_rows = [], []
    image_dir = work_dir / "deepcluster_review_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handles = {}
        for index, role in [*selections, *((index, "audit") for index in audit_indices)]:
            assignment = assignments[index]
            key = source_identity_to_key(assignment)
            review_id = "D" + hashlib.sha256(json.dumps(key).encode("utf-8")).hexdigest()[:16]
            image_path = Path("deepcluster_review_images") / f"{review_id}.png"
            sample = samples[index]
            if sample.path not in handles:
                handles[sample.path] = stack.enter_context(h5py.File(sample.path, "r"))
            save_stft_review_image(handles[sample.path]["stft"][sample.row_idx, 0],
                                       review_id, work_dir / image_path)
            previous = (old_audit if role == "audit" else old_samples).get(key, {})
            row = {"review_id": review_id, "image_path": image_path.as_posix(), "review_role": role,
                   **assignment, "manual_label": previous.get("manual_label", "")}
            if role == "audit":
                audit_rows.append(row)
            else:
                row.update({"decision": previous.get("decision", ""),
                            "confirmed": previous.get("confirmed", "false")})
                development_rows.append(row)
    counts = Counter(int(label) for label in labels)
    clusters = [{"cluster_id": cluster_id, "cluster_size": counts[cluster_id],
                 "decision": "quarantine", "confirmed": "false", "mixed": "false", "review_note": ""}
                for cluster_id in sorted(counts)]
    write_csv(work_dir / "deepcluster_assignments.csv", assignments, assignments[0])
    write_csv(work_dir / "cluster_review.csv", clusters, clusters[0])
    write_csv(work_dir / "sample_review.csv", development_rows, (*review_fields, "decision", "confirmed"))
    write_csv(work_dir / "audit.csv", audit_rows, review_fields)
    return {"development_count": len(development_rows), "audit_count": len(audit_rows),
            "audit_per_class": dict(Counter(row["original_label"] for row in audit_rows)),
            "manual_labels": list(MANUAL_LABELS), "audit_reused": bool(previous_audit),
            "boundary_selection": "largest_assigned_centroid_distance"}

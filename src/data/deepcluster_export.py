"""按人工确认处置导出 DeepCluster 样本；独立审核只计算指标。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np

from src.data.deepcluster_constants import DECISIONS, MANUAL_LABEL_DECISIONS, MANUAL_LABELS as REVIEW_MANUAL_LABELS, SOURCE_IDENTITY_FIELDS
from src.data.drone_rfa_io import EXCLUDED_CLASS_CODES, RESEARCH_LABELS, parse_label
from src.utils.files import compute_file_sha256

MANUAL_LABELS = frozenset(REVIEW_MANUAL_LABELS)
IDENTITY_FIELDS = (*SOURCE_IDENTITY_FIELDS, "cluster_id")


def parse_review_boolean(value: str | bool | int, field: str) -> bool:
    """解析审核记录中的布尔字段。

    输入：CSV 字符串或布尔/整数值 value，以及用于报错的字段名 field。
    输出：bool；true/1 为真，false/0/空字符串为假，忽略大小写和首尾空白。
    其他值抛出 ValueError，不按 Python 的一般真值规则转换。
    """
    text = str(value).strip().lower()  # 将输入转为字符串，去除首尾空白并统一为小写。
    if text not in ("", "true", "false", "1", "0"):
        raise ValueError(f"Invalid {field}: {value}")
    return text in ("true", "1")


def parse_review_decision(row: Mapping[str, object]) -> str:
    """解析审核记录的 decision 字段，忽略大小写和首尾空白。

    输入：审核记录映射 row，可包含 decision 及其他字段。
    输出：keep/drop/quarantine；字段缺失或空白时返回空字符串，其他值抛出 ValueError。
    """
    action = str(row.get("decision", "")).strip().lower()
    if action and action not in DECISIONS:
        raise ValueError(f"Invalid decision: {action}")
    return action


def match_review_to_assignment(
    row: Mapping[str, object],
    assignments: Mapping[int, Mapping[str, object]],
) -> int:
    """核对审核记录与聚类分配的身份及人工标签，返回匹配的全局样本索引。

    输入：审核记录 row，以及以整数 sample_index 为键的聚类分配记录 assignments。
    输出：int 样本索引；未知索引、身份不匹配或非法人工标签均抛出 ValueError。
    身份匹配包括来源字段与 cluster_id；数值字段按整数比较，文件名按字符串比较。
    """
    index = int(row["sample_index"])
    if index not in assignments:
        raise ValueError(f"Unknown review sample_index: {index}")
    for field in IDENTITY_FIELDS:
        actual = row.get(field)
        expected = assignments[index][field]
        if actual is None or (str(actual) != str(expected) if field == "source_file" else int(actual) != int(expected)):
            raise ValueError(f"Review identity mismatch: sample_index={index}, field={field}")
    label = str(row.get("manual_label", "")).strip().lower()
    if label and label not in MANUAL_LABELS:
        raise ValueError(f"Invalid manual_label: {label}; convert manually to video/other/uncertain")
    return index


def resolve_deepcluster_decisions(
    assignment_rows: Iterable[Mapping[str, object]],
    cluster_review_rows: Iterable[Mapping[str, object]],
    sample_review_rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """合并聚类分配与人工审核，生成每个样本的最终处置。

    输入：聚类分配记录 assignment_rows、整簇审核记录 cluster_review_rows、
    逐样本审核记录 sample_review_rows；均为可迭代的字段映射，支持 CSV 字符串值。
    输出：按 sample_index 排列的记录列表，保留分配字段并补充人工标签、审核状态、处置及原因。
    未确认的簇默认隔离；混合簇禁止整簇处置，已确认单样本标签映射的处置优先。
    """
    assignments = {}
    identities = set()
    cluster_counts = Counter()
    for original in assignment_rows:
        row = dict(original)
        index = int(row["sample_index"])
        if index < 0 or index in assignments:
            raise ValueError("Duplicate or negative sample_index")
        filename = row["source_file"]
        if not filename or filename != Path(filename).name or filename in (".", ".."):
            raise ValueError("source_file must be a basename")
        for field in IDENTITY_FIELDS[1:]:
            row[field] = int(row[field])
        if (row["original_label"] not in RESEARCH_LABELS or row["original_label"] == 0
                or row["rf_channel"] not in (0, 1) or row["source_row_idx"] < 0
                or row["source_sample_idx"] < 0 or row["cluster_id"] < 0):
            raise ValueError(f"Invalid assignment identity: sample_index={index}")
        identity = (filename, row["source_row_idx"])
        if identity in identities:
            raise ValueError(f"Duplicate source row: {identity}")
        identities.add(identity)
        row["sample_index"] = index
        assignments[index] = row
        cluster_counts[row["cluster_id"]] += 1
    if not assignments or set(assignments) != set(range(len(assignments))):
        raise ValueError("sample_index must cover 0..N-1")

    cluster_actions = {}
    for row in cluster_review_rows:
        cluster = int(row["cluster_id"])
        if cluster not in cluster_counts or cluster in cluster_actions:
            raise ValueError(f"Unknown or duplicate reviewed cluster: {cluster}")
        if int(row["cluster_size"]) != cluster_counts[cluster]:
            raise ValueError(f"Cluster size mismatch: {cluster}")
        action = parse_review_decision(row)
        confirmed = parse_review_boolean(row.get("confirmed", ""), "confirmed")
        mixed = parse_review_boolean(row.get("mixed", ""), "mixed")
        if mixed:
            cluster_actions[cluster] = ("quarantine", "mixed_cluster")
        elif confirmed and action:
            cluster_actions[cluster] = (action, "confirmed_cluster")
        else:
            cluster_actions[cluster] = ("quarantine", "unconfirmed_cluster")

    sample_reviews = {}
    for row in sample_review_rows:
        index = match_review_to_assignment(row, assignments)
        if index in sample_reviews:
            raise ValueError(f"Duplicate sample review: {index}")
        if "decision" in row:
            raise ValueError("Legacy sample_review decision field; remove it and manually convert to manual_label/confirmed")
        label = str(row.get("manual_label", "")).strip().lower()
        confirmed = parse_review_boolean(row.get("confirmed", ""), "confirmed")
        if confirmed and not label:
            raise ValueError(f"Confirmed sample requires manual_label: {index}")
        sample_reviews[index] = (row, MANUAL_LABEL_DECISIONS[label] if confirmed else "")

    rows = []
    for index in range(len(assignments)):
        row = assignments[index]
        action, reason = cluster_actions.get(row["cluster_id"], ("quarantine", "unreviewed_cluster"))
        review, override = sample_reviews.get(index, ({}, ""))
        if override:
            action, reason = override, "confirmed_sample_override"
        rows.append({**row, "manual_label": str(review.get("manual_label", "")).strip().lower(),
                     "sample_reviewed": index in sample_reviews, "decision": action, "decision_reason": reason})
    return rows


def compute_audit_rates(rows: Sequence[Mapping[str, object]]) -> dict[str, float | None]:
    """计算独立审计标签与最终处置之间的比例指标，不修改处置记录。

    输入：含 audit_label 和 decision 字段的记录序列 rows，可携带其他样本信息。
    输出：视频保留/误删/隔离率、非视频删除召回率、删除精确率及总体隔离率；无分母时为 None。
    负类仅包含 other；删除精确率仅计 video/other，
    总体隔离率统计全部已标注记录，包括 uncertain。
    """
    videos = [row for row in rows if row["audit_label"] == "video"]
    negatives = [row for row in rows if row["audit_label"] == "other"]
    definite_drops = [row for row in videos + negatives if row["decision"] == "drop"]
    return {
        "video_retention": sum(row["decision"] == "keep" for row in videos) / len(videos) if videos else None,
        "video_false_drop_rate": sum(row["decision"] == "drop" for row in videos) / len(videos) if videos else None,
        "video_quarantine_rate": sum(row["decision"] == "quarantine" for row in videos) / len(videos) if videos else None,
        "nonvideo_drop_recall": sum(row["decision"] == "drop" for row in negatives) / len(negatives) if negatives else None,
        "drop_precision": sum(row["audit_label"] != "video" for row in definite_drops) / len(definite_drops) if definite_drops else None,
        "quarantine_rate": sum(row["decision"] == "quarantine" for row in rows) / len(rows) if rows else None,
    }


def compute_deepcluster_audit_report(
    decision_rows: Iterable[Mapping[str, object]],
    audit_rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """核验独立审计记录并计算报告，审计标签不改变已有处置。

    输入：含样本身份、decision 和可选 sample_reviewed 的处置记录 decision_rows，
    含同一身份及可选 manual_label 的审计序列 audit_rows、重采样次数 bootstrap_samples 和 seed。
    输出：含审核计数、比例指标、按源文件 bootstrap 的 95% 置信区间及误删/隔离视频索引的字典。
    空标签仅计入审计总数；重复审计或与开发审核重叠时抛错。
    有标签记录少于两个源文件或重采样次数为零时，置信区间为 None。
    """
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    assignments = {int(row["sample_index"]): row for row in decision_rows}
    reviewed = []
    seen = set()
    for row in audit_rows:
        index = match_review_to_assignment(row, assignments)
        if index in seen or assignments[index].get("sample_reviewed", False):
            raise ValueError(f"Duplicate audit or development/audit overlap: {index}")
        seen.add(index)
        label = str(row.get("manual_label", "")).strip().lower()
        if label:
            reviewed.append({**assignments[index], "audit_label": label})
    rates = compute_audit_rates(reviewed)
    groups = defaultdict(list)
    for row in reviewed:
        groups[row["source_file"]].append(row)
    intervals = {key: None for key in rates}
    if len(groups) >= 2 and bootstrap_samples:
        files = sorted(groups)
        rng = np.random.default_rng(seed)
        values = defaultdict(list)
        for _ in range(bootstrap_samples):
            sample = [row for index in rng.integers(len(files), size=len(files)) for row in groups[files[index]]]
            for key, value in compute_audit_rates(sample).items():
                if value is not None:
                    values[key].append(value)
        for key, samples in values.items():
            intervals[key] = np.quantile(samples, (0.025, 0.975)).tolist()
    return {
        "review_type": "held_out_sample_audit", "audit_count": len(audit_rows),
        "labeled_count": len(reviewed), "source_file_count": len(groups),
        "label_counts": dict(Counter(row["audit_label"] for row in reviewed)),
        "label_decisions": {label: dict(Counter(row["decision"] for row in reviewed if row["audit_label"] == label))
                            for label in sorted(MANUAL_LABELS)},
        **rates, "file_bootstrap_ci95": intervals, "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "false_dropped_video_indices": [row["sample_index"] for row in reviewed if row["audit_label"] == "video" and row["decision"] == "drop"],
        "quarantined_video_indices": [row["sample_index"] for row in reviewed if row["audit_label"] == "video" and row["decision"] == "quarantine"],
    }


def validate_export_sources_and_add_background(
    source_root: Path,
    source_paths: Sequence[Path],
    by_file: defaultdict[str, list[dict[str, object]]],
    decision_rows: Iterable[Mapping[str, object]],
    training_report: Mapping[str, object] | None,
) -> None:
    """核对训练指纹与源行身份，并按文件顺序原地补齐背景的直通处置。

    输入：源目录 source_root、待导出 H5 路径序列 source_paths、按文件名分组的处置记录
    by_file（defaultdict(list)）、UAV 处置记录 decision_rows 和可选的训练报告 training_report。
    输出：None；向 by_file 中追加 T0000 的 keep 记录，不写出目标 H5。
    提供训练报告时核验指纹、文件内容与样本身份；源行覆盖或身份不符时抛出 ValueError。
    """
    selected_files = set(by_file)
    if training_report is not None:
        payload = training_report["fingerprint_payload"]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != training_report["dataset_fingerprint"]:
            raise ValueError("Training dataset fingerprint is inconsistent")
        expected_files = {row["source_file"]: row["sha256"] for row in payload["files"]}
        if len(expected_files) != len(payload["files"]) or set(expected_files) != selected_files:
            raise ValueError("Assignments do not match training source files")
        identity_fields = IDENTITY_FIELDS[:-1]
        expected_samples = [tuple(str(row[field]) for field in identity_fields) for row in payload["samples"]]
        actual_samples = [tuple(str(row[field]) for field in identity_fields)
                          for row in sorted(decision_rows, key=lambda row: int(row["sample_index"]))]
        if expected_samples != actual_samples:
            raise ValueError("Assignments do not match training sample identities")
        for filename, expected_hash in expected_files.items():
            if compute_file_sha256(source_root / filename) != expected_hash:
                raise ValueError(f"Source H5 content changed since training: {filename}")

    # 所有源文件校验通过后才写目标，避免身份错误留下部分导出。
    for path in source_paths:
        with h5py.File(path, "r") as source:
            count = source["stft"].shape[0]
            label = parse_label(path.name)
            for field in ("labels", "rf_channel", "source_sample_idx"):
                if source[field].shape != (count,):
                    raise ValueError(f"Invalid source dataset {field}: {path}")
            if set(int(value) for value in source["labels"][:]) - {label}:
                raise ValueError(f"Source label mismatch: {path}")
            rows = by_file[path.name]
            if label == 0:
                if rows:
                    raise ValueError("T0000 must not appear in assignments")
                for index in range(count):
                    rows.append({"sample_index": "", "source_file": path.name, "source_row_idx": index,
                                 "original_label": 0, "rf_channel": int(source["rf_channel"][index]),
                                 "source_sample_idx": int(source["source_sample_idx"][index]), "cluster_id": "",
                                 "decision": "keep", "decision_reason": "t0000_passthrough"})
                continue
            indices = [int(row["source_row_idx"]) for row in rows]
            if len(indices) != count or set(indices) != set(range(count)):
                raise ValueError(f"Assignments must cover every UAV source row once: {path.name}")
            for row in rows:
                index = int(row["source_row_idx"])
                for field, source_field in (("original_label", "labels"), ("rf_channel", "rf_channel"),
                                            ("source_sample_idx", "source_sample_idx")):
                    if int(row[field]) != int(source[source_field][index]):
                        raise ValueError(f"Source identity mismatch: {path.name}, row={index}, field={field}")


def copy_h5_dataset_rows(
    source: h5py.Dataset,
    destination_file: h5py.File,
    name: str,
    source_rows: list[int],
    *,
    batch_size: int = 16,
) -> None:
    """按第零维的行索引分批复制 H5 dataset。

    输入：源 dataset source、可写目标文件 destination_file、新 dataset 名称 name、
    递增且不重复的行索引列表 source_rows，以及正整数批次大小 batch_size。
    输出：None；创建目标 dataset，保留 dtype 和 dataset 属性。
    非空选择保留压缩配置并按行数调整 chunks；空选择创建零行 dataset，不指定压缩或 chunks。
    """
    shape = (len(source_rows), *source.shape[1:])
    kwargs = {}
    if source.chunks is not None and source_rows:
        kwargs["chunks"] = (min(source.chunks[0], len(source_rows)), *source.chunks[1:])
    if source.compression is not None and source_rows:
        kwargs["compression"] = source.compression
        kwargs["compression_opts"] = source.compression_opts
    destination = destination_file.create_dataset(name, shape=shape, dtype=source.dtype, **kwargs)
    for key, value in source.attrs.items():
        destination.attrs[key] = value
    for start in range(0, len(source_rows), batch_size):
        rows = source_rows[start : start + batch_size]
        destination[start : start + len(rows)] = source[rows]


def write_deepcluster_h5_partition(
    source_path: str | Path,
    output_path: str | Path,
    source_rows: list[int],
    *,
    action: str,
    model_hash: str,
) -> None:
    """按选中源行写出一个 DeepCluster H5 分区。

    输入：源/目标文件路径 source_path/output_path、递增且不重复的行索引 source_rows，
    以及分区处置 action（keep/quarantine）和模型哈希 model_hash。
    输出：None；独占创建目标 H5，复制文件属性并记录清洗方法、处置和模型哈希。
    顶层首维等于 STFT 样本数的数据集按行复制，其他对象原样复制；目标已存在时抛错。
    """
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "x") as output:
        count = source["stft"].shape[0]
        for key, value in source.attrs.items():
            output.attrs[key] = value
        output.attrs["deepcluster_cleaning_method"] = "reviewed_deepcluster"
        output.attrs["deepcluster_cleaning_decision"] = action
        output.attrs["deepcluster_checkpoint_sha256"] = model_hash
        for name, dataset in source.items():
            if isinstance(dataset, h5py.Dataset) and dataset.shape[:1] == (count,):
                copy_h5_dataset_rows(dataset, output, name, source_rows)
            else:
                source.copy(name, output)


def summarize_deepcluster_manifest(
    manifest: Sequence[Mapping[str, object]],
    source_paths: Sequence[Path],
    excluded_files: list[str],
) -> dict[str, object]:
    """汇总 DeepCluster 处置清单，不修改输入记录。

    输入：含 original_label、source_file、rf_channel、decision 等字段的清单序列 manifest，
    已纳入导出的源文件路径序列 source_paths，以及排除文件名列表 excluded_files。
    输出：包含总体处置数、源文件数、排除名单，以及按类别/源文件/RF 通道/距离分组的统计字典。
    空源文件保留在源文件统计中，keep_rate 为 None；距离编码缺失或不唯一时归入 unknown。
    """
    per_class = {}
    for label in sorted({int(row["original_label"]) for row in manifest}):
        rows = [row for row in manifest if int(row["original_label"]) == label]
        per_class[str(label)] = {
            "sample_counts": dict(Counter(row["decision"] for row in rows)),
            "source_file_counts": {action: len({row["source_file"] for row in rows if row["decision"] == action})
                                   for action in sorted(DECISIONS)},
            "source_file_count": len({row["source_file"] for row in rows}),
        }
    grouped = {"per_source_file": defaultdict(Counter), "per_rf_channel": defaultdict(Counter),
               "per_distance": defaultdict(Counter)}
    for path in source_paths:
        grouped["per_source_file"][path.name] = Counter()
    for row in manifest:
        # 只解析明确、唯一的距离编码；缺失或有歧义的文件归入 unknown。
        distances = [part for part in Path(row["source_file"]).stem.split("_") if part in ("D00", "D01", "D10")]
        distance = distances[0] if len(distances) == 1 else "unknown"
        for name, key in (("per_source_file", row["source_file"]), ("per_rf_channel", str(row["rf_channel"])),
                          ("per_distance", distance)):
            grouped[name][key][row["decision"]] += 1
    breakdowns = {
        name: {key: {"sample_count": sum(counts.values()),
                     "decision_counts": {action: counts[action] for action in sorted(DECISIONS)},
                     "keep_rate": counts["keep"] / sum(counts.values()) if counts else None}
               for key, counts in sorted(groups.items())}
        for name, groups in grouped.items()
    }
    return {"decision_counts": dict(Counter(row["decision"] for row in manifest)),
            "source_file_count": len(source_paths), "excluded_source_files": excluded_files,
            "per_class": per_class, **breakdowns}


def export_reviewed_deepcluster_h5(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    decision_rows: Sequence[Mapping[str, object]],
    model_hash: str = "",
    training_report: Mapping[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """按审核处置导出 DeepCluster H5 分区，返回清单与统计。

    输入：原始 H5 目录 data_dir、新输出目录 output_dir、含样本身份及最终处置的记录序列
    decision_rows、模型哈希 model_hash，以及可选训练报告 training_report。
    输出：(清单记录列表, 统计字典)；清单记录含 output_file 和 output_row_idx，不在此写出 CSV/JSON。
    全量核对源行后创建 keep/quarantine H5，drop 仅记入清单，T0000 直通保留。
    输出目录须与源目录互不包含且尚不存在；未选中的 UAV 文件不参与导出。
    """
    source_root = Path(data_dir).expanduser().resolve()
    destination_root = Path(output_dir).expanduser().resolve()
    if destination_root == source_root or source_root in destination_root.parents or destination_root in source_root.parents:
        raise ValueError("output-dir and data-dir must be separate directories")
    if destination_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {destination_root}")
    all_source_paths = [path for path in sorted(source_root.glob("*.h5"))
                        if path.stem.split("_")[0] not in EXCLUDED_CLASS_CODES]
    if not all_source_paths:
        raise ValueError(f"No source H5 found in {source_root}")
    by_file = defaultdict(list)
    for original in decision_rows:
        row = dict(original)
        if row["decision"] not in DECISIONS:
            raise ValueError(f"Invalid decision: {row['decision']}")
        by_file[row["source_file"]].append(row)
    selected_files = set(by_file)
    unknown = selected_files - {path.name for path in all_source_paths}
    if unknown:
        raise ValueError(f"Assignments reference missing source files: {sorted(unknown)}")
    source_paths = [path for path in all_source_paths if path.name in selected_files or path.stem.split("_")[0] == "T0000"]
    excluded_files = [path.name for path in all_source_paths if path not in source_paths]
    validate_export_sources_and_add_background(source_root, source_paths, by_file, decision_rows, training_report)

    destination_root.mkdir(parents=True)
    for action in ("keep", "quarantine"):
        (destination_root / action).mkdir()
    manifest = []
    for path in source_paths:
        rows = sorted(by_file[path.name], key=lambda row: int(row["source_row_idx"]))
        locations = {}
        for action in ("keep", "quarantine"):
            selected = [row for row in rows if row["decision"] == action]
            indices = [int(row["source_row_idx"]) for row in selected]
            output_path = destination_root / action / path.name
            write_deepcluster_h5_partition(path, output_path, indices, action=action, model_hash=model_hash)
            for output_index, row in enumerate(selected):
                locations[int(row["source_row_idx"])] = (f"{action}/{path.name}", output_index)
        for row in rows:
            output_file, output_index = locations.get(int(row["source_row_idx"]), ("", ""))
            manifest.append({**row, "output_file": output_file, "output_row_idx": output_index})
    fields = list(dict.fromkeys(field for row in manifest for field in row))
    manifest = [{field: row.get(field, "") for field in fields} for row in manifest]
    return manifest, summarize_deepcluster_manifest(manifest, source_paths, excluded_files)

"""DeepCluster 特征的 t-SNE 展示与审核后同坐标决策对照。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

from src.data.deepcluster_schema import DECISIONS, SOURCE_IDENTITY_FIELDS
from src.utils.files import write_csv
from src.utils.plot import import_pyplot

MAX_SAMPLES = 10000
FEATURE_SPACE = "PCA-whitened and L2-normalized encoder features"
IDENTITY_FIELDS = ("sample_index", *SOURCE_IDENTITY_FIELDS, "cluster_id")


def _identity(row):
    return tuple(str(row[name]) if name == "source_file" else int(row[name]) for name in IDENTITY_FIELDS)


def _lookup(rows):
    lookup = {}
    sources = set()
    for row in rows:
        identity = _identity(row)
        index = identity[0]
        source = identity[1:-1]
        if index < 0 or index in lookup or source in sources:
            raise ValueError("Duplicate or invalid DeepCluster sample identity")
        lookup[index] = row
        sources.add(source)
    return lookup


def _coordinates(rows):
    coordinates = np.asarray([(float(row["tsne_x"]), float(row["tsne_y"])) for row in rows])
    if not len(rows) or not np.isfinite(coordinates).all():
        raise ValueError("DeepCluster t-SNE coordinates must be non-empty and finite")
    return coordinates


def _plot(path, rows, key, title, total_count):
    coordinates = _coordinates(rows)
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
    padding = np.maximum((upper - lower) * 0.05, 1.0)
    values = np.array([str(row[key]) if key == "decision" else int(row[key]) for row in rows])
    categories = sorted(set(values.tolist()))
    plt = import_pyplot(show=False)
    colors = plt.get_cmap("tab20" if len(categories) <= 20 else "hsv", len(categories))
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    try:
        for order, category in enumerate(categories):
            color = colors(order)
            label = str(category)
            if key == "original_label":
                label = f"T{category:04b}"
            elif key == "decision":
                color = {"keep": "tab:green", "drop": "tab:red", "quarantine": "tab:orange"}[category]
            points = coordinates[values == category]
            ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.65, color=color,
                       label=label, linewidths=0)
        subtitle = f"{len(rows)} / {total_count} samples"
        if np.all(coordinates == coordinates[0]):
            subtitle += " | Degenerate projection: all points identical"
        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
        ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8,
                  ncol=max(1, (len(categories) + 29) // 30))
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def write_deepcluster_tsne(work_dir, features, assignment_rows, *, seed=42):
    """投影最终白化且归一化的特征；features 第 i 行对应 assignment_rows[i]，仅用于展示。"""
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[0] != len(assignment_rows) or not features.size:
        raise ValueError("DeepCluster features must be non-empty (N, D) and aligned with assignments")
    if not np.isfinite(features).all():
        raise ValueError("DeepCluster t-SNE features must be finite")
    _lookup(assignment_rows)
    indices = np.arange(len(assignment_rows))
    if len(indices) > MAX_SAMPLES:
        indices = np.sort(np.random.default_rng(seed).choice(indices, size=MAX_SAMPLES, replace=False))
    selected = features[indices]
    parameters = {"n_components": 2, "perplexity": min(30, len(indices) - 1),
                  "init": "pca" if selected.shape[1] >= 2 else "random", "learning_rate": "auto",
                  "random_state": seed, "max_iter": 1000}
    degenerate = bool(len(indices) == 1 or np.all(selected == selected[0]))
    # 单样本与常量特征没有邻域结构，跳过 t-SNE 以避免 PCA/困惑度异常。
    coordinates = np.zeros((len(indices), 2)) if degenerate else TSNE(**parameters).fit_transform(selected)
    rows = [{**assignment_rows[index], "tsne_x": float(x), "tsne_y": float(y)}
            for index, (x, y) in zip(indices, coordinates)]
    _coordinates(rows)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    csv_name = "deepcluster_tsne.csv"
    write_csv(work_dir / csv_name, rows, list(rows[0]))
    files = [csv_name]
    for key, suffix, title in (("cluster_id", "clusters", "DeepCluster clusters"),
                               ("original_label", "original_labels", "Original UAV classes")):
        name = f"deepcluster_tsne_{suffix}.png"
        _plot(work_dir / name, rows, key, title, len(assignment_rows))
        files.append(name)
    return {"total_sample_count": len(assignment_rows), "plotted_sample_count": len(rows),
            "max_samples": MAX_SAMPLES, "parameters": parameters, "feature_space": FEATURE_SPACE,
            "degenerate": degenerate, "files": files}


def read_deepcluster_tsne(work_dir, assignment_rows):
    """读取展示子集并核验完整身份，防止旧投影与本次聚类结果串用。"""
    path = Path(work_dir) / "deepcluster_tsne.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not {*IDENTITY_FIELDS, "tsne_x", "tsne_y"}.issubset(reader.fieldnames or []):
            raise ValueError("Missing DeepCluster t-SNE identity or coordinate columns")
        rows = list(reader)
    lookup = _lookup(assignment_rows)
    _lookup(rows)
    _coordinates(rows)
    for row in rows:
        index = int(row["sample_index"])
        if index not in lookup or _identity(row) != _identity(lookup[index]):
            raise ValueError(f"DeepCluster t-SNE identity mismatch for sample_index={index}")
        row["tsne_x"], row["tsne_y"] = float(row["tsne_x"]), float(row["tsne_y"])
    return rows


def write_deepcluster_decision_plot(output_dir, coordinate_rows, decision_rows):
    """复用训练时的投影，只改 keep/drop/quarantine 着色，不加入 T0000。"""
    lookup = _lookup([row for row in decision_rows if int(row["original_label"]) != 0])
    _lookup(coordinate_rows)
    _coordinates(coordinate_rows)
    rows = []
    for row in coordinate_rows:
        if int(row["original_label"]) == 0:
            continue
        index = int(row["sample_index"])
        if index not in lookup or _identity(row) != _identity(lookup[index]):
            raise ValueError(f"DeepCluster decision identity mismatch for sample_index={index}")
        decision = lookup[index]["decision"]
        if decision not in DECISIONS:
            raise ValueError(f"Invalid DeepCluster decision: {decision}")
        rows.append({**row, "decision": decision})
    coordinates = _coordinates(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_name, png_name = "deepcluster_tsne_decisions.csv", "deepcluster_tsne_decisions.png"
    write_csv(output_dir / csv_name, rows, list(rows[0]))
    _plot(output_dir / png_name, rows, "decision", "DeepCluster cleaning decisions", len(lookup))
    return {"total_sample_count": len(lookup), "plotted_sample_count": len(rows), "max_samples": MAX_SAMPLES,
            "parameters": {"coordinate_source": "deepcluster_tsne.csv", "projection_reused": True},
            "feature_space": FEATURE_SPACE, "degenerate": bool(np.all(coordinates == coordinates[0])),
            "files": [csv_name, png_name]}

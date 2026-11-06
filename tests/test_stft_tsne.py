"""t-SNE 可视化的索引对齐、边界输入及真实投影验证。"""

import csv

import numpy as np
import pytest
from PIL import Image

from src.data import deepcluster_visualization as analysis


def make_rows(n):
    return [{"sample_index": i, "source_file": "T0010_test.h5", "source_row_idx": i,
             "rf_channel": 0, "source_sample_idx": i, "cluster_id": i % 2,
             "original_label": 2 + i % 2} for i in range(n)]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def test_sampling_and_alignment(monkeypatch, tmp_path):
    n = 10007
    z = np.arange(n * 128, dtype=np.float32).reshape(n, 128)
    before = z.copy()
    calls = []

    class Projection:
        def __init__(self, **kwargs):
            assert kwargs["perplexity"] == 30

        def fit_transform(self, selected):
            calls.append(selected.copy())
            return selected[:, :2]

    monkeypatch.setattr(analysis, "TSNE", Projection)
    rows = make_rows(n)
    for _ in range(2):
        report = analysis.write_deepcluster_tsne(tmp_path, z, rows)
        exported = read_rows(tmp_path / "deepcluster_tsne.csv")
        indices = [int(row["sample_index"]) for row in exported]
        assert len(indices) == len(set(indices)) == 10000
        assert indices == sorted(indices)
        np.testing.assert_array_equal(calls[-1], z[indices])
        for index, row in zip(indices, exported):
            assert float(row["tsne_x"]) == z[index, 0]
            assert int(row["cluster_id"]) == rows[index]["cluster_id"]
    np.testing.assert_array_equal(calls[0], calls[1])
    np.testing.assert_array_equal(z, before)
    assert report["total_sample_count"] == n


@pytest.mark.parametrize("n,degenerate", [(2, False), (12, False), (8, True)])
def test_real_projection_and_images(tmp_path, n, degenerate):
    z = np.zeros((n, 128)) if degenerate else np.random.default_rng(42).normal(size=(n, 128))
    rows = make_rows(n)
    report = analysis.write_deepcluster_tsne(tmp_path, z, rows)
    assert report["degenerate"] == degenerate
    assert report["parameters"]["perplexity"] == n - 1
    exported = read_rows(tmp_path / "deepcluster_tsne.csv")
    xy = np.array([[float(row["tsne_x"]), float(row["tsne_y"])] for row in exported])
    assert np.isfinite(xy).all()
    if degenerate:
        assert not xy.any()
    for name in ("deepcluster_tsne_clusters.png", "deepcluster_tsne_original_labels.png"):
        with Image.open(tmp_path / name) as image:
            assert min(image.size) > 100
            image.verify()

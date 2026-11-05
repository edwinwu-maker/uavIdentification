"""t-SNE 可视化的索引对齐、边界输入及真实投影验证。"""

import csv

import numpy as np
import pytest
from PIL import Image

from src.data import stft_analysis as analysis


def make_rows(n):
    return [{"hdbscan_label": -1 if i % 3 == 0 else i % 2,
             "original_label": 2 + i % 2,
             "fusion_label": ("signal", "background", "uncertain")[i % 3]} for i in range(n)]


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
        report = analysis._write_tsne_visualization(tmp_path, z, rows)
        exported = read_rows(tmp_path / "hdbscan_tsne.csv")
        indices = [int(row["sample_index"]) for row in exported]
        assert len(indices) == len(set(indices)) == 10000
        assert indices == sorted(indices)
        np.testing.assert_array_equal(calls[-1], z[indices])
        for index, row in zip(indices, exported):
            assert float(row["tsne_x"]) == z[index, 0]
            assert int(row["hdbscan_label"]) == rows[index]["hdbscan_label"]
            assert row["fusion_label"] == rows[index]["fusion_label"]
    np.testing.assert_array_equal(calls[0], calls[1])
    np.testing.assert_array_equal(z, before)
    assert report["total_sample_count"] == n


@pytest.mark.parametrize("n,degenerate", [(2, False), (12, False), (8, True)])
def test_real_projection_and_images(tmp_path, n, degenerate):
    z = np.zeros((n, 128)) if degenerate else np.random.default_rng(42).normal(size=(n, 128))
    rows = make_rows(n)
    for row in rows:
        row["hdbscan_label"] = -1
    report = analysis._write_tsne_visualization(tmp_path, z, rows)
    assert report["degenerate"] == degenerate
    assert report["parameters"]["perplexity"] == n - 1
    exported = read_rows(tmp_path / "hdbscan_tsne.csv")
    xy = np.array([[float(row["tsne_x"]), float(row["tsne_y"])] for row in exported])
    assert np.isfinite(xy).all()
    if degenerate:
        assert not xy.any()
    for name in report["files"][1:]:
        with Image.open(tmp_path / name) as image:
            assert image.size == (1500, 1050)
            image.verify()

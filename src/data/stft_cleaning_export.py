"""DeepCluster 导出共用的 H5 行复制、CSV 清单与 JSON 写出工具。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py


def _copy_row_dataset(
    source: h5py.Dataset,
    destination_file: h5py.File,
    name: str,
    source_rows: list[int],
    *,
    batch_size: int = 16,
) -> None:
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


def write_manifest(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty cleaning manifest")
    fieldnames = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    with Path(path).open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, allow_nan=False)

import csv
import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import h5py


def get_filename_with_suffix(file_path: str, suffix: str) -> str:
    """获取替换最后一个扩展名后的文件名，不访问文件系统。

    输入：
        file_path: 文件路径字符串，可包含目录。
        suffix: 包含点号的新后缀，如 ".h5"。
    输出：
        不包含目录的文件名字符串；只替换最后一个扩展名，无扩展名时追加后缀。
    示例：
        get_filename_with_suffix("/data/demo.v1.mat", ".h5") 返回 "demo.v1.h5"。
    """
    name, _ = os.path.splitext(os.path.basename(file_path))
    return name + suffix


def iter_files_by_suffix(directory: str, suffix: str) -> Iterator[str]:
    """按文件名排序逐项返回指定后缀的文件路径，不递归扫描。

    输入：
        directory: 待扫描的目录，目录须存在。
        suffix: 按大小写敏感的原样后缀匹配，如 ".h5"；空字符串匹配所有文件。
    输出：
        单次遍历的惰性迭代器 Iterator[str]，不包含子目录，无匹配文件时不产生路径。
        路径保留输入目录形式：相对目录返回相对路径，绝对目录返回绝对路径。
        创建生成器时不扫描目录，首次迭代时读取并排序全部目录名称；
        目录访问异常也在迭代时抛出，排序仍需存储全部目录名称。
    """
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(suffix):
            continue
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            yield file_path


def list_files_by_suffix(directory: str, suffix: str) -> list[str]:
    """筛选当前目录中指定后缀的文件，不递归扫描。

    输入：
        directory: 待扫描的目录，目录须存在。
        suffix: 按大小写敏感的原样后缀匹配，如 ".h5"；空字符串匹配所有文件。
    输出：
        按文件名排序的文件路径列表 list[str]，不包含子目录，无匹配文件时返回 []。
        路径保留输入目录形式：相对目录返回相对路径，绝对目录返回绝对路径。
    """
    return list(iter_files_by_suffix(directory, suffix))


def read_csv_records(path: str | Path) -> list[dict[str, str]]:
    """一次性读取 UTF-8 CSV ，以首行表头为键返回字典记录列表。

    输入：
        path: CSV 文件路径。
    输出：
        按原始行顺序返回全部记录；空文件或仅含表头时返回 []。
        保留 csv.DictReader 的解析行为，读取异常直接向调用方传播。
    """
    with Path(path).open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def compute_file_sha256(path: str | Path) -> str:
    """按 1 MiB 分块读取文件内容，计算 SHA-256 指纹。

    输入：
        path: 待计算文件的路径，接受字符串或 Path。
    输出：
        64 字符的小写十六进制字符串；文件读取异常直接向调用方传播。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def h5_attr_to_text(value: object) -> str:
    """将 H5 属性值转换为文本。

    输入：
        value: 从 H5 读取的属性值。
    输出：
        bytes 按 UTF-8 解码，其他类型使用 str() 转换；异常直接向调用方传播。
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


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

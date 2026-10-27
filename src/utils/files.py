import os


def list_files_by_suffix(directory: str, suffix: str) -> list[str]:
    """按文件名排序返回当前目录中指定后缀的文件路径，不递归扫描。

    suffix 按大小写敏感的原样后缀匹配，例如 ".h5"；返回路径保留输入目录形式。
    """
    return [
        os.path.join(directory, filename)
        for filename in sorted(os.listdir(directory))
        if filename.endswith(suffix) and os.path.isfile(os.path.join(directory, filename))
    ]

import os


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


def list_files_by_suffix(directory: str, suffix: str) -> list[str]:
    """筛选当前目录中指定后缀的文件，不递归扫描。

    输入：
        directory: 待扫描的目录路径字符串，目录须存在。
        suffix: 按大小写敏感的原样后缀匹配，如 ".h5"；空字符串匹配所有文件。
    输出：
        按文件名排序的文件路径列表 list[str]，不包含子目录，无匹配文件时返回 []。
        路径保留输入目录形式：相对目录返回相对路径，绝对目录返回绝对路径。
    示例：
        若 data 目录包含 b.h5、a.h5 和 c.txt，
        list_files_by_suffix("data", ".h5") 返回 ["data/a.h5", "data/b.h5"]。
    """
    return [
        os.path.join(directory, filename)
        for filename in sorted(os.listdir(directory))
        if filename.endswith(suffix) and os.path.isfile(os.path.join(directory, filename))
    ]

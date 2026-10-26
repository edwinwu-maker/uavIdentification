from pathlib import Path

import torch

from src.utils.paths import model_dir


def default_model_path(model_name: str) -> Path:
    """将模型文件名拼接到项目默认的 outputs/models/ 目录下。"""
    return model_dir() / model_name


def _resolve_model_path(model_path_or_name) -> Path:
    """纯文件名使用默认模型目录；绝对路径或含父目录的相对路径按原样使用。

    相对路径以当前工作目录为基准。
    """
    path = Path(model_path_or_name)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return default_model_path(path.name)


def save_model(state_dict, model_path_or_name):
    """保存传入的模型状态字典，自动创建父目录并返回保存路径。

    同名文件会被覆盖；优化器状态和训练轮次不会自动补入。
    """
    path = _resolve_model_path(model_path_or_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    return path


def load_model(model_path_or_name, map_location=None):
    """按文件名或指定路径加载并返回已保存的对象。

    map_location 透传给 torch.load，用于指定张量加载到的设备，
    例如传入 "cpu" 可在 CPU 上读取 GPU 保存的权重；本函数不构建模型。
    """
    path = _resolve_model_path(model_path_or_name)
    return torch.load(path, map_location=map_location)

import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_raw_data_dir() -> str:
    """按平台返回 DroneRFa 原始 .mat 数据的默认目录。

    输出：
        仅返回路径，不检查目录是否存在，也不创建目录。
    """
    if os.name == "nt":
        return "E:/dataSet/DroneRFa"
    if sys.platform == "darwin":
        return os.path.expanduser("~/Desktop/dataset/droneRFa")
    return "/mnt/data/wurixin/DroneRFa"


def output_dir() -> Path:
    return project_root() / "outputs"


def log_dir() -> Path:
    return output_dir() / "logs"


def model_dir() -> Path:
    return output_dir() / "models"


def figures_dir() -> Path:
    return output_dir() / "figures"


def metrics_dir() -> Path:
    return output_dir() / "metrics"

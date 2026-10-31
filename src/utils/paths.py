import os
import sys
from pathlib import Path


def format_snr_for_filename(snr_db: float) -> str:
    """把 SNR 数值转换成稳定、安全的文件名片段。"""

    snr = float(snr_db)
    if snr < 0:
        prefix = "m"
    elif snr > 0:
        prefix = "p"
    else:
        prefix = ""

    value = abs(snr)
    if value.is_integer():
        value_text = str(int(value))
    else:
        value_text = f"{value:g}".replace(".", "p")
    return f"{prefix}{value_text}"


def build_raw_sample_png_path(
    save_root: Path,
    *,
    filename: str,
    sample_idx: int,
    rf_channel: int,
    snr_db: float | None,
) -> Path:
    """构造 raw 导图路径：<根目录>/<条件>/<类别>/<源文件名>/<PNG 文件名>。

    条件为 clean 或 snr_<编码>db；样本索引至少补齐四位。
    仅拼接路径，不创建目录或写入图片。
    """
    stem = Path(filename).stem
    class_code = stem.split("_")[0]
    condition = "clean" if snr_db is None else f"snr_{format_snr_for_filename(snr_db)}db"
    png_name = f"{stem}_RF{rf_channel}_sample_{sample_idx:04d}_{condition}.png"
    return save_root / condition / class_code / stem / png_name


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

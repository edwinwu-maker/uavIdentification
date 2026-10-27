"""Shared feature metadata for the STFT and CPP/FAM workflows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Type

from torch.utils.data import Dataset

from src.data.drone_rfa_io import default_raw_data_dir
from src.data.cpp_dataset import CppDataset
from src.data.stft_dataset import StftDataset


@dataclass(frozen=True)
class FeatureConfig:
    """保存单种特征的数据集、默认路径和输出命名配置；实例创建后字段不可修改。"""

    name: str                       # 特征标识，如 "stft" 或 "cpp"。
    dataset_class: Type[Dataset]    # 加载对应 H5 特征的数据集类。
    default_data_dir: str           # 预计算特征 H5 文件的默认目录。
    model_name: str                 # 模型权重的默认文件名，不包含目录。
    cm_array_name: str              # 混淆矩阵 NumPy 数组的默认文件名。
    cm_image_name: str              # 混淆矩阵 PNG 图片的默认文件名。


def _build_config() -> dict[str, FeatureConfig]:
    """构建 STFT 与 CPP/FAM 配置，以特征标识为键。

    每次调用均根据当前平台的原始数据目录构造同级特征目录路径，不检查目录是否存在。
    """
    base_dir = default_raw_data_dir()
    return {
        "stft": FeatureConfig(
            name="stft",
            dataset_class=StftDataset,
            default_data_dir=str(Path(base_dir).parent / "DroneRFa_stft_awgn_random_16class_h5"),
            model_name="best_stft_16class_model.pth",
            cm_array_name="stft_16class_confusion_matrix.npy",
            cm_image_name="stft_16class_confusion_matrix.png",
        ),
        "cpp": FeatureConfig(
            name="cpp",
            dataset_class=CppDataset,
            default_data_dir=str(Path(base_dir).parent / "DroneRFa_cpp_awgn_random_16class_h5"),
            model_name="best_cpp_16class_model.pth",
            cm_array_name="cpp_16class_confusion_matrix.npy",
            cm_image_name="cpp_16class_confusion_matrix.png",
        ),
    }


def get_feature_config(feature: str) -> FeatureConfig:
    """按大小写敏感的特征标识获取配置，支持 "stft" 和 "cpp"。

    不支持的标识会触发 ValueError，错误信息列出可选特征。
    """
    specs = _build_config()
    try:
        return specs[feature]
    except KeyError as exc:
        supported = ", ".join(sorted(specs))
        raise ValueError(f"Unsupported feature {feature!r}; choose one of: {supported}") from exc

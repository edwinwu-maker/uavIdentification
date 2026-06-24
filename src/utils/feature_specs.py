"""Shared feature metadata for the STFT and CPP/FAM workflows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Type

from torch.utils.data import Dataset

from src.data.drone_rfa_io import default_raw_data_dir
from src.data.cpp_dataset import CppDataset
from src.data.stft_dataset import StftDataset

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dataset_class: Type[Dataset]
    feature_key: str
    default_data_dir: str
    checkpoint_name: str
    cm_array_name: str
    cm_image_name: str
    train_description: str
    eval_description: str


def _build_specs() -> dict[str, FeatureSpec]:
    base_dir = default_raw_data_dir()
    return {
        "stft": FeatureSpec(
            name="stft",
            dataset_class=StftDataset,
            feature_key="stft",
            default_data_dir=str(Path(base_dir).parent / "DroneRFa_stft_h5"),
            checkpoint_name="best_stft_model.pth",
            cm_array_name="stft_confusion_matrix.npy",
            cm_image_name="stft_confusion_matrix.png",
            train_description="Single-GPU training on pre-computed .h5 stfts",
            eval_description="Test on pre-computed .h5 stfts",
        ),
        "cpp": FeatureSpec(
            name="cpp",
            dataset_class=CppDataset,
            feature_key="cpp",
            default_data_dir=str(Path(base_dir).parent / "DroneRFa_cpp_h5"),
            checkpoint_name="best_cpp_model.pth",
            cm_array_name="cpp_confusion_matrix.npy",
            cm_image_name="cpp_confusion_matrix.png",
            train_description="Single-GPU training on pre-computed CPP/FAM .h5 matrices",
            eval_description="Test on pre-computed CPP/FAM .h5 matrices",
        ),
    }


def get_feature_spec(feature: str) -> FeatureSpec:
    specs = _build_specs()
    try:
        return specs[feature]
    except KeyError as exc:
        supported = ", ".join(sorted(specs))
        raise ValueError(f"Unsupported feature {feature!r}; choose one of: {supported}") from exc

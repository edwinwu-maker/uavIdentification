"""H5 样本导图工具：校验特征和 RF 通道，根据样本信息生成 PNG 路径。"""

from __future__ import annotations

import os

import h5py


def build_sample_png_path(
    save_root: str,
    drone_code: str,
    name_no_extension: str,
    sample_idx: int,
    rf_channel: int,
) -> str:
    """根据源文件和样本信息构造 PNG 输出路径。

    参数：
        save_root: PNG 输出根目录。
        drone_code: 类别编码，如 "T0010"，用作根目录下的类别子目录名。
        name_no_extension: 源 H5 文件去掉目录和扩展名后的名称，用作 PNG 文件名前缀。
        sample_idx: 样本在源 H5 文件中的索引，从 0 开始，文件名中至少补齐四位。
        rf_channel: 当前样本所属的 RF 通道编号，通常为 0 或 1，写入文件名以区分通道。

    返回：
        字符串路径：
        <save_root>/<drone_code>/<name_no_extension>_RF<rf_channel>_sample_<sample_idx:04d>.png。
        仅拼接路径，不创建目录或写入图片，实际保存由绘图函数完成。
    """
    png_name = f"{name_no_extension}_RF{rf_channel}_sample_{sample_idx:04d}.png"
    return os.path.join(save_root, drone_code, png_name)


def validate_h5_samples(h5f: h5py.File, feature_key: str) -> None:
    """校验已打开 H5 中的单通道特征及 RF 通道，不管理文件句柄。

    feature_key（如 stft、cpp）对应的数据须为 (N, 1, H, W)，
    rf_channel 须为 (N,)，且各元素转为整数后为 0 或 1。
    校验失败时抛出异常；只读取特征形状和 RF 通道，不加载特征数组。
    """
    h5_path = h5f.filename
    feature_shape = h5f[feature_key].shape
    if len(feature_shape) != 4 or feature_shape[1] != 1:
        raise ValueError(
            f"Feature dataset must have shape (N, 1, H, W), got {feature_shape} in {h5_path}. "
            "Re-run precomputation; legacy dual-channel caches are not supported."
        )
    if "rf_channel" not in h5f:
        raise ValueError(f"Missing rf_channel dataset in {h5_path}; re-run precomputation")
    rf_channels = h5f["rf_channel"][:]
    if rf_channels.shape != (feature_shape[0],) or not all(
        int(channel) in (0, 1) for channel in rf_channels
    ):
        raise ValueError(f"Invalid rf_channel dataset in {h5_path}; re-run precomputation")

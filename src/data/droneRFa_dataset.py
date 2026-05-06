import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.logger import logger

SAMPLE_POINT_NUM = 10e6

def load_iq_signal(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    logger.info(f"read file {file_path}...")
    with h5py.File(file_path, 'r') as file:
        RF0_I = file['RF0_I'][0]
        RF0_Q = file['RF0_Q'][0]
        data_ch0 = RF0_I + 1j * RF0_Q
        RF1_I = file['RF1_I'][0]
        RF1_Q = file['RF1_Q'][0]
        data_ch1 = RF1_I + 1j * RF1_Q

    logger.debug(f"Signal length of channel 0 is: {len(data_ch0)}, Data Type is: {data_ch0.dtype}")
    logger.debug(f"Signal length of channel 1 is: {len(data_ch1)}, Data Type is: {data_ch1.dtype}")
    return data_ch0, data_ch1


class DroneRFaDataset(Dataset):
    def __init__(self, mat_file_paths, transform=None):
        self.mat_file_paths = mat_file_paths
        self.sample_length = SAMPLE_POINT_NUM
        self.sample_idx_list = self._build_sample_index_list()
        self.transform = transform

        self.label_mapping = {
            "T0000": 0,
            "T0001": 1,
            "T0010": 2,
            "T0011": 3,
            "T0100": 4,
            "T0101": 5,
            "T0110": 6,
            "T0111": 7,
            "T1000": 8,
            "T1001": 9,
            "T1010": 10,
            "T1011": 11,
            "T1100": 12,
            "T1101": 13,
            "T1110": 14,
            "T1111": 15,
            "T10000": 16,
            "T10001": 17,
            "T10010": 18,
            "T10011": 19,
            "T10100": 20,
            "T10101": 21,
            "T10110": 22,
            "T10111": 23,
            "T11000": 24,
        }

    def _build_sample_index_list(self) -> list[tuple[str, int]]:
        """
        返回：list[(文件路径, 偏移量)]
        """
        sample_index = []
        for mat_file_path in self.mat_file_paths:
            try:
                with h5py.File(mat_file_path, "r") as f:
                    total_points = int(f["RF0_I"].shape[1])
                    num_samples = total_points // int(self.sample_length)
                    for i in range(num_samples):
                        offset = i * self.sample_length
                        sample_index.append((mat_file_path, offset))
            except Exception as e:
                logger.error(f"Warning: 跳过损坏文件 {mat_file_path}, {str(e)}")
        return sample_index
    
    def _parse_drone_label(self, file_name) -> int:
        """从文件名解析无人机机型标签"""
        base_name = os.path.basename(file_name)
        type_code = base_name.split("_")[0]
        return self.label_mapping.get(type_code, 0)
    
    def __len__(self) -> int:
        return len(self.sample_idx_list)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # 定位数据
        file_path, offset = self.sample_idx_list[idx]

        # 读取IQ数据（语义化变量名）
        with h5py.File(file_path, "r") as mat_file:
            i_channel_0 = mat_file["RF0_I"][offset : offset + self.sample_length]
            q_channel_0 = mat_file["RF0_Q"][offset : offset + self.sample_length]
            i_channel_1 = mat_file["RF1_I"][offset : offset + self.sample_length]
            q_channel_1 = mat_file["RF1_Q"][offset : offset + self.sample_length]

        # 构造复信号
        complex_signal_ch0 = i_channel_0 + 1j * q_channel_0
        complex_signal_ch1 = i_channel_1 + 1j * q_channel_1
        iq_data = np.stack([complex_signal_ch0, complex_signal_ch1], axis=0)
        # 获取标签
        drone_label = self._parse_drone_label(file_path)
        # 转张量
        iq_data = torch.from_numpy(iq_data).double()
        label = torch.tensor(drone_label, dtype=torch.int64)
        return iq_data, label
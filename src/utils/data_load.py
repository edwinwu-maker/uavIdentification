import h5py
import numpy as np
from .logger import logger

def load_iq_signal(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    logger.info(f"read file {file_path}...")
    file = h5py.File(file_path, 'r')
    RF0_I = file['RF0_I'][0]
    RF0_Q = file['RF0_Q'][0]
    data_ch0 = RF0_I + 1j * RF0_Q
    RF1_I = file['RF1_I'][0]
    RF1_Q = file['RF1_Q'][0]
    data_ch1 = RF1_I + 1j * RF1_Q
    file.close()
    return data_ch0, data_ch1









import glob
import os

from torch.utils.data import DataLoader

from data import droneRFa_dataset
from utils.logger import logger


if __name__ == "__main__":
    if os.name == "nt":
        logger.info("run in windows")
        DATA_DIR = "E:/dataSet/DroneRFa"
    else:
        logger.info("run in linux")
        DATA_DIR = "/mnt/data/wurixin/DroneRFa"
    # 获取所有 mat 文件
    mat_file_list = sorted(glob.glob(f"{DATA_DIR}/*.mat"))
    logger.info(f"find {len(mat_file_list)} .mat files")

    # 构建 Dataset
    dataset =  droneRFa_dataset.DroneRFaDataset(
        mat_file_list,
        transform=None
    )
    logger.info(f"dataset length: {len(dataset)}")

    # 构建 DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    logger.info(f"{len(dataloader)} batches in dataloader")
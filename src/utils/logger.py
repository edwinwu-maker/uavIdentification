import logging
import os
from datetime import datetime

from src.utils.paths import log_dir

def setup_logger():
    logger = logging.getLogger("DroneRFa")
    logger.setLevel(logging.DEBUG)

    # 防止重复追加handler
    if logger.handlers:
        logger.handlers.clear()

    # 日志格式
    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-6s | %(filename)s:%(lineno)-4d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)

    # 文件 Handler
    log_path = log_dir() / f"{datetime.now().strftime('%Y%m%d')}.log"
    os.makedirs(log_path.parent, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)

    # 日志 Handler
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# 全局单例
logger = setup_logger()

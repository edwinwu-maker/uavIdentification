import logging
import os
from datetime import datetime

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
    # 获取当前文件（logger.py）的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 上两级 → 到达项目根目录（main.py 的上一级）
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)

    # 日志 Handler
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# 全局单例
logger = setup_logger()
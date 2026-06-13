import logging
from datetime import datetime

from src.utils.paths import log_dir


def _mark_handler(handler, kind: str):
    handler._drone_rfa_kind = kind
    return handler


def _has_handler(logger, kind: str, *, log_path: str | None = None) -> bool:
    for handler in logger.handlers:
        if getattr(handler, "_drone_rfa_kind", None) != kind:
            continue
        if kind == "file" and log_path is not None:
            if getattr(handler, "baseFilename", None) == log_path:
                return True
            continue
        return True
    return False


def setup_logger():
    logger = logging.getLogger("DroneRFa")
    logger.setLevel(logging.DEBUG)

    # 日志格式
    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-6s | %(filename)s:%(lineno)-4d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 Handler
    if not _has_handler(logger, "console"):
        console_handler = _mark_handler(logging.StreamHandler(), "console")
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    # 文件 Handler
    log_path = log_dir() / f"{datetime.now().strftime('%Y%m%d')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path_str = str(log_path)
    if not _has_handler(logger, "file", log_path=log_path_str):
        file_handler = _mark_handler(
            logging.FileHandler(log_path, encoding="utf-8"),
            "file",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

# 全局单例
logger = setup_logger()

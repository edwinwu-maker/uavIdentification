import logging
from datetime import datetime

from src.utils.paths import log_dir

# 标记本模块创建的 Handler，供重复初始化时识别和去重。
def _mark_handler(handler, kind: str):
    # 给 handler 添加一个自定义属性 _drone_rfa_kind，并将它的值设置为 kind
    handler._drone_rfa_kind = kind
    return handler

# 检查是否已有指定类型的 Handler；指定日志路径时，文件 Handler 还需匹配该路径。
def _has_handler(logger, kind: str, *, log_path: str | None = None) -> bool:
    # 遍历 logger 的所有 handler。
    for handler in logger.handlers: 
        # 如果 handler 没有标记类型，则跳过。
        if getattr(handler, "_drone_rfa_kind", None) != kind:
            continue
        # 如果 handler 是文件类型，并且指定了日志路径，则检查该 handler 的日志路径是否匹配。
        if kind == "file" and log_path is not None:
            if getattr(handler, "baseFilename", None) == log_path:
                return True # file_handler 合规
            continue
        return True # console_handler 合规
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

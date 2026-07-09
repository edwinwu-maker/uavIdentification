import os
import shlex
import subprocess
import sys


def format_current_command() -> str:
    """返回当前 Python 进程对应的可复制命令行。"""

    parts = [sys.executable, *sys.argv]
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def log_current_command(logger) -> None:
    """在脚本启动时打印一次本次执行命令。"""

    logger.info("Command: %s", format_current_command(), stacklevel=2)

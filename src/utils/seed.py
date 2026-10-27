import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 的全局随机种子。

    参数：
        seed: 随机种子整数；CUDA 可用时也为所有 CUDA 设备设置种子。

    返回：
        None；直接更新各随机数生成器的状态。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

import torch
import numpy as np
import random
import os

def set_random_seed(seed: int, using_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if using_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_logger_path(logger_path: str = 'dp_single_version2/logger_path.txt') -> str:
    if not os.path.exists(logger_path): return ValueError(f'logger path: {logger_path} not exsits!')
    with open(logger_path, mode='r', encoding='utf-8') as reader:
        content = reader.read().strip()
    return content

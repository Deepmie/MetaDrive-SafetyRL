import torch
import numpy as np
import random
import os
import json
from typing import List, Dict

def set_random_seed(seed: int, using_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if using_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_logger_path(logger_path: str, method_name: str) -> str:
    if not os.path.exists(logger_path): return ValueError(f'logger path: {logger_path} not exsits!')
    with open(logger_path, mode='r', encoding='utf-8') as reader:
        content: Dict = json.loads(reader.read().strip())
    return str(content.get(method_name))
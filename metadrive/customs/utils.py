import torch
from torch import Tensor
from numpy import ndarray
from typing import List, Union, cast, Tuple, Optional
from io import TextIOWrapper
from dataclasses import dataclass
import os
from datetime import datetime
import matplotlib.pyplot as plt
import swanlab
from metadrive.customs.config import MainConfig


def converto_torch(x: Union[ndarray, Tensor]):
    if isinstance(x, Tensor):
        return x
    x = cast(ndarray, x)
    return torch.from_numpy(x).to(torch.float32)


def converto_ndarray(x: Union[ndarray, Tensor]):
    if isinstance(x, ndarray):
        return x
    x = cast(Tensor, x)
    return x.cpu().detach().numpy()


@dataclass
class LoggerConfig:
    root_path: str = 'log'


@dataclass
class LoggerLevel:
    INFO:  str = '0'
    WARN:  str = '1'
    ERROR: str = '2'


class Logger:
    def __init__(self, config: LoggerConfig):
        self.config = config
        if not os.path.exists(config.root_path):
            os.makedirs(config.root_path, exist_ok=False)
        self.reset()
    
    def reset(self):
        self._create_time: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self._logger_file_path: str = f'log_{self._create_time}.txt'
        self._logger_file: TextIOWrapper = open(os.path.join(self.config.root_path, self._logger_file_path), mode='w', encoding='utf-8')
    
    def _record(self, level: str, message: str):
        self._record_time: str = datetime.now().strftime('%H:%M:%S')
        self._logger_file.write(f'[{level}] {self._record_time} {message}')


def swanlab_init(config: MainConfig) -> bool:
    project_name: str = 'SaftyRL'
    create_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    swanlab.init(
        # 设置将记录此次运行的项目信息
        project = project_name,
        experiment_name=f"Run_{create_time}",
        workspace = 'deepmiemie',
        # 跟踪超参数和运行元数据
        config={
            'learning_rate': config.solver_config.learning_rate,
        }
    )
    return True
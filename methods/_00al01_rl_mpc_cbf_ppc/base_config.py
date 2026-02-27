from dataclasses import dataclass, field, fields
import numpy as np
import torch
from torch.nn import Module, Tanh
from numpy import ndarray
from typing import Dict, Optional, Tuple, Union
from methods.common.type import ActionType
from methods.common.base_config import BaseAlgConfig

METHOD_NAME          = 'rl+mpc+cbf+ppc+traj'
ACTION_DIM           = 4

@dataclass
class AlgConfig(BaseAlgConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_special_attr(self, 'action_dim', ACTION_DIM)
        self.method_name: str = METHOD_NAME


if __name__ == '__main__':
    algconfig = AlgConfig()
    print(algconfig.env_config.action_dim)
    print(algconfig.method_name)
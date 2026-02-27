from dataclasses import dataclass, field
import numpy as np
import torch
from torch.nn import Module, Tanh
from numpy import ndarray
from typing import Dict, Optional, Tuple
from methods.common.type import ActionType
from methods.common.base_config import BaseAlgConfig

METHOD_NAME          = 'rl'

@dataclass
class AlgConfig(BaseAlgConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method_name: str = METHOD_NAME


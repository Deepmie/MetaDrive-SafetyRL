from torch import Tensor
from numpy import ndarray
from methods.common.base_config import BaseAlgConfig
from typing import Tuple, Dict, Union, Optional, List
from abc import ABC, abstractmethod

class BaseAlg(ABC):
    def __init__(self, config: BaseAlgConfig, eval_mode: bool = False):
        ...
    
    @abstractmethod
    def start(self) -> Tuple[bool, str]:
        ...

    @abstractmethod
    def eval_with_checkpoint(self, ckpt_path: str) -> Tuple[float, List, Dict]:
        ...
    
    @abstractmethod
    def final_eval(self, evaluate_path: Optional[str] = None):
        ...
    
    @abstractmethod
    def predict(self, obs: Union[ndarray, Tensor], state: Optional[ndarray] = None, deterministic: bool = False):
        ...
    
    @abstractmethod
    def close(self):
        ...
    
    @abstractmethod
    def load_weight_from_checkpoint(self, load_path: Optional[str] = None) -> Dict:
        ...


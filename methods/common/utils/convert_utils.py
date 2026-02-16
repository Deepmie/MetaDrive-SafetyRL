import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from typing import Union, cast

def converto_torch(x: Union[ndarray, Tensor]):
    if isinstance(x, Tensor):
        return x
    elif isinstance(x, ndarray):
        x = cast(ndarray, x)
        return torch.from_numpy(x).to(torch.float32)

def converto_ndarray(x: Union[float, bool, int, ndarray, Tensor], dtype: Union[np.dtype] = np.float32):
    if isinstance(x, ndarray):
        return x
    elif isinstance(x, float) or isinstance(x, bool):
        return np.array([x], dtype=dtype)
    elif isinstance(x, int):
        return np.array([x], dtype=dtype)
    elif isinstance(x, Tensor):
        x = cast(Tensor, x)
        return x.cpu().detach().numpy()
    else:
        raise TypeError(f'x must include [\'float\', \'bool\', \'int\', \'ndarray\', \'Tensor\'], but now the type of x is {type(x)}')

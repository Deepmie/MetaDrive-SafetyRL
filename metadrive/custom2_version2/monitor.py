from typing import List, Tuple, Dict, cast, Optional, Union, Callable
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
import os

class Monitor:
    def __init__(self, logger_path: str):
        self.ppc_errors: List = list()
        self._ppc_errors_steps: int = 0
        self._best_reward: float = float('inf')
        self._npy_save_path = f'{logger_path}/ppc_errors.npy'
        self._fig_save_path = f'{logger_path}/ppc_errors.png'

    def collect_ppc_errors(self, errors: ndarray, save_freq: int = 5000):
        self.ppc_errors.append(errors.tolist())
        self._ppc_errors_steps += 1

        if self._ppc_errors_steps % save_freq == 0:
            self._ppc_errors_numpy = np.array(self.ppc_errors, dtype=np.float32) # 转为numpy矩阵
            np.save(self._npy_save_path, self._ppc_errors_numpy) # 添加save_freq次保存一次
            self.plot_error()
    
    def plot_error(self):
        assert os.path.exists(self._npy_save_path)
        _ppc_errors_numpy = np.load(self._npy_save_path)

        fig, ax = plt.subplots(figsize=(20, 6))
        ax.axis('off')
        x = np.arange(0, _ppc_errors_numpy.shape[0], 1)

        # ax1
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.scatter(x, _ppc_errors_numpy[:, 0, 0], s=1)
        ax1.set_title('ppc error dim 0')
        ax1.set_xlabel('step'); ax1.set_ylabel('error')

        fig.savefig(self._fig_save_path)

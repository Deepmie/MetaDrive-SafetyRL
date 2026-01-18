from typing import List, Tuple, Dict, cast, Optional, Union, Callable
from metadrive.custom2_version2.ocp.performance_metric_func import PerformetricFunc
from metadrive.custom2_version2.ocp.config import MPConfig
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
        mpc_config: MPConfig = MPConfig()
        self.performance_func = PerformetricFunc(mpc_config)
        self.z_refs: List = list()
        self.z_mpcs: List = list()
        self._z_refs_mpcs_steps: int     = 0
        self._z_refs_save_path: str      = f'{logger_path}/z_refs.npy'
        self._z_mpcs_save_path: str      = f'{logger_path}/z_mpcs.npy'
        self._z_refs_mpcs_fig_save_path: str = f'{logger_path}/z_refs_mpcs.png'

    def collect_ppc_errors(self, errors: ndarray, save_freq: int = 5000):
        self.ppc_errors.append(errors.tolist())
        self._ppc_errors_steps += 1

        if self._ppc_errors_steps % save_freq == 0:
            self._ppc_errors_numpy = np.array(self.ppc_errors, dtype=np.float32) # 转为numpy矩阵
            np.save(self._npy_save_path, self._ppc_errors_numpy) # 添加save_freq次保存一次
            self.plot_error()

    def collect_mpc_and_rl(self, z_ref: ndarray, z_mpc: ndarray, save_freq: int = 5000):
        self.z_refs.append(z_ref)
        self.z_mpcs.append(z_mpc)
        self._z_refs_mpcs_steps += 1

        if self._z_refs_mpcs_steps % save_freq == 0:
            self.z_refs_numpy = np.array(self.z_refs)
            self.z_mpcs_numpy = np.array(self.z_mpcs)
            np.save(self._z_refs_save_path, self.z_refs_numpy); np.save(self._z_mpcs_save_path, self.z_mpcs_numpy)
            self.plot_z_refs_mpcs()
        
    def plot_error(self):
        assert os.path.exists(self._npy_save_path)
        _ppc_errors_numpy = np.load(self._npy_save_path) # (data_nums, env_nums, 2)

        fig, ax = plt.subplots(figsize=(20, 13))
        ax.axis('off')
        N = _ppc_errors_numpy.shape[0]
        x = np.arange(0, N, 1)
        zetas = np.zeros(N, dtype=np.float32)

        for i in range(0, _ppc_errors_numpy.shape[0]):
            zetas[i] = self.performance_func.error_transformation(error=_ppc_errors_numpy[i, 0, 0], step=i)

        # ax1
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.scatter(x, _ppc_errors_numpy[:, 0, 0], s=1)
        ax1.set_title('ppc error dim 0')
        ax1.set_xlabel('step'); ax1.set_ylabel('error')

        # ax2
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(zetas)
        
        fig.savefig(self._fig_save_path)

    def plot_z_refs_mpcs(self):
        assert os.path.exists(self._z_refs_save_path); assert os.path.exists(self._z_mpcs_save_path)
        z_refs_numpy = np.load(self._z_refs_save_path); z_mpcs_numpy = np.load(self._z_mpcs_save_path)

        fig, ax = plt.subplots(figsize=(20, 13))
        ax.axis('off')
        N = z_refs_numpy.shape[0]
        x = np.arange(0, N, 1)

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.scatter(x, z_refs_numpy[:, 0, 0], s=1)
        ax1.set_title('z_ref dim 0')
        ax1.set_xlabel('step'); ax1.set_ylabel('value')

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.scatter(x, z_mpcs_numpy[:, 0, 0], s=1)
        ax2.set_title('z_mpc dim 0')
        ax2.set_xlabel('step'); ax2.set_ylabel('value')

        fig.savefig(self._z_refs_mpcs_fig_save_path)
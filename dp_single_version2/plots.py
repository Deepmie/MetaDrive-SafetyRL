import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
import os
from typing import Tuple, List, Dict

class Ploter:
    def __init__(self):
        self._path_root: str = 'dp_single_version2/figdata'
        self._figsize: Tuple = (15, 6)
        self._set_init_config()
    
    def plot(self, reward_name: str):
        self._fig, self._ax = plt.subplots(figsize=self._figsize)
        self._ax.axis('off')
        self._plot_reward_1()

        self._fig.savefig(os.path.join(self._path_root, f'{reward_name}.png'))

    def _plot_reward_1(self):
        _ax = self._fig.add_subplot(1, 1, 1)
        self._plot_reward_group(_ax, [
            dict(reward_name='avg_reward', method_name='rl_mpc_cbf_traj'),
            dict(reward_name='avg_reward', method_name='rl_mpc_cbf_ppc_traj'),
        ])

        _ax.set_xlabel('step $s$'); _ax.set_ylabel('reward $r$')
        _ax.set_title('AVG REWARD')
        _ax.legend()

    def _plot_reward_group(self, ax: Axes, data_infos: List[Dict]) -> List[ndarray]:
        N: int = float('inf')
        slice_func = lambda x, N: x[0: N]
        datas: List[ndarray] = list()
        # get data
        for info in data_infos:
            data = self._process_reward(self._get_plot_data(info['reward_name'], info['method_name']))
            datas.append(data)
            if data.shape[0] < N: N = data.shape[0]
        
        # slice & plot
        x = np.arange(0, N)
        for info, data in zip(data_infos, datas):
            scatter_config: Dict = dict(s=2, label=info['method_name'])
            ax.scatter(x, slice_func(data, N), **scatter_config)
        return datas
    
    def _get_plot_data(self, reward_name: str, method_name: str) -> ndarray:
        _reward_abs_path: str = os.path.join(self._path_root, method_name, f'{reward_name}.txt')
        with open(_reward_abs_path, mode='r', encoding='utf-8') as reader:
            data: ndarray = np.array(list(filter(lambda x: x.strip(), reader.read().split('\n'))), dtype=np.float32).flatten()
        return data
    
    def _process_reward(self, data: ndarray) -> ndarray: # use mae
        N: int = data.shape[0]
        alpha: float = 0.05
        data_processed: ndarray = data[0] * np.ones_like(data)
        for i in range(1, N):
            data_processed[i] = alpha * data[i] + (1 - alpha) * data_processed[i-1]
        return data_processed
    
    def _set_init_config(self):
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
        })


if __name__ == '__main__':
    ploter: Ploter = Ploter()
    ploter.plot('avg_reward')
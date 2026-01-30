import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
import os
from typing import Tuple, List, Dict

class Ploter:
    def __init__(self):
        self._path_root: str = 'dp_single_version2/figdata'
        self._figsize: Tuple = (20, 6)
        self._set_init_config()
    
    def plot(self, reward_name: str):
        self._fig, self._ax = plt.subplots(figsize=self._figsize)
        self._ax.axis('off')

        if reward_name == 'avg_reward':
            self._plot_reward_1()
        elif reward_name == 'eval_reward':
            self._plot_reward_2()

        self._fig.savefig(os.path.join(self._path_root, f'{reward_name}.png'))

    def _plot_reward_1(self):
        _ax = self._fig.add_subplot(1, 1, 1)
        self._plot_reward_avg_group(_ax, [
            dict(reward_name='avg_reward', method_name='rl_mpc_cbf'),
            dict(reward_name='avg_reward', method_name='rl_mpc_cbf_traj'),
            dict(reward_name='avg_reward', method_name='rl_mpc_cbf_ppc2_traj'),
        ])

        _ax.set_xlabel('step $s$'); _ax.set_ylabel('reward $r$')
        _ax.set_title('AVG REWARD')
        _ax.legend()
    
    def _plot_reward_2(self):
        _ax = self._fig.add_subplot(1, 1, 1)
        self._plot_reward_eval_group(_ax, [
            dict(reward_name='eval_reward', method_name='rl_mpc_cbf'),
            dict(reward_name='eval_reward', method_name='rl_mpc_cbf_traj'),
            dict(reward_name='eval_reward', method_name='rl_mpc_cbf_ppc2_traj'),
        ])
        
        _ax.set_xlabel('step $s$'); _ax.set_ylabel('reward $r$')
        _ax.set_title('EVAL REWARD')
        _ax.legend()

    def _plot_reward_avg_group(self, ax: Axes, data_infos: List[Dict]) -> List[ndarray]:
        N: int = float('inf')
        init_index: int = 0
        slice_func = lambda x, N: x[init_index: N]
        datas: List[ndarray] = list()
        # get data
        for info in data_infos:
            data = self._process_reward_ma(self._get_plot_data(info['reward_name'], info['method_name']))
            # data = self._get_plot_data(info['reward_name'], info['method_name'])
            datas.append(data)
            if data.shape[0] < N: N = data.shape[0]
        
        # slice & plot
        x = np.arange(init_index, N)
        for info, data in zip(data_infos, datas):
            scatter_config: Dict = dict(s=2, label=info['method_name'])
            ax.scatter(x, slice_func(data, N), **scatter_config)
        return datas
    
    def _plot_reward_eval_group(self, ax: Axes, data_infos: List[Dict]) -> List[ndarray]:
        N: int = float('inf')
        datas: List[ndarray] = list()
        init_index: int = 0
        slice_func = lambda x, N: x[init_index: N]
        for info in data_infos:
            data = self._get_plot_data(info['reward_name'], info['method_name'])
            datas.append(data)
            if data.shape[0] < N: N = data.shape[0]
        
        x = np.arange(init_index, N)
        width: float = 0.2
        for idx, (info, data) in enumerate(zip(data_infos, datas)):
            data_sliced = slice_func(data, N)
            x_row: ndarray = x + idx * N
            max_idx: int = np.argmax(data_sliced).item()
            ax.bar(x_row, data_sliced, width, label=info['method_name'])
            ax.text(x_row[max_idx], data_sliced[max_idx], str(data_sliced[max_idx]))
        return datas
    
    def _get_plot_data(self, reward_name: str, method_name: str) -> ndarray:
        _reward_abs_path: str = os.path.join(self._path_root, method_name, f'{reward_name}.txt')
        with open(_reward_abs_path, mode='r', encoding='utf-8') as reader:
            data: ndarray = np.array(list(filter(lambda x: x.strip(), reader.read().split('\n'))), dtype=np.float32).flatten()
        return data
    
    def _process_reward_mae(self, data: ndarray) -> ndarray: # use mae
        N: int = data.shape[0]
        alpha: float = 0.05
        data_processed: ndarray = data[0] * np.ones_like(data)
        for i in range(1, N):
            data_processed[i] = alpha * data[i] + (1 - alpha) * data_processed[i-1]
        return data_processed
    
    def _process_reward_ma(self, data: ndarray, W: int = 50) -> ndarray: # use ma
        N: int = data.shape[0]
        data_processd: ndarray = np.zeros(N-W+1)
        for i in range(0, N-W+1):
            data_processd[i] = sum(data[i: i+(W-1)])
        return data_processd
    
    def _set_init_config(self):
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
        })


if __name__ == '__main__':
    ploter: Ploter = Ploter()
    ploter.plot('avg_reward')
    ploter.plot('eval_reward')
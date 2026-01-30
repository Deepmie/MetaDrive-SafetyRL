import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
import os
from typing import Tuple, List, Dict

class Ploter:
    def __init__(self):
        self._path_root: str = 'dp_single_version2/figdata'

        self._set_init_config()
    
    def plot(self, reward_name: str, figsize: Tuple):
        self._fig, self._ax = plt.subplots(figsize=figsize)
        self._ax.axis('off')

        if reward_name == 'avg_reward':
            self._plot_reward_1()
        elif reward_name == 'eval_reward':
            self._plot_reward_2()
        elif reward_name == 'physic_state':
            self._plot_physic_state()

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

    def _plot_physic_state(self):
        method_infos: List[Dict] = [
            dict(method_name='rl_mpc_cbf'),
            dict(method_name='rl_mpc_cbf_traj'),
            dict(method_name='rl_mpc_cbf_ppc2_traj'),
        ]
        _ax1 = self._fig.add_subplot(2, 3, 1)
        _ax2 = self._fig.add_subplot(2, 3, 2)
        _ax3 = self._fig.add_subplot(2, 3, 3)
        _ax3 = self._fig.add_subplot(2, 3, 3)
        _ax4 = self._fig.add_subplot(2, 3, 4)
        _ax5 = self._fig.add_subplot(2, 3, 5)
        for idx, info in enumerate(method_infos):
            method_name: str = info['method_name']
            data_path: str = os.path.join(self._path_root, method_name, 'phsical_state.txt')
            data: ndarray = self._preprocess_physical_data(data_path)
            _ax1.plot(data[:, 0: 1], data[:, 1: 2], label=str(info['method_name']))
            _ax2.plot(data[:, 2: 3], label=str(info['method_name']))
            _ax3.plot(data[:, 3: 4], label=str(info['method_name']))
            _ax4.plot(data[:, 4: 5], label=str(info['method_name']))
            _ax5.plot(data[:, 5: 6], label=str(info['method_name']))
        
        _ax1.set_xlabel('pos $x$'); _ax1.set_ylabel('pos $y$'); _ax1.set_title('POSITION');   _ax1.legend()
        _ax2.set_xlabel('step $s$'); _ax2.set_ylabel('$v$'); _ax2.set_title('VELOCITY');   _ax2.legend()
        _ax3.set_xlabel('step $s$'); _ax3.set_ylabel('$\theta$'); _ax3.set_title('THETA');      _ax3.legend()
        _ax4.set_xlabel('step $s$'); _ax4.set_ylabel('$a$'); _ax4.set_title('ACCELERATE'); _ax4.legend()
        _ax5.set_xlabel('step $s$'); _ax5.set_ylabel('$\delta$'); _ax5.set_title('DELTA');      _ax5.legend()

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
    
    def _preprocess_physical_data(self, data_path: str) -> ndarray:
        with open(data_path, mode='r', encoding='utf-8') as f:
            data: str = np.array([[float(s) for s in d.strip().split(',')] for d in f.read().strip().split('\n') if d.strip()], dtype=np.float32)
        return data
    
    def _set_init_config(self):
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
        })


if __name__ == '__main__':
    ploter: Ploter = Ploter()
    # ploter.plot('avg_reward')
    # ploter.plot('eval_reward')
    ploter.plot('physic_state', figsize=(16, 10))
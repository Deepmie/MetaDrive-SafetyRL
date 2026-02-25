import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
import os
from typing import Tuple, List, Dict
from itertools import accumulate

METHODS_LIST: List[str] = [
    'rl_mpc_cbf_ppc_traj',
    'rl_mpc_cbf_traj',
    'rl_mpc_cbf',
    'rl_mpc',
    'rl'
]

class Ploter:
    def __init__(self):
        self._path_root: str = 'deep/data/'
        self._set_init_config()
    
    def plot(self, reward_name: str, figsize: Tuple):
        self._fig, self._ax = plt.subplots(figsize=figsize)
        self._ax.axis('off')

        if reward_name == 'avg_reward':
            self._plot_avg_rwards()
        elif reward_name == 'eval_reward':
            self._plot_eval_rewards()
        elif reward_name == 'physic_state':
            self._plot_physic_state_copy()
        elif reward_name == 'position':
            self._plot_position()
        elif reward_name == 'success_rate':
            self._plot_success_rate()
        elif reward_name == 'plot_msdv':
            self._plot_msdv()
        
        self._fig.savefig(os.path.join(self._path_root, f'{reward_name}.svg'))

    def _plot_avg_rwards(self):
        plots: List[Dict] = list()
        for method_name in METHODS_LIST:
            plots.append(dict(reward_name='avg_reward', method_name=method_name))

        _ax = self._fig.add_subplot(1, 1, 1)
        self._plot_reward_avg_group(_ax, plots)

        _ax.set_xlabel('step $s$'); _ax.set_ylabel('reward $r$')
        _ax.set_title('AVG REWARD')
        _ax.legend()
    
    def _plot_eval_rewards(self):
        plots: List[Dict] = list()
        for method_name in METHODS_LIST:
            plots.append(dict(reward_name='eval_reward', method_name=method_name))

        _ax = self._fig.add_subplot(1, 1, 1)
        self._plot_reward_eval_group(_ax, plots)
        
        _ax.set_ylim(0, 190)
        _ax.set_xlabel('step $s$'); _ax.set_ylabel('reward $r$')
        _ax.set_title('EVAL REWARD')
        _ax.legend()
    
    def _plot_success_rate(self):
        plots: List[Dict] = list()
        for method_name in METHODS_LIST:
            plots.append(dict(reward_name='avg_reward', method_name=method_name))

        _ax = self._fig.add_subplot(1, 1, 1)
        self._plot_success_rate_group(_ax, plots)
        
        _ax.set_xlabel('step $s$'); _ax.set_ylabel('rate $r$')
        _ax.set_title('SUCCESS RATE')
        _ax.legend()

    def _plot_msdv(self):
        plots: List[Dict] = list()
        for method_name in METHODS_LIST:
            plots.append(dict(method_name=method_name))
        msdv_func = lambda x: np.sqrt(np.mean(x ** 2))

        _ax1 = self._fig.add_subplot(1, 2, 1)
        _ax2 = self._fig.add_subplot(1, 2, 2)
        msdv_axs = []; msdv_ays = []
        for idx, info in enumerate(plots):
            method_name: str = info['method_name']
            data_path: str = os.path.join(self._path_root, method_name, 'phsical_state.txt')
            data: ndarray = self._preprocess_physical_data(data_path)
            ax = data[:, 4:5]
            msdv_axs.append(msdv_func(ax))
            velocity = data[:, 2:3].flatten()
            delta = data[:, 5:6].flatten()
            ay = np.diff(delta) * velocity[0:-1]
            msdv_ays.append(msdv_func(ay))
        _ax1.bar(list(range(len(plots))), msdv_axs)
        _ax2.bar(list(range(len(plots))), msdv_ays)
        print(msdv_axs)
        print(msdv_ays)
        _ax1.set_title('$MSDV_V$'); _ax2.set_title('$MSDV_L$')
        _ax1.set_ylabel('value'); _ax2.set_ylabel('value')

    def _plot_position(self):
        plots: List[Dict] = list()
        for method_name in METHODS_LIST:
            plots.append(dict(method_name=method_name))
        
        _ax = self._fig.add_subplot(1, 1, 1)
        for idx, info in enumerate(plots):
            method_name: str = info['method_name']
            data_path: str = os.path.join(self._path_root, method_name, 'phsical_state.txt')
            data: ndarray = self._preprocess_physical_data(data_path)
            _ax.plot(data[:, 0: 1], data[:, 1: 2], label=method_name)
        _ax.legend(); _ax.set_xlabel('$x$'); _ax.set_ylabel('$y$')


    def _plot_physic_state_copy(self):
        plt.subplots_adjust(wspace=0.0, hspace=0)
        property_list: Dict = {
            'VELOCITY': '.',
            'THETA': '+',
            'ACCELERATE': '^',
            'DELTA': 's'
        }
        plots: List[Dict] = list()
        for method_name in METHODS_LIST:
            plots.append(dict(method_name=method_name))
        
        axs: List[List[Axes]] = [[None for _ in range(5)] for _ in range(len(plots))]
        init_idx: int   = 1
        delta_d: int    = 20
        for i in range(len(plots)):
            for j in range(4):
                axs[i][j] = self._fig.add_subplot(len(plots), 4, init_idx); init_idx += 1
        
        for idx, info in enumerate(plots):
            method_name: str = info['method_name']
            data_path: str = os.path.join(self._path_root, method_name, 'phsical_state.txt')
            data: ndarray = self._preprocess_physical_data(data_path)
            # print(data.shape)
            for i, (key, value) in enumerate(property_list.items()):
                axs[idx][i].plot(data[:, 2+i: 3+i])
                axs[idx][i].tick_params(axis='y', direction='in', pad=-15)
                if idx == 0: axs[idx][i].set_title(key)
                axs[idx][i].set_xlim(0-delta_d, 350+delta_d)
                if idx != len(plots)-1: axs[idx][i].set_xticks([])
                if i == 0:
                    axs[idx][i].set_ylim(-1, 13)
                elif i == 1:
                    axs[idx][i].set_ylim(-1.5, 1.5)
                elif i == 2:
                    axs[idx][i].set_ylim(-1.5, 1.5)
                elif i == 3:
                    axs[idx][i].set_ylim(-1.2, 1.2)
                    axs[idx][i].set_yticks([-1, 0, 1])
                    axs[idx][i].set_yticks([-1, 0, 1])
                # if i != 0: axs[idx][i].set_yticks([])
                if i == 0: axs[idx][i].set_ylabel(method_name)
                

    def _plot_physic_state(self):
        plots: List[Dict] = list()
        for method_name in METHODS_LIST:
            plots.append(dict(method_name=method_name))

        _ax1 = self._fig.add_subplot(2, 2, 1)
        _ax2 = self._fig.add_subplot(2, 2, 2)
        _ax3 = self._fig.add_subplot(2, 2, 3)
        _ax4 = self._fig.add_subplot(2, 2, 4)
        # _ax5 = self._fig.add_subplot(2, 3, 5)
        for idx, info in enumerate(plots):
            method_name: str = info['method_name']
            data_path: str = os.path.join(self._path_root, method_name, 'phsical_state.txt')
            data: ndarray = self._preprocess_physical_data(data_path)
            # _ax1.plot(data[:, 0: 1], data[:, 1: 2], label=str(info['method_name']))
            _ax1.plot(data[:, 2: 3], label=str(info['method_name']))
            _ax2.plot(data[:, 3: 4], label=str(info['method_name']))
            _ax3.plot(data[:, 4: 5], label=str(info['method_name']))
            _ax4.plot(data[:, 5: 6], label=str(info['method_name']))
        
        # _ax1.set_xlabel('pos $x$'); _ax1.set_ylabel('pos $y$'); _ax1.set_title('POSITION');   _ax1.legend()
        _ax1.set_xlabel('step $s$'); _ax1.set_ylabel('$v$')     ; _ax1.set_title('VELOCITY')  ; _ax1.legend()
        _ax2.set_xlabel('step $s$'); _ax2.set_ylabel('$\theta$'); _ax2.set_title('THETA')     ; _ax2.legend()
        _ax3.set_xlabel('step $s$'); _ax3.set_ylabel('$a$')     ; _ax3.set_title('ACCELERATE'); _ax3.legend()
        _ax4.set_xlabel('step $s$'); _ax4.set_ylabel('$\delta$'); _ax4.set_title('DELTA')     ; _ax4.legend()

    def _plot_reward_avg_group(self, ax: Axes, data_infos: List[Dict]) -> List[ndarray]:
        N: int = float('inf')
        init_index: int = 4000
        slice_func = lambda x, N: x[init_index: N]
        datas: List[ndarray] = list()
        # get data
        for info in data_infos:
            data = self._process_reward_ma(self._get_plot_data(info['reward_name'], info['method_name']))
            # data = self._get_plot_data(info['reward_name'], info['method_name'])
            datas.append(data)
            if data.shape[0] < N: N = data.shape[0]
        
        # slice & 
        N: int = min(6250, N)
        x = np.arange(init_index, N)
        for info, data in zip(data_infos, datas):
            scatter_config: Dict = dict(markersize=2, label=info['method_name'])
            data_slice = slice_func(data, N)
            max_idx: int = np.argmax(data_slice).item()
            max_x = x[max_idx]; max_data = data_slice[max_idx]
            ax.plot(x, data_slice, **scatter_config)
            # ax.text(max_x, max_data, s=f'{max_data:.2f}')
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
    
    def _plot_success_rate_group(self, ax: Axes, data_infos: List[Dict]):
        N: int = float('inf')
        datas: List[ndarray] = list()
        init_index: int = 0
        slice_func = lambda x, N: x[init_index: N]
        for info in data_infos:
            data = self._get_plot_data(info['reward_name'], info['method_name'])
            datas.append(data)
            if data.shape[0] < N: N = data.shape[0]
        
        window_size: int = 80
        for idx, (info, data) in enumerate(zip(data_infos, datas)):
            data_sliced: ndarray = slice_func(data, N)
            suc: List[int] = (data_sliced > 150).astype(np.long).tolist()
            suc_rate: List[int] = list()
            for i in range(0, N, window_size):
                sr = sum(suc[i: i+window_size]) / window_size
                suc_rate.append(sr)
            marker_config: Dict = dict(marker='o', markerfacecolor='w', markersize=4)
            ax.plot(suc_rate, **marker_config, label=info['method_name'])
            print(max(suc_rate))
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
    
    def _process_reward_ma(self, data: ndarray, W: int = 250) -> ndarray: # use ma
        N: int = data.shape[0]
        data_processd: ndarray = np.zeros(N-W+1)
        for i in range(0, N-W+1):
            data_processd[i] = sum(data[i: i+(W-1)]) / W
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
    ploter.plot('avg_reward'  , figsize=(6, 4))
    # ploter.plot('eval_reward' , figsize=(10, 4))
    # ploter.plot('success_rate', figsize=(10, 4))
    # ploter.plot('physic_state', figsize=(25, 10))
    # ploter.plot('plot_msdv', figsize=(10, 4))
    # ploter.plot('position', figsize=(10, 4))
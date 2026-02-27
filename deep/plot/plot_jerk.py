import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
import os
from typing import Tuple, List, Dict
from itertools import accumulate
from common import set_init_config

PATH_ROOT = 'deep/data/'
INFOS: List[Dict] = [
    dict(method_name='rl_mpc_cbf_ppc_traj', style=dict(linewidth=1.5, zorder=99, color='#0868ac')),
    dict(method_name='rl_mpc_cbf_traj'    , style=dict(linewidth=1  , zorder=98, color='#43a2ca')),
    dict(method_name='rl_mpc_cbf'         , style=dict(linewidth=1  , zorder=97, color='#7bccc4')),
    dict(method_name='rl_mpc'             , style=dict(linewidth=1  , zorder=96, color='#bae4bc')),
    dict(method_name='rl'                 , style=dict(linewidth=1  , zorder=95, color="#ecfedd")),
]

def get_plot_data(file_name: str, method_name: str) -> ndarray:
    _reward_abs_path: str = os.path.join(PATH_ROOT, method_name, f'{file_name}.txt')
    with open(_reward_abs_path, mode='r', encoding='utf-8') as reader:
        data: ndarray = np.array(list(filter(lambda x: x.strip(), reader.read().split('\n'))), dtype=np.float32).flatten()
    return data

def preprocess_physical_data(data_path: str) -> ndarray:
    with open(data_path, mode='r', encoding='utf-8') as f:
        data: str = np.array([[float(s) for s in d.strip().split(',')] for d in f.read().strip().split('\n') if d.strip()], dtype=np.float32)
    return data

def jerk_func(x: ndarray, k: int=1) -> float:
    x = x.flatten()
    N: int = int(x.shape[0])
    diff_list: List[float] = []
    for i in range(0, N, k):
        v = ((x[min(i+k, N-1)] - x[i]) > 0.5) ** 2
        diff_list.append(v.item())
    return np.sqrt(sum(diff_list)).item()

def main():
    set_init_config()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    N = len(INFOS)
    jerk_accx = []; jerk_accy = []
    for idx, info in enumerate(INFOS):
        method_name: str = info.get('method_name')
        data_path: str = os.path.join(PATH_ROOT, method_name, 'phsical_state.txt')
        data: ndarray = preprocess_physical_data(data_path)
        accx = data[:, 4:5]
        jerk_accx.append(jerk_func(accx))
        velocity = data[:, 2:3].flatten()
        delta = data[:, 5:6].flatten()
        accy = np.diff(delta) * velocity[0:-1]
        jerk_accy.append(jerk_func(accy))
        # print(f'accx: {accx:.3f}, accy: {accy:.3f}')
    ax1.bar(list(range(N)), jerk_accx)
    ax2.bar(list(range(N)), jerk_accy)
    
    for idx in range(N):
        ax1.text(idx, jerk_accx[idx], f'{jerk_accx[idx]:.3f}')
        ax2.text(idx, jerk_accy[idx], f'{jerk_accy[idx]:.3f}')
    
    ax1.set_title('$JERK_V$'); ax2.set_title('$JERK_L$')
    ax1.set_ylabel('value'); ax2.set_ylabel('value')
    fig.savefig(os.path.join(PATH_ROOT, 'plot_jerk.svg'))


if __name__ == '__main__':
    main()
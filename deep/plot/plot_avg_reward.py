import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
import os
from typing import Tuple, List, Dict
from itertools import accumulate
from common import set_init_config

PATH_ROOT = 'deep/data/'
INFOS = [
    dict(method_name='rl_mpc_cbf_ppc_traj', style=dict(linewidth=1.5, zorder=99, )), #color='#0868ac')),
    # dict(method_name='rl_mpc_cbf_traj'    , style=dict(linewidth=1  , zorder=98, )), #color='#43a2ca')),
    dict(method_name='rl_mpc_cbf_ppc'     , style=dict(linewidth=1  , zorder=97, )), #color="#ca6343")),
    dict(method_name='rl_mpc_cbf'         , style=dict(linewidth=1  , zorder=96, )), #color='#7bccc4')),
    # dict(method_name='rl_mpc'             , style=dict(linewidth=1  , zorder=95, )), #color='#bae4bc')),
    # dict(method_name='rl'                 , style=dict(linewidth=1  , zorder=94, )), #color="#ecfedd")),
]

def get_plot_data(file_name: str, method_name: str) -> ndarray:
    _reward_abs_path: str = os.path.join(PATH_ROOT, method_name, f'{file_name}.txt')
    with open(_reward_abs_path, mode='r', encoding='utf-8') as reader:
        data: ndarray = np.array(list(filter(lambda x: x.strip(), reader.read().split('\n'))), dtype=np.float32).flatten()
    return data

def process_reward_ma(data: ndarray, W: int = 250) -> ndarray: # use ma
    N: int = data.shape[0]
    data_processd: ndarray = np.zeros(N-W+1)
    for i in range(0, N-W+1):
        data_processd[i] = sum(data[i: i+(W-1)]) / W
    return data_processd

def main():
    set_init_config()
    fig, ax = plt.subplots(figsize=(10, 4))

    N: int = float('inf')
    init_index: int = 0
    slice_func = lambda x, N: x[init_index: N]
    datas: List[ndarray] = list()
    # get data
    for info in INFOS:
        data = process_reward_ma(get_plot_data('avg_reward', info.get('method_name')))
        datas.append(data)
        if data.shape[0] < N: N = data.shape[0]
    
    # slice & 
    N: int = min(6250, N)
    x = np.arange(init_index, N)
    for info, data in zip(INFOS, datas):
        style: Dict = info.get('style')
        style['label'] = info.get('method_name')
        data_slice = slice_func(data, N)
        max_idx: int = np.argmax(data_slice).item()
        max_x = x[max_idx]; max_data = data_slice[max_idx]
        ax.plot(x, data_slice, **style)
        ax.text(max_x, max_data, s=f'{max_data:.2f}')

    ax.set_xlabel('step $s$'); ax.set_ylabel('reward $r$')
    ax.set_title('AVG REWARD')
    ax.legend()

    fig.savefig(os.path.join(PATH_ROOT, 'avg_reward_global.svg'))


if __name__ == '__main__':
    main()
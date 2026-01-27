import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import os

def plot_init_config():
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
    })

def plot_rewards(reward_name: str, method_name:str = 'rl_mpc_cbf_traj', path_root: str = 'dp_single_version2/figdata'):
    reward_real_path: str = os.path.join(path_root, method_name, f'{reward_name}.txt')

    with open(reward_real_path, mode='r', encoding='utf-8') as reader:
        data: ndarray = np.array(list(filter(lambda x: x.strip(), reader.read().split('\n'))), dtype=np.float32).flatten()

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis('off')
    
    ax1 = fig.add_subplot(1, 1, 1)
    x: ndarray = np.arange(0, data.shape[0])
    ax1.scatter(x, data, s=2)
    ax1.set_xlabel('step'); ax1.set_ylabel('reward')
    ax1.set_title('method: rl_mpc_cbf_traj')
    
    fig.savefig(os.path.join(path_root, method_name, f'{reward_name}.png'))


if __name__ == '__main__':
    # plot_init_config()
    plot_rewards('avg_reward')
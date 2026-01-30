import sys
sys.path.append('/workspace/metadrive-github/')
from metadrive.custom2_version2.base_config import PPOConfig
from metadrive.custom2_version2.ppo import PPO
from metadrive.custom2_version2.utils import set_random_seed
from typing import List, Tuple, Dict
from metadrive.custom2_version2.type import VehicleState
import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np
import os

def main():
    set_random_seed(0)
    ppo_config: PPOConfig  = PPOConfig()
    ppo: PPO               = PPO(ppo_config, eval_mode=True)
    root_path: str         = 'dp_single_version2/figdata'
    method_infos: List[Dict] = [
        {'method_name': 'rl_mpc_cbf'},
        {'method_name': 'rl_mpc_cbf_traj'},
        {'method_name': 'rl_mpc_cbf_ppc2_traj'},
    ]

    fig, ax = plt.subplots(figsize=(6, 18))
    ax.axis('off')
    for i, m in enumerate(method_infos):
        ax_sub = fig.add_subplot(3, 1, i+1)
        method_name: str = m['method_name']
        rewards, extract_infos, metadata = ppo.eval_with_checkpoint(os.path.join(root_path, method_name, 'policy_best.pth'))
        print(f'The reward of `{method_name}` is: {rewards}')
        x: ndarray = np.zeros(len(extract_infos))
        y: ndarray = np.zeros(len(extract_infos))
        for j, info in enumerate(extract_infos):
            state: VehicleState = info['vehicle_state']
            x[j] = state.x; y[j] = state.y
        ax_sub.plot(x, y)
    
    fig.savefig('dp_single_version2/figdata/physic.png')
    ppo.close()

if __name__ == '__main__':
    main()
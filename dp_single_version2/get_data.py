import sys
sys.path.append('/workspace/metadrive-github/')
from metadrive.custom2_version2.base_config import PPOConfig
from metadrive.custom2_version2.ppo import PPO
from metadrive.custom2_version2.utils import set_random_seed
from metadrive.custom2_version2.type import VehicleState
from numpy import ndarray
import numpy as np
import os

if __name__ == '__main__':
    set_random_seed(0)
    ppo_config: PPOConfig = PPOConfig()
    ppo: PPO              = PPO(ppo_config, eval_mode=True)
    root_path: str        = 'dp_single_version2/figdata/'
    method_name: str      = 'rl_mpc_cbf_ppc2_traj'
    method_path: str      = os.path.join(root_path, method_name)

    reward_eval, extract_infos, metadata = ppo.eval_with_checkpoint(os.path.join(method_path, 'policy_best.pth'))
    file_writted = open(os.path.join(method_path, 'phsical_state.txt'), mode='w', encoding='utf-8')
    N: int = len(extract_infos)
    for info in extract_infos:
        state: VehicleState = info['vehicle_state']
        file_writted.write(f'{state.x},{state.y},{state.v},{state.theta},{state.a},{state.theta}\n')
        file_writted.flush()
    file_writted.close()
import sys
sys.path.append('/workspace/metadrive-github/')
from metadrive.custom2_version2.base_config import PPOConfig
from metadrive.custom2_version2.create_env import create_env
from metadrive.custom2_version2.ppo import PPO
from metadrive.custom2_version2.controller import EvalController
from datetime import datetime
from IPython.display import clear_output
import numpy as np

if __name__ == '__main__':
    ppo_config = PPOConfig()
    ppo = PPO(ppo_config)
    is_suc, info = ppo.start()
    
    if not is_suc:
        print(info)
    else:
        print('Training have finished! Start to final evaluate...')
        ppo.final_eval()
        ppo.close()

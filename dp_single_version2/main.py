import sys
sys.path.append('/workspace/metadrive-github/')
from metadrive.custom2_version2.base_config import PPOConfig
from metadrive.custom2_version2.ppo import PPO
from metadrive.custom2_version2.utils import set_random_seed

if __name__ == '__main__':
    set_random_seed(0)
    ppo_config = PPOConfig()
    ppo = PPO(ppo_config)
    is_suc, info = ppo.start()
    
    if not is_suc:
        print(info)
    else:
        print('Training have finished! Start to final evaluate...')
        ppo.final_eval()
        ppo.close()

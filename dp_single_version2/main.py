import sys
sys.path.append('/workspace/metadrive-github/')
from metadrive.custom2_version2.base_config import PPOConfig
from metadrive.custom2_version2.ppo import PPO
from metadrive.custom2_version2.utils import set_random_seed, get_logger_path
import os

if __name__ == '__main__':
    set_random_seed(0)
    ppo_config = PPOConfig()
    ppo = PPO(ppo_config)
    logger_path: str = get_logger_path()
    is_suc, info = ppo.start()
    
    if not is_suc:
        print(info)
    else:
        print('Training have finished! Start to final evaluate...')
        ppo.final_eval(os.path.join(logger_path, 'eval_final.gif'))
        ppo.load_weight_from_checkpoint(os.path.join(logger_path, ppo_config.best_policy_checkpoint_pth))
        ppo.final_eval(os.path.join(logger_path, 'eval_best.gif'))
        ppo.close()
        
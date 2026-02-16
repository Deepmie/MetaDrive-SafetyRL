import sys
from methods.common.base_config import PPOConfig
from methods.common.ppo import PPO
from methods.common.utils import set_random_seed
from methods.common.utils.utils import get_logger_path
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate process')
    parser.add_argument('--type', type=str, help='evaluate type')

    args = parser.parse_args()

    set_random_seed(0)
    ppo_config: PPOConfig = PPOConfig()
    ppo: PPO              = PPO(ppo_config, eval_mode=True)
    logger_path: str      = get_logger_path()
    
    if args.type == 'newest':
        ckpt_path = os.path.join(logger_path, ppo_config.policy_checkpoint_pth)
    elif args.type == 'best':
        ckpt_path = os.path.join(logger_path, ppo_config.best_policy_checkpoint_pth)
    else:
        raise TypeError(f'Arg `type` must in [newest, best], but you give {args.type} not in list.')
    
    metadata = ppo.load_weight_from_checkpoint(ckpt_path)
    print(f'metadata: \n{metadata}')
    ppo.final_eval(os.path.join(logger_path, 'evaluate.gif'))
    ppo.close()
    
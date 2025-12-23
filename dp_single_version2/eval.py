import sys
sys.path.append('/workspace/metadrive-github/')
from metadrive.custom2_version2.base_config import PPOConfig
from metadrive.custom2_version2.ppo import PPO
from metadrive.custom2_version2.utils import set_random_seed
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate process')
    parser.add_argument('--type', type=str, help='evaluate type')

    args = parser.parse_args()

    set_random_seed(0)
    ppo_config = PPOConfig()
    ppo = PPO(ppo_config, eval_mode=True)
    
    if args.type == 'newest':
        ckpt_path = ppo_config.policy_checkpoint_pth
    elif args.type == 'best':
        ckpt_path = ppo_config.best_policy_checkpoint_pth
    else:
        raise TypeError(f'Arg `type` must in [newest, best], but you give {args.type} not in list.')
    
    metadata = ppo.load_weight_from_checkpoint(ckpt_path)
    print(f'metadata: \n{metadata}')
    ppo.final_eval()
    ppo.close()

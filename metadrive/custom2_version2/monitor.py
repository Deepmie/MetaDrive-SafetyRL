from typing import List, Tuple, Dict, cast, Optional, Union, Callable
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
import os
from metadrive.custom2_version2.collector import Collector

class Monitor:
    path_root: str = 'dp_single_version2/figdata'
    def __init__(self, method_name: str):
        self.method_name = method_name
        self._init_dir(); self._create_collector()
        self.accum_rewards: ndarray = np.zeros([4, ], dtype=np.float32)
        self.steps: ndarray = np.zeros([4, ], dtype=np.float32)
    
    def collect_rewards(self, rewards: ndarray, dones: ndarray):
        self.accum_rewards += rewards
        self.steps += 1
        
        for idx, done in enumerate(dones.tolist()):
            if done:
                avg_reward: float = self.accum_rewards[idx].item()
                self.avg_reward_collector.collect_data(avg_reward)
                # 重置
                self.accum_rewards[idx] = 0
                self.steps[idx] = 0
    
    def collect_evaluate_reward(self, reward: float):
        self.eval_reward_collector.collect_data(reward)

    def close(self):
        self.avg_reward_collector.close()
        self.eval_reward_collector.close()

    def _init_dir(self):
        self.dir_path: str = os.path.join(self.path_root, self.method_name)
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

    def _create_collector(self):
        self.avg_reward_collector = Collector(name='avg_reward', freq=10, path_root=self.dir_path)
        self.eval_reward_collector = Collector(name='eval_reward', freq=10, path_root=self.dir_path)
    


if __name__ == '__main__':
    m = Monitor(method_name='xxxx')
    m.plot_rewards()
from torch import Tensor
from numpy import ndarray
from typing import Tuple, Union
import torch.nn as nn
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.distributions import Distribution
from metadrive.customs_parallel.utils import converto_ndarray, converto_torch
from metadrive.customs_parallel.config import PolicyConfig
from metadrive.customs_parallel.net.version2 import FeatureExtractor, ActorNet, CriticNet

class Policy(nn.Module):
    def __init__(self, config: PolicyConfig):
        super(Policy, self).__init__()
        self.config = config
        # 初始化三个神经网络
        self.feature_extractor = FeatureExtractor(config.state_dim, config.hidden_dim)
        self.actor = ActorNet(config.hidden_dim, config.action_dim, config.std_config)
        self.critic = CriticNet(config.hidden_dim)

    
    def forward(self, x: Tensor) -> Tuple[Distribution, Tensor]: # (batch_size, state_dim) -> [dist, (batch_size, 1)]
        feature = self.feature_extractor(x)  # (batch_size, hidden_dim)
        actor_output = self.actor(feature)   # dist
        critic_output = self.critic(feature) # (batch_size, 1)
        return actor_output, critic_output


    def select_action(self, obs: Union[ndarray, Tensor]) -> Tuple[ndarray, float, float]:
        obs: Tensor = converto_torch(obs)

        # 判断是否需要扩维度
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(dim = 0)

        with torch.no_grad():
            dist, value = self.forward(obs)
            action: Tensor = dist.sample()           # 采样得到动作, (batch_size, action_dim)
            log_prob: Tensor = dist.log_prob(action) # 对数概率向量, (batch_size, )
        
        action = self._map_action(action.flatten())
        action: ndarray = converto_ndarray(action)   # 动作转换成ndarray
        return action, log_prob.item(), value.item()
    
    
    def evaluate_action(self, obs: Union[ndarray, Tensor], action: Union[ndarray, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        obs: Tensor = converto_torch(obs)
        action: Tensor = converto_torch(action)
        dist, value = self.forward(obs)
        return dist.log_prob(action), dist.entropy(), value

    
    def act_mean(self, obs: Union[ndarray, Tensor]) -> Tensor:
        with torch.no_grad():
            obs: Tensor = converto_torch(obs)
            dist, _ = self.forward(obs)
            return dist.mean
    
    
    def _map_action(self, action: Tensor) -> Tensor:
        # 给定v_ref的取值范围    : [v_min, v_max]
        # 给定theta_ref的取值范围: [theta_min, theta_max]
        action_map = torch.empty_like(action)
        action_map[0] = self._map_func(action[0], self.config.v_min, self.config.v_max)
        action_map[1] = self._map_func(action[1], self.config.theta_min, self.config.theta_max)
        return action_map

    def _map_func(self, x: Tensor, a: float, b: float) -> Tensor:
        x = x.tanh()                       # [-inf, +inf] -> [-1, 1]
        x = (b + a) / 2 + (b - a) / 2 * x  # [-1, 1]      -> [ a, b]
        return x


class PolicyManager:
    def __init__(self, config: PolicyConfig):
        self.config = config
        self.cpu = Policy(self.config)
        self.gpu = None
        self.cpu.share_memory()

    def init_ddp(self, rank: int):
        self.gpu = Policy(self.config).to(rank)
        self.gpu = DDP(self.gpu, device_ids=[rank, ])
        self.sync()

    def sync(self): # gpu更新完毕后同步到cpu上
        if self.gpu is not None:
            with torch.no_grad():
                self.cpu.load_state_dict(self.gpu.module.state_dict())

    def close(self):
        dist.destroy_process_group()
        torch.cuda.empty_cache()
    
    def optimizer(self, opt_func):
        if self.gpu is None:
            return None
        return \
        opt_func([
            {'params': self.gpu.module.feature_extractor.parameters(), },
            {'params': self.gpu.module.actor.parameters() , 'lr': self.config.learning_rate_actor},
            {'params': self.gpu.module.critic.parameters(), 'lr': self.config.learning_rate_critic},
        ], lr=self.config.learning_rate)

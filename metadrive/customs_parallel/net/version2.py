import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal
from torch.nn.functional import softplus
from typing import Tuple

class FeatureExtractor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
    
    def forward(self, x: Tensor) -> Tensor: # (batch_size, state_dim) -> (batch_size, hidden_dim)
        x = self.net(x)
        return x


class ActorNet(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, std_config: Tuple):
        super(ActorNet, self).__init__()
        self.mu_head = nn.Linear(in_features=hidden_dim, out_features=action_dim, dtype = torch.float32)
        self.log_std_head = nn.Linear(in_features=hidden_dim, out_features=action_dim)
        self.std_config = std_config
        self._init_weight()

    def forward(self, x: Tensor) -> Distribution: # (batch_size, hidden_dim) -> N(mu, )
        mu: Tensor = self.mu_head(x)
        log_std: Tensor = self.log_std_head(x)

        if self.std_config[0]:
            log_std: Tensor = torch.clamp(log_std, min=self.std_config[1], max=self.std_config[2])
        
        std = log_std.exp()
        # sigma = torch.diag_embed(std)        # 协方差矩阵sigma
        # dist = MultivariateNormal(mu, sigma) # 构造分布N(mu, sigma)
        dist = torch.distributions.Normal(mu, std)
        dist = torch.distributions.Independent(dist, 1)
        return dist
    
    def _init_weight(self):
        nn.init.constant_(self.log_std_head.bias, -1.0)
    

class CriticNet(nn.Module):
    def __init__(self, hidden_dim: int):
        super(CriticNet, self).__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: Tensor) -> Tensor: # (batch_size, hidden_dim) -> (batch_size, 1)
        x = self.value_head(x)
        return x

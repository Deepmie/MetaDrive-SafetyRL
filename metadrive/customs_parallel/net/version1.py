import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal
from torch.nn.functional import softplus

class FeatureExtractor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(in_features=state_dim, out_features=hidden_dim, dtype=torch.float32)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, dtype=torch.float32)
        self.ac = nn.Tanh()
    
    def forward(self, x: Tensor) -> Tensor: # (batch_size, state_dim) -> (batch_size, hidden_dim)
        x = self.ac(self.fc1(x)) # (batch_size, hidden_dim)
        x = self.ac(self.fc2(x)) # (batch_size, hidden_dim)
        return x


class ActorNet(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, action_std_init: float):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(in_features = hidden_dim, out_features = action_dim, dtype = torch.float32)
        self.var = nn.Parameter(action_std_init ** 2 * torch.ones(action_dim))
    
    def forward(self, x: Tensor) -> Distribution: # (batch_size, hidden_dim) -> N(mu, )
        mu = self.fc1(x) # 均值mu
        sigma = torch.diag(softplus(self.var)) # 协方差sigma
        dist = MultivariateNormal(mu, sigma) # 构造分布N(mu, sigma)
        return dist


class CriticNet(nn.Module):
    def __init__(self, hidden_dim: int):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(in_features = hidden_dim, out_features = 1, dtype = torch.float32)
    
    def forward(self, x: Tensor) -> Tensor: # (batch_size, hidden_dim) -> (batch_size, 1)
        x = self.fc1(x)
        return x

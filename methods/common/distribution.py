from torch.distributions import Categorical, Normal
import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod


def sum_independent_dims(t: Tensor):
    if len(t.shape) > 1:
        return t.sum(dim=1)
    return t.sum()

class Distribution(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        ...

    @abstractmethod
    def proba_distribution(self, action_logits: Tensor) -> nn.Module:
        ...

    @abstractmethod
    def log_prob(self, action: Tensor) -> Tensor:
        ...

    def get_action(self, deterministic: bool = False) -> Tensor:
        if deterministic:
            return self.mode()
        return self.sample()

    @abstractmethod
    def sample(self) -> Tensor:
        ...
    
    @abstractmethod
    def mode(self) -> Tensor:
        ...

    @abstractmethod
    def entropy(self) -> Tensor:
        ...

# 离散
class CategoricalDistribution(Distribution):
    def __init__(self, action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, self.action_dim)
    
    def proba_distribution(self, action_logits: Tensor) -> nn.Module:
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, action: Tensor) -> Tensor:
        return self.distribution.log_prob(action)

    def sample(self) -> Tensor:
        return self.distribution.sample()
    
    def mode(self) -> Tensor:
        return torch.argmax(self.distribution.probs, dim=1)
    
    def entropy(self) -> Tensor:
        return self.distribution.entropy()


# 多维离散
class MultiCategoricalDistribution(Distribution):
    def __init__(self, action_dims: List[int]):
        super(MultiCategoricalDistribution, self).__init__()
        self.action_dims = action_dims
    
    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, sum(self.action_dims))
    
    def proba_distribution(self, action_logits: Tensor) -> nn.Module:
        self.distribution = [Categorical(logits=split) for split in torch.split(action_logits, list(self.action_dims), dim=1)]
        return self
    
    def log_prob(self, action: Tensor) -> Tensor:
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, torch.unbind(action, dim=1))]
        ).sum(dim=1)
    
    def sample(self) -> Tensor:
        return torch.stack([dist.sample() for dist in self.distribution], dim=1)
    
    def mode(self) -> Tensor:
        return torch.stack([torch.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)
    
    def entropy(self) -> Tensor:
        return torch.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)


# 连续
class DiagGaussianDistribution(Distribution):
    def __init__(self, action_dim: int):
        super(DiagGaussianDistribution).__init__()
        self.action_dim = action_dim
        self.mean_action = None
        self.log_std = None
    
    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> nn.Module:
        action_net = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True)
        return action_net, log_std
    
    def proba_distribution(self, mean_action: Tensor, log_std: Tensor) -> nn.Module:
        action_std = torch.ones_like(mean_action) * log_std.exp()
        self.distribution = Normal(mean_action, action_std)
        return self
    
    def log_prob(self, action: Tensor) -> Tensor:
        log_prob = self.distribution.log_prob(action)
        return self._sum(log_prob)
    
    def sample(self) -> Tensor:
        return self.distribution.rsample()
    
    def mode(self) -> Tensor:
        return self.distribution.mean
    
    def entropy(self) -> Tensor:
        return self._sum(self.distribution.entropy())

    def _sum(self, t: Tensor) -> Tensor:
        if len(t.shape) > 1:
            return t.sum(dim=1)
        return t

# Use sde
class StateDependentNoiseDistribution(Distribution):
    def __init__(
            self,
            action_dim: int,
            learn_features: bool = False,
            epsilon: float = 1e-6,
        ):
        super(StateDependentNoiseDistribution, self).__init__()
        self.action_dim = action_dim
        self.learn_features = learn_features
        self.epsilon = epsilon
    
    def proba_distribution_net(self, latent_dim: int, log_std_init: float = -2.0, latent_sde_dim: Optional[int] = None) -> Tuple[nn.Module, nn.Parameter]:
        mean_action_net = nn.Linear(latent_dim, self.action_dim)
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        log_std = torch.ones(self.latent_sde_dim, self.action_dim)
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        self.sample_weight(log_std)
        return mean_action_net, log_std
    
    def proba_distribution(self, mean_action: Tensor, log_std: Tensor, latent_sde: Tensor) -> nn.Module:
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        variance = torch.mm(self._latent_sde ** 2, self.get_std(log_std) ** 2)
        self.distribution = Normal(mean_action, torch.sqrt(variance + self.epsilon))
        return self
    
    def log_prob(self, action: Tensor) -> Tensor:
        return self._sum(self.distribution.log_prob(action))
    
    def sample(self) -> Tensor:
        noise = self.get_noise(self._latent_sde)
        action = self.distribution.mean + noise
        return action
    
    def mode(self) -> Tensor:
        action = self.distribution.mean
        return action
    
    def entropy(self) -> Tensor:
        return self._sum(self.distribution.entropy())
    
    def get_noise(self, latent_sde: Tensor) -> Tensor:
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return torch.mm(latent_sde, self.exploration_mat)
        latent_sde = latent_sde.unsqueeze(dim=1)
        noise = torch.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(dim=1)
    
    def sample_weight(self, log_std: Tensor, batch_size: int = 1) -> None:
        std = self.get_std(log_std)
        self.weight_dist = Normal(torch.zeros_like(std), std)
        self.exploration_mat = self.weight_dist.rsample()
        self.exploration_matrices = self.weight_dist.rsample((batch_size, ))
    
    def get_std(self, log_std: Tensor) -> Tensor:
        std = torch.exp(log_std)
        return std
    
    def _sum(self, t: Tensor) -> Tensor: # (batch_size, state_dim) -> (batch_size, )
        return t.sum(dim=1) if len(t.shape) > 1 else t
    

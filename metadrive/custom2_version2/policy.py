from torch import Tensor
from numpy import ndarray
import numpy as np
from typing import Tuple, Union, Dict, cast, Optional
import torch.nn as nn
from torch.nn.functional import softplus
import torch
from functools import partial
from metadrive.custom2_version2.distribution import \
Distribution, MultiCategoricalDistribution, CategoricalDistribution, DiagGaussianDistribution
from metadrive.custom2_version2.base_config import PolicyConfig
from metadrive.custom2_version2.utils import converto_ndarray, converto_torch
from metadrive.custom2_version2.type import ActionType

class Policy(nn.Module):
    def __init__(self, config: PolicyConfig):
        super(Policy, self).__init__()
        self.config = config
        self.mlp_policy = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            config.activate_func(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            config.activate_func(),
        )
        
        self.mlp_value = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            config.activate_func(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            config.activate_func(),
        )
        
        # 构造action分布和action网络
        self._make_proba_distribution()

        self.value_net = nn.Linear(config.hidden_dim, 1)
        # self._load_state_dict_from_baseline() # 加载权重.pth
        self._load_state_dict_from_checkpoint() if self.config.is_load else self._init_state_dict() # 初始化

    def forward(self, x: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        latent_value = self.mlp_value(x)
        value = self.value_net(latent_value).flatten()

        latent_policy = self.mlp_policy(x)
        distribution: Distribution = self._get_distribution_from_latent(latent_policy)
        action: Tensor = distribution.get_action(deterministic=deterministic)
        log_prob: Tensor = distribution.log_prob(action)
        return action, value, log_prob

    def select_action(self, obss: Union[ndarray, Tensor]) -> Tuple[ndarray, float, float]:
        obss: Tensor = converto_torch(obss)

        # 判断是否需要扩维度
        if len(obss.shape) == 1:
            obss = obss.unsqueeze(dim = 0)

        with torch.no_grad():
            actions, values, log_probs = self.forward(obss)
        
        actions: ndarray = converto_ndarray(actions)   # 动作转换成ndarray
        return actions, log_probs, values
    
    def evaluate_action(self, obss: Union[ndarray, Tensor], actions: Union[ndarray, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        obss: Tensor = converto_torch(obss)
        latent_value  = self.mlp_value(obss)
        values = self.value_net(latent_value)

        latent_policy = self.mlp_policy(obss)
        distributaion: Distribution = self._get_distribution_from_latent(latent_policy)
        log_probs = distributaion.log_prob(actions)
        entropys = distributaion.entropy()
        return values, log_probs, entropys

    def predict(self, obs: Union[ndarray, Tensor], state: Optional[ndarray] = None, deterministic: bool = False):
        obs: Tensor = converto_torch(obs).unsqueeze(dim=0)
        with torch.no_grad():
            action: Tensor = self.get_distribution(obs).get_action(deterministic=deterministic)
        action: ndarray = converto_ndarray(action).flatten()
        return action, state

    def predict_value(self, obss: Union[ndarray, Tensor]) -> Tensor:
        obss: Tensor = converto_torch(obss)
        latent_value: Tensor = self.mlp_value(obss)
        return self.value_net(latent_value).flatten()
    
    def get_distribution(self, obss: Union[ndarray, Tensor]) -> Distribution:
        latent_policy = self.mlp_policy(obss)
        return self._get_distribution_from_latent(latent_policy)
    
    def _get_distribution_from_latent(self, latent: Tensor):
        mean_action: Tensor = self.action_net(latent)
        action_space: int = self.config.distribution_config.action_space
        
        if action_space == ActionType.discrete or action_space == ActionType.multi_discrete:
            distributaion: Distribution = self.action_dist.proba_distribution(mean_action)
        elif action_space == ActionType.box:
            distributaion: DiagGaussianDistribution = self.action_dist.proba_distribution(mean_action, self.log_std)
        return distributaion

    def _make_proba_distribution(self, *args, **kwargs):
        action_space: int = self.config.distribution_config.action_space
        init_kwargs: Dict = self.config.distribution_config.action_init_kwargs
        if action_space == ActionType.discrete:
            self.action_dist = CategoricalDistribution(**init_kwargs.get('discrete', None))
            self.action_net = self.action_dist.proba_distribution_net(self.config.hidden_dim, *args, **kwargs)
        elif action_space == ActionType.multi_discrete:
            self.action_dist = MultiCategoricalDistribution(**init_kwargs.get('multi_discrete', None))
            self.action_net = self.action_dist.proba_distribution_net(self.config.hidden_dim, *args, **kwargs)
        elif action_space == ActionType.box:
            self.action_dist = DiagGaussianDistribution(**init_kwargs.get('box', None))
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(self.config.hidden_dim, *args, **kwargs)
        else:
            raise TypeError(f'action space must be in [discrete, multi discrete, box], but now action space is {action_space}')
    
    def _init_state_dict(self):
        module_gains: Dict[nn.Module, Union[ndarray, float, int]] = {
            self.mlp_policy: np.sqrt(2),
            self.mlp_value: np.sqrt(2),
            self.action_net: 0.01,
            self.value_net: 1,
        }
        for module, gain in module_gains.items():
            module.apply(partial(self._init_weight, gain=gain))
    
    def _init_weight(self, module: nn.Module, gain: float = 1):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def _load_state_dict_from_checkpoint(self, ckp_pth: str):
        self.load_state_dict(torch.load(ckp_pth))

    def _load_state_dict_from_baseline(self):
        self.mlp_policy.load_state_dict(torch.load('/workspace/model_weight/mlp_policy_net.pth', weights_only=True))
        self.mlp_value.load_state_dict(torch.load('/workspace/model_weight/mlp_value_net.pth', weights_only=True))
        
        self.action_net.load_state_dict(torch.load('/workspace/model_weight/action_net.pth', weights_only=True))
        self.value_net.load_state_dict(torch.load('/workspace/model_weight/value_net.pth', weights_only=True))
    
    def _preprocess_dict(self, state_dict: Dict) -> Dict:
        new_state_dict = dict()
        
        # 逐帧处理key
        for key, value in state_dict.items():
            new_state_dict[f'model.{key}'] = value
        return new_state_dict
    
    def _wrapped_action(self, action: Tensor):
        if self.config.distribution_config.action_space == ActionType.box:
            action = np.clip(action, self.config.distribution_config.low, self.config.distribution_config.high)
        return action


if __name__ == '__main__':
    import numpy as np
    config = PolicyConfig()
    policy = Policy(config)

    obs = np.random.rand(4)
    ac = policy.select_action(obs)

    print(ac)
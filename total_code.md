./metadrive_project/main.py:
```
import sys
sys.path.append('/workspace/metadrive-github/')
from metadrive.custom2_version2.base_config import PPOConfig
from metadrive.custom2_version2.create_env import create_env
from metadrive.custom2_version2.ppo import PPO
from metadrive.custom2_version2.controller import EvalController
from datetime import datetime
from IPython.display import clear_output
import numpy as np

if __name__ == '__main__':
    ppo_config = PPOConfig()
    ppo = PPO(ppo_config)
    is_suc, info = ppo.start()
    
    if not is_suc:
        print(info)
    else:
        print('Training have finished! Start to final evaluate...')
        ppo.final_eval()
        ppo.close()
```

./metadrive_project/custom2_version2/distribution.py:
```
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
        mean_action = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_action, log_std
    
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
```

./metadrive_project/custom2_version2/create_env.py:
```
from metadrive.envs import MetaDriveEnv
from metadrive.utils.doc_utils import generate_gif
from metadrive.custom2_version2.base_config import MetaDriveEnvConfig
from numpy import ndarray
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import asdict

def create_env(config: MetaDriveEnvConfig) -> MetaDriveEnv:
    env = MetaDriveEnv(asdict(config))
    return env

def create_render_config(text: Optional[Dict] = None) -> Dict:
    row_config = dict(
        mode            = 'topdown',
        screen_record   = True,
        window          = False,
        screen_size     = (850, 850),
        camera_position = (83, 10),
    )
    
    if text is not None:
        row_config['text'] = text
    return row_config


def check_env(env: MetaDriveEnv):
    env.reset()
    frames: List[ndarray] = list()
    try:
        for step in range(1):
            if step < 50 : action = [0, 0.5]
            elif 50 <= step < 400: action = [0, -0.1]
            obs, reward, done, _, _ = env.step(action)
            frames.append(env.render(**create_render_config(text={'step': env.engine.episode_step, 'mode': 'Test'})))
        generate_gif(frames, f'dp_single_version2/check/check.gif')
    finally:
        env.close()


if __name__ == '__main__':
    metadriveenv_config = MetaDriveEnvConfig()
    check_env(create_env(metadriveenv_config))
```

./metadrive_project/custom2_version2/controller.py:
```
from metadrive.custom2_version2.ocp import MPC, CBF
from metadrive.custom2_version2.ocp.config import MPConfig, CBFconfig
from metadrive.custom2_version2.envs import ParallelEnv, SingleEnv
from metadrive.custom2_version2.type import VehicleState
from metadrive.custom2_version2.base_config import ControllerConfig
from metadrive import MetaDriveEnv
from metadrive.component.vehicle.default_vehicle import DefaultVehicle
from typing import List, Dict, Tuple, cast
import numpy as np
from numpy import ndarray
from copy import deepcopy
from abc import ABC

class ControllerResult:
    def __init__(self, config: ControllerConfig, eval_mode: bool = False):
        self.config = config
        self._num   = self.config.n_process if not eval_mode else 1
        self.reset()

    def reset(self):
        self._env_idx: int = 0
        self.state_values          = np.empty([self._num, self.config.vehicle_state_dim])
        self.state_values_modified = np.empty([self._num, self.config.vehicle_state_dim])
        self.control_values        = np.empty([self._num, self.config.control_dim])
        self.delta_control_values  = np.empty([self._num, self.config.control_dim])
    
    def push(self, x_mpc: ndarray, x_cbf: ndarray, u_mpc: ndarray, du_cbf: ndarray):
        self.state_values[self._env_idx, :]          = x_mpc
        self.state_values_modified[self._env_idx, :] = x_cbf
        self.control_values[self._env_idx, :]        = u_mpc
        self.delta_control_values[self._env_idx, :]  = du_cbf
        self._env_idx += 1

    @property
    def control_values_modified(self) -> ndarray:
        return self.control_values + self.delta_control_values




class Controller:
    def __init__(self, env: ParallelEnv, config: ControllerConfig):
        self.env = env
        self.config = config
        self.control_values_prev = np.zeros([self.config.n_process, self.config.control_dim])
        self.controller_result = ControllerResult(self.config)
        self._build_controller()

    def control(self, actions: ndarray, dones: ndarray) -> ControllerResult:
        # 获得初始状态
        vehicle_states_init = self._get_vehicle_state()
        infos, masks        = self.env.get_all_vehicle_position()
        self.controller_result.reset()
        
        for idx, (state, info, mask, ) in enumerate(zip(vehicle_states_init, infos, masks)):
            x0 = np.array([state.x, state.y, state.v, state.theta])
            z_ref = actions[idx, :]
            u_prev = self.control_values_prev[idx, :]
            u_mpc, x_mpc = self.mpc_controller(x0, z_ref, u_prev)
            u_mpc, x_mpc = cast(ndarray, u_mpc), cast(ndarray, x_mpc)
            self._check_solve_results(x0, x_mpc[0, :], 'mpc x')
            
            # 求解修正的控制量
            u_cbf, du_cbf, x_cbf = self.cbf_controller(x0, u_mpc[0, :], info, mask)
            self._check_solve_results(x0, x_cbf[0, :], 'cbf x')
            self._check_solve_results(u_mpc[0, :], u_cbf[0, :], 'cbf u')
            
            # 存入结果类中
            self.controller_result.push(x_mpc[1, :], x_cbf[1, :], u_mpc[0, ::-1], du_cbf[0, ::-1])
        
        self.control_values_prev = (1 - dones.reshape(-1, 1)) * self.controller_result.control_values_modified
        return deepcopy(self.controller_result)

    def _build_controller(self):
        metadata = self.env.get_metadata()
        mpc_config = MPConfig(); cbf_config = CBFconfig()
        self.mpc_controller = MPC(mpc_config, metadata)
        self.cbf_controller = CBF(cbf_config, metadata)

    def _get_vehicle_state(self) -> List[VehicleState]:
        vehicle_states: List[Dict] = self.env.get_state()
        vehicle_states_extracted: List[VehicleState] = list()
        for state in vehicle_states:
            vehicle_states_extracted.append(self._extract_row_state(state))
        return vehicle_states_extracted

    def _extract_row_state(self, state: Dict) -> VehicleState:
        vehicle_state = VehicleState()
        pos = state.get('position', None)
        vel = state.get('velocity')
        
        vehicle_state.x = pos[0]
        vehicle_state.y = pos[1]
        vehicle_state.v = np.linalg.norm(vel, 2) # 取速度的二范数
        vehicle_state.theta = state.get('heading_theta', None)
        vehicle_state.a = state.get('throttle_brake', None)
        vehicle_state.delta = state.get('steering', None)
        return vehicle_state
    
    def _check_solve_results(self, a: ndarray, b: ndarray, sign: str, delta: float = 0.1):
        assert np.linalg.norm(a - b) < delta, \
        f'process of {sign}, first != second, first: {a.tolist()}, second: {b.tolist()}'



class EvalController:
    def __init__(self, env: SingleEnv, config: ControllerConfig):
        self.env = env
        self.config = config
        self.control_value_prev = np.zeros([self.config.control_dim])
        self.controller_result  = ControllerResult(self.config, eval_mode=True)
        self._build_controller()

    def control(self, action: ndarray, done: ndarray) -> Tuple[ndarray, ndarray]: # [action_dim]
        # 获得初始状态
        vehicle_states_init = self._get_vehicle_state()
        info, mask          = self.env.get_all_vehicle_position()
        self.controller_result.reset()
        
        # control_values: [[delta, a], ...]
        x0 = np.array([vehicle_states_init.x, vehicle_states_init.y, vehicle_states_init.v, vehicle_states_init.theta])
        z_ref = action
        u_prev = self.control_value_prev
        u_mpc, x_mpc = self.mpc_controller(x0, z_ref, u_prev)
        u_mpc, x_mpc = cast(ndarray, u_mpc), cast(ndarray, x_mpc)
        
        # 检查mpc求解是否合理
        assert np.linalg.norm(x0 - x_mpc[0, :]) < 0.1, f'x0 != x[0, :], x0={x0.tolist()}, x[0, :]={x_mpc[0, :].tolist()}'
        
        u_cbf, du_cbf, x_cbf = self.cbf_controller(x0, u_mpc[0, :], info, mask)

        self.controller_result.push(x_mpc[1, :], x_cbf[1, :], u_mpc[0, :], du_cbf[0, ::-1])
        
        self.control_value_prev = (1 - done) * self.controller_result.control_values_modified
        return deepcopy(self.controller_result)

    def _build_controller(self):
        metadata = self.env.get_metadata()
        mpc_config = MPConfig(); cbf_config = CBFconfig()
        self.mpc_controller = MPC(mpc_config, metadata)
        self.cbf_controller = CBF(cbf_config, metadata)

    def _get_vehicle_state(self) -> VehicleState:
        vehicle_state: Dict = self.env.get_state()
        return self._extract_row_state(vehicle_state)

    def _extract_row_state(self, state: Dict) -> VehicleState:
        vehicle_state = VehicleState()
        pos = state.get('position', None)
        vel = state.get('velocity')
        
        vehicle_state.x = pos[0]
        vehicle_state.y = pos[1]
        vehicle_state.v = np.linalg.norm(vel, 2) # 取速度的二范数
        vehicle_state.theta = state.get('heading_theta', None)
        vehicle_state.a = state.get('throttle_brake', None)
        vehicle_state.delta = state.get('steering', None)
        return vehicle_state
```

./metadrive_project/custom2_version2/utils/logger.py:
```
from metadrive.custom2_version2.base_config import LoggerConfig
from typing import Union, List, Tuple, Dict, Optional
from datetime import datetime
import os
import importlib
from dataclasses import dataclass
import time

@dataclass
class SYMBOL:
    space: str = ' '
    line: str  = '-'

CONFIG_DICT = {
    'base_config.MetaDriveEnvConfig': [
        'map', 'traffic_density', 'traffic_mode', 
        'horizon',
    ],
    'base_config.PPOConfig': [
        'n_process', 'sample_steps',
        'epoch', 'batch_size',
        'evaluate_steps', 'learning_rate',
    ],
    'ocp.config.MPConfig': [
        'np', 'mu', 'a_min', 'a_max',
        'delta_min', 'delta_max',
    ]

}

class Logger:
    def __init__(self, config: LoggerConfig):
        self.config = config
        self.now_datetime = datetime.now()
        self.logger_path = os.path.join(config.logger_path_root, 'log_{}.txt'.format(self.now_datetime.strftime("%Y_%m_%d_%H_%M_%S")))
        self.logger_file = open(self.logger_path, mode='w', encoding='utf-8')
        self._is_first_write_reward: bool = True
        self._is_first_write_addparam: bool = True
        self._reward_index: int = 0
        self._last_write_pos: Optional[int] = None
        self.write_init()
    
    def close(self):
        self.logger_file.close()
    
    def write_init(self):
        self._write(f'Write in {self.now_datetime.strftime("%Y.%m.%d %H: %M: %S")}\n')
        self.write_table()
    
    def write_table(self):
        config_dict: Dict[str, Dict] = self._get_config_modules_dynamic()
        self._write('\nExperiment Args: \n')
        for idx, (config_name, single_config_dict) in enumerate(config_dict.items()):
            self._write(self._get_line_string_with_name(config_name))
            for jdx, (attr_name, attr_value) in enumerate(single_config_dict.items()):
                self._write(self._get_attr_table_string(attr_name, attr_value))
                if jdx < len(single_config_dict)-1: self._write(self._get_line_string())
        
        self._last_write_pos = self._tell()
        self._write(self._get_line_string())

    def write_tabel_additional_params(self, param_names: Union[str, List[str]], param_values: Union[List, Tuple, Dict, float, int, str, ]):
        if self._is_first_write_addparam:
            self._write_addparam_init()
            self._is_first_write_addparam = False
        
        if not isinstance(param_names, list): # 单个的
            param_names = [param_names]
            param_values = [param_values]

        for idx, (param_name, param_value) in enumerate(zip(param_names, param_values)):
            self._write(self._get_attr_table_string(param_name, self._process_attr_value(param_value, param_name)))
            if idx >= len(param_names)-1: self._last_write_pos = self._tell()
            self._write(self._get_line_string())

    def write_reward(self, reward_value: float, best_reward: Optional[float] = None):
        if self._is_first_write_reward:
            self._write_reward_init()
            self._is_first_write_reward = False
        else:
            self._check_last_write_pos()
            self._seek(self._last_write_pos)
        
        # 每次调用书写
        self._write(self._get_attr_table_string(str(self._reward_index), self._process_attr_value(reward_value, f'reward_{self._reward_index}')))
        self._last_write_pos = self._tell()
        self._write(self._get_line_string())
        
        # 是否要添加最优奖励
        if best_reward is not None: 
            self._write(self._get_attr_table_string('best reward', self._process_attr_value(best_reward, 'best_reward')))
            self._write(self._get_line_string())
        
        self._reward_index += 1

    def write_time(self, time_result: Dict, process_name: Optional[str]):
        self._write(f'\n{process_name}\'s time recorded: \n')
        for idx, (time_unit_string, time_value) in enumerate(time_result.items()):
            self._write(f'{time_value: .4f}{time_unit_string}')
            if idx < len(time_result)-1: self._write(' ') 
        self._write('\n')

    def _write(self, s: str):
        self.logger_file.write(s)
        self.logger_file.flush()

    def _seek(self, pos: int):
        self.logger_file.seek(pos)
        self.logger_file.truncate()

    def _tell(self) -> int:
        return self.logger_file.tell()
    
    def _write_reward_init(self):
        self._write('\nRewards: \n')
        self._write(self._get_line_string())
        self._write(self._get_attr_table_string('idx', 'value'))
        self._write(self._get_line_string())

    def _write_addparam_init(self):
        self._check_last_write_pos()
        self._seek(self._last_write_pos)
        self._write(self._get_line_string_with_name('AdditionalParams'))

    def _check_last_write_pos(self):
        if self._last_write_pos is None:
            raise Exception('logic have error! self._last_write_pos not should be None!')

    def _get_line_string(self) -> str:
        return SYMBOL.line * (self.config.max_name_length + self.config.max_value_length + self.config.bias) + '\n'
    
    def _get_line_string_with_name(self, name: str) -> str:
        total_length = self.config.max_name_length + self.config.max_value_length + self.config.bias
        return self._generate_end_string(name, total_length, SYMBOL.line) + '\n'

    def _get_attr_table_string(self, attr_name: str, attr_value: Union[List, Tuple, Dict, float, int, str]) -> str:
        # generate name's end-string
        attr_name_end_string = self._generate_end_string(attr_name, self.config.max_name_length, SYMBOL.space)
        
        # generate value's end-string
        attr_value_string: str = self._process_attr_value(attr_value, attr_name)
        attr_value_end_string = self._generate_end_string(attr_value_string, self.config.max_value_length, SYMBOL.space)
        return f'|{attr_name_end_string}|{attr_value_end_string}|\n'

    def _get_config_modules_dynamic(self) -> Dict[str, Dict]:
        config_dict = dict()
        for key, value in CONFIG_DICT.items():
            key_splited = key.split('.')
            main_module = '.'.join(key_splited[0:-1])
            config_name = key_splited[-1]
            module_total_name = f'metadrive.custom2_version2.{main_module}'
            module = importlib.import_module(module_total_name)
            
            if not hasattr(module, config_name):
                raise NameError(f'Config name: {config_name} not Founded in {module_total_name}.')

            single_config_dict = dict()
            config_cls = getattr(module, config_name)
            
            for attr_name in value:
                if not hasattr(config_cls, attr_name):
                    raise NameError(f'{attr_name} not Founded in {config_name}.')
                attr_value = getattr(config_cls, attr_name)
                single_config_dict[attr_name] = attr_value # 赋值
            config_dict[config_name] = single_config_dict
        return config_dict

    def _generate_end_string(self, s: str, max_length: int, symbol: str) -> str:
        res_length = max_length - len(s)
        split_string = (res_length // 2, res_length // 2 + int(res_length % 2 != 0))
        return f'{split_string[0] * symbol}{s}{split_string[1] * symbol}'

    def _process_attr_value(self, value: Union[List, Dict, Tuple, float, int, str, ], name: str) -> str:
        if isinstance(value, float):
            return f'{value :.4}'
        elif isinstance(value, (int, List, Dict, Tuple, )):
            return str(value)
        elif isinstance(value, str):
            return value
        else:
            raise TypeError(f'the type of attr {name} is {type(value)}, not in [float, int, str, list, dict, tuple, ].')

if __name__ == '__main__':
    logger_config = LoggerConfig()
    logger = Logger(logger_config)
    logger.write_reward(1.2331, 3.213)
    time.sleep(2)
    logger.write_reward(2.23431, 3.213)
    time.sleep(2)
    logger.write_reward(3.213, 3.213)
    time.sleep(2)
    logger.close()
```

./metadrive_project/custom2_version2/utils/timer.py:
```
import time
from typing import Optional, Tuple, Dict

class Timer:
    def __init__(self):
        ...

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end()
        return False
    
    def start(self, name: Optional[str] = None):
        row_string = 'Start to time!'
        if name is not None: row_string = f'In the `{name}` processing..., ' + row_string.lower()
        print(row_string)
        self.start_time = time.time()
    
    def end(self) -> Dict[str, float]:
        end_time = time.time()
        duration = end_time - self.start_time
        dur_min, dur_s = self._second_to_minute(duration)
        print(f'Time out! Total Time is {dur_min:.3f} min {dur_s:.3f} s')
        return {'min': dur_min, 's': dur_s, }

    def _second_to_minute(self, duration: float):
        return duration // 60, duration % 60
```

./metadrive_project/custom2_version2/utils/__init__.py:
```
from .utils import set_random_seed
from .convert_utils import converto_ndarray, converto_torch
from .logger import Logger
from .timer import Timer
```

./metadrive_project/custom2_version2/utils/utils.py:
```
import torch
import numpy as np
import random

def set_random_seed(seed: int, using_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if using_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

./metadrive_project/custom2_version2/utils/convert_utils.py:
```
import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from typing import Union, cast

def converto_torch(x: Union[ndarray, Tensor]):
    if isinstance(x, Tensor):
        return x
    elif isinstance(x, ndarray):
        x = cast(ndarray, x)
        return torch.from_numpy(x).to(torch.float32)

def converto_ndarray(x: Union[float, bool, int, ndarray, Tensor], dtype: Union[np.dtype] = np.float32):
    if isinstance(x, ndarray):
        return x
    elif isinstance(x, float) or isinstance(x, bool):
        return np.array([x], dtype=dtype)
    elif isinstance(x, int):
        return np.array([x], dtype=dtype)
    elif isinstance(x, Tensor):
        x = cast(Tensor, x)
        return x.cpu().detach().numpy()
    else:
        raise TypeError(f'x must include [\'float\', \'bool\', \'int\', \'ndarray\', \'Tensor\'], but now the type of x is {type(x)}')
```

./metadrive_project/custom2_version2/base_config.py:
```
from dataclasses import dataclass, field
import numpy as np
import torch
from torch.nn import Module, Tanh
from numpy import ndarray
from typing import Dict, Optional, List, Tuple
from metadrive.custom2_version2.type import ActionType
from metadrive.custom2_version2.distribution import CategoricalDistribution, MultiCategoricalDistribution

TEST_MODE            = True
STATE_DIM            = 259
ACTION_DIM           = 2
TORCH_DTYPE          = torch.float32
TORCH_DEVICE         = torch.device(device='cuda:0') if torch.cuda.is_available() else torch.device(device='cpu')
# TORCH_DEVICE         = torch.device(device='cpu')
MAX_EPS_COUNTS       = 100
N_PROCESS            = 4
GAE_LAMBDA           = 0.95
UPDATE_FREQ          = 1
IS_LOAD              = False

# Paper Param:
GAMMA                = 0.99
LEARNING_RATE        = 3e-4
EPOCH                = 25
# MAX_BUFFER_SIZE  = 4096 if not TEST_MODE else 256
MAX_BUFFER_SIZE      = 6600    if not TEST_MODE else 256
BATCH_SIZE           = 64
EVALUATE_STEPS       = 100000  if not TEST_MODE else 10
EVALUATE_TOTAL_STEPS = 2000
TOTAL_STEPS          = 2000000 if not TEST_MODE else 1000


@dataclass
class EnvConfig:
    name: str        = 'random test'
    state_dim: int   = STATE_DIM
    action_dim: int  = ACTION_DIM


@dataclass
class MetaDriveEnvConfig:
    map: str                      = 'O'        # 地图形状
    traffic_density: float        = 0.3        # 交通状况
    # 交通模式, option: trigger, respawn, hybrid, basic
    traffic_mode: str             = 'respawn'
    horizon: int                  = 1000       # ego存活序列数
    random_spawn_lane_index: bool = False      # 车辆是否随机生成在某个车道上
    num_scenarios: int            = 1          # 场景的数量
    start_seed: int               = 6          # 开始采样的种子id
    accident_prob: int            = 0          # 事故发生概率
    use_lateral_reward: bool      = True       # 是否启用横向奖励
    log_level: int                = 50         # 日志等级


@dataclass
class ParallelEnvConfig:
    name: str         = 'parallel metadrive env'
    state_dim: int    = STATE_DIM
    action_dim: int   = ACTION_DIM
    start_method: str = 'forkserver'
    n_process: int    = N_PROCESS


@dataclass
class DistributionConfig:
    action_space: int = ActionType.box
    action_init_kwargs: Dict = field(default_factory=lambda: dict(
        discrete = dict(action_dim = 3 * 3),
        multi_discrete = dict(action_dim = [3, 3]),
        box = dict(action_dim = ACTION_DIM),
        sde = dict(action_dim = ACTION_DIM, learn_features = False, epsilon = 1e-6),
    ))
    low: ndarray  = field(default_factory=lambda: np.array([-1.0, -1.0], dtype=np.float32))
    high: ndarray = field(default_factory=lambda: np.array([1.0, 1.0], dtype=np.float32))


@dataclass
class PolicyConfig:
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    hidden_dim: int  = 64
    activate_func: Module = Tanh
    distribution_config: DistributionConfig = field(default_factory=DistributionConfig)
    is_load: bool      = IS_LOAD
    dtype: torch.dtype = TORCH_DTYPE
    device: torch.device = TORCH_DEVICE


@dataclass
class RolloutBufferConfig:
    state_dim: int       = STATE_DIM
    action_dim: int      = ACTION_DIM
    high_state_dim: int  = 2
    batch_size: int      = BATCH_SIZE
    device: torch.device = TORCH_DEVICE
    max_buffer_size: int = MAX_BUFFER_SIZE
    gamma: float         = GAMMA           # 计算优势函数的参数之一
    gae_lambda: float    = GAE_LAMBDA      # 计算优势函数的参数之一
    n_process: int       = N_PROCESS       # 进程数量


@dataclass
class ControllerConfig:
    n_process: int         = N_PROCESS
    control_dim: int       = 2
    vehicle_state_dim: int = 4

@dataclass
class LoggerConfig:
    logger_path_root: str = 'dp_single_version2/logger'
    max_name_length: int  = 35
    max_value_length: int = 35
    bias: int             = 3


@dataclass
class PPOConfig:
    env_config: EnvConfig = field(default_factory = EnvConfig)
    metadriveenv_config: MetaDriveEnvConfig = field(default_factory = MetaDriveEnvConfig)
    parallel_env_config: ParallelEnvConfig = field(default_factory = ParallelEnvConfig)
    policy_config: PolicyConfig = field(default_factory = PolicyConfig)
    buffer_config: RolloutBufferConfig = field(default_factory = RolloutBufferConfig)
    distribution_config: DistributionConfig = field(default_factory=DistributionConfig)
    controller_config: ControllerConfig = field(default_factory=ControllerConfig)
    logger_config: LoggerConfig = field(default_factory=LoggerConfig)

    n_process: int                   = N_PROCESS
    sample_steps: int                = MAX_BUFFER_SIZE   # 一次的采样长度
    total_steps: int                 = TOTAL_STEPS       # 总的期望采样长度
    max_eps_counts: int              = MAX_EPS_COUNTS    # 一个回合最大的采样数量
    update_freq: int                 = UPDATE_FREQ       # 经过多少步才更新
    epoch: int                       = EPOCH
    batch_size: int                  = BATCH_SIZE
    epsilon: float                   = 0.2
    entropy_coef: float              = 0.0
    bc_coef: float                   = 1.0
    value_loss_coef: float           = 0.5
    learning_rate: float             = 3e-4
    max_grad_norm: float             = 0.5
    update_freq: int                 = UPDATE_FREQ       # 经过多少步才更新
    target_kl: Optional[float]       = None
    evaluate_steps: int              = EVALUATE_STEPS
    evaluate_total_steps: int        = EVALUATE_TOTAL_STEPS
    is_load: bool                    = IS_LOAD
    delta_bc: float                  = 1.0
    logger_save_root: str            = 'dp_single_version2/logger'
    policy_checkpoint_pth: str       = 'dp_single_version2/ckp_pth/policy.pth'
    best_policy_checkpoint_pth: str  = 'dp_single_version2/ckp_pth/best_policy.pth'
    evaluate_save_root: str          = 'dp_single_version2/eval'
    device: torch.device             = TORCH_DEVICE
```

./metadrive_project/custom2_version2/schedule.py:
```
class ConstantSchedule:
    def __init__(self, val: float):
        self.val = val
    
    def __call__(self, _: float) -> float:
        return self.val
```

./metadrive_project/custom2_version2/ppo.py:
```
import sys
sys.path.append('/workspace/metadrive-github/metadrive')
import numpy as np
from torch import Tensor
from numpy import ndarray
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
from metadrive.custom2_version2.base_config import PPOConfig
from metadrive.custom2_version2.policy import Policy
from metadrive.custom2_version2.buffer import RolloutBuffer, RolloutBatchData
from metadrive.custom2_version2.envs import ParallelEnv, SingleEnv
from metadrive.custom2_version2.schedule import ConstantSchedule
from metadrive.custom2_version2.utils import set_random_seed, converto_ndarray, converto_torch, Logger, Timer
from metadrive.custom2_version2.create_env import create_env, create_render_config
from metadrive.custom2_version2.type import ActionType, RenderClass
from metadrive.custom2_version2.controller2 import Controller
from metadrive.utils.doc_utils import generate_gif
from typing import Tuple, Dict, Union, Optional, List, cast
from tqdm import tqdm
from functools import partial
from datetime import datetime
import traceback
import os

class PPO:
    def __init__(self, config: PPOConfig, eval_mode: bool = False):
        set_random_seed(0) # 设置随机种子
        self.config = config
        self.now_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.final_evaluate_path = f'{self.config.evaluate_save_root}/demo_{self.now_datetime}.gif'

        self.env: ParallelEnv = ParallelEnv(
            [partial(create_env, config.metadriveenv_config) for _ in range(self.config.parallel_env_config.n_process)],
            self.config.parallel_env_config,
        )
        self.env_eval: SingleEnv = SingleEnv(
            partial(create_env, config.metadriveenv_config),
            self.config.parallel_env_config,
        )
        
        self.policy: Policy = Policy(self.config.policy_config)
        self.buffer: RolloutBuffer = RolloutBuffer(self.config.buffer_config)
        self._last_obss  = self.env.reset() # 初始化第一次的env
        self.env_eval.reset()
        
        self.controller: Controller      = Controller(self.env, self.config.controller_config, eval_mode=False)
        self.controller_eval: Controller = Controller(self.env_eval, self.config.controller_config, eval_mode=True)
        self.timer: Timer = Timer()
        self.render_class: RenderClass = RenderClass()

        if not eval_mode:
            self._create_logger()
        
        self.schedule    = ConstantSchedule(self.config.epsilon)

        # ======== 一些常量 ======== #
        self._last_dones = np.array([True, ]) # 初始化第一次的done
        self._remaining  = 1.0
        self.num_steps   = 0.0
        self._start_successful: Optional[bool] = None

        # ====== 训练的初始化 ====== #
        self.optimizer = Adam(self.policy.parameters(), lr=self.config.learning_rate)
        # self.optimizer.load_state_dict(torch.load('/workspace/model_weight/optimizer.pth'))
    
    def start(self) -> Tuple[bool, str]:
        _process_name_ = 'Sample & Train'
        self.timer.start(_process_name_)
        is_suc, info = self._start()
        if is_suc: self.logger.write_time(self.timer.end(), _process_name_)
        return is_suc, info
    
    def _start(self) -> Tuple[bool, str]:
        # try:
        self.policy.train()
        pbar = tqdm(total=self.config.total_steps, desc='total')
        iterations = 0
        evaluate_idx = 0
        best_reward = -float('inf')
        while self.num_steps < self.config.total_steps:
            print('\nstart to sample...')
            self._sample()
            iterations += 1
            self.num_steps = self.config.n_process * iterations * self.config.sample_steps
            self._update_remaining(self.num_steps)
            pbar.update(self.config.n_process * self.config.sample_steps)

            print('\nstart to train...')
            self._train()

            if self.num_steps >= (evaluate_idx + 1) * self.config.evaluate_steps:
                print('\nstart to evaluate & save...')
                reward_eval = self._evaluate()
                self.logger.write_reward(reward_eval)
                evaluate_idx += 1
                self._save(ckp_pth=self.config.policy_checkpoint_pth)
                if reward_eval > best_reward: self._save(ckp_pth=self.config.best_policy_checkpoint_pth); best_reward = reward_eval
        
        self._start_successful = True
        start_info = 'Successful!'
        # except Exception:
        #     self._start_successful = False
        #     start_info = traceback.format_exc()
        # finally:
        #     if not self._start_successful:
        #         self.close()
        # return self._start_successful, start_info
    
    def final_eval(self):
        reward_eval = self._evaluate(True, self.final_evaluate_path)
        print(f'episode_reward {reward_eval}')
        print('gif generation is finished ...')
    
    def predict(self, obs: Union[ndarray, Tensor], state: Optional[ndarray] = None, deterministic: bool = False):
        action, state = self.policy.predict(obs, state, deterministic)
        return action, state
    
    def close(self):
        self.env.close()
        self.env_eval.close()
        if hasattr(self, 'logger'): self.logger.close()

    def load_weight_from_checkpoint(self, load_path: Optional[str] = None):
        if load_path is None:
            load_path = self.config.policy_checkpoint_pth
        self.policy.load_state_dict(torch.load(f=load_path))
        print(f'load weight from `{load_path}` successfully!')

    def _sample(self):
        pbar = tqdm(total=self.config.sample_steps, desc='sample')
        # set_random_seed(0, True) # 对齐用

        curr_step: int = 0
        self.buffer.reset() # 采样前先清空buffer

        while curr_step < self.config.sample_steps:
            curr_step += 1
            pbar.update(1)
            
            # 上层决策
            actions, log_probs, values = self.policy.select_action(self._last_obss)
            
            # 底层控制
            controller_result = self.controller.control(actions, self._last_dones)

            obs_nexts, rewards, dones, step_infos = self.env.step(controller_result.control_values_modified)
            rewards = self._bootstraping(rewards, dones, step_infos)
            
            # 存入buffer
            self.buffer.push(self._last_obss, actions, rewards, self._last_dones, log_probs, values,  # 正常ppo的
                             controller_result.state_values[:, 2::], controller_result.state_values_modified[:, 2::], )
            
            self._last_obss = obs_nexts # update observation
            self._last_dones = dones    # update done

        with torch.no_grad():
            values = self.policy.predict_value(converto_torch(obs_nexts))
        dones = torch.from_numpy(dones).to(torch.long)
        
        self.buffer.compute_advantage(values, dones)
        pbar.close()
    
    def _train(self):
        total_sample_steps = self.config.sample_steps * self.config.n_process
        pbar = tqdm(total=self.config.epoch * (total_sample_steps // self.config.batch_size + int(total_sample_steps % self.config.batch_size != 0)), desc='train')
        clip_range = self._clip_range()
        self.policy.to(self.config.device)
        
        # policy_loss_list: List = []
        # value_loss_list: List = []
        # loss_list: List = []

        for epoch in range(self.config.epoch):
            for rollout_data in self.buffer.get():
                rollout_data.move_to_device(self.config.device)
                actions = rollout_data.actions
                
                # 如果是离散动作, 展平
                if self.config.distribution_config.action_space == ActionType.discrete:
                    actions = actions.flatten()
                
                values, log_probs, entropys = self.policy.evaluate_action(rollout_data.obss, actions)
                
                # advantage批内归一化
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_probs - rollout_data.old_log_probs)
                
                policy_loss = self._get_policy_loss(rollout_data.obss, advantages, ratio, rollout_data.z_mpcs, rollout_data.z_cbfs, clip_range)
                value_loss = mse_loss(rollout_data.returns.flatten(), values.flatten())
                entropy_loss = -torch.mean(entropys)

                loss = policy_loss + self.config.entropy_coef * entropy_loss + self.config.value_loss_coef * value_loss

                # early stop机制
                if self._early_stop(log_probs, rollout_data.old_log_probs): break

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # ===============extra info==================#
                pbar.update(1)

                # policy_loss_list.append(policy_loss.item())
                # value_loss_list.append(value_loss.item())
                # loss_list.append(loss.item())
                # ===========================================#
        
        self.policy.to(torch.device(device='cpu'))
        pbar.close()

    def _evaluate(self, is_render: bool = False, evaluate_save_path: str = '') -> Tuple:
        self.policy.eval()
        obs = self.env_eval.reset()
        last_done = np.ones(shape=[1, ])
        total_reward = 0.0
        if is_render: render_row_text = self._create_render_text()
        
        for _ in range(self.config.evaluate_total_steps):
            action, _ = self.predict(obs, deterministic=True)
            controller_result = self.controller_eval.control(action, last_done)
            obs, reward, done, step_info = self.env_eval.step(controller_result.control_values_modified)
            
            total_reward += reward
            if is_render: self.render_class.add_frame(self._render(render_row_text))
            
            if done:
                break
            
            last_done = np.array([done], dtype=np.long)
        
        if is_render: self.render_class.generate_gif(evaluate_save_path)
        return total_reward
    
    def _save(self, ckp_pth: str):
        # 保存checkpoint
        torch.save(self.policy.state_dict(), ckp_pth)

    def _render(self, text: Dict) -> ndarray:
        if 'step' in text: text['step'] = self.render_class.render_index
        return self.env_eval.render(**create_render_config(text = text))
    
    def _create_render_text(self) -> Dict:
        metadriveenv_config = self.config.metadriveenv_config
        return dict(traffic_mode = metadriveenv_config.traffic_mode, step = None)
    
    def _get_policy_loss(self, obss: Tensor, advantages: Tensor, ratio: Tensor, z_mpcs: Tensor, z_cbfs: Tensor, clip_range: float) -> Tensor:
        # create index
        bc_index = torch.norm(z_cbfs - z_mpcs, p=1, dim=1) > self.config.delta_bc
        standard_index = (bc_index == False)
        
        # caculate standard loss:
        if (~standard_index).all():
            standard_loss = 0.0
        else:
            standard_loss_1 = advantages[standard_index] * ratio[standard_index]
            standard_loss_2 = advantages[standard_index] * torch.clamp(ratio[standard_index], 1 - clip_range, 1 + clip_range)
            standard_loss = -torch.min(standard_loss_1, standard_loss_2).mean()
        
        # caculate bc loss:
        if (~bc_index).all():
            bc_loss = 0.0
        else:
            pi_action = self.policy.act_mean(obss[bc_index])
            omega = 1 + torch.exp(torch.norm(z_cbfs[bc_index] - z_mpcs[bc_index], p=2, dim=1))
            bc_loss = (omega * torch.norm(z_cbfs[bc_index] - pi_action, p=2, dim=1)).mean()

        policy_loss = standard_loss + self.config.bc_coef * bc_loss
        return policy_loss

    def _bootstraping(self, rewards: ndarray, dones: ndarray, step_infos: List[Dict]) -> ndarray:
        '''
        根据done和step_info的情况考虑是否修正reward, 如果是因为timelimit导致的done, 则考虑用value(obs)修正当前的reward
        '''
        for idx, done in enumerate(dones):
            done = cast(ndarray, done)
            if (
                done.item() # 完成了(截断 or 成功)
                and step_infos[idx].get('terminal_observation', None) is not None # (有终端观测值)
                and step_infos[idx].get('TimeLimit.truncated', False) # (是截断)
            ):
                with torch.no_grad():
                    terminal_value = self.policy.predict_value(step_infos[idx].get('terminal_observation'))
                    terminal_value: ndarray = converto_ndarray(terminal_value)
                rewards[idx] += self.config.buffer_config.gamma * terminal_value
            return rewards
    
    def _early_stop(self, log_prob: Tensor, old_log_probs: RolloutBatchData) -> bool:
        with torch.no_grad():
            log_ratio = log_prob - old_log_probs
            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
        
        if self.config.target_kl is not None and approx_kl_div > 1.5 * self.config.target_kl:
            return True
        return False

    def _update_remaining(self, num_step: int):
        self._remaining = 1.0 - float(num_step / self.config.total_steps)

    
    def _clip_range(self) -> float:
        return self.schedule(self._remaining)
    
    def _create_logger(self):
        self.logger: Logger = Logger(self.config.logger_config)

        if hasattr(self, 'final_evaluate_path'):
            self.logger.write_tabel_additional_params(['final_evaluate_path'], [os.path.basename(self.final_evaluate_path)])



if __name__ == '__main__':
    ppo_config = PPOConfig()
    ppo = PPO(ppo_config)
    
    # 开始执行ppo算法
    ppo.start()
```

./metadrive_project/custom2_version2/policy.py:
```
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
        action_mean: Tensor = distribution.get_action(deterministic=True)
        return action, value, log_prob, action_mean

    def select_action(self, obss: Union[ndarray, Tensor]) -> Tuple[ndarray, float, float]:
        obss: Tensor = converto_torch(obss)

        # 判断是否需要扩维度
        if len(obss.shape) == 1:
            obss = obss.unsqueeze(dim = 0)

        with torch.no_grad():
            actions, values, log_probs, _ = self.forward(obss)
        
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
    
    def act_mean(self, obss: Union[ndarray, Tensor]) -> Tensor:
        obss: Tensor = converto_torch(obss)
        with torch.no_grad():
            _, _, _, action_means = self.forward(obss)
        return action_means

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
```

./metadrive_project/custom2_version2/controller2.py:
```
from metadrive.custom2_version2.ocp import MPC, CBF
from metadrive.custom2_version2.ocp.config import MPConfig, CBFconfig
from metadrive.custom2_version2.envs import ParallelEnv, SingleEnv
from metadrive.custom2_version2.type import VehicleState
from metadrive.custom2_version2.base_config import ControllerConfig
from metadrive import MetaDriveEnv
from metadrive.component.vehicle.default_vehicle import DefaultVehicle
from typing import List, Dict, Tuple, Union, cast
import numpy as np
from numpy import ndarray
from copy import deepcopy
from abc import ABC

class ControllerResult:
    def __init__(self, config: ControllerConfig, eval_mode: bool = False):
        self.config = config
        self._num   = self.config.n_process if not eval_mode else 1
        self.reset()

    def reset(self):
        self._env_idx: int = 0
        self.state_values          = np.empty([self._num, self.config.vehicle_state_dim])
        self.state_values_modified = np.empty([self._num, self.config.vehicle_state_dim])
        self.control_values        = np.empty([self._num, self.config.control_dim])
        self.delta_control_values  = np.empty([self._num, self.config.control_dim])
        self.control_values_prev   = np.empty([self._num, self.config.control_dim])
    
    def push(self, x_mpc: ndarray, x_cbf: ndarray, u_mpc: ndarray, du_cbf: ndarray):
        self.state_values[self._env_idx, :]          = x_mpc
        self.state_values_modified[self._env_idx, :] = x_cbf
        self.control_values[self._env_idx, :]        = u_mpc
        self.delta_control_values[self._env_idx, :]  = du_cbf
        self._env_idx += 1

    def update_control_values_prev(self, dones: ndarray):
        self.control_values_prev = (1 - dones.reshape(-1, 1)) * self.control_values_modified

    @property
    def control_values_modified(self) -> ndarray:
        return self.control_values + self.delta_control_values



class Controller:
    def __init__(self, env: Union[ParallelEnv, SingleEnv], config: ControllerConfig, eval_mode: bool = False):
        self.env = env
        self.config = config
        self.eval_mode = eval_mode
        self.controller_result = ControllerResult(self.config, eval_mode=eval_mode)
        self._build_controller()

    def control(self, actions: ndarray, dones: ndarray) -> ControllerResult:
        # 获得初始状态
        vehicle_states_init = self._get_vehicle_state()
        infos, masks        = self._get_all_vehicle_position()
        self.controller_result.reset()
        
        for idx, (state, info, mask, ) in enumerate(zip(vehicle_states_init, infos, masks)):
            x0 = np.array([state.x, state.y, state.v, state.theta])
            z_ref = actions[idx, :]
            u_prev = self.controller_result.control_values_prev[idx, :]
            u_mpc, x_mpc, solve_info_mpc = self.mpc_controller(x0, z_ref, u_prev)
            u_mpc, x_mpc = cast(ndarray, u_mpc), cast(ndarray, x_mpc)
            self._check_solve_results(x0, x_mpc[0, :], 'mpc x')
            
            # 求解修正的控制量
            u_cbf, du_cbf, x_cbf, solve_info_cbf = self.cbf_controller(x0, u_mpc[0, :], info, mask)
            self._check_solve_results(x0, x_cbf[0, :], 'cbf x')
            self._check_solve_results(u_mpc[0, :], u_cbf[0, :], 'cbf u')
            
            if not bool(solve_info_cbf.get('success')): # cbf解不出来(无论怎样都满足不了...)
                u_cbf = None; x_cbf = deepcopy(x_mpc)
                du_cbf = np.zeros([1, self.mpc_config.nu])

            # 存入结果类中
            self.controller_result.push(x_mpc[1, :], x_cbf[1, :], u_mpc[0, ::-1], du_cbf[0, ::-1])
        
        # 更新control的values
        self.controller_result.update_control_values_prev(dones)
        return deepcopy(self.controller_result)

    def _build_controller(self):
        metadata = self.env.get_metadata()
        self.mpc_config = MPConfig(); self.cbf_config = CBFconfig()
        self.mpc_controller = MPC(self.mpc_config, metadata)
        self.cbf_controller = CBF(self.cbf_config, metadata)

    def _get_vehicle_state(self) -> List[VehicleState]:
        vehicle_states: List[Dict] = self.env.get_state() if not self.eval_mode else [self.env.get_state()]
        vehicle_states_extracted: List[VehicleState] = list()
        for state in vehicle_states:
            vehicle_states_extracted.append(self._extract_row_state(state))
        return vehicle_states_extracted
    
    def _get_all_vehicle_position(self) -> Tuple[List[ndarray], List[ndarray]]:
        infos, masks = self.env.get_all_vehicle_position()
        if self.eval_mode:
            infos = [infos]; masks = [masks]
        return infos, masks

    def _extract_row_state(self, state: Dict) -> VehicleState:
        vehicle_state = VehicleState()
        pos = state.get('position', None)
        vel = state.get('velocity')
        
        vehicle_state.x = pos[0]
        vehicle_state.y = pos[1]
        vehicle_state.v = np.linalg.norm(vel, 2) # 取速度的二范数
        vehicle_state.theta = state.get('heading_theta', None)
        vehicle_state.a = state.get('throttle_brake', None)
        vehicle_state.delta = state.get('steering', None)
        return vehicle_state
    
    def _check_solve_results(self, a: ndarray, b: ndarray, sign: str, delta: float = 0.1):
        assert np.linalg.norm(a - b) < delta, \
        f'process of {sign}, first != second, first: {a.tolist()}, second: {b.tolist()}'
```

./metadrive_project/custom2_version2/buffer.py:
```
from metadrive.custom2_version2.base_config import RolloutBufferConfig
from dataclasses import dataclass
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader

@dataclass
class RolloutBatchData:
    obss: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    z_mpcs: Tensor
    z_cbfs: Tensor
    
    def move_to_device(self, device: torch.device):
        self.obss          = self.obss.to(device)
        self.actions       = self.actions.to(device)
        self.rewards       = self.rewards.to(device)
        self.dones         = self.dones.to(device)
        self.old_log_probs = self.old_log_probs.to(device)
        self.old_values    = self.old_values.to(device)
        self.advantages    = self.advantages.to(device)
        self.returns       = self.returns.to(device)
        self.z_mpcs        = self.z_mpcs.to(device)
        self.z_cbfs        = self.z_cbfs.to(device)


class RolloutBuffer:
    def __init__(self, config: RolloutBufferConfig):
        self.config = config
        self.reset()
    
    def push(self, states: ndarray, actions: ndarray, rewards: ndarray, dones: ndarray, log_probs: Tensor, values: Tensor, z_mpcs: ndarray, z_cbfs: ndarray):
        if self._idx >= self.config.max_buffer_size:
            raise IndexError(f'index out of range, max buffer size = {self.config.max_buffer_size}.')
        
        _slice = slice(self._idx, self._idx + self.config.n_process, 1)

        self.states[_slice, ...]   = torch.from_numpy(states).to(torch.float32)
        self.actions[_slice, ...]  = torch.from_numpy(actions).to(torch.float32)
        self.rewards[_slice, :]    = torch.from_numpy(rewards).to(torch.float32)
        self.dones[_slice, :]      = torch.from_numpy(dones).to(torch.long)
        self.log_probs[_slice, :]  = log_probs.clone().detach().to(torch.float32)
        self.values[_slice, :]     = values.clone().detach().to(torch.float32)
        self.z_mpcs[_slice, ...]   = torch.from_numpy(z_mpcs).to(torch.float32)
        self.z_cbfs[_slice, ...]   = torch.from_numpy(z_cbfs).to(torch.float32)
        self._idx += 1 # 索引后移n位

    def compute_advantage(self, values: Tensor, dones: Tensor):
        advantage_lasts: int = 0
        for t in range(self._idx-1, -1, -1):
            if t == self._idx-1: # 初始条件
                next_non_terminals = 1.0 - dones
                next_values = values
            else:
                next_non_terminals = 1.0 - self.dones[t+1]
                next_values = self.values[t+1]

            # 计算优势函数
            delta_t = self.rewards[t] + self.config.gamma * next_non_terminals * next_values - self.values[t]
            advantage_lasts = delta_t + self.config.gamma * self.config.gae_lambda * next_non_terminals * advantage_lasts
            self.advantages[t] = advantage_lasts
        
        # 累积奖励
        self.returns = self.advantages + self.values

    def reset(self):
        self.states: Tensor     = self._init_tensor(self.config.max_buffer_size, self.config.state_dim, dtype=torch.float32)        # 2d vector
        self.actions: Tensor    = self._init_tensor(self.config.max_buffer_size, self.config.action_dim, dtype=torch.float32)       # 2d vector
        self.rewards: Tensor    = self._init_tensor(self.config.max_buffer_size, dtype=torch.float32)                               # 1d vector
        self.dones: Tensor      = self._init_tensor(self.config.max_buffer_size, dtype=torch.long)                                  # 1d vector
        self.log_probs: Tensor  = self._init_tensor(self.config.max_buffer_size, dtype=torch.float32)                               # 1d vector
        self.values: Tensor     = self._init_tensor(self.config.max_buffer_size, dtype=torch.float32)                               # 1d vector
        self.z_mpcs: Tensor     = self._init_tensor(self.config.max_buffer_size, self.config.high_state_dim, dtype=torch.float32)   # 2d vector
        self.z_cbfs: Tensor     = self._init_tensor(self.config.max_buffer_size, self.config.high_state_dim, dtype=torch.float32)   # 2d vector
        
        self.advantages: Tensor = self._init_tensor(self.config.max_buffer_size, dtype=torch.float32)                               # 1d vector
        self.returns: Tensor    = self._init_tensor(self.config.max_buffer_size, dtype=torch.float32)                               # 1d vector
        

        # ============other data========== #
        self._idx = 0
        self._next_value = 0
        self._have_flatten = False
        # ================================ #

    def get(self): # 构造一个生成器
        _len = self._idx * self.config.n_process
        indices = np.random.permutation(_len)

        if not self._have_flatten:
            self._flatten_tensor() # 展平所有的变量
            self._have_flatten = True
        
        start_idx = 0
        while start_idx < _len:
            end_idx = min(start_idx + self.config.batch_size, _len)
            indice = indices[start_idx: end_idx]
            start_idx += self.config.batch_size
            yield \
            RolloutBatchData(
                obss          = self.states[indice],
                actions       = self.actions[indice],
                rewards       = self.rewards[indice].flatten(),
                dones         = self.dones[indice].flatten(),
                old_log_probs = self.log_probs[indice].flatten(),
                old_values    = self.values[indice].flatten(),
                advantages    = self.advantages[indice].flatten(),
                returns       = self.returns[indice].flatten(),
                z_mpcs        = self.z_mpcs[indice],
                z_cbfs        = self.z_cbfs[indice],
            )

    def _flatten_tensor(self):
        self.states        = self._flatten(self.states)
        self.actions       = self._flatten(self.actions)
        self.rewards       = self._flatten(self.rewards)
        self.dones         = self._flatten(self.dones)
        self.log_probs     = self._flatten(self.log_probs)
        self.values        = self._flatten(self.values)
        self.advantages    = self._flatten(self.advantages)
        self.returns       = self._flatten(self.returns)
        self.z_mpcs        = self._flatten(self.z_mpcs)
        self.z_cbfs        = self._flatten(self.z_cbfs)

    def _flatten(self, t: Tensor): # t.shape = 2 or 3
        if len(t.shape) < 3:
            t = t.unsqueeze(dim=-1)
        return t.permute(1, 0, 2).reshape(-1, t.shape[-1])
    
    def _reset(self, t: Tensor):
        t = t.zero_().reshape()
    
    def _init_tensor(self, num: int, dim: Optional[int] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        if dim is None: # 1维创建1维向量
            return torch.zeros(size=[num, self.config.n_process, ], dtype=dtype)
        return torch.zeros(size=[num, self.config.n_process, dim], dtype=dtype)
```

./metadrive_project/custom2_version2/ocp/cbf.py:
```
from metadrive.custom2_version2.type import VehicleState
from metadrive.custom2_version2.ocp.config import CBFconfig
from metadrive.custom2_version2.ocp.base import OCP
from typing import Dict, cast, Tuple
import casadi as ca
import numpy as np
from numpy import ndarray

class CBF(OCP):
    def __init__(self, config: CBFconfig, metadata: Dict):
        '''
        config: CBF的配置,
        metadata: 一些常量配置, 不应该在update中改变
        '''
        self.lf = metadata.get('lf', None)
        self.lr = metadata.get('lr', None)
        super(CBF, self).__init__(config, metadata)
    
    def __call__(self, x0, u0, mask, info):
        res: Dict = super(CBF, self).__call__(x0, u0, mask, info)
        return self._parse_result(res)
    
    def _build_numeric_problem(self):
        # 决策设定决策变量的初值, 和值的上下界
        w0 = np.zeros(self._nlp_metadata['w_dim']) if not hasattr(self, '_last_w') else self._last_w

        lbw = np.concatenate([
            np.tile([self.config.a_min, self.config.delta_min], self._nlp_metadata['u_dim'] // 2),
            np.tile([self.config.a_min, self.config.delta_min], self._nlp_metadata['du_dim'] // 2),
            -np.inf * np.ones(self._nlp_metadata['x_dim']),
        ])
        ubw = np.concatenate([
            np.tile([self.config.a_max, self.config.delta_max], self._nlp_metadata['u_dim'] // 2),
            np.tile([self.config.a_max, self.config.delta_max], self._nlp_metadata['du_dim'] // 2),
            np.inf * np.ones(self._nlp_metadata['x_dim']),
        ])

        # 设定约束条件的上下界
        lbg = np.zeros(self._nlp_metadata['g_equal_dim'] + self._nlp_metadata['g_unequal_dim_1'])
        ubg = np.concatenate([
            np.zeros(self._nlp_metadata['g_equal_dim']),
            np.inf * np.ones(self._nlp_metadata['g_unequal_dim_1']),
        ])

        self._num_prob = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg, }

    def _caculate_cost_and_conditions(self):
        # ------- 添加决策变量 ----- #
        U  = ca.MX.sym( 'U', self.config.nu)
        DU = ca.MX.sym('DU', self.config.nu)
        X  = ca.MX.sym( 'X', 2 * self.config.nx)

        # -------- 添加常量 ------- #
        x0 = ca.MX.sym('x0', self.config.nx)
        u0 = ca.MX.sym('u0', self.config.nu)

        self._g = list()
        self._p = [x0, u0, ]
        cost = ca.sumsqr(DU)

        # cons0[=]: 初始状态约束
        xk = X[0: self.config.nx]
        uk = U[0: self.config.nu]
        du = DU
        
        self._g.append(xk - x0)
        self._g.append(uk - u0)

        # cons1[=]: 状态转移方程约束
        xk_next = X[self.config.nx: 2 * self.config.nx]
        xk_step = self._f(xk, uk, du)
        self._g.append(xk_next - xk_step)

        # cons2[>=]: 添加安全约束
        self._constraint_safety_lane_change_collision(xk[0: 2], xk_next[0: 2])

        self._nlp = \
        {
            'x': ca.vertcat(U, DU, X),
            'f': cost,
            'g': ca.vertcat(*self._g),
            'p': ca.vertcat(*self._p),
        }

        self._nlp_metadata = \
        {
            'w_dim' : U.shape[0] + DU.shape[0] + X.shape[0],
            'u_dim' : U.shape[0],
            'du_dim': DU.shape[0],
            'x_dim' : X.shape[0],
            'g_equal_dim' : self.config.nx + self.config.nu + self.config.nx,
            'g_unequal_dim_1': self.config.N,
        }

    def _define_state_update_equation(self):
        x  = ca.MX.sym( 'x', self.config.nx)
        u  = ca.MX.sym( 'u', self.config.nu)
        du = ca.MX.sym('du', self.config.nu)

        u_modify = u + du
        # 状态更新方程
        beta = ca.atan(ca.tan(u_modify[1]) * self.lr / (self.lf + self.lr))
        x_next = ca.vertcat(
            x[0] + x[2] * ca.cos(x[3] + beta) * self.config.Ts,
            x[1] + x[2] * ca.sin(x[3] + beta) * self.config.Ts,
            x[2] + u_modify[0] * self.config.Ts,
            x[3] + x[2] / self.lr * ca.sin(beta) * self.config.Ts,
        )
        self._f = ca.Function('f', [x, u, du], [x_next])
    
    def _constraint_safety_lane_change_collision(self, pk, pk_next):
        '''
        给出安全约束之, 变道碰撞的约束. 当ego变道到另一个道路的时候,
        可能与另一个车辆相撞, 该约束通过限定两车之间的距离大于零而避免相撞.

        Args:
            pk: 当前时刻ego的位置信息, 应该是2维;
            pk_next: 下一时刻ego的位置信息, 应该是2维;
        '''
        mask = ca.MX.sym('mask', self.config.N)
        info = ca.MX.sym('info', self.config.N * self.config.info_dim)
        h = lambda pk, pk_s: ca.sqrt(ca.sumsqr(pk - pk_s) + 1e-6) - self.config.dist_min

        for i in range(self.config.N):
            pk_s = info[i * self.config.info_dim: (i+1) * self.config.info_dim]
            self._g.append(ca.if_else(
                mask[i] > 0, mask[i] * (h(pk_next, pk_s) - (1 + self.config.gamma) * h(pk, pk_s)), 0
            ))
        
        self._p.append(mask)
        self._p.append(info)

    def _parse_result(self, res: Dict) -> Tuple[ndarray, ndarray, ndarray, Dict]:
        '''
        解析求解器返回的结果.

        Args:
            res: 求解器solver返回的结果
        
        Returns:
            元组, 第一项是u, 形状是(1, nu); 第二项是du, 形状是(1, nu); 第三项是x, 形状是(2, nx).
        '''
        w: ndarray  = res.get('x', None).full().flatten()
        u: ndarray  = w[0: self._nlp_metadata['u_dim']].reshape(1, self.config.nu)
        du: ndarray = w[self._nlp_metadata['u_dim']: self._nlp_metadata['u_dim']+self._nlp_metadata['du_dim']].reshape(1, self.config.nu)
        x: ndarray  = w[self._nlp_metadata['u_dim']+self._nlp_metadata['du_dim']::].reshape(2, self.config.nx)
        solve_info = self._get_stats()
        self._last_w = w
        return u, du, x, solve_info
```

./metadrive_project/custom2_version2/ocp/__init__.py:
```
from .cbf import CBF
from .mpc import MPC
from .config import CBFconfig, MPConfig
```

./metadrive_project/custom2_version2/ocp/base.py:
```
from metadrive.custom2_version2.ocp.config import OCPconfig
from metadrive.custom2_version2.type import VehicleState
from abc import ABC, abstractmethod
import casadi as ca
from numpy import ndarray
from typing import cast, List, Dict, Tuple
import numpy as np

class OCP(ABC):
    def __init__(self, config: OCPconfig, metadata: Dict):
        self.config = config
        self.metadata = metadata
        self._is_first_created = True
        self._define_state_update_equation()

    def __call__(self, *args, **kwargs) -> ndarray:
        if self._is_first_created:
            opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.print_user_options': 'yes'}
            self._caculate_cost_and_conditions()
            self._build_numeric_problem()

            self.solver = ca.nlpsol('solver', 'ipopt', self._nlp, opts)
            self._is_first_created = False
        
        # 添加p的参数
        p_args = []
        
        for a in list(args):
            p_args.extend(self._converto_list(a))
        
        for key in sorted(kwargs.keys()):
            p_args.extend(self._converto_list(kwargs[key]))
        
        p_args = np.array(p_args, dtype=np.float32)
        return self.solver(**self._num_prob, p=p_args)

    @ abstractmethod
    def _build_numeric_problem(self):
        '''
        构建数值问题, 从ca的符号变量转化到数值变量
        '''
    
    @ abstractmethod
    def _caculate_cost_and_conditions(self):
        '''
        计算代价和约束条件的模块
        '''

    @ abstractmethod
    def _define_state_update_equation(self):
        '''
        定义状态转移方程用
        '''

    def _converto_dm(self, args) -> ca.DM:
        if isinstance(args, ndarray) or isinstance(args, list):
            return ca.DM(args)
        elif isinstance(args, float) or isinstance(args, int) or isinstance(args, bool):
            return ca.DM([args])
        return args
    
    def _converto_list(self, args) -> List:
        if isinstance(args, ndarray):
            return args.tolist()
        if isinstance(args, int) or isinstance(args, float) or isinstance(args, bool):
            return [args]
        return args
    
    def _get_stats(self) -> Tuple:
        stats: Dict = self.solver.stats()
        return {'success': stats.get('success'), 'return_status': stats.get('return_status')}
```

./metadrive_project/custom2_version2/ocp/mpc.py:
```
import casadi as ca
from casadi import DM, MX
import numpy as np
from numpy import ndarray
from typing import Tuple, Dict
from metadrive.custom2_version2.type import VehicleState
from metadrive.custom2_version2.ocp.config import MPConfig
from metadrive.custom2_version2.ocp.base import OCP

class MPC(OCP):
    def __init__(self, config: MPConfig, metadata: Dict):
        '''
        config: MPC的配置,
        metadata: 一些常量配置, 不应该在update中改变
        '''
        self.lf = metadata.get('lf', None)
        self.lr = metadata.get('lr', None)
        super(MPC, self).__init__(config, metadata)
    
    def __call__(self, x0, z_ref, u_prev) -> Tuple[ndarray, ndarray]:
        '''
        x0: 当前车辆的状态: [横坐标x, 纵坐标y, 车辆速度v, 车辆角度theta],
        z_ref: 车辆的跟踪轨迹: [参考速度v_ref, 参考角度theta_ref],
        u_prev: 上一次的控制值: [加速度a_prev, 转向角delta_prev]
        '''
        res: Dict = super(MPC, self).__call__(x0, z_ref, u_prev)
        return self._parse_result(res)

    def _build_numeric_problem(self):
        # 设定决策变量的初值, 和值的上下界
        w0 = np.zeros(self._nlp_metadata['w_dim']) if not hasattr(self, '_last_w') else self._last_w
        
        lbw = np.concatenate([
            np.tile([self.config.a_min, self.config.delta_min], self._nlp_metadata['u_dim'] // 2),
            -np.inf * np.ones(self._nlp_metadata['x_dim']),
        ])
        ubw = np.concatenate([
            np.tile([self.config.a_max, self.config.delta_max], self._nlp_metadata['u_dim'] // 2),
            np.inf * np.ones(self._nlp_metadata['x_dim']),
        ])

        # 设定约束条件的上下界
        lbg = np.zeros(self._nlp_metadata['g_dim'])
        ubg = np.zeros(self._nlp_metadata['g_dim'])

        self._num_prob = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg, }

    def _caculate_cost_and_conditions(self):
        # ------- 添加决策变量 ----- #
        U = MX.sym('U', self.config.nu * self.config.mu)
        X = MX.sym('X', self.config.nx * (self.config.np + 1))

        # -------- 添加常量 ------- #
        x0 = MX.sym('x0', self.config.nx)
        z_ref = MX.sym('z_ref', 2)
        u_prev = MX.sym('u_prev', self.config.nu)

        Q  = DM(np.diag([5, 50])) # error
        R  = DM(np.diag([1, 1]))  # cost
        Rd = DM(np.diag([0, 0]))  # delta
        
        g = list()
        cost = 0
        
        g.append(X[0: self.config.nx] - x0)
        u_prev_sym = u_prev
        for k in range(self.config.np):
            xk = X[k * self.config.nx: (k+1) * self.config.nx]
            
            if k < self.config.mu:
                uk = U[k * self.config.nu: (k+1) * self.config.nu]
            else:
                uk = U[(self.config.mu - 1) * self.config.nu: self.config.mu * self.config.nu]
            
            zk = xk[2::] # 取x的后两项
            
            # 状态更新
            xk_next = self._f(xk, uk)

            # 约束和代价更新
            g.append(X[(k+1) * self.config.nx: (k+2) * self.config.nx] - xk_next)
            
            zk_e = z_ref - zk
            cost += ca.mtimes([zk_e.T, Q, zk_e])
            cost += ca.mtimes([uk.T, R, uk])
            
            du = uk - u_prev_sym
            cost += ca.mtimes([du.T, Rd, du])
            
            # 控制变量更新
            u_prev_sym = uk
        
        self._nlp = \
        {
            'x': ca.vertcat(U, X),
            'f': cost,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(x0, z_ref, u_prev)
        }

        self._nlp_metadata = \
        {
            'w_dim': U.shape[0] + X.shape[0],
            'u_dim': U.shape[0],
            'x_dim': X.shape[0],
            'g_dim': (self.config.np + 1) * self.config.nx,
        }

    def _define_state_update_equation(self):
        x = MX.sym('x', self.config.nx)
        u = MX.sym('u', self.config.nu)
        
        # 状态更新方程
        beta = ca.atan(ca.tan(u[1]) * self.lr / (self.lf + self.lr))
        x_next = ca.vertcat(
            x[0] + x[2] * ca.cos(x[3] + beta) * self.config.Ts,
            x[1] + x[2] * ca.sin(x[3] + beta) * self.config.Ts,
            x[2] + u[0] * self.config.Ts,
            x[3] + x[2] / self.lr * ca.sin(beta) * self.config.Ts,
        )
        self._f = ca.Function('f', [x, u], [x_next])

    def _parse_result(self, res: Dict) -> Tuple[ndarray, ndarray, Dict]:
        '''
        Args:
        res: 求解器solver返回的结果
        Returns:
        元组, 第一项是u, 形状是(np, nu); 第二项是x, 形状是(np+1, nx).
        '''

        w: ndarray = res.get('x', None).full().flatten()
        u: ndarray = w[0: self._nlp_metadata['u_dim']].reshape(self.config.mu, self.config.nu)
        x: ndarray = w[self._nlp_metadata['u_dim']::].reshape(self.config.np + 1, self.config.nx)
        solve_info = self._get_stats()
        self._last_w = w
        return u, x, solve_info
```

./metadrive_project/custom2_version2/ocp/config.py:
```
from dataclasses import dataclass
from numpy import pi


@dataclass
class OCPconfig:
    np: int          = 3         # step的数量
    mu: int          = 3         # u的控制时域
    nx: int          = 4         # 状态的维度, [x, y, v, theta]
    nu: int          = 2         # 控制的维度, [a, delta]
    Ts: int          = 0.02 * 5  # 周期
    a_min: float     = -1        # a的最小控制值
    a_max: float     = 1         # a的最大控制值
    delta_min: float = -pi / 6   # delta的最小控制值
    delta_max: float =  pi / 6   # delta的最大控制值


@dataclass
class MPConfig(OCPconfig):
    ...


@dataclass
class CBFconfig(OCPconfig):
    N: int           = 20        # 假设当前地图最多50辆车
    info_dim: int    = 2         # 需要车辆的信息维度
    dist_min: int    = 1.5       # 需要保持的最小安全距离
    gamma: float     = -0.5      # cbf参数之一
```

./metadrive_project/custom2_version2/envs/single/worker.py:
```
from metadrive.custom2_version2.envs.utils import CloudpickleWrapper, Commond
from metadrive.custom2_version2.utils import set_random_seed
from metadrive.component.vehicle.default_vehicle import DefaultVehicle
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.base_class.base_object import BaseObject
from metadrive.custom2_version2.ocp import CBFconfig
from metadrive import MetaDriveEnv
from multiprocessing.connection import Connection
import traceback
from typing import Dict, Callable, Any, Tuple, cast
from numpy import ndarray
import numpy as np

class Worker:
    def __init__(
        self,
        par_remotes: Connection,
        sub_remotes: Connection,
        env_wrapper: CloudpickleWrapper,
        env_idx: int,
    ):
        self.par_remotes = par_remotes
        self.sub_remotes = sub_remotes
        self.par_remotes.close()
        self.env: MetaDriveEnv = env_wrapper.env() # 获取环境
        self.env_idx = env_idx
        self.running = False
        self.func_names: Dict[str, Callable] = {func_name: getattr(self, func_name) for func_name in dir(self) if not func_name.startswith('_')}
        self.cbf_config = CBFconfig()

    def step(self, action: ndarray) -> Tuple:
        obs, reward, terminated, truncated, step_info = self.env.step(action)
        done: bool = terminated or truncated
        reset_info = dict()
        step_info['TimeLimit.truncated'] = truncated and not terminated
        return obs, reward, done, step_info, reset_info
    
    def reset(self) -> Tuple:
        # set_random_seed(42 * (self.env_idx + 1))
        obs, reset_info = self.env.reset()
        return obs, reset_info
    
    def render(self, mode, screen_record, window, screen_size, camera_position, text) -> ndarray:
        return self.env.render(mode=mode, screen_record=screen_record, window=window, screen_size=screen_size, camera_position=camera_position, text=text)
    
    def get_state(self) -> Dict:
        agent: DefaultVehicle = self.env.agent
        state: Dict = agent.get_state()
        return state
    
    def get_metadata(self) -> Dict:
        agent: DefaultVehicle = self.env.agent
        return dict(
            l  = agent.FRONT_WHEELBASE + agent.REAR_WHEELBASE,
            lf = agent.FRONT_WHEELBASE,
            lr = agent.REAR_WHEELBASE,
        )
    
    def get_all_vehicle_position(self) -> Tuple[ndarray, ndarray]:
        # help function
        def _filter_other_object(obj: BaseObject):
            if not isinstance(obj, BaseVehicle):
                return False
            obj = cast(BaseVehicle, obj)
            if obj.id == self.env.agent.id:
                return False
            return True
        
        info = np.zeros([self.cbf_config.N, self.cbf_config.info_dim])
        mask = np.zeros([self.cbf_config.N, ])
        for idx, (oid, obj) in enumerate(self.env.engine.get_objects(filter=_filter_other_object).items()):
            obj = cast(BaseVehicle, obj)
            info[idx, 0: 2] = obj.position
            mask[idx] = 1
        return info, mask
    
    def close(self) -> str:
        self.env.close()
        self.running = False
        return 'close successful!'
    
    def run(self):
        self.par_remotes.close()
        self.running = True
        while self.running:
            try:
                cmd: Commond = self.sub_remotes.recv()
                
                if cmd.name == 'close': # exit条件
                    self._send(self.close())
                    break
                
                func: Callable = self.func_names.get(cmd.name, None)
                self._send(func(*cmd.args)) # 执行cmd
                
            except Exception:
                traceback.print_exc()
        self.sub_remotes.close()

    def _send(self, data: Any):
        self.sub_remotes.send(data)


# 
def worker_func(
    par_remotes: Connection,
    sub_remotes: Connection,
    env_wrapper: CloudpickleWrapper,
    env_idx: int,
):
    worker = Worker(par_remotes, sub_remotes, env_wrapper, env_idx)
    worker.run()
```

./metadrive_project/custom2_version2/envs/single/env.py:
```
from metadrive.custom2_version2.envs.utils import MetaProcess, Commond, CloudpickleWrapper
from metadrive.custom2_version2.envs.single.worker import worker_func
from metadrive.custom2_version2.base_config import ParallelEnvConfig
import multiprocessing as mp
from multiprocessing.context import BaseContext
from multiprocessing import Process
from typing import List, Tuple, Dict, cast
from numpy import ndarray
import numpy as np


class SingleEnv:
    def __init__(self, env, config: ParallelEnvConfig):
        self.config = config
        ctx: BaseContext = mp.get_context(config.start_method)
        self.waiting = False
        self.closed = False

        # 只创建一个进程
        par_remotes, sub_remotes = ctx.Pipe()
        process: Process = ctx.Process(target=worker_func, args=(par_remotes, sub_remotes, CloudpickleWrapper(env), 0), daemon=True)
        process.start()
        self.meta_process = MetaProcess(par_remotes, sub_remotes, process)
        sub_remotes.close()
    
    def step(self, action: ndarray) -> Tuple:
        self.step_async(action)
        return self.step_wait()
    
    def step_async(self, action: ndarray): # (action_dim, )
        self.meta_process.par_remotes.send(Commond(name='step', args=(action, )))
        self.waiting = True

    def step_wait(self) -> Tuple:
        obs, reward, done, step_info, self.reset_info = self.meta_process.par_remotes.recv()
        self.waiting = False
        return obs, reward, done, step_info
    
    def reset(self) -> Tuple:
        self.meta_process.par_remotes.send(Commond(name='reset'))
        obs, self.reset_info = self.meta_process.par_remotes.recv()
        return obs
    
    def render(self, mode, screen_record, window, screen_size, camera_position, text):
        self.meta_process.par_remotes.send(Commond(name='render', args=(mode, screen_record, window, screen_size, camera_position, text)))
        return self.meta_process.par_remotes.recv()
    
    def get_state(self) -> Dict:
        self.meta_process.par_remotes.send(Commond(name='get_state'))
        return self.meta_process.par_remotes.recv()

    def get_metadata(self) -> Dict:
        self.meta_process.par_remotes.send(Commond(name='get_metadata'))
        return self.meta_process.par_remotes.recv()

    def get_all_vehicle_position(self) -> Tuple[ndarray, ndarray]:
        self.meta_process.par_remotes.send(Commond(name='get_all_vehicle_position'))
        info, mask = self.meta_process.par_remotes.recv()
        info = cast(ndarray, info); mask = cast(ndarray, mask)
        return info, mask
    
    def close(self):
        if self.closed:
            return 

        if self.waiting:
            self.meta_process.par_remotes.recv()
        
        self.meta_process.par_remotes.send(Commond(name='close'))
        self.meta_process.process.join()
        self.closed = True
```

./metadrive_project/custom2_version2/envs/__init__.py:
```
from .parallel.env import ParallelEnv
from .single.env import SingleEnv
```

./metadrive_project/custom2_version2/envs/utils.py:
```
import cloudpickle
from metadrive import MetaDriveEnv
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing import Process
from typing import Tuple


class CloudpickleWrapper:
    def __init__(self, env: MetaDriveEnv):
        self.env = env
    
    def __getstate__(self) -> MetaDriveEnv:
        return cloudpickle.dumps(self.env)

    def __setstate__(self, env: MetaDriveEnv) -> None:
        self.env = cloudpickle.loads(env)


@dataclass
class MetaProcess:
    par_remotes: Connection
    sub_remotes: Connection
    process: Process


@dataclass
class Commond:
    name: str
    args: Tuple = field(default_factory=tuple)
```

./metadrive_project/custom2_version2/envs/parallel/worker.py:
```
from metadrive.custom2_version2.envs.utils import CloudpickleWrapper, Commond
from metadrive.custom2_version2.utils import set_random_seed
from metadrive.component.vehicle.default_vehicle import DefaultVehicle
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.base_class.base_object import BaseObject
from metadrive.custom2_version2.ocp import CBFconfig
from metadrive import MetaDriveEnv
from multiprocessing.connection import Connection
import traceback
from typing import Dict, Callable, Any, Tuple, cast
from numpy import ndarray
import numpy as np


class Worker:
    def __init__(
        self,
        par_remotes: Connection,
        sub_remotes: Connection,
        env_wrapper: CloudpickleWrapper,
        env_idx: int,
    ):
        self.par_remotes = par_remotes
        self.sub_remotes = sub_remotes
        self.par_remotes.close()
        self.env: MetaDriveEnv = env_wrapper.env() # 获取环境
        self.env_idx = env_idx
        self.running = False
        self.func_names: Dict[str, Callable] = {func_name: getattr(self, func_name) for func_name in dir(self) if not func_name.startswith('_')}
        self.cbf_config = CBFconfig()

    def step(self, action: ndarray) -> Tuple:
        obs, reward, terminated, truncated, step_info = self.env.step(action)
        done: bool = terminated or truncated
        reset_info = dict()
        step_info['TimeLimit.truncated'] = truncated and not terminated
        if done:
            step_info['terminal_observation'] = obs
            obs, reset_info = self.env.reset()
        return obs, reward, done, step_info, reset_info
    
    def reset(self) -> Tuple:
        # set_random_seed(42 * (self.env_idx + 1))
        obs, reset_info = self.env.reset()
        return obs, reset_info
    
    def render(self, mode, screen_record, window, screen_size, camera_position) -> ndarray:
        return self.env.render(mode, screen_record, window, screen_size, camera_position)
    
    def get_state(self) -> Dict:
        agent: DefaultVehicle = self.env.agent
        state: Dict = agent.get_state()
        return state
    
    def get_metadata(self) -> Dict:
        agent: DefaultVehicle = self.env.agent
        return dict(
            l  = agent.FRONT_WHEELBASE + agent.REAR_WHEELBASE,
            lf = agent.FRONT_WHEELBASE,
            lr = agent.REAR_WHEELBASE,
        )
    
    def get_all_vehicle_position(self) -> Tuple[ndarray, ndarray]:
        # help function
        def _filter_other_object(obj: BaseObject):
            if not isinstance(obj, BaseVehicle):
                return False
            obj = cast(BaseVehicle, obj)
            if obj.id == self.env.agent.id:
                return False
            return True
        
        info = np.zeros([self.cbf_config.N, self.cbf_config.info_dim])
        mask = np.zeros([self.cbf_config.N, ])
        for idx, (oid, obj) in enumerate(self.env.engine.get_objects(filter=_filter_other_object).items()):
            obj = cast(BaseVehicle, obj)
            info[idx, 0: 2] = obj.position
            mask[idx] = 1
        return info, mask
    
    def close(self) -> str:
        self.env.close()
        self.running = False
        return 'close successful!'
    
    def run(self):
        self.par_remotes.close()
        self.running = True
        while self.running:
            try:
                cmd: Commond = self.sub_remotes.recv()
                
                if cmd.name == 'close': # exit条件
                    self._send(self.close())
                    break
                
                func: Callable = self.func_names.get(cmd.name, None)
                self._send(func(*cmd.args)) # 执行cmd
                
            except Exception:
                traceback.print_exc()
        self.sub_remotes.close()

    def _send(self, data: Any):
        self.sub_remotes.send(data)


# 
def worker_func(
    par_remotes: Connection,
    sub_remotes: Connection,
    env_wrapper: CloudpickleWrapper,
    env_idx: int,
):
    worker = Worker(par_remotes, sub_remotes, env_wrapper, env_idx)
    worker.run()
```

./metadrive_project/custom2_version2/envs/parallel/env.py:
```
from metadrive.custom2_version2.envs.utils import MetaProcess, Commond, CloudpickleWrapper
from metadrive.custom2_version2.envs.parallel.worker import worker_func
from metadrive.custom2_version2.base_config import ParallelEnvConfig
import multiprocessing as mp
from multiprocessing.context import BaseContext
from multiprocessing import Process
from typing import List, Tuple, Dict, cast
from numpy import ndarray
import numpy as np


class ParallelEnv:
    def __init__(self, envs: List, config: ParallelEnvConfig):
        self.config = config
        ctx: BaseContext = mp.get_context(config.start_method)
        self.meta_processs: List[MetaProcess] = []
        self.waiting = False
        self.closed = False
        self.reset_infos = list()

        for idx in range(self.config.n_process):
            par_remotes, sub_remotes = ctx.Pipe()
            process: Process = ctx.Process(target=worker_func, args=(par_remotes, sub_remotes, CloudpickleWrapper(envs[idx]), idx), daemon=True)
            process.start()
            meta_process = MetaProcess(par_remotes, sub_remotes, process)
            self.meta_processs.append(meta_process)
            sub_remotes.close()
    
    def step(self, actions: ndarray) -> Tuple:
        self.step_async(actions)
        return self.step_wait()
    
    def step_async(self, actions: ndarray): # (env_idx, action_dim)
        for meta_process, action in zip(self.meta_processs, actions):
            meta_process.par_remotes.send(Commond(name='step', args=(action, )))
        self.waiting = True

    def step_wait(self) -> Tuple:
        obss = np.empty([self.config.n_process, self.config.state_dim], dtype=np.float32)
        rewards = np.empty([self.config.n_process], dtype=np.float32)
        dones = np.empty([self.config.n_process], dtype=np.long)
        step_infos = list()
        self.reset_infos = list()

        for idx, meta_process in enumerate(self.meta_processs):
            obs, reward, done, step_info, reset_info = meta_process.par_remotes.recv()
            obss[idx, :]    = obs
            rewards[idx] = reward
            dones[idx]   = done
            step_infos.append(step_info)
            self.reset_infos.append(reset_info)
        self.waiting = False
        return obss, rewards, dones, step_infos
    
    def reset(self) -> Tuple:
        obss = np.empty([self.config.n_process, self.config.state_dim], dtype=np.float32)
        self.reset_infos = list()

        for meta_process in self.meta_processs:
            meta_process.par_remotes.send(Commond(name='reset'))
        
        for idx, meta_process in enumerate(self.meta_processs):
            obs, reset_info = meta_process.par_remotes.recv()
            obss[idx, :] = obs
            self.reset_infos.append(reset_info)
        return obss
    
    def get_state(self) -> List[Dict]:
        vehicle_states = list()
        for meta_process in self.meta_processs:
            meta_process.par_remotes.send(Commond(name='get_state'))
        
        for meta_process in self.meta_processs:
            state = meta_process.par_remotes.recv()
            vehicle_states.append(state)
        return vehicle_states

    def get_metadata(self) -> Dict:
        meta_process = self.meta_processs[0]
        meta_process.par_remotes.send(Commond(name='get_metadata'))
        return meta_process.par_remotes.recv()
    
    def get_all_vehicle_position(self) -> Tuple[List[ndarray], List[ndarray]]:
        infos = list()
        masks = list()
        for meta_process in self.meta_processs:
            meta_process.par_remotes.send(Commond(name='get_all_vehicle_position'))

        for meta_process in self.meta_processs:
            info, mask = meta_process.par_remotes.recv()
            info = cast(ndarray, info); mask = cast(ndarray, mask)
            infos.append(info.flatten()); masks.append(mask)
        return infos, masks

    def close(self):
        if self.closed:
            return 

        if self.waiting:
            for meta_process in self.meta_processs:
                meta_process.par_remotes.recv()
        
        for meta_process in self.meta_processs:
            meta_process.par_remotes.send(Commond(name='close'))
        
        for meta_process in self.meta_processs:
            meta_process.process.join()
        
        self.closed = True
```

./metadrive_project/custom2_version2/type.py:
```
from dataclasses import dataclass, field
from metadrive.utils.doc_utils import generate_gif
from typing import Optional, List
from numpy import ndarray
import numpy as np

@dataclass
class ActionType:
    discrete: int       = 0
    multi_discrete: int = 1
    box: int            = 2

@dataclass
class ActionSpace:
    action_type: str = ActionType.box

@dataclass
class VehicleState:
    # --------state-------- #
    x: Optional[float]     = None  # x坐标
    y: Optional[float]     = None  # y坐标
    v: Optional[float]     = None  # 速度, [标量]
    theta: Optional[float] = None  # 朝向
    # -----------u--------- #
    a: Optional[float]     = None  # 加速度
    delta: Optional[float] = None  # 转向角
    # # ---------utils------- #
    # l: Optional[float]     = None  # 轴距


@dataclass
class RenderClass:
    frames: List[ndarray]  = field(default_factory=list)
    render_index: int      = 0

    def add_frame(self, frame: ndarray):
        self.frames.append(frame)
        self.render_index += 1

    def reset(self):
        self.frames        = list()
        self.render_index  = 0

    def generate_gif(self, gif_name: str):
        generate_gif(self.frames, gif_name=gif_name)
        self.reset()
```


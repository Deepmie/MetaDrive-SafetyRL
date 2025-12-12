from dataclasses import dataclass, field
from typing import Optional, Dict
from numpy import ndarray, pi
from metadrive.customs.ocp import MPConfig, CBFconfig
import torch


STATE_DIM        = 259
ACTION_DIM       = 2
TORCH_DTYPE      = torch.float32
TORCH_DEVICE     = torch.device(device='cuda:0') if torch.cuda.is_available() else torch.device(device='cpu')
BATCH_SIZE       = 128
MAX_SAMPLE_STEPS = int(1e6)
MAX_EPS_COUNTS   = 500
UPDATE_FREQ      = 10
EVALUATE_FREQ    = 2
GAMMA            = 0.995
LAMBDA           = 0.95
USE_SWANLAB      = True


def get_metadrive_env_config() -> Dict:
    return dict(
        use_render = False,
        traffic_density = 0.1,
        traffic_mode = 'respawn',
        map = 'O',
        use_lateral_reward = True,
        vehicle_config = dict(
            vehicle_model = 'with_solver',
            spawn_lane_index = ('>>', '>>>', 0),
        ),
    )


@dataclass
class EnvConfig:
    name: str        = 'dpenv test'
    state_dim: int   = STATE_DIM
    action_dim: int  = ACTION_DIM
    metadrive_env_config: Dict = field(default_factory=get_metadrive_env_config)


@dataclass
class RolloutBufferConfig:
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    batch_size: int = BATCH_SIZE
    device: torch.device = TORCH_DEVICE
    max_buffer_size: int = UPDATE_FREQ
    gamma: float = GAMMA  # 计算优势函数的参数之一
    lamb: float = LAMBDA  # 计算优势函数的参数之一


@dataclass
class PolicyConfig:
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    action_std_init: float = 0.1
    hidden_dim: int  = 512
    dtype: torch.dtype = TORCH_DTYPE
    device: torch.device = TORCH_DEVICE
    mpc_config: MPConfig = field(default_factory=MPConfig)
    cbf_config: CBFconfig = field(default_factory=CBFconfig)
    v_min: float     = 0.0
    v_max: float     = 12.0
    theta_min: float = -1
    theta_max: float = 1



@dataclass
class SolverConfig:
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)  # policy的配置
    mpc_config: MPConfig        = field(default_factory=MPConfig)      # mpc的配置
    cbf_config: CBFconfig       = field(default_factory=CBFconfig)     # cbf的配置
    env_config: EnvConfig       = field(default_factory = EnvConfig)
    policy_config: PolicyConfig = field(default_factory = PolicyConfig)
    buffer_config: RolloutBufferConfig = field(default_factory=RolloutBufferConfig)
    epoch: int                  = 1000
    epsilon: float              = 0.15
    entropy_coef: float         = 0.01
    value_loss_coef: float      = 0.5
    bc_coef: float              = 1.0
    learning_rate: float        = 3e-4
    learning_rate_actor: float  = 3e-4
    learning_rate_critic: float = 1e-3
    update_freq: int            = UPDATE_FREQ       # 经过多少步才更新
    delta_bc: float             = 0.1
    use_swanlab: bool           = USE_SWANLAB         # 是否启用wandb
    device: torch.device        = TORCH_DEVICE


@dataclass
class MainConfig:
    env_config: EnvConfig   = field(default_factory = EnvConfig)
    buffer_config: RolloutBufferConfig = field(default_factory = RolloutBufferConfig)
    solver_config: SolverConfig = field(default_factory=SolverConfig)
    max_sample_steps: int   = MAX_SAMPLE_STEPS  # 总的期望采样长度
    max_eps_counts: int     = MAX_EPS_COUNTS    # 一个回合最大的采样数量
    update_freq: int        = UPDATE_FREQ       # 经过多少步才更新
    evaluate_freq: int      = EVALUATE_FREQ     # 经过多少步才评估
    max_evaluate_steps: int = MAX_EPS_COUNTS    # 一个回合最大的评估步骤
    use_swanlab: bool       = USE_SWANLAB         # 是否启用wandb


from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from numpy import ndarray, pi
from metadrive.customs_parallel.ocp import MPConfig, CBFconfig
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT

import torch
import logging

STATE_DIM             = 259
ACTION_DIM            = 2
TORCH_DTYPE           = torch.float32
TORCH_DEVICE          = torch.device(device='cuda:0') if torch.cuda.is_available() else torch.device(device='cpu')
BATCH_SIZE            = 64
MAX_SAMPLE_STEPS      = int(1e6)
MAX_EPS_COUNTS        = 1500
SAMPLE_SUBPROCESS_NUM = 13
GPU_NUM               = 4

TOTAL_FREQ            = int(30000)
SINGLE_UPDATE_FREQ    = int(TOTAL_FREQ // SAMPLE_SUBPROCESS_NUM)
UPDATE_FREQ           = int(SINGLE_UPDATE_FREQ * SAMPLE_SUBPROCESS_NUM)
EVALUATE_UPDATE_RATIO = 1
GAMMA                 = 0.99
LAMBDA                = 0.95
USE_SWANLAB           = False


def get_metadrive_env_config() -> Dict:
    return dict(
        num_scenarios      = 1,
        start_seed         = 0,
        use_render         = False,
        traffic_density    = 0.0,
        traffic_mode       = 'respawn',
        map                = 'O',
        
        discrete_action=True,
        discrete_throttle_dim=3,
        discrete_steering_dim=3,
        
        random_spawn_lane_index=False,
        agent_configs  = {
            DEFAULT_AGENT: dict(
                use_special_color=True,
                spawn_lane_index = ('>>', '>>>', 1),
            )
        },
        vehicle_config = dict(
            vehicle_model = 'with_solver_parallel',
            spawn_lane_index = ('>>', '>>>', 1),
            spawn_longitude = -5.0,
        ),
        use_lateral_reward = True,
        on_continuous_line_done = False,
        log_level = logging.WARNING,
    )


@dataclass
class EnvConfig:
    name: str        = 'dpenv test'
    seed: int        = 42
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
    sample_subprocess_num: int = SAMPLE_SUBPROCESS_NUM
    single_buffer_size: int    = SINGLE_UPDATE_FREQ


@dataclass
class PolicyConfig:
    state_dim: int              = STATE_DIM
    action_dim: int             = ACTION_DIM
    std_config: Tuple           = (True, -20.0, 2.0) # (是否启用clamp, min, max)
    hidden_dim: int             = 512
    dtype: torch.dtype          = TORCH_DTYPE
    device: torch.device        = TORCH_DEVICE
    mpc_config: MPConfig        = field(default_factory=MPConfig)
    cbf_config: CBFconfig       = field(default_factory=CBFconfig)
    v_min: float                = 0.5
    v_max: float                = 12.0
    theta_min: float            = -1
    theta_max: float            = 1
    learning_rate: float        = 3e-4
    learning_rate_actor: float  = 3e-4
    learning_rate_critic: float = 1e-3



@dataclass
class SolverConfig:
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)  # policy的配置
    mpc_config: MPConfig        = field(default_factory=MPConfig)      # mpc的配置
    cbf_config: CBFconfig       = field(default_factory=CBFconfig)     # cbf的配置
    env_config: EnvConfig       = field(default_factory = EnvConfig)
    policy_config: PolicyConfig = field(default_factory = PolicyConfig)
    buffer_config: RolloutBufferConfig = field(default_factory=RolloutBufferConfig)
    epoch: int                  = 10
    epsilon: float              = 0.15
    entropy_coef: float         = 0.01
    value_loss_coef: float      = 0.5
    bc_coef: float              = 1.0
    learning_rate: float        = 3e-4
    learning_rate_actor: float  = 3e-4
    learning_rate_critic: float = 3e-4
    update_freq: int            = UPDATE_FREQ       # 经过多少步才更新
    delta_bc: float             = 10.0
    use_swanlab: bool           = USE_SWANLAB         # 是否启用wandb
    device: torch.device        = TORCH_DEVICE

@dataclass
class TrainerConfig:
    epoch: int                  = 200
    delta_bc: float             = 10.0
    epsilon: float              = 0.15
    bc_coef: float              = 1.0
    value_loss_coef: float      = 0.5
    entropy_coef: float         = 0.01
    gpu_num: int                = GPU_NUM
    max_grad_num: float         = 0.5
    use_swanlab: bool           = USE_SWANLAB         # 是否启用wandb


@dataclass
class MainConfig:
    env_config: EnvConfig   = field(default_factory = EnvConfig)
    buffer_config: RolloutBufferConfig = field(default_factory = RolloutBufferConfig)
    solver_config: SolverConfig  = field(default_factory=SolverConfig)
    trainer_config: TrainerConfig  = field(default_factory=TrainerConfig)
    mpc_config: MPConfig        = field(default_factory=MPConfig)
    cbf_config: CBFconfig       = field(default_factory=CBFconfig)

    max_sample_steps: int        = MAX_SAMPLE_STEPS       # 总的期望采样长度
    max_eps_counts: int          = MAX_EPS_COUNTS         # 一个回合最大的采样数量
    update_freq: int             = UPDATE_FREQ            # 经过多少步才更新
    evaluate_update_ratio: int   = EVALUATE_UPDATE_RATIO  # 经过多少步才评估
    max_evaluate_steps: int      = MAX_EPS_COUNTS         # 一个回合最大的评估步骤
    sample_subprocess_num: int   = SAMPLE_SUBPROCESS_NUM  # 进程的总数量
    single_update_freq: int      = SINGLE_UPDATE_FREQ     # 单个进程更新的步数
    gpu_num: int                 = GPU_NUM                # GPU的数量
    use_swanlab: bool            = USE_SWANLAB            # 是否启用wandb



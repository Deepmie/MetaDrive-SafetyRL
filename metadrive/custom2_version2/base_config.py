from dataclasses import dataclass, field
import numpy as np
import torch
from torch.nn import Module, Tanh
from numpy import ndarray
from typing import Dict, Optional, Tuple
from metadrive.custom2_version2.type import ActionType

METHOD_NAME          = 'rl_mpc_cbf_ppc_traj'
TEST_MODE            = False
STATE_DIM            = 259
ACTION_DIM           = 4
TORCH_DTYPE          = torch.float32
TORCH_DEVICE         = torch.device(device='cuda:0') if torch.cuda.is_available() else torch.device(device='cpu')
# TORCH_DEVICE         = torch.device(device='cpu')
MAX_EPS_COUNTS       = 100
N_PROCESS            = 4
GAE_LAMBDA           = 0.95
UPDATE_FREQ          = 1
MONITOR_SAVE_FREQ    = 5000 if not TEST_MODE else 100
IS_LOAD              = False

# Paper Param:
GAMMA                = 0.99
LEARNING_RATE        = 3e-4
DELTA_BC             = 0.01
EPOCH                = 20
# MAX_BUFFER_SIZE  = 4096 if not TEST_MODE else 256
MAX_BUFFER_SIZE      = 4096    if not TEST_MODE else 256
BATCH_SIZE           = 64
EVALUATE_TOTAL_STEPS = 2000   # 一轮评估的最大步长
EVALUATE_STEPS       = 20000  if not TEST_MODE else 50
TOTAL_STEPS          = 2000000 if not TEST_MODE else 9000


@dataclass
class EnvConfig:
    name: str        = 'random test'
    state_dim: int   = STATE_DIM
    action_dim: int  = ACTION_DIM


@dataclass
class MetaDriveEnvConfig:
    map: str                      = 'O'        # 地图形状
    traffic_density: float        = 0.2        # 交通状况
    # 交通模式, option: trigger, respawn, hybrid, basic
    traffic_mode: str             = 'respawn'  # 
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
    delta_bc: float      = DELTA_BC
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
    filter_num: int        = 5
    ppc_coef: float        = 0.2

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

    method_name: str                 = METHOD_NAME
    n_process: int                   = N_PROCESS
    sample_steps: int                = MAX_BUFFER_SIZE   # 一次的采样长度
    total_steps: int                 = TOTAL_STEPS       # 总的期望采样长度
    max_eps_counts: int              = MAX_EPS_COUNTS    # 一个回合最大的采样数量
    update_freq: int                 = UPDATE_FREQ       # 经过多少步才更新
    monitor_save_freq: int           = MONITOR_SAVE_FREQ # 经过多少步更新monitor的权重
    epoch: int                       = EPOCH
    batch_size: int                  = BATCH_SIZE
    epsilon: float                   = 0.2
    entropy_coef: float              = 0.0
    bc_coef: float                   = 1.0
    value_loss_coef: float           = 0.5
    lambda_coef: float               = 0.2
    learning_rate: float             = 3e-4
    max_grad_norm: float             = 0.5
    action_space_range: Tuple        = (-1, 1)

    # parameters' range
    v_min: float                     = 0.0
    v_max: float                     = 15.0
    theta_min: float                 = -np.pi / 3
    theta_max: float                 =  np.pi / 3
    alpha_min: float                 = 0.8
    alpha_max: float                 = 1.0
    
    update_freq: int                 = UPDATE_FREQ       # 经过多少步才更新
    target_kl: Optional[float]       = None
    evaluate_steps: int              = EVALUATE_STEPS
    evaluate_total_steps: int        = EVALUATE_TOTAL_STEPS
    is_load: bool                    = IS_LOAD
    delta_bc: float                  = DELTA_BC
    logger_save_root: str            = 'dp_single_version2/logger'
    policy_checkpoint_pth: str       = 'policy.pth'
    best_policy_checkpoint_pth: str  = 'policy_best.pth'
    device: torch.device             = TORCH_DEVICE



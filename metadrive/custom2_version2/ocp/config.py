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
    N: int           = 50        # 假设当前地图最多50辆车
    info_dim: int    = 2         # 需要车辆的信息维度
    dist_min: int    = 0.5       # 需要保持的最小安全距离
    gamma: float     = 0.1       # cbf参数之一
from dataclasses import dataclass, field, asdict
from numpy import pi
from typing import Dict, cast
from metadrive.component.vehicle.default_vehicle import DefaultVehicle

VEHICLE_LENGTH = 4.515

@dataclass
class IpoptConfig:
    print_level: int                  = 0
    sb: str                           = 'yes'
    warm_start_init_point: str        = 'yes'
    warm_start_bound_push: float      = 1e-6
    warm_start_mult_bound_push: float = 1e-6
    mu_init: float                    = 1e-2
    tol: float                        = 1e-4
    acceptable_tol: float             = 1e-3
    max_iter: int                     = 50
    acceptable_iter: int              = 5
    linear_solver: str                = 'mumps'  # 可以尝试'ma27', 'ma57', 'ma86'


@dataclass
class OptimConfig:
    ipopt: IpoptConfig = field(default_factory=IpoptConfig)
    print_time: int    = 0

    def asdict(self) -> Dict:
        new_res: Dict = dict()
        res = asdict(self)
        for key, value in res.items():
            if isinstance(value, Dict):
                value = cast(Dict, value)
                for sub_key, sub_value in value.items():
                    new_res[f'{key}.{sub_key}'] = sub_value
            else:
                new_res[key] = value
        return new_res


@dataclass
class OCPconfig:
    np: int                   = 3         # step的数量
    mu: int                   = 3         # u的控制时域
    nx: int                   = 4         # 状态的维度, [x, y, v, theta]
    nu: int                   = 2         # 控制的维度, [a, delta]
    Ts: int                   = 0.02 * 5  # 周期
    a_min: float              = -1        # a的最小控制值
    a_max: float              = 1         # a的最大控制值
    delta_min: float          = -pi / 6   # delta的最小控制值
    delta_max: float          =  pi / 6   # delta的最大控制值
    optim_config: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class MPConfig(OCPconfig):
    delta_L: float = 0.2
    delta_R: float = 0.2
    alpha: float   = 0.2
    p_0: float     = 3
    p_inf: float   = 0.1
    lota: float    = 0.1


@dataclass
class CBFconfig(OCPconfig):
    N: int           = 20                    # 假设当前地图最多20辆车
    info_dim: int    = 3                     # 需要车辆的信息维度
    dist_min: int    = 0.5                   # 需要保持的最小安全距离
    gamma: float     = -0.5                  # cbf参数之一
    filter_num: int  = 5                     # 只取得前5辆车考虑


if __name__ == '__main__':
    oc = OptimConfig()
    print(asdict(oc))
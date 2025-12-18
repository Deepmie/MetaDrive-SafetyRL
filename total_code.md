./metadrive_project/ocp/cbf.py:
```
from metadrive.custom2_version2.ocp.config import CBFconfig
from metadrive.custom2_version2.ocp.base import OCP
from typing import Dict, Tuple
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

./metadrive_project/ocp/__init__.py:
```
from .cbf import CBF
from .mpc import MPC
from .config import CBFconfig, MPConfig
```

./metadrive_project/ocp/base.py:
```
from metadrive.custom2_version2.ocp.config import OCPconfig
from abc import ABC, abstractmethod
import casadi as ca
from numpy import ndarray
from typing import List, Dict, Tuple
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

./metadrive_project/ocp/mpc.py:
```
import casadi as ca
from casadi import DM, MX
import numpy as np
from numpy import ndarray
from typing import Tuple, Dict
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

./metadrive_project/ocp/config.py:
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


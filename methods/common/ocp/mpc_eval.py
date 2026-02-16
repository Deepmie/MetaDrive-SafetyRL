import casadi as ca
from casadi import DM, MX
import numpy as np
from numpy import ndarray
from typing import Tuple, Dict, cast
from methods.common.ocp.config import MPConfig
from methods.common.ocp.base import OCP

class MPCEval(OCP):
    def __init__(self, config: MPConfig, metadata: Dict):
        '''
        config: MPC的配置,
        metadata: 一些常量配置, 不应该在update中改变
        '''
        self.lf = metadata.get('lf', None)
        self.lr = metadata.get('lr', None)
        super(MPCEval, self).__init__(config, metadata)
        self.config: MPConfig = cast(MPConfig, self.config)

    def __call__(self, x0, z_ref, u_prev) -> Tuple[ndarray, ndarray]:
        '''
        x0: 当前车辆的状态: [横坐标x, 纵坐标y, 车辆速度v, 车辆角度theta],
        z_ref: 车辆的跟踪轨迹: [参考速度v_ref, 参考角度theta_ref],
        u_prev: 上一次的控制值: [加速度a_prev, 转向角delta_prev],
        '''
        res: Dict = super(MPCEval, self).__call__(x0, z_ref, u_prev)
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
        x0        = MX.sym('x0', self.config.nx)
        z_ref     = MX.sym('z_ref', 2)
        u_prev    = MX.sym('u_prev', self.config.nu)

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

            if k >= 1:
                if k == 1:
                    zk_e = zk - z_ref
                else:
                    zk_e = zk - zk_last
                cost += ca.mtimes([zk_e.T, Q, zk_e])
            
            cost += ca.mtimes([uk.T, R, uk])
            
            du = uk - u_prev_sym
            cost += ca.mtimes([du.T, Rd, du])
            
            # 控制变量更新
            u_prev_sym = uk
            zk_last    = zk
        
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
        return u, x, solve_info

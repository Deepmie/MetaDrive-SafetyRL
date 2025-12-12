import casadi as ca
from casadi import DM
import numpy as np
from numpy import ndarray
from typing import Tuple, Dict
from metadrive.customs.type import AgentInfo
from metadrive.customs.ocp.config import MPConfig
from metadrive.customs.ocp.base import OCP

class MPC(OCP):
    def __init__(self, config: MPConfig, agent_info: AgentInfo):
        super(MPC, self).__init__(config, agent_info)
    
    def __call__(self, x0, z_ref) -> Tuple[ndarray, ndarray]:
        res: Dict = super(MPC, self).__call__(x0, z_ref)
        return self._parse_result(res)

    def _build_numeric_problem(self):
        # 设定决策变量的初值, 和值的上下界
        w0 = np.zeros(self._nlp_metadata['w_dim'])
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
        U = ca.MX.sym('U', self.config.nu * self.config.np)
        X = ca.MX.sym('X', self.config.nx * (self.config.np + 1))

        # -------- 添加常量 ------- #
        x0 = ca.MX.sym('x0', self.config.nx)
        z_ref = ca.MX.sym('z_ref', 2)
        wz = 100
        wu = 0.01

        g = list()
        cost = 0

        g.append(X[0: self.config.nx] - x0)
        for k in range(self.config.np):
            xk = X[k * self.config.nx: (k+1) * self.config.nx]
            uk = U[k * self.config.nu: (k+1) * self.config.nu]
            zk = xk[2::] # 取x的后两项

            # 状态更新
            xk_next = self._f(xk, uk)

            # 约束和代价更新
            g.append(X[(k+1) * self.config.nx: (k+2) * self.config.nx] - xk_next)
            cost += wz * ca.sumsqr(z_ref - zk) + wu * ca.sumsqr(uk)

        self._nlp = \
        {
            'x': ca.vertcat(U, X),
            'f': cost,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(x0, z_ref)
        }

        self._nlp_metadata = \
        {
            'w_dim': U.shape[0] + X.shape[0],
            'u_dim': U.shape[0],
            'x_dim': X.shape[0],
            'g_dim': (self.config.np + 1) * self.config.nx,
        }

    def _define_state_update_equation(self):
        x = ca.MX.sym('x', self.config.nx)
        u = ca.MX.sym('u', self.config.nu)

        # 状态更新方程
        beta = ca.atan(1 / 2 * ca.tan(u[1]))
        x_next = ca.vertcat(
            x[0] + (x[2] * ca.cos(x[3] + beta)) * self.config.Ts,
            x[1] + (x[2] * ca.sin(x[3] + beta)) * self.config.Ts,
            x[2] + u[0] * self.config.Ts,
            x[3] + x[2] / self.agent_info.l * ca.sin(beta) * self.config.Ts,
        )
        self._f = ca.Function('f', [x, u], [x_next])

    def _parse_result(self, res: Dict) -> Tuple[ndarray, ndarray]:
        '''
        解析求解器返回的结果.

        Args:
            res: 求解器solver返回的结果
        
        Returns:
            元组, 第一项是u, 形状是(np, nu); 第二项是x, 形状是(np+1, nx).
        '''

        w: ndarray = res.get('x', None).full().flatten()
        u: ndarray = w[0: self._nlp_metadata['u_dim']].reshape(self.config.np, self.config.nu)
        x: ndarray = w[self._nlp_metadata['u_dim']::].reshape(self.config.np + 1, self.config.nx)

        return (u, x)

        

        
        


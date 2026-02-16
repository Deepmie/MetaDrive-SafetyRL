import casadi as ca
from casadi import DM, MX
import numpy as np
from numpy import ndarray
from typing import Tuple, Dict, cast
from methods.common.ocp.config import MPConfig
from methods.common.ocp.base import OCP

class DefaultMPC(OCP):
    def __init__(self, config: MPConfig, metadata: Dict):
        '''
        config: MPC的配置,
        metadata: 一些常量配置, 不应该在update中改变
        '''
        self.lf = metadata.get('lf', None)
        self.lr = metadata.get('lr', None)
        super(DefaultMPC, self).__init__(config, metadata)
        self.config: MPConfig = cast(MPConfig, self.config)

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
        ...

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

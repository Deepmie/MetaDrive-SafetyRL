from metadrive.customs.ocp.base import OCP
from metadrive.customs.ocp.config import CBFconfig
from metadrive.customs.type import AgentInfo
from metadrive.engine.base_engine import BaseEngine
from typing import Dict, cast
import casadi as ca
import numpy as np
from numpy import ndarray

class CBF(OCP):
    def __init__(self, config: CBFconfig, agent_info: AgentInfo):
        super(CBF, self).__init__(config, agent_info)
        self.config = cast(CBFconfig, self.config)
    
    def __call__(self, x0, u0, mask, info):
        res: Dict = super(CBF, self).__call__(x0, u0, mask, info)
        return self._parse_result(res)
    
    def _build_numeric_problem(self):
        # 决策设定决策变量的初值, 和值的上下界
        w0 = np.zeros(self._nlp_metadata['w_dim'])
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
        cost = 0.5 * ca.sumsqr(DU)

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

        # 状态更新方程
        beta = ca.atan(1 / 2 * ca.tan(u[1] + du[1]))
        x_next = ca.vertcat(
            x[0] + (x[2] * ca.cos(x[3] + beta)) * self.config.Ts,
            x[1] + (x[2] * ca.sin(x[3] + beta)) * self.config.Ts,
            x[2] + (u[0] + du[0]) * self.config.Ts,
            x[3] + x[2] / self.agent_info.l * ca.sin(beta) * self.config.Ts,
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


    def _parse_result(self, res: Dict):
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

        return (u, du, x)
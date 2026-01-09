from metadrive.custom2_version2.ocp.config import CBFconfig
from metadrive.custom2_version2.ocp.base import OCP
from metadrive.custom2_version2.ocp.cbf_func import CBFunctionsCasadi
from typing import Dict, Tuple, cast
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
        self.config = cast(CBFconfig, self.config)
        self.cbf_functions = CBFunctionsCasadi(self.config)
    
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
            np.zeros(self._nlp_metadata['s_dim']),
        ])
        ubw = np.concatenate([
            np.tile([self.config.a_max, self.config.delta_max], self._nlp_metadata['u_dim'] // 2),
            np.tile([self.config.a_max, self.config.delta_max], self._nlp_metadata['du_dim'] // 2),
            np.inf * np.ones(self._nlp_metadata['x_dim']),
            np.inf * np.ones(self._nlp_metadata['s_dim']),
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
        S  = ca.MX.sym( 'S', self.config.filter_num)

        # -------- 添加常量 ------- #
        x0 = ca.MX.sym('x0', self.config.nx)
        u0 = ca.MX.sym('u0', self.config.nu)

        self._g = list()
        self._p = [x0, u0, ]
        cost = ca.sumsqr(DU) + 1000 * ca.sumsqr(S)

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
        self._constraint_safety_lane_change_collision(xk, xk_next, S)

        self._nlp = \
        {
            'x': ca.vertcat(U, DU, X, S),
            'f': cost,
            'g': ca.vertcat(*self._g),
            'p': ca.vertcat(*self._p),
        }

        self._nlp_metadata = \
        {
            'w_dim' : U.shape[0] + DU.shape[0] + X.shape[0] + S.shape[0],
            'u_dim' : U.shape[0],
            'du_dim': DU.shape[0],
            'x_dim' : X.shape[0],
            's_dim' : S.shape[0],
            'g_equal_dim' : self.config.nx + self.config.nu + self.config.nx,
            'g_unequal_dim_1': self.config.filter_num,
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
    
    def _constraint_safety_lane_change_collision(self, state_k, state_k_next, S):
        '''
        安全约束: ego和其他车辆的椭圆距离应该大于dist_min
        Args:
            state_k     : 时刻k, [x, y, v, theta]的值
            state_k_next: 时刻k+1, 上述变量的值;
        '''
        N: int = self.config.filter_num
        masks  = ca.MX.sym('mask', N)
        infos_other  = ca.MX.sym('info', N * self.config.info_dim)
        
        info_k = ca.vertcat(state_k[0: 2], state_k[3])
        info_k_next = ca.vertcat(state_k_next[0: 2], state_k_next[3])
        h_dist = self.cbf_functions.distance_contrains

        for i in range(N):
            info_other = infos_other[i * self.config.info_dim: (i+1) * self.config.info_dim]
            # self._g.append(ca.if_else(
            #     mask[i] > 0, mask[i] * (h(pk_next, pk_s) - (1 + self.config.gamma) * h(pk, pk_s)), 0
            # ))
            self._g.append(h_dist(info_k_next, info_other) - (1 + self.config.gamma) * h_dist(info_k, info_other) + S[i: i+1])
        
        self._p.append(masks)
        self._p.append(infos_other)

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
        
        du: ndarray = w[self._nlp_metadata['u_dim']:
                        self._nlp_metadata['u_dim']+self._nlp_metadata['du_dim']].reshape(1, self.config.nu)
        
        x: ndarray  = w[self._nlp_metadata['u_dim']+self._nlp_metadata['du_dim']: 
                        self._nlp_metadata['u_dim']+self._nlp_metadata['du_dim']+self._nlp_metadata['x_dim']].reshape(2, self.config.nx)
        
        s: ndarray  = w[self._nlp_metadata['u_dim']+self._nlp_metadata['du_dim']+self._nlp_metadata['x_dim']::]
        solve_info = self._get_stats()
        return u, du, x, s, solve_info
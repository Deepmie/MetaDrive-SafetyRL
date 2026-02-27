import casadi as ca
from casadi import DM, MX
import numpy as np
from numpy import ndarray
from typing import Tuple, Dict, cast
from methods.common.ocp.config import MPConfig
from methods.common.ocp import DefaultMPC

class MPC(DefaultMPC):
    def __call__(self, x0, z_ref, u_prev) -> Tuple[ndarray, ndarray]:
        '''
        x0: 当前车辆的状态: [横坐标x, 纵坐标y, 车辆速度v, 车辆角度theta],
        z_ref: 车辆的跟踪轨迹: [参考速度v_ref, 参考角度theta_ref],
        u_prev: 上一次的控制值: [加速度a_prev, 转向角delta_prev],
        '''
        res: Dict = super(MPC, self).__call__(x0, z_ref, u_prev)
        return self._parse_result(res)

    def _caculate_cost_and_conditions(self):
        # ------- 添加决策变量 ----- #
        U = MX.sym('U', self.config.nu * self.config.mu)
        X = MX.sym('X', self.config.nx * (self.config.np+1))

        # -------- 添加常量 ------- #
        x0        = MX.sym('x0', self.config.nx)
        z_ref     = MX.sym('z_ref', 2 * (self.config.np + 1))
        u_prev    = MX.sym('u_prev', self.config.nu)

        Q  = DM(np.diag([5, 50])) # error
        R  = DM(np.diag([1, 1]))  # cost
        Rd = DM(np.diag([0, 0]))  # delta
        
        g = list()
        cost = 0
        
        g.append(X[0: self.config.nx] - x0)
        u_prev_sym = u_prev
        
        for k in range(0, self.config.np): # k_max = np-1, cons_u~k = mu-1
            xk = X[k * self.config.nx: (k + 1) * self.config.nx]
            uk = U[k * self.config.nu: (k + 1) * self.config.nu] if k < self.config.mu else U[(self.config.mu - 1) * self.config.nu: self.config.mu * self.config.nu]
            # 状态转移 & 添加约束进来
            xk_next = self._f(xk, uk); g.append(X[(k + 1) * self.config.nx: (k + 2) * self.config.nx] - xk_next)
            
            if k >= 1:
                zk_e = xk[2::] - z_ref[k * 2: (k + 1) * 2]
                error_trans = ca.mtimes([zk_e.T, Q, zk_e])
                cost += error_trans
            cost += ca.mtimes([uk.T, R, uk])

            # duk = uk - u_prev_sym
            # cost += ca.times([duk.T, Rd, duk])
            
            # update control varible #
            u_prev_sym = uk
        
        zk_e_terminal = xk_next[2::] - z_ref[(self.config.np-1) * 2 : self.config.np * 2]
        cost += ca.mtimes([zk_e_terminal.T, Q, zk_e_terminal])
        
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
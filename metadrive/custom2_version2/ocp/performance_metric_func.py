from metadrive.custom2_version2.ocp.config import MPConfig
from typing import List, Optional
import numpy as np
import casadi as ca
from casadi import DM
from numpy import ndarray
import matplotlib.pyplot as plt

class PerformetricFunc:
    '''
    性能度量函数. 用于全过程中'收紧'误差
    '''
    def __init__(self, config: MPConfig):
        self.config = config
        self._set_attr()
    
    def error_transformation(self, error: float, step: int) -> float:
        p: float = self.caculate(step)
        error_norm: float = error / (p * self.delta_L)
        # return 1 / 2 * np.log((error_norm + self.delta_L) / (self.delta_R - error_norm))
        return np.exp(self.alpha * error_norm ** 2) - 1

    def caculate(self, step: int) -> float:
        return (self.p_0 - self.p_inf) * np.exp(-self.lota * step) + self.p_inf

    def show_explot(self, total_step: int = 10000, errors: Optional[ndarray] = None):
        # simulate
        res: ndarray = np.empty(shape=[total_step, ], dtype=np.float32)
        rer: ndarray = np.empty(shape=[total_step, ], dtype=np.float32)
        index: ndarray = np.arange(total_step, dtype=np.long)
        for i in range(total_step):
            res[i] = self.caculate(i)
            if errors is not None: rer[i] = self.error_transformation(errors[i], i)
        
        # 求解左右两侧的res
        res_left = - self.delta_L * res
        res_right = self.delta_R * res
        
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.axis('off')
        # add axis1
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(index, res_left, linewidth=3)
        ax1.plot(index, res_right, linewidth=3)
        
        # plot baseline
        delta_value = 2
        left_value  = -delta_value
        right_value = total_step + delta_value
        ax1.plot([left_value, right_value], [0, 0], linestyle='--', color='#000000', linewidth=1)
        ax1.set_title('Performance Metric Function')
        ax1.set_xlim([left_value, right_value])
        ax1.set_xlabel('Steps (n)'); ax1.set_ylabel('Value')

        # add axis2
        if errors is not None:
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(index, rer, linewidth=3)
            ax2.set_title('Performance Error')

        fig.savefig('dp_single_version2/check/check_performance_metric_func.png')
    
    def _set_attr(self):
        self.p_0   = self.config.p_0
        self.p_inf = self.config.p_inf
        self.lota  = self.config.lota
        self.delta_L = self.config.delta_L
        self.delta_R = self.config.delta_R
        self.alpha   = self.config.alpha
        self._eps    = 1e-6


class PerformetricFuncCasadi:
    def __init__(self, config: MPConfig):
        self.config = config
        self._set_attr()
    
    def error_transformation(self, error, step):
        p = self.caculate(step)
        error_norm = error / (p * self.delta_L)
        # return 1 / 2 * ca.log((error_norm + self.delta_L + self._eps) / (self.delta_R - error_norm + self._eps))
        return ca.exp(self.alpha * error_norm ** 2) - 1

    def caculate(self, step):
        return (self.p_0 - self.p_inf) * ca.exp(-self.lota * step) + self.p_inf
    
    def _set_attr(self):
        self.p_0     = DM(self.config.p_0)
        self.p_inf   = DM(self.config.p_inf)
        self.lota    = DM(self.config.lota)
        self.delta_L = DM(self.config.delta_L)
        self.delta_R = DM(self.config.delta_R)
        self.alpha   = DM(self.config.alpha)
        self._eps    = 1e-6



if __name__ == '__main__':
    mpc_config: MPConfig = MPConfig()
    performetric_func: PerformetricFunc = PerformetricFunc(mpc_config)
    l = np.zeros(shape=[10000], dtype=np.float32)
    l[0] = -15.0
    l[1] = 2.5
    l[1000] = -15.0
    l[2000] = -15.0
    performetric_func.show_explot(errors=np.array(l))
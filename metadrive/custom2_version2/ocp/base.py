from metadrive.custom2_version2.ocp.config import OCPconfig
from metadrive.custom2_version2.type import VehicleState
from abc import ABC, abstractmethod
import casadi as ca
from numpy import ndarray
from typing import cast, List, Dict, Tuple
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
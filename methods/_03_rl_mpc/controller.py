from methods.common.envs import ParallelEnv, SingleEnv
from methods.common.base_config import ControllerConfig
from methods.common.controller import DefaultController, DefaultControllerResult
from methods.common.ocp import DefaultCBF
from methods._03_rl_mpc.mpc import MPC
from typing import Dict, Tuple, Union, Optional, cast
import numpy as np
from numpy import ndarray
from copy import deepcopy

class ControllerResult(DefaultControllerResult):
    def reset(self):
        self._env_idx: int = 0
        self.state_values            = np.empty([self._num, self.config.vehicle_state_dim])
        self.control_values          = np.empty([self._num, self.config.control_dim])
    
    def push(self, x_mpc: ndarray, u_mpc: ndarray):
        self.state_values[self._env_idx, :]             = x_mpc
        self.control_values[self._env_idx, :]           = u_mpc
        self._env_idx += 1

    def update_control_values_prev(self, dones: ndarray):
        self.control_values_prev = (1 - dones.reshape(-1, 1)) * self.control_values
    
    def get_state_values_modified(self, is_split: bool = True) -> ndarray:
        res = self.state_values
        if is_split: res = res[:, 2::]
        return res
    
    def get_control_values_modified(self, is_reverse: bool = True) -> ndarray:
        res = self.control_values
        return self._process_control_value(res, is_reverse)


class Controller(DefaultController):
    def __init__(
            self, env: Union[ParallelEnv, SingleEnv], config: ControllerConfig, 
            mpc_cls = MPC, cbf_cls = DefaultCBF, eval_mode: bool = False
        ):
        super(Controller, self).__init__(env, config, mpc_cls, cbf_cls, eval_mode)
        self.mpc_controller = cast(MPC, self.mpc_controller)

    def control(self, actions: ndarray, dones: ndarray) -> Tuple[DefaultControllerResult, Optional[Dict]]:
        # 获得初始状态
        vehicle_states_init = self._get_vehicle_state()
        infos, masks        = self._get_all_vehicle_position()
        actions, dones      = self._preprocess_var(actions), self._preprocess_var(dones)
        self.controller_result.reset()

        # ====== update if done include `True` ===== #
        self.controller_result.update_control_values_prev(dones)
        # ========================================== #
        
        for idx, (state, info, mask, ) in enumerate(zip(vehicle_states_init, infos, masks)):
            x0 = np.array([state.x, state.y, state.v, state.theta])
            state_ref = actions[idx, 0: 2]
            u_prev = self.controller_result.control_values_prev[idx, :]

            # if not self.eval_mode and curr_step is not None:
            assert isinstance(self.mpc_controller, MPC), 'Type of MPC mismatch!'
            u_mpc, x_mpc, solve_info_mpc = self.mpc_controller(x0, state_ref, u_prev)
            u_mpc, x_mpc = cast(ndarray, u_mpc), cast(ndarray, x_mpc)

            if not bool(solve_info_mpc.get('success')):
                raise Exception('MPC solve infeasible!')
            else:
                self._check_solve_results(x0, x_mpc[0, :], 'mpc x')
            
            # 存入结果类中
            self.controller_result.push(x_mpc[1, :], u_mpc[0, :])
            
            if self.eval_mode:
                extra_info: Dict = dict()
                extra_info['success'] = False
                extra_info['vehicle_state'] = vehicle_states_init[0]
            else:
                extra_info = None
        return deepcopy(self.controller_result), extra_info

    def _create_controller_result(self, eval_mode: bool) -> ControllerResult:
        return ControllerResult(self.config, eval_mode=eval_mode)
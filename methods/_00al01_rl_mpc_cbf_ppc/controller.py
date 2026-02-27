from methods.common.envs import ParallelEnv, SingleEnv
from methods.common.base_config import ControllerConfig
from methods.common.controller import DefaultController, DefaultControllerResult
from methods.common.ocp import DefaultCBF
from methods._00_rl_mpc_cbf_ppc_traj.mpc import MPC
from typing import Dict, Tuple, Union, Optional, cast
import numpy as np
from numpy import ndarray
from copy import deepcopy


class ControllerResult(DefaultControllerResult):
    def reset(self):
        super(ControllerResult, self).reset()
        self.performetrics = np.empty([self._num, 2])

    def push(self, x_mpc: ndarray, x_cbf: ndarray, u_mpc: ndarray, u_cbf: ndarray, performetric: ndarray):
        self.state_values[self._env_idx, :]             = x_mpc
        self.state_values_modified[self._env_idx, :]    = x_cbf
        self.control_values[self._env_idx, :]           = u_mpc
        self.control_values_modified[self._env_idx, :]  = u_cbf
        self.performetrics[self._env_idx, :]            = performetric
        self._env_idx += 1
    
    def get_performetrics(self) -> ndarray:
        return self.performetrics


class Controller(DefaultController):
    def __init__(
            self, env: Union[ParallelEnv, SingleEnv], config: ControllerConfig, 
            mpc_cls = MPC, cbf_cls = DefaultCBF, eval_mode: bool = False
        ):
        super(Controller, self).__init__(env, config, mpc_cls, cbf_cls, eval_mode)
        self.performetrics: ndarray = self.mpc_config.p_0 * np.ones([self._num, 2], dtype=np.float32)
        self.mpc_controller = cast(MPC, self.mpc_controller)

    def control(self, actions: ndarray, dones: ndarray) -> Tuple[ControllerResult, Optional[Dict]]:
        # 获得初始状态
        vehicle_states_init = self._get_vehicle_state()
        infos, masks        = self._get_all_vehicle_position()
        actions, dones      = self._preprocess_var(actions), self._preprocess_var(dones)
        self.controller_result.reset()

        # ====== update if done include `True` ===== #
        self.controller_result.update_control_values_prev(dones)
        dones_num: int = int(dones.sum().item())
        if dones_num > 0:
            dones_index = dones.flatten().astype(np.bool)
            self.performetrics[dones_index] = self.mpc_config.p_0 * np.ones([dones_num, 2], dtype=np.float32)
        # ========================================== #
        
        for idx, (state, info, mask, ) in enumerate(zip(vehicle_states_init, infos, masks)):
            x0 = np.array([state.x, state.y, state.v, state.theta])
            state_ref = actions[idx, 0: 2]; alpha = actions[idx, 2: 4]
            u_prev = self.controller_result.control_values_prev[idx, :]
            
            # if not self.eval_mode and curr_step is not None:
            assert isinstance(self.mpc_controller, MPC), 'Type of MPC mismatch!'
            u_mpc, x_mpc, solve_info_mpc = self.mpc_controller(x0, state_ref, u_prev, self.performetrics[idx])
            u_mpc, x_mpc = cast(ndarray, u_mpc), cast(ndarray, x_mpc)

            if not bool(solve_info_mpc.get('success')):
                raise Exception('MPC solve infeasible!')
            else:
                self._check_solve_results(x0, x_mpc[0, :], 'mpc x')
            
            # 求解修正的控制量
            info, mask = self._filter_info_mask(np.array([state.x, state.y, state.theta]), info, mask, filter_num=self.config.filter_num)
            u_cbf, x_cbf, solve_info_cbf = self.cbf_controller(x0, u_mpc[0, :], info.flatten(), mask)

            if not bool(solve_info_cbf.get('success')): # cbf解不出来(无论怎样都满足不了...)
                # print('no success!')
                u_cbf = u_mpc; x_cbf = deepcopy(x_mpc)
            else: # 如果成功了检查...
                self._check_solve_results(x0, x_cbf[0, :], 'cbf x')
                # self._check_solve_results(u_mpc[0, :], u_cbf[0, :], 'cbf u')
            
            # 存入结果类中
            self.controller_result.push(x_mpc[1, :], x_cbf[1, :], u_mpc[0, :], u_cbf[0, :], self.performetrics[idx, :])
            
            if self.eval_mode:
                extra_info: Dict = self._get_extra_test_info(x0, info, state, u_cbf)
                extra_info['success'] = bool(solve_info_cbf.get('success'))
                extra_info['vehicle_state'] = vehicle_states_init[0]
            else:
                extra_info = None
        
        # update performance metrics
        self.performetrics += self.mpc_config.p_inf * alpha.reshape(1, 2)
        return deepcopy(self.controller_result), extra_info
    
    def _create_controller_result(self, eval_mode: bool) -> ControllerResult:
        return ControllerResult(self.config, eval_mode=eval_mode)
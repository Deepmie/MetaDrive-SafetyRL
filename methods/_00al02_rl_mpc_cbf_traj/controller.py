from methods.common.envs import ParallelEnv, SingleEnv
from methods.common.base_config import ControllerConfig
from methods.common.controller import DefaultController, DefaultControllerResult
from methods.common.ocp import DefaultCBF
from methods._01_rl_mpc_cbf_traj.mpc import MPC
from typing import Dict, Tuple, Union, Optional, cast
import numpy as np
from numpy import ndarray
from copy import deepcopy

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
            z_ref: ndarray = self.traj_generator.generate(state.theta, state_ref)
            assert isinstance(self.mpc_controller, MPC), 'Type of MPC mismatch!'
            u_mpc, x_mpc, solve_info_mpc = self.mpc_controller(x0, z_ref, u_prev)
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
            self.controller_result.push(x_mpc[1, :], x_cbf[1, :], u_mpc[0, :], u_cbf[0, :])
            
            if self.eval_mode:
                extra_info: Dict = self._get_extra_test_info(x0, info, state, u_cbf)
                extra_info['success'] = bool(solve_info_cbf.get('success') and solve_info_mpc.get('success'))
                extra_info['vehicle_state'] = vehicle_states_init[0]
            else:
                extra_info = None
        return deepcopy(self.controller_result), extra_info
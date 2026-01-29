from metadrive.custom2_version2.ocp import MPC, CBF
from metadrive.custom2_version2.ocp.config import MPConfig, CBFconfig
from metadrive.custom2_version2.ocp.cbf_func import CBFunctions
from metadrive.custom2_version2.envs import ParallelEnv, SingleEnv
from metadrive.custom2_version2.type import VehicleState
from metadrive.custom2_version2.base_config import ControllerConfig
from metadrive.custom2_version2.traj_generator import TrajGenerator
from typing import List, Dict, Tuple, Union, cast, Optional
import numpy as np
from numpy import ndarray
from copy import deepcopy

class ControllerResult:
    def __init__(self, config: ControllerConfig, eval_mode: bool = False):
        self.config = config
        self.eval_mode = eval_mode
        self._num   = self.config.n_process if not eval_mode else 1
        self.reset()
        self.control_values_prev   = np.empty([self._num, self.config.control_dim])

    def reset(self):
        self._env_idx: int = 0
        self.state_values            = np.empty([self._num, self.config.vehicle_state_dim])
        self.state_values_modified   = np.empty([self._num, self.config.vehicle_state_dim])
        self.control_values          = np.empty([self._num, self.config.control_dim])
        self.control_values_modified = np.empty([self._num, self.config.control_dim])
        self.performetrics           = np.empty([self._num, 1])
    
    def push(self, x_mpc: ndarray, x_cbf: ndarray, u_mpc: ndarray, u_cbf: ndarray, performetric: ndarray):
        self.state_values[self._env_idx, :]             = x_mpc
        self.state_values_modified[self._env_idx, :]    = x_cbf
        self.control_values[self._env_idx, :]           = u_mpc
        self.control_values_modified[self._env_idx, :]  = u_cbf
        self.performetrics[self._env_idx, :]            = performetric
        self._env_idx += 1

    def update_control_values_prev(self, dones: ndarray):
        self.control_values_prev = (1 - dones.reshape(-1, 1)) * self.control_values_modified

    def get_control_values(self, is_reverse: bool = True) -> ndarray:
        res = self.control_values
        return self._process_control_value(res, is_reverse)

    def get_control_values_modified(self, is_reverse: bool = True) -> ndarray:
        res = self.control_values_modified
        return self._process_control_value(res, is_reverse)

    def get_delta_control_values(self, is_reverse: bool = True) -> ndarray:
        res = self.control_values_modified - self.control_values
        return self._process_control_value(res, is_reverse)
    
    def get_state_values(self, is_split: bool = True) -> ndarray:
        res = self.state_values
        if is_split: res = res[:, 2::]
        return res
    
    def get_state_values_modified(self, is_split: bool = True) -> ndarray:
        res = self.state_values_modified
        if is_split: res = res[:, 2::]
        return res

    def _process_control_value(self, res: ndarray, is_reverse: bool) -> ndarray:
        if is_reverse: res = res[:, ::-1]
        if self.eval_mode: res = res.flatten()
        return res
    
    def get_performetrics(self) -> ndarray:
        return self.performetrics



class Controller:
    def __init__(self, env: Union[ParallelEnv, SingleEnv], config: ControllerConfig, eval_mode: bool = False):
        self.env = env
        self.config = config
        self.eval_mode = eval_mode
        self._num = self.config.n_process if not eval_mode else 1
        self.controller_result = ControllerResult(self.config, eval_mode=eval_mode)
        self._build_controller()
        self.performetrics: ndarray = self.mpc_config.p_0 * np.ones([self._num, 2], dtype=np.float32)

    def control(self, actions: ndarray, dones: ndarray) -> Tuple[ControllerResult, Union[Dict]]:
        # 获得初始状态
        vehicle_states_init = self._get_vehicle_state()
        infos, masks        = self._get_all_vehicle_position()
        actions, dones      = self._preprocess_var(actions), self._preprocess_var(dones)
        self.controller_result.reset()

        # ====== update if done include `True` ===== #
        self.controller_result.update_control_values_prev(dones)
        dones_num: int = int(dones.sum().item())
        # ========================================== #
        
        for idx, (state, info, mask, ) in enumerate(zip(vehicle_states_init, infos, masks)):
            x0 = np.array([state.x, state.y, state.v, state.theta])
            state_ref = actions[idx, 0: 2]; alpha = actions[idx, 2: 4]
            u_prev = self.controller_result.control_values_prev[idx, :]

            # if not self.eval_mode and curr_step is not None:
            z_ref: ndarray = self.traj_generator.generate(state.theta, state_ref)
            assert isinstance(self.mpc_controller, MPC), 'Type of MPC mismatch!'
            u_mpc, x_mpc, solve_info_mpc = self.mpc_controller(x0, z_ref, u_prev, self.performetrics[idx, 0: 1])
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
            self.controller_result.push(x_mpc[1, :], x_cbf[1, :], u_mpc[0, :], u_cbf[0, :], self.performetrics[idx, 0])
            
            if self.eval_mode:
                extra_info: Dict = self._get_extra_test_info(x0, info, state, u_cbf)
                extra_info['success'] = bool(solve_info_cbf.get('success'))
            else:
                extra_info = None
        
        # update performance metrics
        self.performetrics += self.mpc_config.p_inf * alpha.reshape(1, 2)
        return deepcopy(self.controller_result), extra_info

    def _build_controller(self):
        metadata = self.env.get_metadata()
        self.mpc_config = MPConfig(); self.cbf_config = CBFconfig()
        self.mpc_controller = MPC(self.mpc_config, metadata)
        self.cbf_controller = CBF(self.cbf_config, metadata)
        self.traj_generator = TrajGenerator(self.mpc_config)
        self.cbf_functions: CBFunctions = CBFunctions(self.cbf_config)

    def _get_vehicle_state(self) -> List[VehicleState]:
        vehicle_states: List[Dict] = self.env.get_state() if not self.eval_mode else [self.env.get_state()]
        vehicle_states_extracted: List[VehicleState] = list()
        for state in vehicle_states:
            vehicle_states_extracted.append(self._extract_row_state(state))
        return vehicle_states_extracted
    
    def _get_all_vehicle_position(self) -> Tuple[List[ndarray], List[ndarray]]:
        infos, masks = self.env.get_all_vehicle_position()
        if self.eval_mode:
            infos = [infos]; masks = [masks]
        return infos, masks

    def _filter_info_mask(self, info_ego: ndarray, info: ndarray, mask: ndarray, filter_num: int = 5) -> Tuple[ndarray, ndarray]:
        info_masked = info[mask == 1]
        info_res, mask_res = np.zeros([filter_num, self.cbf_config.info_dim]), np.zeros([filter_num, ])

        dists: ndarray = np.zeros([info_masked.shape[0]], dtype=np.float32)
        for i in range(info_masked.shape[0]):
            dists[i] = self.cbf_functions.caculate_distance(info_ego, info_masked[i, :])

        sort_idx = np.sort(np.argsort(dists)[0: filter_num])
        select_index = min(sort_idx.shape[0], filter_num)
        info_res[0: select_index, :] = info_masked[sort_idx, :]
        mask_res[0: select_index] = np.ones(select_index)
        return info_res, mask_res

    def _extract_row_state(self, state: Dict) -> VehicleState:
        vehicle_state = VehicleState()
        pos = state.get('position', None)
        vel = state.get('velocity')
        
        vehicle_state.x = pos[0]
        vehicle_state.y = pos[1]
        vehicle_state.v = np.linalg.norm(vel, 2) # 取速度的二范数
        vehicle_state.theta = state.get('heading_theta', None)
        vehicle_state.a     = state.get('throttle_brake', None)
        vehicle_state.delta = state.get('steering', None)
        return vehicle_state
    
    def _check_solve_results(self, a: ndarray, b: ndarray, sign: str, delta: float = 5):
        assert np.linalg.norm(a - b) / int(a.shape[0]) < delta, \
        f'process of {sign}, first != second, first: {a.tolist()}, second: {b.tolist()}'

    def _preprocess_var(self, v: ndarray) -> ndarray:
        if len(v.shape) == 1:
            return v.reshape(1, -1)
        return v
    
    def _get_extra_test_info(self, x0: ndarray, info: ndarray, state: VehicleState, u_cbf: ndarray) -> Dict:
        dist: ndarray           = np.zeros([self.cbf_config.filter_num], dtype=np.float32)
        x0_next: ndarray        = self._state_update_equation(x0, u_cbf[0, :])
        constr: ndarray         = np.zeros([self.cbf_config.filter_num], dtype=np.float32)
        cbf_real_value: ndarray = np.zeros([self.cbf_config.filter_num], dtype=np.float32)
        
        # 计算距离
        for i in range(info.shape[0]):
            dist[i] = self.cbf_functions.caculate_distance(np.array([state.x, state.y, state.theta]), info[i, :])
        # 计算约束
        for i in range(info.shape[0]):
            constr[i] = self.cbf_functions.distance_contrains(np.array([x0_next[0], x0_next[1], x0_next[3]]), info[i, :]) - \
            (1 + self.cbf_config.gamma) * self.cbf_functions.distance_contrains(np.array([x0[0], x0[1], x0[3]]), info[i, :])
        # 计算真实值
        for i in range(info.shape[0]):
            cbf_real_value[i] = self.cbf_functions.distance_contrains(np.array([x0[0], x0[1], x0[3]]), info[i, :])
        return dict(dist=dist, constrants=constr, cbf_real_value=cbf_real_value)

    def _state_update_equation(self, x: ndarray, u: ndarray) -> ndarray:
        lr = self.cbf_controller.lr; lf = self.cbf_controller.lf
        # 状态更新方程
        beta = np.atan(np.tan(u[1]) * lr / (lf + lr))
        x_next = np.array([
            x[0] + x[2] * np.cos(x[3] + beta) * self.cbf_config.Ts,
            x[1] + x[2] * np.sin(x[3] + beta) * self.cbf_config.Ts,
            x[2] + u[0] * self.cbf_config.Ts,
            x[3] + x[2] / lr * np.sin(beta) * self.cbf_config.Ts,
        ])
        return x_next



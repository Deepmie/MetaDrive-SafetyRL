from metadrive.custom2_version2.ocp import MPC, CBF
from metadrive.custom2_version2.ocp.config import MPConfig, CBFconfig
from metadrive.custom2_version2.envs import ParallelEnv, SingleEnv
from metadrive.custom2_version2.type import VehicleState
from metadrive.custom2_version2.base_config import ControllerConfig
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
        self.state_values          = np.empty([self._num, self.config.vehicle_state_dim])
        self.state_values_modified = np.empty([self._num, self.config.vehicle_state_dim])
        self.control_values        = np.empty([self._num, self.config.control_dim])
        self.delta_control_values  = np.empty([self._num, self.config.control_dim])
    
    def push(self, x_mpc: ndarray, x_cbf: ndarray, u_mpc: ndarray, du_cbf: ndarray):
        self.state_values[self._env_idx, :]          = x_mpc
        self.state_values_modified[self._env_idx, :] = x_cbf
        self.control_values[self._env_idx, :]        = u_mpc
        self.delta_control_values[self._env_idx, :]  = du_cbf
        self._env_idx += 1

    def update_control_values_prev(self, dones: ndarray):
        self.control_values_prev = (1 - dones.reshape(-1, 1)) * self.control_values_modified

    @property
    def control_values_modified(self, is_reverse: bool = True) -> ndarray:
        res = self.control_values + self.delta_control_values
        if is_reverse: res = res[:, ::-1]
        if self.eval_mode: res = res.flatten()
        return res



class Controller:
    def __init__(self, env: Union[ParallelEnv, SingleEnv], config: ControllerConfig, eval_mode: bool = False):
        self.env = env
        self.config = config
        self.eval_mode = eval_mode
        self.controller_result = ControllerResult(self.config, eval_mode=eval_mode)
        self._build_controller()

    def control(self, actions: ndarray, dones: ndarray) -> ControllerResult:
        # 获得初始状态
        vehicle_states_init = self._get_vehicle_state()
        infos, masks        = self._get_all_vehicle_position()
        actions, dones      = self._preprocess_var(actions), self._preprocess_var(dones)
        self.controller_result.reset()
        
        for idx, (state, info, mask, ) in enumerate(zip(vehicle_states_init, infos, masks)):
            x0 = np.array([state.x, state.y, state.v, state.theta])
            z_ref = actions[idx, :]
            u_prev = self.controller_result.control_values_prev[idx, :]
            u_mpc, x_mpc, solve_info_mpc = self.mpc_controller(x0, z_ref, u_prev)
            u_mpc, x_mpc = cast(ndarray, u_mpc), cast(ndarray, x_mpc)
            self._check_solve_results(x0, x_mpc[0, :], 'mpc x')
            
            # 求解修正的控制量
            info, mask = self._filter_info_mask(state.x, state.y, info, mask, filter_num=self.config.filter_num)
            u_cbf, du_cbf, x_cbf, solve_info_cbf = self.cbf_controller(x0, u_mpc[0, :], info.flatten(), mask)
            
            if not bool(solve_info_cbf.get('success')): # cbf解不出来(无论怎样都满足不了...)
                u_cbf = None; x_cbf = deepcopy(x_mpc)
                du_cbf = np.zeros([1, self.mpc_config.nu])
            else: # 如果成功了检查...
                self._check_solve_results(x0, x_cbf[0, :], 'cbf x')
                self._check_solve_results(u_mpc[0, :], u_cbf[0, :], 'cbf u')
            
            # 存入结果类中
            self.controller_result.push(x_mpc[1, :], x_cbf[1, :], u_mpc[0, :], du_cbf[0, :])
        
        # 更新control的values
        self.controller_result.update_control_values_prev(dones)
        return deepcopy(self.controller_result)

    def _build_controller(self):
        metadata = self.env.get_metadata()
        self.mpc_config = MPConfig(); self.cbf_config = CBFconfig()
        self.mpc_controller = MPC(self.mpc_config, metadata)
        self.cbf_controller = CBF(self.cbf_config, metadata)

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

    def _filter_info_mask(self, x: float, y: float, info: ndarray, mask: ndarray, filter_num: int = 5):
        pos = np.array([x, y], dtype=np.float32).reshape(1, -1)
        sort_idx = np.argsort(np.linalg.norm(info - pos, ord=2, axis=1))[0: filter_num]
        return info[sort_idx, :], mask[sort_idx]

    def _extract_row_state(self, state: Dict) -> VehicleState:
        vehicle_state = VehicleState()
        pos = state.get('position', None)
        vel = state.get('velocity')
        
        vehicle_state.x = pos[0]
        vehicle_state.y = pos[1]
        vehicle_state.v = np.linalg.norm(vel, 2) # 取速度的二范数
        vehicle_state.theta = state.get('heading_theta', None)
        vehicle_state.a = state.get('throttle_brake', None)
        vehicle_state.delta = state.get('steering', None)
        return vehicle_state
    
    def _check_solve_results(self, a: ndarray, b: ndarray, sign: str, delta: float = 5):
        assert np.linalg.norm(a - b) / int(a.shape[0]) < delta, \
        f'process of {sign}, first != second, first: {a.tolist()}, second: {b.tolist()}'

    def _preprocess_var(self, v: ndarray) -> ndarray:
        if len(v.shape) == 1:
            return v.reshape(1, -1)
        return v


from metadrive.custom2_version2.ocp import MPC, CBF
from metadrive.custom2_version2.ocp.config import MPConfig, CBFconfig
from metadrive.custom2_version2.envs import ParallelEnv, SingleEnv
from metadrive.custom2_version2.type import VehicleState
from metadrive.custom2_version2.base_config import ControllerConfig
from metadrive import MetaDriveEnv
from metadrive.component.vehicle.default_vehicle import DefaultVehicle
from typing import List, Dict, Tuple, cast
import numpy as np
from numpy import ndarray
from copy import deepcopy
from abc import ABC

class ControllerResult:
    def __init__(self, config: ControllerConfig, eval_mode: bool = False):
        self.config = config
        self._num   = self.config.n_process if not eval_mode else 1
        self.reset()

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

    @property
    def control_values_modified(self) -> ndarray:
        return self.control_values + self.delta_control_values




class Controller:
    def __init__(self, env: ParallelEnv, config: ControllerConfig):
        self.env = env
        self.config = config
        self.control_values_prev = np.zeros([self.config.n_process, self.config.control_dim])
        self.controller_result = ControllerResult(self.config)
        self._build_controller()

    def control(self, actions: ndarray, dones: ndarray) -> ControllerResult:
        # 获得初始状态
        vehicle_states_init = self._get_vehicle_state()
        infos, masks        = self.env.get_all_vehicle_position()
        self.controller_result.reset()
        
        for idx, (state, info, mask, ) in enumerate(zip(vehicle_states_init, infos, masks)):
            x0 = np.array([state.x, state.y, state.v, state.theta])
            z_ref = actions[idx, :]
            u_prev = self.control_values_prev[idx, :]
            u_mpc, x_mpc = self.mpc_controller(x0, z_ref, u_prev)
            u_mpc, x_mpc = cast(ndarray, u_mpc), cast(ndarray, x_mpc)
            self._check_solve_results(x0, x_mpc[0, :], 'mpc x')
            
            # 求解修正的控制量
            u_cbf, du_cbf, x_cbf = self.cbf_controller(x0, u_mpc[0, :], info, mask)
            self._check_solve_results(x0, x_cbf[0, :], 'cbf x')
            self._check_solve_results(u_mpc[0, :], u_cbf[0, :], 'cbf u')
            
            # 存入结果类中
            self.controller_result.push(x_mpc[1, :], x_cbf[1, :], u_mpc[0, ::-1], du_cbf[0, ::-1])
        
        self.control_values_prev = (1 - dones.reshape(-1, 1)) * self.controller_result.control_values_modified
        return deepcopy(self.controller_result)

    def _build_controller(self):
        metadata = self.env.get_metadata()
        mpc_config = MPConfig(); cbf_config = CBFconfig()
        self.mpc_controller = MPC(mpc_config, metadata)
        self.cbf_controller = CBF(cbf_config, metadata)

    def _get_vehicle_state(self) -> List[VehicleState]:
        vehicle_states: List[Dict] = self.env.get_state()
        vehicle_states_extracted: List[VehicleState] = list()
        for state in vehicle_states:
            vehicle_states_extracted.append(self._extract_row_state(state))
        return vehicle_states_extracted

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
    
    def _check_solve_results(self, a: ndarray, b: ndarray, sign: str, delta: float = 0.1):
        assert np.linalg.norm(a - b) < delta, \
        f'process of {sign}, first != second, first: {a.tolist()}, second: {b.tolist()}'



class EvalController:
    def __init__(self, env: SingleEnv, config: ControllerConfig):
        self.env = env
        self.config = config
        self.control_value_prev = np.zeros([self.config.control_dim])
        self.controller_result  = ControllerResult(self.config, eval_mode=True)
        self._build_controller()

    def control(self, action: ndarray, done: ndarray) -> Tuple[ndarray, ndarray]: # [action_dim]
        # 获得初始状态
        vehicle_states_init = self._get_vehicle_state()
        info, mask          = self.env.get_all_vehicle_position()
        self.controller_result.reset()
        
        # control_values: [[delta, a], ...]
        x0 = np.array([vehicle_states_init.x, vehicle_states_init.y, vehicle_states_init.v, vehicle_states_init.theta])
        z_ref = action
        u_prev = self.control_value_prev
        u_mpc, x_mpc = self.mpc_controller(x0, z_ref, u_prev)
        u_mpc, x_mpc = cast(ndarray, u_mpc), cast(ndarray, x_mpc)
        
        # 检查mpc求解是否合理
        assert np.linalg.norm(x0 - x_mpc[0, :]) < 0.1, f'x0 != x[0, :], x0={x0.tolist()}, x[0, :]={x_mpc[0, :].tolist()}'
        
        u_cbf, du_cbf, x_cbf = self.cbf_controller(x0, u_mpc[0, :], info, mask)

        self.controller_result.push(x_mpc[1, :], x_cbf[1, :], u_mpc[0, :], du_cbf[0, ::-1])
        
        self.control_value_prev = (1 - done) * self.controller_result.control_values_modified
        return deepcopy(self.controller_result)

    def _build_controller(self):
        metadata = self.env.get_metadata()
        mpc_config = MPConfig(); cbf_config = CBFconfig()
        self.mpc_controller = MPC(mpc_config, metadata)
        self.cbf_controller = CBF(cbf_config, metadata)

    def _get_vehicle_state(self) -> VehicleState:
        vehicle_state: Dict = self.env.get_state()
        return self._extract_row_state(vehicle_state)

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
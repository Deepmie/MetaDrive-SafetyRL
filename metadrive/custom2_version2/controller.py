from metadrive.custom2_version2.ocp import MPC
from metadrive.custom2_version2.ocp.config import MPConfig
from metadrive.custom2_version2.envs import ParallelEnv, SingleEnv
from metadrive.custom2_version2.type import VehicleState
from metadrive.custom2_version2.base_config import ControllerConfig
from metadrive import MetaDriveEnv
from metadrive.component.vehicle.default_vehicle import DefaultVehicle
from typing import List, Dict, Tuple, cast
import numpy as np
from numpy import ndarray

class Controller:
    def __init__(self, env: ParallelEnv, config: ControllerConfig):
        self.env = env
        self.config = config
        self.control_values_prev = np.zeros([self.config.n_process, self.config.control_dim])
        self._build_mpc_controller()

    def control(self, actions: ndarray, dones: ndarray) -> Tuple[ndarray, ndarray]: # [env_num, action_dim]
        # 获得初始状态
        vehicle_states_init = self._get_vehicle_state()
        
        # control_values: [[delta, a], ...]
        control_values = np.empty([self.config.n_process, self.config.control_dim])
        state_values = np.empty([self.config.n_process, self.config.vehicle_state_dim])

        for idx, state in enumerate(vehicle_states_init): 
            x0 = np.array([state.x, state.y, state.v, state.theta])
            z_ref = actions[idx, :]
            u_prev = self.control_values_prev[idx, :]
            u, x = self.mpc_controller(x0, z_ref, u_prev)
            u, x = cast(ndarray, u), cast(ndarray, x)
            assert np.linalg.norm(x0 - x[0, :]) < 0.1, f'x0 != x[0, :], x0={x0.tolist()}, x[0, :]={x[0, :].tolist()}'
            control_values[idx, :] = u[0, ::-1]
            state_values[idx, :] = x[1, :] # 第0项是初始状态
        
        self.control_values_prev = (1 - dones.reshape(-1, 1)) * control_values
        return control_values, state_values

    def _build_mpc_controller(self):
        metadata = self.env.get_metadata()
        mpc_config = MPConfig()
        self.mpc_controller = MPC(mpc_config, metadata)

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





class EvalController:
    def __init__(self, env: SingleEnv, config: ControllerConfig):
        self.env = env
        self.config = config
        self.control_value_prev = np.zeros([self.config.control_dim])
        self._build_mpc_controller()

    def control(self, action: ndarray, done: ndarray) -> Tuple[ndarray, ndarray]: # [action_dim]
        # 获得初始状态
        vehicle_states_init = self._get_vehicle_state()
        
        # control_values: [[delta, a], ...]
        x0 = np.array([vehicle_states_init.x, vehicle_states_init.y, vehicle_states_init.v, vehicle_states_init.theta])
        z_ref = action
        u_prev = self.control_value_prev
        u, x = self.mpc_controller(x0, z_ref, u_prev)
        u, x = cast(ndarray, u), cast(ndarray, x)
        assert np.linalg.norm(x0 - x[0, :]) < 0.1, f'x0 != x[0, :], x0={x0.tolist()}, x[0, :]={x[0, :].tolist()}'
        control_value = u[0, ::-1]
        state_value   = x[1, :] # 第0项是初始状态
        
        self.control_value_prev = (1 - done) * control_value
        return control_value, state_value

    def _build_mpc_controller(self):
        metadata = self.env.get_metadata()
        mpc_config = MPConfig()
        self.mpc_controller = MPC(mpc_config, metadata)

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
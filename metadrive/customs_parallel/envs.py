from metadrive import MetaDriveEnv
from metadrive.utils.config import Config
from metadrive.component.vehicle.default_vehicle import DefaultVehicle
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.customs_parallel.config import MainConfig
from metadrive.customs_parallel.vehicle import SolverVehicle
from typing import cast, List
import numpy as np

config: MainConfig = MainConfig()
policy_config = config.solver_config.policy_config

DP_DEFAULT_CONFIG = dict(
    v_max = policy_config.v_max,
    v_min = policy_config.v_min,
    theta_max = policy_config.theta_max,
    theta_min = policy_config.theta_min,
)

class DpEnv(MetaDriveEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = super(DpEnv, cls).default_config()
        config.update(DP_DEFAULT_CONFIG)
        return config

    def __init__(self, config):
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        super(DpEnv, self).__init__(config)
    
    
    def reward_function(self, vehicle_id: str):
        vehicle: SolverVehicle = self.agents[vehicle_id]
        step_info = dict()
        step_info['step']        = vehicle.get_step_num
        step_info['v']           = vehicle.agent_info.v
        step_info['change lane'] = vehicle.change_lane
        
        # 到达checkpoint的奖励
        r1, coor_curr = self._is_arrive_route_points(vehicle)
        r1 *= 1.2
        
        # 速度惩罚
        r2 = 1 / self.config['v_max'] * (vehicle.agent_info.v - 2)
        r2 *= 1
        
        # 压线, 碰到其他obj的惩罚
        r3 = -5 if (vehicle.crash_sidewalk or vehicle.crash_building or vehicle.crash_human or  self._is_out_of_road(vehicle)) else 0
        r3 *= 1
        
        # 到达目的地奖励
        r4 = 5.0 if self._is_arrive_destination(vehicle) else 0.0

        # 鼓励往正确的道路上走
        in_right: bool = self._in_right_lane(vehicle)
        if in_right:
            r5 = self._compute_progress(vehicle)
        else:
            r5 = -0.5
        
        step_info['in_right'] = in_right

        reward = r1 + r2 + r3 + r4 + r5
        return reward, step_info

    # 离散写法
    # def _is_arrive_route_points(self, vehicle: DefaultVehicle, min_dist: float = 1.0):
    #     # 如果没有就创建一个已达节点列表
    #     if not hasattr(self, 'have_arrived_points'):
    #         self.have_arrived_points: List = list()
        
    #     curr_lane = vehicle.navigation.current_lane
    #     curr_lane = cast(AbstractLane, curr_lane)
    #     result: bool = False
    #     route_point_idx = None
        
    #     # get 局部坐标系
    #     coor_curr = vehicle.position
    #     coor_curr = np.array(coor_curr)
    #     for idx, route_point in enumerate(vehicle.navigation.route_points):
    #         dist_err = np.linalg.norm(route_point - coor_curr, ord=2).item()
    #         if dist_err < min_dist:
    #             result = True
    #             self.have_arrived_points.append(idx)
    #             break
        
    #     if result and route_point_idx is not None: # 类似吃过就会消失的感觉
    #         vehicle.navigation.route_points.pop(route_point_idx)

    #     return result, coor_curr

    def _is_arrive_route_points(self, vehicle: DefaultVehicle):
        curr_lane = vehicle.navigation.current_lane
        curr_lane = cast(AbstractLane, curr_lane)
        
        coor_curr = vehicle.position
        coor_curr = np.array(coor_curr)
        dist_min = float('inf')
        for route_point in vehicle.navigation.route_points:
            dist_err = np.linalg.norm(route_point - coor_curr, ord=2).item()
            if dist_err < dist_min:
                dist_min = dist_err

        reward = np.exp(-dist_min)
        return reward, coor_curr

    def _compute_progress(self, vehicle: DefaultVehicle):
        curr_lane = vehicle.navigation.current_lane
        curr_lane = cast(AbstractLane, curr_lane)

        # 说明没变道, 给予前进奖励
        long_last, _ = curr_lane.local_coordinates(vehicle.last_position)
        long_curr, _ = curr_lane.local_coordinates(vehicle.position)
        reward = (long_curr - long_last)
        return reward

    def _in_right_lane(self, vehicle: DefaultVehicle):
        right_lane = vehicle.navigation.right_lane
        if vehicle.lane in right_lane:
            return True
        return False

    def _is_same_lane(self, lane1: AbstractLane, lane2: AbstractLane):
        if (lane1.index[0] != lane2.index[0]) or (lane1.index[1] != lane2.index[2]):
            return False
        return True
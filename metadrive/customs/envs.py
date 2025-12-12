from metadrive import MetaDriveEnv
from metadrive.envs import ScenarioEnv
from metadrive.component.navigation_module.trajectory_navigation import TrajectoryNavigation
from metadrive.utils.math import clip, wrap_to_pi
from metadrive.utils.config import Config
from metadrive.customs.vehicle import SolverVehicle
from metadrive.component.lane.abs_lane import AbstractLane
import numpy as np
from typing import cast

DP_DEFAULT_CONFIG = dict()

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
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle: SolverVehicle = self.agents[vehicle_id]
        step_info = dict()

        step_info['step'] = vehicle.get_step_num
        step_info['v']    = vehicle.agent_info.v
        step_info['change lane'] = vehicle.change_lane

        r1 = - (10 + vehicle.agent_info.v / 12) if vehicle.crash_vehicle else 0
        r2 = -0.5 if vehicle.change_lane else 0
        r3 = -5 if (vehicle.crash_sidewalk or vehicle.crash_building or vehicle.crash_human or  self._is_out_of_road(vehicle)) else 0
        r4 = 2 * 1 / 12 * (vehicle.agent_info.v - 2)
        # r5 = - 0.01 * (vehicle.get_step_num / 10)
        r5 = 0
        r6 = 5 if self._is_arrive_destination(vehicle) else 0

        # extra
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            curr_lane = vehicle.lane
            positive_road = 1
        else:
            curr_lane = vehicle.navigation.current_ref_lanes[0]
            curr_road = vehicle.navigation.current_road
            positive_road = 1 if not curr_road.is_negative_road() else -1
        
        curr_lane = cast(AbstractLane, curr_lane)
        _, lateral = curr_lane.local_coordinates(vehicle.position)
        r7 = 0.5 * clip(1 - 2 * abs(lateral) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0) * positive_road

        step_info['lateral'] = lateral

        reward = r1 + r2 + r3 + r4 + r5 + r6 + r7
        return reward, step_info
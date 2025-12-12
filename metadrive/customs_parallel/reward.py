from metadrive.customs_parallel.vehicle import SolverVehicle
from typing import cast
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive import MetaDriveEnv


# 奖励函数1
def reward_function_version1(self: MetaDriveEnv, vehicle_id: str):
    vehicle: SolverVehicle = self.agents[vehicle_id]
    step_info = dict()
    
    step_info['step']        = vehicle.get_step_num
    step_info['v']           = vehicle.agent_info.v
    step_info['change lane'] = vehicle.change_lane

    # 车辆碰撞的惩罚
    r1 = - (10 + vehicle.agent_info.v / self.config['v_max']) if vehicle.crash_vehicle else 0
    r1 *= 1
    
    # 变道的惩罚
    r2 = -0.5 if vehicle.change_lane else 0
    r2 *= 1
    
    # 压线, 碰到其他obj的惩罚
    r3 = -5 if (vehicle.crash_sidewalk or vehicle.crash_building or vehicle.crash_human or  self._is_out_of_road(vehicle)) else 0
    r3 *= 1

    # 速度过慢的惩罚
    r4 = 1 / self.config['v_max'] * (vehicle.agent_info.v - 2)
    r4 *= 2
    
    # 完成任务过慢的惩罚
    # r5 = - 0.01 * (vehicle.get_step_num / 10)
    r5 = 0
    
    # 成功到达目的地的奖励
    r6 = 5 if self._is_arrive_destination(vehicle) else 0

    # ==================保持在车道中心的奖励=========================== #
    if vehicle.lane in vehicle.navigation.current_ref_lanes:
        curr_lane = vehicle.lane
        positive_road = 1
    else:
        curr_lane = vehicle.navigation.current_ref_lanes[0]
        curr_road = vehicle.navigation.current_road
        positive_road = 1 if not curr_road.is_negative_road() else -1
    
    curr_lane = cast(AbstractLane, curr_lane)
    long_last, _ = curr_lane.local_coordinates(vehicle.last_position)
    long_curr, lateral_curr = curr_lane.local_coordinates(vehicle.position)
    r7 = (long_curr - long_last) * positive_road
    r7 *= 2.0
    # ============================================================= #
    lane_width = vehicle.navigation.get_current_lane_width()
    lateral_norm = abs(lateral_curr) / (lane_width / 2)
    r8 = - (lateral_norm ** 2)
    r8 *= 12.0
    
    step_info['lateral'] = lateral_curr
    step_info['long_reward'] = (long_curr - long_last) * positive_road
    step_info['lateral_reward'] =  - (lateral_norm ** 2)
    
    reward = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8
    return reward, step_info



# 奖励函数2
def reward_function_version2(self: MetaDriveEnv, vehicle_id: str):
    vehicle: SolverVehicle = self.agents[vehicle_id]
    step_info = dict()

    


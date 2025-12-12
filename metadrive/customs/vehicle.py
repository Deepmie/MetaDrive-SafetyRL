from metadrive.component.vehicle.default_vehicle import DefaultVehicle
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils.config import Config
from metadrive.customs.config import SolverConfig
from metadrive.customs.type import AgentInfo
from metadrive.customs.ocp import MPC, CBF
from metadrive.customs.policy import Policy
from metadrive.customs.buffer import RolloutBuffer
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch import Tensor
import numpy as np
from numpy import ndarray
import swanlab
from datetime import datetime
from typing import List, Tuple, Dict, cast
from tqdm import tqdm


class SolverVehicle(DefaultVehicle):
    def __init__(self, *args, **kwargs):
        super(SolverVehicle, self).__init__(*args, **kwargs)
        self.agent_info = AgentInfo()
        self._update_agent_info()
        self.solver_config: SolverConfig = SolverConfig()
        self._init_extra_info()
        
        self.policy = Policy(self.solver_config.policy_config)
        self.mpc = MPC(self.solver_config.mpc_config, self.agent_info)
        self.cbf = CBF(self.solver_config.cbf_config, self.agent_info)
        self.buffer = RolloutBuffer(self.solver_config.buffer_config)

        # 初始化训练相关的参数
        self.mse_loss = MSELoss()
        self.optimizer = Adam([
            {'params': self.policy.feature_extractor.parameters(), },
            {'params': self.policy.actor.parameters() , 'lr': self.solver_config.learning_rate_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.solver_config.learning_rate_critic},
        ], lr=self.solver_config.learning_rate)
        self.global_count = 0

    def _init_extra_info(self):
        # extra info
        self._lane_list: List[Tuple] = [] # 保存一前一后的lane
        self._step_num: int = 0 # 保存时间步的数量

    def reset(self, *args, **kwargs):
        super(SolverVehicle, self).reset(*args, **kwargs)
        self._init_extra_info()

    def before_step(self, *args, **kwargs) -> Dict:
        step_info = super(SolverVehicle, self).before_step(*args, **kwargs)
        self._update_agent_info() # 获得当前状态下agent的状态
        return step_info
    
    
    def after_step(self, *args, **kwargs):
        super(SolverVehicle, self).after_step(*args, **kwargs)
        curr_lane: Tuple = self.navigation.current_lane.index

        if len(self._lane_list) < 2:
            self._lane_list.append(curr_lane)
        elif len(self._lane_list) == 2:
            self._lane_list.pop(0)
            self._lane_list.append(curr_lane)
        else:
            raise ValueError(f'self._lane_list length must <= 2! now is {len(self._lane_list)}.')

        self._step_num += 1


    def select_action(self, obs: ndarray) -> Tuple[ndarray]:
        # step1. 基于策略网络预测[v_ref, theta_ref]
        z_ref, log_prob, value = self.policy.select_action(obs)

        # step2. 基于v_ref和theta_ref进行MPC预测多步轨迹
        x0 = np.array([self.agent_info.x, self.agent_info.y, self.agent_info.v, self.agent_info.theta])
        u_mpc, x_mpc = self.mpc(x0, z_ref)
        u_mpc = u_mpc[0, :] # 只取最优控制序列的第一项

        # step3. 基于安全约束用CBF修正MPC解出的控制
        u0 = u_mpc
        mask, info = self._get_mask_info()
        _, du_cbf, x_cbf = self.cbf(x0, u0, mask, info)
        du_cbf = du_cbf[0, :]

        # 转换u_mpc和du_cbf为[delta, a]
        z_mpc = x_mpc[1, 2::]
        z_cbf = x_cbf[1, 2::] # 取cbf后迭代的第一步状态
        u_mpc = np.ascontiguousarray(u_mpc[::-1])
        du_cbf = np.ascontiguousarray(du_cbf[::-1])
        return z_ref, z_mpc, z_cbf, u_mpc, du_cbf, log_prob, value
    
    
    def update(self) -> Tuple:
        # 模型: CPU -> GPU
        self.policy = self.policy.to(self.solver_config.device)
        self.policy.train()
        pbar = tqdm(total=self.solver_config.epoch, desc='Update', unit='step')

        count = 0
        for epoch in range(self.solver_config.epoch):
            avg_loss_epoch = 0
            for obs, z_mpc, z_cbf, action, action_cbf, reward, done, old_log_prob, value, advantage, reward_accum in self.buffer.dataloader:
                # 数据: CPU -> GPU
                obs, z_mpc, z_cbf, action, action_cbf, old_log_prob, value, advantage, reward_accum = \
                obs.to(self.solver_config.device), z_mpc.to(self.solver_config.device), z_cbf.to(self.solver_config.device), \
                action.to(self.solver_config.device), action_cbf.to(self.solver_config.device), old_log_prob.to(self.solver_config.device), \
                value.to(self.solver_config.device), advantage.to(self.solver_config.device), reward_accum.to(self.solver_config.device)

                log_prob, entropy, value_new = self.policy.evaluate_action(obs, action)
                ratios = torch.exp(log_prob - old_log_prob)
                
                # create index
                bc_index = torch.norm(z_cbf, p=1, dim=1) > self.solver_config.delta_bc
                stand_index = (bc_index == False)

                # standard loss
                if (~stand_index).all():
                    standard_loss = 0
                else:
                    surr1 = ratios[stand_index] * advantage[stand_index]
                    surr2 = torch.clamp(ratios[stand_index], 1-self.solver_config.epsilon, 1+self.solver_config.epsilon) * advantage[stand_index]
                    standard_loss = -torch.min(surr1, surr2).mean()

                # bc loss
                if (~bc_index).all():
                    bc_loss = 0
                else:
                    pi_action = self.policy.act_mean(obs[bc_index])
                    omega = 1 + torch.exp(torch.norm(z_cbf[bc_index] - z_mpc[bc_index], p=2, dim=1))
                    bc_loss = (omega * torch.norm(z_cbf[bc_index] - pi_action, p=2, dim=1)).mean()

                actor_loss = standard_loss + self.solver_config.bc_coef * bc_loss
                critic_loss = 0.5 * self.mse_loss(value_new.reshape(-1), reward_accum)

                loss: Tensor = actor_loss + self.solver_config.value_loss_coef * critic_loss - self.solver_config.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_loss_epoch += loss.item()
                count += 1

            avg_loss_epoch /= self.buffer.dataloader_len
            
            if self.solver_config.use_swanlab:
                swanlab.log({f'loss/update{self.global_count}': avg_loss_epoch})
            pbar.update(1)
        
        self.global_count += 1
        self.policy = self.policy.to(torch.device('cpu'))

    
    def _update_agent_info(self):
        state = self.get_state()
        pos = state.get('position', None)
        vel = state.get('velocity')
        
        self.agent_info.x = pos[0]
        self.agent_info.y = pos[1]
        self.agent_info.v = np.linalg.norm(vel, 2) # 取速度的二范数
        self.agent_info.theta = state.get('heading_theta', None)
        self.agent_info.a = state.get('throttle_brake', None)
        self.agent_info.delta = state.get('steering', None)
        self.agent_info.l = self.FRONT_WHEELBASE + self.REAR_WHEELBASE
        # self.agent_info.l = state.get('length')


    def _get_mask_info(self) -> Tuple:
        def filter_func(obj):
            if not isinstance(obj, BaseVehicle): # 确保是车辆
                return False
            obj = cast(BaseVehicle, obj)
            if obj.id == self.id: # 确保不是自车
                return False
            return True

        mask = np.zeros([self.solver_config.cbf_config.N])
        info = np.zeros([self.solver_config.cbf_config.N, self.solver_config.cbf_config.info_dim])
        
        for idx, (oid, obj) in enumerate(self.engine.get_objects(filter=filter_func).items()):
            obj = cast(BaseVehicle, obj)
            if self._is_next_lane(obj):
                mask[idx] = 1
                info[idx, 0: 2] = obj.get_state().get('position')[0: 2] # 简单地将位置作为信息传递给求解器
        return mask, info.flatten()

            
    def _is_next_lane(self, other_vehicle: BaseVehicle) -> bool:
        lane1 = self.navigation.current_lane.index
        lane2 = other_vehicle.navigation.current_lane.index
        
        if lane1[0] == lane2[0] and lane1[1] == lane2[1] and abs(lane1[2] - lane2[2]) == 1:
            return True
        
        return False
    
    @property
    def change_lane(self) -> bool:
        if len(self._lane_list) < 2:
            return False
        return self._lane_list[0][-1] != self._lane_list[1][-1] # 如果index不等说明变道了
    
    @property
    def get_step_num(self) -> int:
        return self._step_num
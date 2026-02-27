from methods.common.alg import DefaultAlg
from methods._03_rl_mpc.controller import Controller
from methods._03_rl_mpc.mpc import MPC
from methods.common.ocp import DefaultCBF
from methods.common.envs import SingleEnv, ParallelEnv
from numpy import ndarray
import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch import Tensor
from methods.common.envs import ParallelEnv, SingleEnv
from methods.common.type import ActionType

from typing import Tuple, Dict, Union, Optional, List, cast
from tqdm import tqdm


class Alg(DefaultAlg):
    def _train(self):
        self.policy.train()
        total_sample_steps = self.config.sample_steps * self.config.n_process
        pbar = tqdm(total=self.config.epoch * (total_sample_steps // self.config.batch_size + int(total_sample_steps % self.config.batch_size != 0)), desc='train')
        clip_range = self._clip_range()
        self.policy.to(self.config.device)
        
        # policy_loss_list: List = []
        # value_loss_list: List = []
        # loss_list: List = []

        for epoch in range(self.config.epoch):
            for rollout_data in self.buffer.get():
                rollout_data.move_to_device(self.config.device)
                actions = rollout_data.actions
                
                # 如果是离散动作, 展平
                if self.config.distribution_config.action_space == ActionType.discrete:
                    actions = actions.flatten()
                
                values, log_probs, entropys = self.policy.evaluate_action(rollout_data.obss, actions)
                
                # advantage批内归一化
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_probs - rollout_data.old_log_probs)
                
                policy_loss = self._get_policy_loss(advantages, ratio, clip_range)
                value_loss = mse_loss(rollout_data.returns.flatten(), values.flatten())
                entropy_loss = -torch.mean(entropys)

                loss = policy_loss + self.config.entropy_coef * entropy_loss + self.config.value_loss_coef * value_loss
                
                # early stop机制
                if self._early_stop(log_probs, rollout_data.old_log_probs): break

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # ===============extra info==================#
                pbar.update(1)

                # policy_loss_list.append(policy_loss.item())
                # value_loss_list.append(value_loss.item())
                # loss_list.append(loss.item())
                # ===========================================#
        
        self.policy.to(torch.device(device='cpu'))
        pbar.close()

    def _get_policy_loss(self, advantages: Tensor, ratio: Tensor, clip_range: float):
        standard_loss_1 = advantages * ratio
        standard_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        return -torch.min(standard_loss_1, standard_loss_2).mean()
    
    def _trans_rl_to_control(self, actions: ndarray) -> ndarray:
        if len(actions.shape) == 1: actions = actions.reshape(1, -1) # 扩充维度
        low, high = self.config.action_space_range # 剪裁动作到[-1, 1]之间
        clipped_actions: ndarray = np.clip(actions, low, high)
        
        # 反归一化
        new_actions = np.empty_like(actions)
        new_actions[:, 0] = self._renorm(clipped_actions[:, 0], self.config.v_max, self.config.v_min)
        new_actions[:, 1] = self._renorm(clipped_actions[:, 1], self.config.theta_max, self.config.theta_min)
        return new_actions
    
    def _create_controller(self, env: Union[SingleEnv, ParallelEnv], eval_mode: bool) -> Controller:
        return Controller(env, config = self.config.controller_config,
            mpc_cls = MPC, cbf_cls = DefaultCBF, eval_mode = eval_mode)
from methods.common.alg import DefaultAlg
from methods.common.ocp import DefaultCBF
from methods.common.envs import SingleEnv, ParallelEnv
from numpy import ndarray
import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch import Tensor
from methods.common.envs import ParallelEnv, SingleEnv
from methods.common.type import ActionType
from methods.common.utils import converto_torch
from methods.common.ocp.config import OCPconfig
from methods.common.base_config import BaseAlgConfig
from methods.common.create_env import create_env, create_render_config
from typing import Tuple, Dict, Union, Optional, List, cast
from tqdm import tqdm
from functools import partial
from methods.common.type import VehicleState
from methods.common.controller import DefaultController

class Alg(DefaultAlg):
    def __init__(self, config: BaseAlgConfig, eval_mode: bool = False):
        super().__init__(config, eval_mode)
        self.ocp_config: OCPconfig = OCPconfig()

    def _sample(self):
        pbar = tqdm(total=self.config.sample_steps, desc='sample')
        # set_random_seed(0, True) # 对齐用
        
        curr_step: int    = 0
        self.buffer.reset() # 采样前先清空buffer
        state_temp: ndarray = np.zeros(shape=[self.config.n_process, 2])
        
        while curr_step < self.config.sample_steps:
            # 上层决策
            actions, log_probs, values = self.policy.select_action(self._last_obss)
            transed_actions = self._trans_rl_to_control(actions)
            
            obs_nexts, rewards, dones, step_infos = self.env.step(transed_actions)
            rewards = self._bootstraping(rewards, dones, step_infos)
            
            # 存入buffer
            self.buffer.push(self._last_obss, actions, rewards, self._last_dones, log_probs, values,  # 正常ppo的
                             state_temp, state_temp, )
            
            # 存入监视器
            self.monitor.collect_rewards(rewards, dones)

            self._last_obss = obs_nexts # update observation
            self._last_dones = dones    # update done
            curr_step += 1              # update curr step
            pbar.update(1)              # update pbar

        with torch.no_grad():
            values = self.policy.predict_value(converto_torch(obs_nexts))
        dones = torch.from_numpy(dones).to(torch.long)
        
        self.buffer.compute_advantage(values, dones)
        pbar.close()
    
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

    
    def _evaluate(self, is_render: bool = False, evaluate_save_path: str = '') -> Tuple[float, List[Dict]]:
        self.policy.eval()
        # 创建一个新的环境
        env_eval: SingleEnv = SingleEnv(
            partial(create_env, self.config.metadriveenv_config),
            self.config.parallel_env_config,
        )
        obs = env_eval.reset()

        # 创建一个新的控制器
        self.controller_eval: DefaultController = self._create_controller(env=env_eval, eval_mode=True)

        last_done = np.ones(shape=[1, ])
        total_reward = 0.0
        extra_infos: List[Dict] = list()
        if is_render: render_row_text = self._create_render_text()
        
        for _ in range(self.config.evaluate_total_steps):
            action, _ = self.predict(obs, deterministic=True)
            transed_action = self._trans_rl_to_control(action)
            obs, reward, done, step_info = env_eval.step(transed_action.reshape(-1))
            
            total_reward += reward
            extra_infos.append(self._get_extra_info())
            if is_render: self.render_class.add_frame(self._render(render_row_text, env_eval))
            
            if done:
                break
            
            last_done = np.array([done], dtype=np.long)
        
        if is_render: self.render_class.generate_gif(evaluate_save_path)
        # 清除用于评估的环境
        env_eval.close()
        del env_eval
        return total_reward, extra_infos

    def _get_extra_info(self) -> Dict:
        vehicle_state: VehicleState = self.controller_eval._get_vehicle_state()[0]
        return {'vehicle_state': vehicle_state}

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
        new_actions[:, 0] = self._renorm(clipped_actions[:, 0], self.ocp_config.a_max, self.ocp_config.a_min)
        new_actions[:, 1] = self._renorm(clipped_actions[:, 1], self.ocp_config.delta_max, self.ocp_config.delta_min)
        return new_actions
    
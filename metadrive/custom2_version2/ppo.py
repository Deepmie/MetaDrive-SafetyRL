import sys
sys.path.append('/workspace/metadrive-github/metadrive')
import numpy as np
from torch import Tensor
from numpy import ndarray
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
from metadrive.custom2_version2.base_config import PPOConfig
from metadrive.custom2_version2.policy import Policy
from metadrive.custom2_version2.buffer import RolloutBuffer, RolloutBatchData
from metadrive.custom2_version2.envs import ParallelEnv, SingleEnv
from metadrive.custom2_version2.schedule import ConstantSchedule
from metadrive.custom2_version2.utils import converto_ndarray, converto_torch, Logger, Timer
from metadrive.custom2_version2.create_env import create_env, create_render_config
from metadrive.custom2_version2.type import ActionType, RenderClass
from metadrive.custom2_version2.controller import Controller
from metadrive.custom2_version2.monitor import Monitor
from typing import Tuple, Dict, Union, Optional, List, cast
from tqdm import tqdm
from functools import partial
from datetime import datetime
import os

class PPO:
    def __init__(self, config: PPOConfig, eval_mode: bool = False):
        self.config = config
        self.now_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # self.final_evaluate_path = f'{self.config.evaluate_save_root}/demo_{self.now_datetime}.gif'
        self.eval_mode = eval_mode

        self.env: ParallelEnv = ParallelEnv(
            [partial(create_env, config.metadriveenv_config) for _ in range(self.config.parallel_env_config.n_process)],
            self.config.parallel_env_config,
        )
        # self.env_eval: SingleEnv = SingleEnv(
        #     partial(create_env, config.metadriveenv_config),
        #     self.config.parallel_env_config,
        # )
        self._create_logger()
        
        self.policy: Policy        = Policy(self.config.policy_config)
        self.buffer: RolloutBuffer = RolloutBuffer(self.config.buffer_config, logger=self.logger)
        self._last_obss            = self.env.reset() # 初始化第一次的env
        # self.env_eval.reset()
        
        self.controller: Controller      = Controller(self.env, self.config.controller_config, eval_mode=False)
        self.timer: Timer                = Timer()
        self.render_class: RenderClass   = RenderClass()
        self.monitor: Monitor            = Monitor(self.logger_path)
        
        self.schedule    = ConstantSchedule(self.config.epsilon)

        # ======== 一些常量 ======== #
        self._last_dones = np.array([True, ]) # 初始化第一次的done
        self._remaining  = 1.0
        self.num_steps   = 0.0
        self._start_successful: Optional[bool] = None

        # ====== 训练的初始化 ====== #
        self.optimizer = Adam(self.policy.parameters(), lr=self.config.learning_rate)
        # self.optimizer.load_state_dict(torch.load('/workspace/model_weight/optimizer.pth'))
    
    def start(self) -> Tuple[bool, str]:
        _process_name_ = 'Sample & Train'
        self.timer.start(_process_name_)
        is_suc, info = self._start()
        if is_suc: self.logger.write_time(self.timer.end(), _process_name_)
        return is_suc, info
    
    def _start(self) -> Tuple[bool, str]:
        # try:
        pbar = tqdm(total=self.config.total_steps, desc='total')
        iterations   = 0
        evaluate_idx = 0
        best_reward  = -float('inf')
        while self.num_steps < self.config.total_steps:
            print('\nstart to sample...')
            self._sample()
            iterations += 1
            self.num_steps = self.config.n_process * iterations * self.config.sample_steps
            self._update_remaining(self.num_steps)
            pbar.update(self.config.n_process * self.config.sample_steps)

            print('\nstart to train...')
            self._train()

            if self.num_steps >= (evaluate_idx + 1) * self.config.evaluate_steps:
                print('\nstart to evaluate & save...')
                evaluate_reward = self._evaluate()
                if evaluate_reward > best_reward: self._save(evaluate_reward=evaluate_reward, ckp_pth=self.config.best_policy_checkpoint_pth); best_reward = evaluate_reward
                self.logger.write_reward(evaluate_reward, best_reward=best_reward)
                evaluate_idx += 1

                self._save(evaluate_reward=evaluate_reward, ckp_pth=self.config.policy_checkpoint_pth)
        
        self._start_successful = True
        start_info = 'Successful!'
        # except Exception:
        #     self._start_successful = False
        #     start_info = traceback.format_exc()
        # finally:
        #     if not self._start_successful:
        #         self.close()
        return self._start_successful, start_info
    
    def final_eval(self, evaluate_path: Optional[str] = None):
        reward_eval = self._evaluate(True, evaluate_path)
        print(f'episode_reward {reward_eval}')
        print('gif generation is finished ...')
    
    def predict(self, obs: Union[ndarray, Tensor], state: Optional[ndarray] = None, deterministic: bool = False):
        action, state = self.policy.predict(obs, state, deterministic)
        return action, state
    
    def close(self):
        self.env.close()
        # self.env_eval.close()
        if hasattr(self, 'logger') and self.logger is not None: self.logger.close()

    def load_weight_from_checkpoint(self, load_path: Optional[str] = None) -> Dict:
        if load_path is None:
            load_path = self.config.policy_checkpoint_pth
        meatadata = self._load(load_path)
        print(f'load weight from `{load_path}` successfully!')
        return meatadata

    def _sample(self):
        pbar = tqdm(total=self.config.sample_steps, desc='sample')
        # set_random_seed(0, True) # 对齐用
        
        curr_step: int = 0
        self.buffer.reset() # 采样前先清空buffer

        while curr_step < self.config.sample_steps:
            # 上层决策
            actions, log_probs, values = self.policy.select_action(self._last_obss)
            transed_actions = self._trans_rl_to_control(actions)
            
            # 底层控制
            controller_result, _ = self.controller.control(transed_actions, self._last_dones, curr_step)
            
            obs_nexts, rewards, dones, step_infos = self.env.step(controller_result.get_control_values_modified())
            rewards = self._bootstraping(rewards, dones, step_infos)
            ppc_rewards_errors: ndarray = controller_result.get_ppc_rewards_errors()
            rewards += ppc_rewards_errors[:, 0]
            
            # 存入buffer
            self.buffer.push(self._last_obss, actions, rewards, self._last_dones, log_probs, values,  # 正常ppo的
                             controller_result.get_state_values(), controller_result.get_state_values_modified(), )
            
            # 存入监视器
            self.monitor.collect_ppc_errors(ppc_rewards_errors[:, 1::], save_freq=self.config.monitor_save_freq)

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
                
                policy_loss = self._get_policy_loss(rollout_data.bc_index, rollout_data.standard_index, rollout_data.obss, advantages, ratio, rollout_data.z_mpcs, rollout_data.z_cbfs, clip_range)
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

    def _evaluate(self, is_render: bool = False, evaluate_save_path: str = '') -> Tuple:
        self.policy.eval()
        # 创建一个新的环境
        env_eval: SingleEnv = SingleEnv(
            partial(create_env, self.config.metadriveenv_config),
            self.config.parallel_env_config,
        )
        obs = env_eval.reset()

        # 创建一个新的控制器
        controller_eval: Controller = Controller(env_eval, self.config.controller_config, eval_mode=True)

        last_done = np.ones(shape=[1, ])
        total_reward = 0.0
        if is_render: render_row_text = self._create_render_text()
        
        for _ in range(self.config.evaluate_total_steps):
            action, _ = self.predict(obs, deterministic=True)
            transed_action = self._trans_rl_to_control(action)
            controller_result, extra_info = controller_eval.control(transed_action, last_done)
            obs, reward, done, step_info = env_eval.step(controller_result.get_control_values_modified())
            
            total_reward += reward
            if is_render: self.render_class.add_frame(self._render(render_row_text, env_eval))
            
            if done:
                break
            
            last_done = np.array([done], dtype=np.long)
        
        if is_render: self.render_class.generate_gif(evaluate_save_path)
        # 清除用于评估的环境
        env_eval.close()
        del env_eval
        return total_reward
    
    def _save(self, evaluate_reward: float, ckp_pth: str):
        # 保存checkpoint
        data = dict(policy_state_dict = self.policy.state_dict(), metadata = dict(evaluate_reward = evaluate_reward), )
        torch.save(data, ckp_pth)
    
    def _load(self, load_path: str) -> Dict:
        data: Dict = torch.load(f=load_path)
        self.policy.load_state_dict(data.get('policy_state_dict'))
        return data.get('metadata')
    
    def _render(self, text: Dict, env: SingleEnv) -> ndarray:
        if 'step' in text: text['step'] = self.render_class.render_index
        return env.render(**create_render_config(text = text))
    
    def _create_render_text(self) -> Dict:
        metadriveenv_config = self.config.metadriveenv_config
        return dict(traffic_mode = metadriveenv_config.traffic_mode, step = None)
    
    def _get_policy_loss(self, bc_index: Tensor, standard_index: Tensor, obss: Tensor, advantages: Tensor, ratio: Tensor, z_mpcs: Tensor, z_cbfs: Tensor, clip_range: float) -> Tensor:
        # caculate standard loss:
        if (~standard_index).all():
            standard_loss = 0.0
        else:
            standard_loss_1 = advantages[standard_index] * ratio[standard_index]
            standard_loss_2 = advantages[standard_index] * torch.clamp(ratio[standard_index], 1 - clip_range, 1 + clip_range)
            standard_loss = -torch.min(standard_loss_1, standard_loss_2).mean()
        
        # caculate bc loss:
        if (~bc_index).all():
            bc_loss = 0.0
        else:
            pi_action = self.policy.act_mean(obss[bc_index])
            omega = 1 + torch.exp(torch.norm(z_cbfs[bc_index] - z_mpcs[bc_index], p=2, dim=1))
            bc_loss = (omega * torch.norm(z_cbfs[bc_index] - pi_action, p=2, dim=1)).mean()

        policy_loss = standard_loss + self.config.bc_coef * bc_loss
        return policy_loss

    def _bootstraping(self, rewards: ndarray, dones: ndarray, step_infos: List[Dict]) -> ndarray:
        '''
        根据done和step_info的情况考虑是否修正reward, 如果是因为timelimit导致的done, 则考虑用value(obs)修正当前的reward
        '''
        for idx, done in enumerate(dones):
            done = cast(ndarray, done)
            if (
                done.item() # 完成了(截断 or 成功)
                and step_infos[idx].get('terminal_observation', None) is not None # (有终端观测值)
                and step_infos[idx].get('TimeLimit.truncated', False) # (是截断)
            ):
                with torch.no_grad():
                    terminal_value = self.policy.predict_value(step_infos[idx].get('terminal_observation'))
                    terminal_value: ndarray = converto_ndarray(terminal_value)
                rewards[idx] += self.config.buffer_config.gamma * terminal_value
        return rewards
    
    def _early_stop(self, log_prob: Tensor, old_log_probs: RolloutBatchData) -> bool:
        with torch.no_grad():
            log_ratio = log_prob - old_log_probs
            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
        
        if self.config.target_kl is not None and approx_kl_div > 1.5 * self.config.target_kl:
            return True
        return False

    def _update_remaining(self, num_step: int):
        self._remaining = 1.0 - float(num_step / self.config.total_steps)
    
    def _clip_range(self) -> float:
        return self.schedule(self._remaining)
    
    def _trans_rl_to_control(self, actions: ndarray) -> ndarray:
        if len(actions.shape) == 1: actions = actions.reshape(1, -1) # 扩充维度
        low, high = self.config.action_space_range # 剪裁动作到[-1, 1]之间
        clipped_actions: ndarray = np.clip(actions, low, high)
        
        # 反归一化
        new_actions = np.empty_like(actions)
        new_actions[:, 0] = (clipped_actions[:, 0] + 1) / 2 * (self.config.v_max - self.config.v_min) + self.config.v_min
        new_actions[:, 1] = (clipped_actions[:, 1] + 1) / 2 * (self.config.theta_max - self.config.theta_min) + self.config.theta_min
        return new_actions
    
    def _create_logger(self):
        if not self.eval_mode:
            self.logger: Logger   = Logger(self.config.logger_config)
            self.logger_path: str = self.logger.get_logger_path().strip()

            if hasattr(self, 'final_evaluate_path'):
                self.logger.write_tabel_additional_params(['final_evaluate_path'], [os.path.basename(self.final_evaluate_path)])

            with open('dp_single_version2/logger_path.txt', mode='w', encoding='utf-8') as writer:
                writer.write(self.logger_path)
        else:
            self.logger = None



if __name__ == '__main__':
    ppo_config = PPOConfig()
    ppo = PPO(ppo_config)
    
    # 开始执行ppo算法
    ppo.start()
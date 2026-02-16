from methods.common.alg import DefaultAlg
from methods._00_rl_mpc_cbf_ppc_traj.controller import Controller
from methods._00_rl_mpc_cbf_ppc_traj.mpc import MPC
from methods.common.ocp import DefaultCBF
from methods.common.envs import SingleEnv, ParallelEnv
from numpy import ndarray
import numpy as np
from typing import Union

class Alg(DefaultAlg):
    def _trans_rl_to_control(self, actions: ndarray) -> ndarray:
        if len(actions.shape) == 1: actions = actions.reshape(1, -1) # 扩充维度
        low, high = self.config.action_space_range # 剪裁动作到[-1, 1]之间
        clipped_actions: ndarray = np.clip(actions, low, high)
        
        # 反归一化
        new_actions = np.empty_like(actions)
        new_actions[:, 0] = self._renorm(clipped_actions[:, 0], self.config.v_max, self.config.v_min)
        new_actions[:, 1] = self._renorm(clipped_actions[:, 1], self.config.theta_max, self.config.theta_min)
        new_actions[:, 2] = self._renorm(clipped_actions[:, 2], self.config.alpha_max, self.config.alpha_min)
        new_actions[:, 3] = self._renorm(clipped_actions[:, 3], self.config.alpha_max, self.config.alpha_min)
        return new_actions

    def _create_controller(self, env: Union[SingleEnv, ParallelEnv], eval_mode: bool) -> Controller:
        return Controller(env, config = self.config.controller_config,
            mpc_cls = MPC, cbf_cls = DefaultCBF, eval_mode = eval_mode)
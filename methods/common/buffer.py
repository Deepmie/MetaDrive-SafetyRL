from methods.common.base_config import RolloutBufferConfig
from methods.common.utils.logger import Logger
from dataclasses import dataclass
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from typing import Optional, List

@dataclass
class RolloutBatchData:
    obss: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    z_mpcs: Tensor
    z_cbfs: Tensor
    bc_index: Tensor
    standard_index: Tensor
    
    def move_to_device(self, device: torch.device):
        self.obss           = self.obss.to(device)
        self.actions        = self.actions.to(device)
        self.rewards        = self.rewards.to(device)
        self.dones          = self.dones.to(device)
        self.old_log_probs  = self.old_log_probs.to(device)
        self.old_values     = self.old_values.to(device)
        self.advantages     = self.advantages.to(device)
        self.returns        = self.returns.to(device)
        self.z_mpcs         = self.z_mpcs.to(device)
        self.z_cbfs         = self.z_cbfs.to(device)
        self.bc_index       = self.bc_index.to(device)
        self.standard_index = self.standard_index.to(device)


class RolloutBuffer:
    def __init__(self, config: RolloutBufferConfig, logger: Optional[Logger]):
        self.config = config
        if logger is not None:
            self.logger = logger
        self.quantile_list: List = [0.5, 0.75, 0.9, 0.95, 0.99] # 选择的分位点列表
        self.reset()
    
    def push(self, states: ndarray, actions: ndarray, rewards: ndarray, dones: ndarray, log_probs: Tensor, values: Tensor, z_mpcs: ndarray, z_cbfs: ndarray):
        if self._idx >= self.config.max_buffer_size:
            raise IndexError(f'index out of range, max buffer size = {self.config.max_buffer_size}.')
        
        _slice = slice(self._idx, self._idx + self.config.n_process, 1)

        self.states[_slice, ...]   = torch.from_numpy(states).to(torch.float32)
        self.actions[_slice, ...]  = torch.from_numpy(actions).to(torch.float32)
        self.rewards[_slice, :]    = torch.from_numpy(rewards).to(torch.float32)
        self.dones[_slice, :]      = torch.from_numpy(dones).to(torch.long)
        self.log_probs[_slice, :]  = log_probs.clone().detach().to(torch.float32)
        self.values[_slice, :]     = values.clone().detach().to(torch.float32)
        self.z_mpcs[_slice, ...]   = torch.from_numpy(z_mpcs).to(torch.float32)
        self.z_cbfs[_slice, ...]   = torch.from_numpy(z_cbfs).to(torch.float32)
        self._idx += 1 # 索引后移n位

    def compute_advantage(self, values: Tensor, dones: Tensor):
        advantage_lasts: int = 0
        for t in range(self._idx-1, -1, -1):
            if t == self._idx-1: # 初始条件
                next_non_terminals = 1.0 - dones
                next_values = values
            else:
                next_non_terminals = 1.0 - self.dones[t+1]
                next_values = self.values[t+1]

            # 计算优势函数
            delta_t = self.rewards[t] + self.config.gamma * next_non_terminals * next_values - self.values[t]
            advantage_lasts = delta_t + self.config.gamma * self.config.gae_lambda * next_non_terminals * advantage_lasts
            self.advantages[t] = advantage_lasts
        
        # 累积奖励
        self.returns = self.advantages + self.values

    def reset(self):
        self.states: Tensor     = self._init_tensor(self.config.max_buffer_size, self.config.state_dim, dtype=torch.float32)        # 2d vector
        self.actions: Tensor    = self._init_tensor(self.config.max_buffer_size, self.config.action_dim, dtype=torch.float32)       # 2d vector
        self.rewards: Tensor    = self._init_tensor(self.config.max_buffer_size, dtype=torch.float32)                               # 1d vector
        self.dones: Tensor      = self._init_tensor(self.config.max_buffer_size, dtype=torch.long)                                  # 1d vector
        self.log_probs: Tensor  = self._init_tensor(self.config.max_buffer_size, dtype=torch.float32)                               # 1d vector
        self.values: Tensor     = self._init_tensor(self.config.max_buffer_size, dtype=torch.float32)                               # 1d vector
        self.z_mpcs: Tensor     = self._init_tensor(self.config.max_buffer_size, self.config.high_state_dim, dtype=torch.float32)   # 2d vector
        self.z_cbfs: Tensor     = self._init_tensor(self.config.max_buffer_size, self.config.high_state_dim, dtype=torch.float32)   # 2d vector
        
        self.advantages: Tensor = self._init_tensor(self.config.max_buffer_size, dtype=torch.float32)                               # 1d vector
        self.returns: Tensor    = self._init_tensor(self.config.max_buffer_size, dtype=torch.float32)                               # 1d vector

        # ============other data========== #
        self._idx = 0
        self._next_value = 0
        self._have_flatten = False
        # ================================ #

    def get(self): # 构造一个生成器
        _len = self._idx * self.config.n_process
        indices = np.random.permutation(_len)

        if not self._have_flatten:
            self._flatten_tensor() # 展平所有的变量
            self._have_flatten = True
        
        start_idx = 0
        while start_idx < _len:
            end_idx = min(start_idx + self.config.batch_size, _len)
            indice = indices[start_idx: end_idx]
            start_idx += self.config.batch_size
            yield \
            RolloutBatchData(
                obss           = self.states[indice],
                actions        = self.actions[indice],
                rewards        = self.rewards[indice].flatten(),
                dones          = self.dones[indice].flatten(),
                old_log_probs  = self.log_probs[indice].flatten(),
                old_values     = self.values[indice].flatten(),
                advantages     = self.advantages[indice].flatten(),
                returns        = self.returns[indice].flatten(),
                z_mpcs         = self.z_mpcs[indice],
                z_cbfs         = self.z_cbfs[indice],
                bc_index       = self.bc_index[indice],
                standard_index = self.standard_index[indice],
            )

    def _flatten_tensor(self):
        self.states      = self._flatten(self.states)
        self.actions     = self._flatten(self.actions)
        self.rewards     = self._flatten(self.rewards)
        self.dones       = self._flatten(self.dones)
        self.log_probs   = self._flatten(self.log_probs)
        self.values      = self._flatten(self.values)
        self.advantages  = self._flatten(self.advantages)
        self.returns     = self._flatten(self.returns)
        self.z_mpcs      = self._flatten(self.z_mpcs)
        self.z_cbfs      = self._flatten(self.z_cbfs)
        self._get_bc_index()   # 计算bc_index
    
    def _get_bc_index(self):
        cbf_mpc_delta = torch.norm(self.z_cbfs - self.z_mpcs, p=1, dim=1)
        self.bc_index = cbf_mpc_delta > self.config.delta_bc
        self.standard_index = (self.bc_index == False)
        
        if self.logger is not None: # 如果不是None, 则记录ratio
            ratio: float = self.bc_index.sum().item() / self.bc_index.shape[0]
            metadata = dict(
                max  = cbf_mpc_delta.max().item(), min  = cbf_mpc_delta.min().item(),
                mean = cbf_mpc_delta.mean().item(), non_zero_mean = cbf_mpc_delta[cbf_mpc_delta > 1e-6].mean().item(),
            )
            quantile_value: Tensor = torch.quantile(cbf_mpc_delta, torch.tensor(self.quantile_list, device=cbf_mpc_delta.device))
            for ql, qv in zip(self.quantile_list, quantile_value.tolist()): metadata[f'{int(ql * 100)}%'] = float(qv)
            self.logger.write_cbf_ratio(r = ratio, metadata=metadata)
    
    def _flatten(self, t: Tensor): # t.shape = 2 or 3
        if len(t.shape) < 3:
            t = t.unsqueeze(dim=-1)
        return t.permute(1, 0, 2).reshape(-1, t.shape[-1])
    
    def _reset(self, t: Tensor):
        t = t.zero_().reshape()
    
    def _init_tensor(self, num: int, dim: Optional[int] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        if dim is None: # 1维创建1维向量
            return torch.zeros(size=[num, self.config.n_process, ], dtype=dtype)
        return torch.zeros(size=[num, self.config.n_process, dim], dtype=dtype)
    


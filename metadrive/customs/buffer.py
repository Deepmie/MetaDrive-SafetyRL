from dataclasses import dataclass
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from metadrive.customs.config import RolloutBufferConfig

class RolloutBuffer(Dataset):
    def __init__(self, config: RolloutBufferConfig):
        self.config = config
        self.state: Tensor = self._build_init_vector(config.max_buffer_size, config.state_dim, dtype=torch.float32)   # state_dim d vector
        self.z_mpc: Tensor = self._build_init_vector(config.max_buffer_size, 2, dtype=torch.float32)                  # 2d vector
        self.z_cbf: Tensor = self._build_init_vector(config.max_buffer_size, 2, dtype=torch.float32)                  # 2d vector
        self.action: Tensor = self._build_init_vector(config.max_buffer_size, config.action_dim, dtype=torch.float32) # 2d vector
        self.action_cbf: Tensor = self._build_init_vector(config.max_buffer_size, config.action_dim, dtype=torch.float32) # 2d vector
        self.reward: Tensor = self._build_init_vector(config.max_buffer_size, dtype=torch.float32)                    # 1d vector
        self.done: Tensor = self._build_init_vector(config.max_buffer_size, dtype=torch.long)                         # 1d vector
        self.log_prob: Tensor = self._build_init_vector(config.max_buffer_size, dtype=torch.float32)                  # 1d vector
        self.value: Tensor = self._build_init_vector(config.max_buffer_size, dtype=torch.float32)                     # 1d vector
        self.advantage: Tensor = self._build_init_vector(config.max_buffer_size, dtype=torch.float32)                 # 1d vector
        self.reward_accum: Tensor = self._build_init_vector(config.max_buffer_size, dtype=torch.float32)              # 1d vector
        self._idx = 0
        self._next_value = 0
    
    def push(self, state: ndarray, z_mpc: ndarray, z_cbf: ndarray, action: ndarray, action_cbf: ndarray, reward: float, done: int, log_prob: float, value: float):
        if self._idx > self.config.max_buffer_size-1:
            raise IndexError(f'index out of range, max buffer size = {self.config.max_buffer_size}.')
        
        self.state[self._idx, :] = torch.from_numpy(state).to(torch.float32)
        self.z_mpc[self._idx, :] = torch.from_numpy(z_mpc).to(torch.float32)
        self.z_cbf[self._idx, :] = torch.from_numpy(z_cbf).to(torch.float32)
        self.action[self._idx, :] = torch.from_numpy(action).to(torch.float32)
        self.action_cbf[self._idx, :] = torch.from_numpy(action_cbf).to(torch.float32)
        self.reward[self._idx] = torch.tensor(reward, dtype=torch.float32)
        self.done[self._idx] = torch.tensor(done, dtype=torch.long)
        self.log_prob[self._idx] = torch.tensor(log_prob, dtype=torch.float32)
        self.value[self._idx] = torch.tensor(value, dtype=torch.float32)
        self._idx += 1 # 索引后移一位
    
    def _build_init_vector(self, num: int, dim: Optional[int] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        if dim is None: # 1维创建1维向量
            return torch.zeros(size=[num, ], dtype=dtype)
        return torch.zeros(size=[num, dim], dtype=dtype)
    
    def __getitem__(self, idx: int):
        return (
            self.state[idx, :],      # t时刻的状态 s
            self.z_mpc[idx, :],      # t+1时刻的状态s (mpc预测的)
            self.z_cbf[idx, :],      # t+1时刻的状态s (cbf纠正后的)
            self.action[idx, :],     # t时刻的动作 a
            self.action_cbf[idx, :], # t时刻的修正动作 a_cbf
            self.reward[idx],        # t时刻的奖励 r
            self.done[idx],          # t时刻是否完成游戏
            self.log_prob[idx],      # t时刻的对数概率值 pi(a|s)
            self.value[idx],         # t时刻的基线值 V(s)
            self.advantage[idx],     # t时刻的优势函数 A(s, a)
            self.reward_accum[idx],  # t时刻的奖励累积值 sum gamma * r
        )
    
    def __len__(self):
        return self._idx
    
    def _build_dataloader(self) -> DataLoader: # 构造dataloader
        dataloader = DataLoader(dataset=self, batch_size=self.config.batch_size, shuffle=True)
        self.dataloader_len = len(dataloader)
        return dataloader
    
    @property
    def dataloader(self):
        self._compute_advantage()
        return self._build_dataloader()
    
    def set_next_value(self, value: float):
        self._next_value = value

    def _compute_advantage(self):
        advantage_temp = self._build_init_vector(self._idx, dtype=torch.float32)
        
        for t in range(self._idx-1, -1, -1):
            if t == self._idx-1: # 初始条件
                advantage_temp[t] = self.reward[t] + self.config.gamma * self._next_value - self.value[t]
                self.reward_accum[t] = self.reward[t]
                self._next_value = 0
            else:
                # 计算优势函数
                delta_t = self.reward[t] + (1 - self.done[t+1]) * self.config.gamma * self.value[t+1] - self.value[t]
                advantage_temp[t] = delta_t + (1 - self.done[t+1]) * self.config.gamma * self.config.lamb  * advantage_temp[t+1]

                # 计算累加奖励
                self.reward_accum[t] = self.reward[t] + self.config.gamma * (1 - self.done[t+1]) * self.reward_accum[t+1]
            
        # 正则化
        advantage_temp = (advantage_temp - advantage_temp.mean()) / (advantage_temp.std() + 1e-8)
        self.advantage[0: self._idx] = advantage_temp
    
    def reset(self):
        self._idx = 0
        self._next_value = 0
        self.state.zero_()
        self.action.zero_()
        self.reward.zero_()
        self.done.zero_()
        self.log_prob.zero_()
        self.value.zero_()
        self.advantage.zero_()
        self.reward_accum.zero_()


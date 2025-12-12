from dataclasses import dataclass
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
import torch.multiprocessing as mp
from typing import List, Optional
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from metadrive.customs_parallel.config import RolloutBufferConfig

class RolloutBuffer(Dataset):
    def __init__(self, config: RolloutBufferConfig):
        self.config = config
        total_buffer_size = config.sample_subprocess_num * config.single_buffer_size
        
        self.state: Tensor        = self._build_init_vector(total_buffer_size, config.state_dim, dtype=torch.float32)   # state_dim d vector
        self.z_mpc: Tensor        = self._build_init_vector(total_buffer_size, 2, dtype=torch.float32)                  # 2d vector
        self.z_cbf: Tensor        = self._build_init_vector(total_buffer_size, 2, dtype=torch.float32)                  # 2d vector
        self.action: Tensor       = self._build_init_vector(total_buffer_size, config.action_dim, dtype=torch.float32)  # 2d vector
        self.action_cbf: Tensor   = self._build_init_vector(total_buffer_size, config.action_dim, dtype=torch.float32)  # 2d vector
        self.reward: Tensor       = self._build_init_vector(total_buffer_size, dtype=torch.float32)                     # 1d vector
        self.done: Tensor         = self._build_init_vector(total_buffer_size, dtype=torch.long)                        # 1d vector
        self.log_prob: Tensor     = self._build_init_vector(total_buffer_size, dtype=torch.float32)                     # 1d vector
        self.value: Tensor        = self._build_init_vector(total_buffer_size, dtype=torch.float32)                     # 1d vector
        self.advantage: Tensor    = self._build_init_vector(total_buffer_size, dtype=torch.float32)                     # 1d vector
        self.reward_accum: Tensor = self._build_init_vector(total_buffer_size, dtype=torch.float32)                     # 1d vector

        self._len = mp.Value('i', 0)
        self._dataloader: Optional[DataLoader] = None
    
    def push(
            self, idx: int, state: ndarray, z_mpc: ndarray, z_cbf: ndarray, action: ndarray,
            action_cbf: ndarray, reward: float, done: int, log_prob: float, value: float
        ):
        
        self.state[idx, :]      = torch.from_numpy(state).to(torch.float32)
        self.z_mpc[idx, :]      = torch.from_numpy(z_mpc).to(torch.float32)
        self.z_cbf[idx, :]      = torch.from_numpy(z_cbf).to(torch.float32)
        self.action[idx, :]     = torch.from_numpy(action).to(torch.float32)
        self.action_cbf[idx, :] = torch.from_numpy(action_cbf).to(torch.float32)
        self.reward[idx]        = torch.tensor(reward, dtype=torch.float32)
        self.done[idx]          = torch.tensor(done, dtype=torch.long)
        self.log_prob[idx]      = torch.tensor(log_prob, dtype=torch.float32)
        self.value[idx]         = torch.tensor(value, dtype=torch.float32)
        
        with self._len.get_lock():
            self._len.value += 1
    
    def _build_init_vector(self, num: int, dim: Optional[int] = None, dtype: torch.dtype = torch.float32, is_shared: bool = True) -> Tensor:
        if dim is None: # 1维创建1维向量
            v = torch.zeros(size=[num, ], dtype=dtype)
        else:
            v = torch.zeros(size=[num, dim], dtype=dtype)
        if is_shared: v.share_memory_()
        return v
    
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
        return self._len.value
    
    def build_dataloader(self, world_size: int, rank: int): # 构造dataloader
        n = self._len.value
        if rank == 0:
            self.advantage[0: n] = self._normalization_advantage(self.advantage[0: n])
        
        sampler = DistributedSampler(self, num_replicas=world_size, rank=rank)
        self._dataloader = DataLoader(dataset=self, batch_size=self.config.batch_size, sampler=sampler)
    
    @property
    def dataloader(self):
        return self._dataloader

    @property
    def dataloader_len(self):
        return len(self._dataloader)

    def compute_advantage(self, start_idx: int, end_idx: int, value_next: float):
        # s: 0, e: 10
        duration = end_idx - start_idx
        advantage_temp = self._build_init_vector(duration, dtype=torch.float32, is_shared=False)
        
        for t in range(duration-1, -1, -1):
            idx = start_idx + t
            if t == duration-1: # 初始条件
                advantage_temp[t] = self.reward[idx] + self.config.gamma * value_next - self.value[idx]
                self.reward_accum[idx] = self.reward[idx]
            else:
                # 计算优势函数
                delta_t = self.reward[idx] + (1 - self.done[idx+1]) * self.config.gamma * self.value[idx+1] - self.value[idx]
                advantage_temp[t] = delta_t + (1 - self.done[idx+1]) * self.config.gamma * self.config.lamb  * advantage_temp[t+1]

                # 计算累加奖励
                self.reward_accum[idx] = self.reward[idx] + self.config.gamma * (1 - self.done[idx+1]) * self.reward_accum[idx+1]
        
        # 正则化
        # advantage_temp = (advantage_temp - advantage_temp.mean()) / (advantage_temp.std() + 1e-8)
        self.advantage[start_idx: end_idx] = advantage_temp
    
    def reset(self):
        with self._len.get_lock():
            self._len.value = 0
        
        self._dataloader = None
        self.state.zero_()
        self.action.zero_()
        self.reward.zero_()
        self.done.zero_()
        self.log_prob.zero_()
        self.value.zero_()
        self.advantage.zero_()
        self.reward_accum.zero_()
    
    def _normalization_advantage(self, advan: Tensor) -> Tensor:
        return (advan - advan.mean()) / (advan.std() + 1e-8)


from metadrive.customs_parallel.policy import PolicyManager
from metadrive.customs_parallel.config import TrainerConfig
from metadrive.customs_parallel.buffer import RolloutBuffer
from typing import Optional
import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import MSELoss
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import swanlab
from tqdm import tqdm


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.policy = None
        self.config = config
        self.mse_loss = MSELoss()
        self.opt_func = Adam
        self.global_count = 0

    def use_policy(self, policy: Optional[PolicyManager]):
        self.policy = policy
        self.optimizer = self.policy.optimizer(self.opt_func)

    def train(self, buffer: RolloutBuffer, rank: int):
        self.policy.gpu.train()
        buffer.build_dataloader(self.config.gpu_num, rank) # 构造dataloader
        
        if rank == 0:
            pbar = tqdm(total=buffer.dataloader_len * self.config.gpu_num * self.config.epoch, desc='Training', ncols=120)
        
        for epoch in range(self.config.epoch):
            avg_loss_epoch = 0
            for obs, z_mpc, z_cbf, action, action_cbf, reward, done, old_log_prob, value, advantage, reward_accum in buffer.dataloader:
                # 数据: CPU -> GPU
                obs, z_mpc, z_cbf, action, action_cbf, old_log_prob, value, advantage, reward_accum = \
                obs.to(rank), z_mpc.to(rank), z_cbf.to(rank), \
                action.to(rank), action_cbf.to(rank), old_log_prob.to(rank), \
                value.to(rank), advantage.to(rank), reward_accum.to(rank)

                log_prob, entropy, value_new = self.policy.gpu.module.evaluate_action(obs, action)
                ratios = torch.exp(log_prob - old_log_prob)
                
                # create index
                bc_index = torch.norm(z_cbf, p=1, dim=1) > self.config.delta_bc
                stand_index = (bc_index == False)

                # standard loss
                if (~stand_index).all():
                    standard_loss = 0
                else:
                    surr1 = ratios[stand_index] * advantage[stand_index]
                    surr2 = torch.clamp(ratios[stand_index], 1-self.config.epsilon, 1+self.config.epsilon) * advantage[stand_index]
                    standard_loss = -torch.min(surr1, surr2).mean()

                # bc loss
                if (~bc_index).all():
                    bc_loss = 0
                else:
                    pi_action = self.policy.gpu.module.act_mean(obs[bc_index])
                    omega = 1 + torch.exp(torch.norm(z_cbf[bc_index] - z_mpc[bc_index], p=2, dim=1))
                    bc_loss = (omega * torch.norm(z_cbf[bc_index] - pi_action, p=2, dim=1)).mean()

                actor_loss = standard_loss + self.config.bc_coef * bc_loss
                critic_loss = 0.5 * self.mse_loss(value_new.reshape(-1), reward_accum)

                loss: Tensor = actor_loss + self.config.value_loss_coef * critic_loss - self.config.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                if self.config.max_grad_num is not None: clip_grad_norm_(self.policy.gpu.parameters(), self.config.max_grad_num)
                self.optimizer.step()

                avg_loss_epoch += loss
                if rank == 0:
                    pbar.update(self.config.gpu_num)
            
            with torch.no_grad():
                dist.all_reduce(avg_loss_epoch, op=dist.ReduceOp.SUM)
                avg_loss_epoch /= buffer.dataloader_len * self.config.gpu_num
            
            if self.config.use_swanlab and rank == 0:
                swanlab.log({f'loss/update{self.global_count}': avg_loss_epoch.item()})
        
        dist.barrier() # 确保所有rank完成训练

        for param in self.policy.gpu.parameters():
            dist.broadcast(param.data, src=0)
        
        dist.barrier()

        if rank == 0:
            self.global_count += 1
            self.policy.sync()
            pbar.close()
        
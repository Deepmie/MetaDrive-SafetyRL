from metadrive.custom2_version2.envs.utils import MetaProcess, Commond, CloudpickleWrapper
from metadrive.custom2_version2.envs.parallel.worker import worker_func
from metadrive.custom2_version2.base_config import ParallelEnvConfig
import multiprocessing as mp
from multiprocessing.context import BaseContext
from multiprocessing import Process
from typing import List, Tuple, Dict, cast
from numpy import ndarray
import numpy as np


class ParallelEnv:
    def __init__(self, envs: List, config: ParallelEnvConfig):
        self.config = config
        ctx: BaseContext = mp.get_context(config.start_method)
        self.meta_processs: List[MetaProcess] = []
        self.waiting = False
        self.closed = False
        self.reset_infos = list()

        for idx in range(self.config.n_process):
            par_remotes, sub_remotes = ctx.Pipe()
            process: Process = ctx.Process(target=worker_func, args=(par_remotes, sub_remotes, CloudpickleWrapper(envs[idx]), idx), daemon=True)
            process.start()
            meta_process = MetaProcess(par_remotes, sub_remotes, process)
            self.meta_processs.append(meta_process)
            sub_remotes.close()
    
    def step(self, actions: ndarray) -> Tuple:
        self.step_async(actions)
        return self.step_wait()
    
    def step_async(self, actions: ndarray): # (env_idx, action_dim)
        for meta_process, action in zip(self.meta_processs, actions):
            meta_process.par_remotes.send(Commond(name='step', args=(action, )))
        self.waiting = True

    def step_wait(self) -> Tuple:
        obss = np.empty([self.config.n_process, self.config.state_dim], dtype=np.float32)
        rewards = np.empty([self.config.n_process], dtype=np.float32)
        dones = np.empty([self.config.n_process], dtype=np.long)
        step_infos = list()
        self.reset_infos = list()

        for idx, meta_process in enumerate(self.meta_processs):
            obs, reward, done, step_info, reset_info = meta_process.par_remotes.recv()
            obss[idx, :]    = obs
            rewards[idx] = reward
            dones[idx]   = done
            step_infos.append(step_info)
            self.reset_infos.append(reset_info)
        self.waiting = False
        return obss, rewards, dones, step_infos
    
    def reset(self) -> Tuple:
        obss = np.empty([self.config.n_process, self.config.state_dim], dtype=np.float32)
        self.reset_infos = list()

        for meta_process in self.meta_processs:
            meta_process.par_remotes.send(Commond(name='reset'))
        
        for idx, meta_process in enumerate(self.meta_processs):
            obs, reset_info = meta_process.par_remotes.recv()
            obss[idx, :] = obs
            self.reset_infos.append(reset_info)
        return obss
    
    def get_state(self) -> List[Dict]:
        vehicle_states = list()
        for meta_process in self.meta_processs:
            meta_process.par_remotes.send(Commond(name='get_state'))
        
        for meta_process in self.meta_processs:
            state = meta_process.par_remotes.recv()
            vehicle_states.append(state)
        return vehicle_states

    def get_metadata(self) -> Dict:
        meta_process = self.meta_processs[0]
        meta_process.par_remotes.send(Commond(name='get_metadata'))
        return meta_process.par_remotes.recv()
    
    def get_all_vehicle_position(self) -> Tuple[List[ndarray], List[ndarray]]:
        infos = list()
        masks = list()
        for meta_process in self.meta_processs:
            meta_process.par_remotes.send(Commond(name='get_all_vehicle_position'))

        for meta_process in self.meta_processs:
            info, mask = meta_process.par_remotes.recv()
            info = cast(ndarray, info); mask = cast(ndarray, mask)
            infos.append(info); masks.append(mask)
        return infos, masks

    def close(self):
        if self.closed:
            return 

        if self.waiting:
            for meta_process in self.meta_processs:
                meta_process.par_remotes.recv()
        
        for meta_process in self.meta_processs:
            meta_process.par_remotes.send(Commond(name='close'))
        
        for meta_process in self.meta_processs:
            meta_process.process.join()
        
        self.closed = True
        

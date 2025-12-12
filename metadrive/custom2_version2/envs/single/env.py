from metadrive.custom2_version2.envs.utils import MetaProcess, Commond, CloudpickleWrapper
from metadrive.custom2_version2.envs.single.worker import worker_func
from metadrive.custom2_version2.base_config import ParallelEnvConfig
import multiprocessing as mp
from multiprocessing.context import BaseContext
from multiprocessing import Process
from typing import List, Tuple, Dict
from numpy import ndarray
import numpy as np


class SingleEnv:
    def __init__(self, env, config: ParallelEnvConfig):
        self.config = config
        ctx: BaseContext = mp.get_context(config.start_method)
        self.waiting = False
        self.closed = False

        # 只创建一个进程
        par_remotes, sub_remotes = ctx.Pipe()
        process: Process = ctx.Process(target=worker_func, args=(par_remotes, sub_remotes, CloudpickleWrapper(env), 0), daemon=True)
        process.start()
        self.meta_process = MetaProcess(par_remotes, sub_remotes, process)
        sub_remotes.close()
    
    def step(self, action: ndarray) -> Tuple:
        self.step_async(action)
        return self.step_wait()
    
    def step_async(self, action: ndarray): # (action_dim, )
        self.meta_process.par_remotes.send(Commond(name='step', args=(action, )))
        self.waiting = True

    def step_wait(self) -> Tuple:
        obs, reward, done, step_info, self.reset_info = self.meta_process.par_remotes.recv()
        self.waiting = False
        return obs, reward, done, step_info
    
    def reset(self) -> Tuple:
        self.meta_process.par_remotes.send(Commond(name='reset'))
        obs, self.reset_info = self.meta_process.par_remotes.recv()
        return obs
    
    def render(self, mode, screen_record, window, screen_size, camera_position, text):
        self.meta_process.par_remotes.send(Commond(name='render', args=(mode, screen_record, window, screen_size, camera_position, text)))
        return self.meta_process.par_remotes.recv()
    
    def get_state(self) -> Dict:
        self.meta_process.par_remotes.send(Commond(name='get_state'))
        return self.meta_process.par_remotes.recv()

    def get_metadata(self) -> Dict:
        self.meta_process.par_remotes.send(Commond(name='get_metadata'))
        return self.meta_process.par_remotes.recv()
    
    def close(self):
        if self.closed:
            return 

        if self.waiting:
            self.meta_process.par_remotes.recv()
        
        self.meta_process.par_remotes.send(Commond(name='close'))
        self.meta_process.process.join()
        self.closed = True
        

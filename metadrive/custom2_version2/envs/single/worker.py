from metadrive.custom2_version2.envs.utils import CloudpickleWrapper, Commond
from metadrive.component.vehicle.default_vehicle import DefaultVehicle
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.custom2_version2.utils import set_random_seed
from metadrive.base_class.base_object import BaseObject
from metadrive.custom2_version2.ocp import CBFconfig
from metadrive import MetaDriveEnv
from multiprocessing.connection import Connection
import traceback
from typing import Dict, Callable, Any, Tuple, cast
from numpy import ndarray
import numpy as np

class Worker:
    def __init__(
        self,
        par_remotes: Connection,
        sub_remotes: Connection,
        env_wrapper: CloudpickleWrapper,
        env_idx: int,
    ):
        self.par_remotes = par_remotes
        self.sub_remotes = sub_remotes
        self.par_remotes.close()
        self.env: MetaDriveEnv = env_wrapper.env() # 获取环境
        self.env_idx = env_idx
        self.running = False
        self.func_names: Dict[str, Callable] = {func_name: getattr(self, func_name) for func_name in dir(self) if not func_name.startswith('_')}
        self.cbf_config = CBFconfig()

    def step(self, action: ndarray) -> Tuple:
        obs, reward, terminated, truncated, step_info = self.env.step(action)
        done: bool = terminated or truncated
        reset_info = dict()
        step_info['TimeLimit.truncated'] = truncated and not terminated
        return obs, reward, done, step_info, reset_info
    
    def reset(self) -> Tuple:
        set_random_seed(0) # 重置随机数
        obs, reset_info = self.env.reset()
        return obs, reset_info
    
    def render(self, mode, screen_record, window, screen_size, camera_position, text) -> ndarray:
        return self.env.render(mode=mode, screen_record=screen_record, window=window, screen_size=screen_size, camera_position=camera_position, text=text)
    
    def get_state(self) -> Dict:
        agent: DefaultVehicle = self.env.agent
        state: Dict = agent.get_state()
        return state
    
    def get_metadata(self) -> Dict:
        agent: DefaultVehicle = self.env.agent
        return dict(
            l  = agent.FRONT_WHEELBASE + agent.REAR_WHEELBASE,
            lf = agent.FRONT_WHEELBASE,
            lr = agent.REAR_WHEELBASE,
        )
    
    def get_all_vehicle_position(self) -> Tuple[ndarray, ndarray]:
        # help function
        def _filter_other_object(obj: BaseObject):
            if not isinstance(obj, BaseVehicle):
                return False
            obj = cast(BaseVehicle, obj)
            if obj.id == self.env.agent.id:
                return False
            return True
        
        info = np.zeros([self.cbf_config.N, self.cbf_config.info_dim])
        mask = np.zeros([self.cbf_config.N, ])
        for idx, (oid, obj) in enumerate(self.env.engine.get_objects(filter=_filter_other_object).items()):
            obj = cast(BaseVehicle, obj)
            info[idx, 0: 2] = obj.position
            mask[idx] = 1
        return info, mask
    
    def close(self) -> str:
        self.env.close()
        self.running = False
        return 'close successful!'
    
    def run(self):
        self.par_remotes.close()
        self.running = True
        while self.running:
            try:
                cmd: Commond = self.sub_remotes.recv()
                
                if cmd.name == 'close': # exit条件
                    self._send(self.close())
                    break
                
                func: Callable = self.func_names.get(cmd.name, None)
                self._send(func(*cmd.args)) # 执行cmd
                
            except Exception:
                traceback.print_exc()
        self.sub_remotes.close()

    def _send(self, data: Any):
        self.sub_remotes.send(data)


# 
def worker_func(
    par_remotes: Connection,
    sub_remotes: Connection,
    env_wrapper: CloudpickleWrapper,
    env_idx: int,
):
    worker = Worker(par_remotes, sub_remotes, env_wrapper, env_idx)
    worker.run()




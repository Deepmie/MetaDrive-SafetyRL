import cloudpickle
from metadrive import MetaDriveEnv
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing import Process
from typing import Tuple


class CloudpickleWrapper:
    def __init__(self, env: MetaDriveEnv):
        self.env = env
    
    def __getstate__(self) -> MetaDriveEnv:
        return cloudpickle.dumps(self.env)

    def __setstate__(self, env: MetaDriveEnv) -> None:
        self.env = cloudpickle.loads(env)


@dataclass
class MetaProcess:
    par_remotes: Connection
    sub_remotes: Connection
    process: Process


@dataclass
class Commond:
    name: str
    args: Tuple = field(default_factory=tuple)


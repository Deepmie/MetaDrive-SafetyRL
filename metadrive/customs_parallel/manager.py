from multiprocessing.managers import BaseManager
from metadrive.customs_parallel.buffer import RolloutBuffer
from metadrive.customs_parallel.config import MainConfig
from metadrive.customs_parallel.type import Counter, Event

class DpManager(BaseManager):
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

pre_register_names = {
    'RolloutBuffer': RolloutBuffer,
    'Counts'       : Counter,
    'MainConfig'   : MainConfig,
    'Event'        : Event,
}

for key, value in pre_register_names.items():
    if key not in DpManager._registry:
        DpManager.register(key, value)

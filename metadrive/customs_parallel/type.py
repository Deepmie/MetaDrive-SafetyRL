from dataclasses import dataclass, field
from typing import Optional, List
import torch.multiprocessing as mp

@dataclass
class AgentInfo:
    # --------state-------- #
    x: Optional[float]     = None  # x坐标
    y: Optional[float]     = None  # y坐标
    v: Optional[float]     = None  # 速度, [标量]
    theta: Optional[float] = None  # 朝向
    
    # -----------u--------- #
    a: Optional[float]     = None  # 加速度
    delta: Optional[float] = None  # 转向角

    # ---------utils------- #
    l: Optional[float]     = None  # 轴距

class Counter:
    def __init__(self):
        self.counts: int = 0
    def __add__(self, v: int) -> 'Counter':
        self.counts = self.counts + v
        return self
    def clean(self):
        self.counts: int = 0
    @property
    def value(self):
        return self.counts
    def __lt__(self, v: int) -> bool: # <
        return self.counts < v
    def __gt__(self, v: int) -> bool: # >
        return self.counts > v
    def __eq__(self, v: int) -> bool: # ==
        return self.counts == v
    def __ne__(self, v: int) -> bool:
        return self.counts != v
    def __ge__(self, v: int) -> bool: # >=
        return self > v or self == v
    def __le__(self, v: int) -> bool: # <=
        return self < v or self == v

class Event:
    def __init__(self):
        self.event = False
    def set(self):
        self.event = True
    def is_set(self) -> bool:
        return self.event
    def clear(self):
        self.event = False

class Value:
    def get_lock(self):
        ...
    @property
    def value(self):
        ...


@dataclass
class EventManager:
    can_update: List[Event]      = field(default_factory=list)      # 是否能开始更新
    can_sample: List[Event]      = field(default_factory=list)      # 是否能开始采样
    is_finished: Event           = field(default_factory=mp.Event)  # 是否要终止整个程序

    def __post_init__(self):
        for c_event in self.can_update: c_event.clear()
        for c_event in self.can_sample: c_event.clear()
        self.is_finished.clear()
    
    def create_can_sample(self, N: int):
        self.can_sample = [mp.Event() for _ in range(N)]
    
    def create_can_update(self, N: int):
        self.can_update = [mp.Event() for _ in range(N)]


if __name__ == '__main__':
    c = Counter()
    c += 10
    
    print(c > 9)
    print(c > 11)
    print(c >= 10)


    
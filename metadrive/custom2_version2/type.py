from dataclasses import dataclass, field
from metadrive.utils.doc_utils import generate_gif
from typing import Optional, List
from numpy import ndarray
import numpy as np

@dataclass
class ActionType:
    discrete: int       = 0
    multi_discrete: int = 1
    box: int            = 2

@dataclass
class ActionSpace:
    action_type: str = ActionType.box

@dataclass
class VehicleState:
    # --------state-------- #
    x: Optional[float]     = None  # x坐标
    y: Optional[float]     = None  # y坐标
    v: Optional[float]     = None  # 速度, [标量]
    theta: Optional[float] = None  # 朝向
    # -----------u--------- #
    a: Optional[float]     = None  # 加速度
    delta: Optional[float] = None  # 转向角
    # # ---------utils------- #
    # l: Optional[float]     = None  # 轴距


@dataclass
class RenderClass:
    frames: List[ndarray]  = field(default_factory=list)
    render_index: int      = 0

    def add_frame(self, frame: ndarray):
        self.frames.append(frame)
        self.render_index += 1

    def reset(self):
        self.frames        = list()
        self.render_index  = 0

    def generate_gif(self, gif_name: str):
        generate_gif(self.frames, gif_name=gif_name)
        self.reset()


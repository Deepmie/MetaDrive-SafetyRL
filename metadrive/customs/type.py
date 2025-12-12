from dataclasses import dataclass
from typing import Optional

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
import numpy as np
from numpy import ndarray
import casadi as ca
from metadrive.custom2_version2.ocp.config import CBFconfig

class CBFunctions:
    LENGTH: float = 4.515  # meters
    WIDTH:  float = 1.852  # meters
    def __init__(self, config: CBFconfig):
        self.a = self.LENGTH / np.sqrt(2)
        self.b = self.WIDTH  / np.sqrt(2)
        self.config = config
    
    def distance_contrains(self, info: ndarray, info_other: ndarray): # h(x)
        return self.caculate_distance(info, info_other) - self.config.dist_min

    def caculate_distance(self, info: ndarray, info_other: ndarray) -> float:
        d = (info[0: 2] - info_other[0: 2]).reshape(-1, 1)
        norm_d = np.linalg.norm(d, ord=2, axis=0).flatten().item()
        u = d / (norm_d + 1e-8)
        t = self.caculate_tvar(info, u, self.a, self.b)
        t_other = self.caculate_tvar(info_other, u, self.a, self.b)
        dist: float = norm_d - t - t_other
        return dist

    def caculate_tvar(self, info: ndarray, u: ndarray, a: float, b: float) -> float:
        theta = info[2]
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ], dtype=np.float32)
        D = np.array([
            [1 / (a ** 2), 0.0       ],
            [0.0       , 1 / (b ** 2)],
        ], dtype=np.float32)
        Q = R @ D @ R.T
        t = 1 / np.sqrt(max(u.T @ Q @ u, 1e-6)).flatten().item()
        return t


class CBFunctionsCasadi:
    LENGTH: float = 4.515  # meters
    WIDTH:  float = 1.852  # meters
    def __init__(self, config: CBFconfig):
        self.a = self.LENGTH / np.sqrt(2)
        self.b = self.WIDTH  / np.sqrt(2)
        self.config = config

    def distance_contrains(self, info, info_other): # h(x)
        return self.caculate_distance(info, info_other) - self.config.dist_min

    def caculate_distance(self, info, info_other):
        d = info[0: 2] - info_other[0: 2]
        norm_d = ca.sqrt(ca.dot(d, d) + 1e-6)
        u = d / (norm_d + 1e-8)
        t = self.caculate_tvar(info, u, self.a, self.b)
        t_other = self.caculate_tvar(info_other, u, self.a, self.b)
        dist = norm_d - t - t_other
        return dist

    def caculate_tvar(self, info, u, a: float, b: float):
        theta = info[2]
        c = ca.cos(theta); s = ca.sin(theta)
        R = ca.vertcat(
            ca.horzcat(c, -s),
            ca.horzcat(s,  c),
        )
        D = ca.diag(ca.vertcat(1 / a ** 2, 1 / b ** 2))
        Q = ca.mtimes([R, D, R.T])
        t = 1 / ca.sqrt(ca.fmax(ca.mtimes([u.T, Q, u]), 1e-6))
        return t
import numpy as np
from numpy import ndarray
from metadrive.custom2_version2.ocp import MPConfig

class TrajGenerator:
    def __init__(self, config: MPConfig):
        self.config = config

    def generate(self, theta_0: float, state_ref: ndarray) -> ndarray:
        '''
        -> state_ref = [v_ref, kappa_ref], shape=(2)
           theta_0, shape=(1, )
        <- z_ref = [v_refs, theta_refs], shape=(2 * (np + 1))
        '''
        if len(state_ref.shape) >= 1: state_ref.flatten()
        
        v_ref, kappa_ref = state_ref.tolist()
        z_ref = np.zeros(shape=[self.config.np + 1, 2], dtype=np.float32)
        
        for i in range(self.config.np + 1):
            z_ref[i, 0] = v_ref
            z_ref[i, 1] = z_ref[i, 1] + v_ref * kappa_ref * self.config.Ts if i > 0 else theta_0
        return z_ref.flatten()
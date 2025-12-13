from metadrive.envs import MetaDriveEnv
from metadrive.utils.doc_utils import generate_gif
from metadrive.custom2_version2.base_config import MetaDriveEnvConfig
from numpy import ndarray
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import asdict

def create_env(config: MetaDriveEnvConfig) -> MetaDriveEnv:
    env = MetaDriveEnv(asdict(config))
    return env

def create_render_config(text: Optional[Dict] = None) -> Dict:
    row_config = dict(
        mode            = 'topdown',
        screen_record   = True,
        window          = False,
        screen_size     = (850, 850),
        camera_position = (83, 10),
    )
    
    if text is not None:
        row_config['text'] = text
    return row_config


def check_env(env: MetaDriveEnv):
    env.reset()
    frames: List[ndarray] = list()
    try:
        for step in range(1):
            if step < 50 : action = [0, 0.5]
            elif 50 <= step < 400: action = [0, -0.1]
            obs, reward, done, _, _ = env.step(action)
            frames.append(env.render(**create_render_config(text={'step': env.engine.episode_step, 'mode': 'Test'})))
        generate_gif(frames, f'dp_single_version2/check/check.gif')
    finally:
        env.close()


if __name__ == '__main__':
    metadriveenv_config = MetaDriveEnvConfig()
    check_env(create_env(metadriveenv_config))
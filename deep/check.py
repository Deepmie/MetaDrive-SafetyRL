from metadrive.envs import MetaDriveEnv
from metadrive.utils.doc_utils import generate_gif
from numpy import ndarray
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
ax.axis('off')
env = MetaDriveEnv(dict(
    traffic_mode="trigger",
    traffic_density=0.2,
    map="O",
))
env.reset(seed=0)
try:
    for i in range(1):
        o,r,d,_,_ = env.step([0,-0.2] if i < 100 or i> 150 else [0, 0.2])
        frame = env.render(
            mode           = "topdown", 
            scaling        = 3.5, 
            camera_position= (100, 5), 
            screen_size    = (500, 500),
            screen_record  = True,
            window         = False,
        )
        ax.imshow(frame)
    # env.top_down_renderer.generate_gif()
finally:
    env.close()
    fig.savefig('deep/check.svg')
    # clear_output()
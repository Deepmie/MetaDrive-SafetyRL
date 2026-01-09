import os
from typing import List
import re
import numpy as np
import matplotlib.pyplot as plt

logger_root: str = 'dp_single_version2/logger'
logger_name: str = 'log_2025_12_30_21_20_27'
logger_path: str = f'{logger_root}/{logger_name}'
cbf_ratio_path: str = f'{logger_path}/cbf_ratio.txt'

with open(cbf_ratio_path, mode='r', encoding='utf-8') as f:
    content = f.read()

pattern: str = r'(\d+):\s+(\d+\.\d+)%'
res = re.findall(pattern, content)

res_split = np.array(list(map(lambda t: t[1], res)), dtype=np.float32)

fig, ax = plt.subplots()
ax.axis('off')

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(res_split, marker='.')
ax1.set_title('cbf ratio')
ax1.set_xlabel('step')
ax1.set_ylabel('ratio')

fig.savefig(f'{logger_path}/eval.png')
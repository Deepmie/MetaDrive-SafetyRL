import torch
import numpy as np
import random
import os
from typing import Dict, Union
import json

def set_random_seed(seed: int, using_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if using_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def insert_metadata_path(key: str, value: Union[str, int, float], metadata_path: str = 'dp_single_version2/metadata.json'):
    if os.path.exists(metadata_path):
        with open(metadata_path, mode='r', encoding='utf-8') as reader:
            content: Dict = json.loads(reader.read())
    else:
        content = dict()
    
    # insert
    content[key] = value
    
    # re write
    with open(metadata_path, mode='w', encoding='utf-8') as writer:
        writer.write(json.dumps(content))

def get_metadata_from_path(metadata_path: str = 'dp_single_version2/metadata.json') -> Dict:
    if not os.path.exists(metadata_path): raise Exception(f'please get metadata after creating `{metadata_path}`')
    with open(metadata_path, mode='r', encoding='utf-8') as reader:
        return json.loads(reader.read())

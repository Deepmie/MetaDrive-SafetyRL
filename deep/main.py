from methods.common.base_config import BaseAlgConfig
from methods.common.alg import DefaultAlg
from methods.common.utils import set_random_seed, get_logger_path
import os
import importlib
import json
from typing import Dict, List

METHODS_LIST: List[str] = [
    'rl+mpc+cbf+ppc+traj',
    'rl+mpc+cbf+traj',
    'rl+mpc+cbf',
]

class Mainer:
    def __init__(self):
        self._default_config()

    def __call__(self):
        for method_name in METHODS_LIST:
            path_name: str = self._get_pathname_from_name(method_name) # 转换为pathname
            print(f'method: {method_name}, training...')
            self._train_single_method(method_name, self._load_method(path_name), self._load_method_config(path_name))
    
    def _train_single_method(self, method_name: str, DefaultAlg_CLS: DefaultAlg, DefaultAlg_CONFIG_CLS: BaseAlgConfig):
        alg_config: BaseAlgConfig = DefaultAlg_CONFIG_CLS()
        alg: DefaultAlg = DefaultAlg_CLS(alg_config)
        logger_path: str = get_logger_path(alg_config.logger_config.record_path, method_name)
        is_suc, info = alg.start()
        
        if not is_suc:
            print(info)
        else:
            print('Training have finished! Start to final evaluate...')
            alg.final_eval(os.path.join(logger_path, 'eval_final.gif'))
            alg.load_weight_from_checkpoint(os.path.join(logger_path, alg_config.best_policy_checkpoint_pth))
            alg.final_eval(os.path.join(logger_path, 'eval_best.gif'))
            alg.close()
    
    def _eval_single_method(self, DefaultAlg_CLS: DefaultAlg, BaseAlg_CONFIG_CLS: BaseAlgConfig):
        alg_config: BaseAlgConfig = BaseAlg_CONFIG_CLS()
        alg: DefaultAlg = DefaultAlg_CLS(alg_config, eval_mode=True)
    
    def _load_method(self, path_name: str) -> DefaultAlg:
        module: str = importlib.import_module(f'methods.{path_name}.alg')
        return getattr(module, 'Alg')
    
    def _load_method_config(self, path_name: str) -> BaseAlgConfig:
        module: str = importlib.import_module(f'methods.{path_name}.base_config')
        return getattr(module, 'AlgConfig')
    
    def _get_pathname_from_name(self, name: str) -> str:
        path_name_list = [m.get('file_name') for m in self.methods if m.get('method_name') == name]
        if len(path_name_list) == 0: raise f'{name} not in useful method list.'
        return path_name_list[0]

    def _default_config(self):
        set_random_seed(0) # set seed
        with open('methods/method_list.json', encoding='utf-8', mode='r') as f:
            self.methods: List[Dict] = json.loads(f.read())


# main progress
if __name__ == '__main__':
    mainer = Mainer()
    mainer() # run
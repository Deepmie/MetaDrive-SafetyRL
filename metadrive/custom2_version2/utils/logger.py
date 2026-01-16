from metadrive.custom2_version2.base_config import LoggerConfig
from typing import Union, List, Tuple, Dict, Optional
from datetime import datetime
import os
from _io import TextIOWrapper
import importlib
from dataclasses import dataclass, asdict
import time

@dataclass
class SYMBOL:
    space: str = ' '
    line: str  = '-'

CONFIG_DICT = {
    'base_config.MetaDriveEnvConfig': [
        'map', 'traffic_density', 'traffic_mode', 
        'horizon',
    ],
    'base_config.PPOConfig': [
        'n_process', 'sample_steps',
        'epoch', 'batch_size',
        'evaluate_steps', 'learning_rate',
    ],
    'ocp.config.MPConfig': [
        'np', 'mu', 'a_min', 'a_max',
        'delta_min', 'delta_max',
    ],
    'ocp.config.OptimConfig': [
        'ipopt', 'print_time'
    ],
}

class Logger:
    def __init__(self, config: LoggerConfig):
        self.config = config
        self.now_datetime = datetime.now()
        
        # =================日志路径名================= #
        logger_path: str           = os.path.join(config.logger_path_root, 'log_{}'.format(self.now_datetime.strftime("%Y_%m_%d_%H_%M_%S")))
        record_path: str           = os.path.join(logger_path, 'record_rl_reward.txt')
        cbf_ratio_record_path: str = os.path.join(logger_path, 'record_cbf_ratio.txt')
        # ========================================== #
        
        os.mkdir(logger_path) # 创建文件夹
        self.logger_path = logger_path
        self.record_file = open(record_path, mode='w', encoding='utf-8')
        self.ratio_record_file = open(cbf_ratio_record_path, mode='w', encoding='utf-8')

        # =================其他变量=================== #
        self._is_first_write_reward: bool = True
        self._is_first_write_addparam: bool = True
        self._reward_index: int = 0
        self._cbf_ratio_record_index: int = 0
        self._last_write_pos: Optional[int] = None
        # ========================================== #
        self.write_init()
    
    def close(self):
        self.record_file.close()
    
    def write_init(self):
        self._write(f'Write in {self.now_datetime.strftime("%Y.%m.%d %H: %M: %S")}\n')
        self._write(f'Start to record CBF use ratio, in {self.now_datetime.strftime("%Y.%m.%d %H: %M: %S")}\n\n', file_name='ratio_record')
        self.write_table()
    
    def write_table(self):
        config_dict: Dict[str, Dict] = self._get_config_modules_dynamic()
        self._write('\nExperiment Args: \n')
        for idx, (config_name, single_config_dict) in enumerate(config_dict.items()):
            self._write(self._get_line_string_with_name(config_name))
            for jdx, (attr_name, attr_value) in enumerate(single_config_dict.items()):
                if isinstance(attr_value, Dict):
                    self._write(self._get_dictattr_table_string(attr_name, attr_value))
                else:
                    self._write(self._get_attr_table_string(attr_name, attr_value))
                if jdx < len(single_config_dict)-1: self._write(self._get_line_string())
        
        self._last_write_pos = self._tell()
        self._write(self._get_line_string())

    def write_tabel_additional_params(self, param_names: Union[str, List[str]], param_values: Union[List, Tuple, Dict, float, int, str, ]):
        if self._is_first_write_addparam:
            self._write_addparam_init()
            self._is_first_write_addparam = False
        
        if not isinstance(param_names, list): # 单个的
            param_names = [param_names]
            param_values = [param_values]

        for idx, (param_name, param_value) in enumerate(zip(param_names, param_values)):
            self._write(self._get_attr_table_string(param_name, self._process_attr_value(param_value, param_name)))
            if idx >= len(param_names)-1: self._last_write_pos = self._tell()
            self._write(self._get_line_string())

    def write_reward(self, reward_value: float, best_reward: Optional[float] = None):
        if self._is_first_write_reward:
            self._write_reward_init()
            self._is_first_write_reward = False
        else:
            self._check_last_write_pos()
            self._seek(self._last_write_pos)
        
        # 每次调用书写
        self._write(self._get_attr_table_string(str(self._reward_index), self._process_attr_value(reward_value, f'reward_{self._reward_index}')))
        self._last_write_pos = self._tell()
        self._write(self._get_line_string())
        
        # 是否要添加最优奖励
        if best_reward is not None: 
            self._write(self._get_attr_table_string('best reward', self._process_attr_value(best_reward, 'best_reward')))
            self._write(self._get_line_string())
        
        self._reward_index += 1

    def write_time(self, time_result: Dict, process_name: Optional[str]):
        self._write(f'\n{process_name}\'s time recorded: \n')
        for idx, (time_unit_string, time_value) in enumerate(time_result.items()):
            self._write(f'{time_value: .4f}{time_unit_string}')
            if idx < len(time_result)-1: self._write(' ') 
        self._write('\n')
    
    def write_cbf_ratio(self, r: float, metadata: Dict):
        # if r < 1e-5: r = 0.0 # 过小直接置为0
        metadata_string: str = ''
        for idx, (key, value) in enumerate(metadata.items()):
            metadata_string += f'{key}: {value: .4e}'
            if idx < len(metadata) - 1: metadata_string += ', '
        self._write(f'{self._cbf_ratio_record_index}: {r * 100 : .4f}%, {metadata_string}\n', file_name='ratio_record')
        self._cbf_ratio_record_index += 1
    
    def get_logger_path(self) -> str:
        if not hasattr(self, 'logger_path'): raise Exception('Please init Logger class first!')
        return self.logger_path
    
    def _get_file(self, file_name: str = 'record') -> TextIOWrapper:
        file_name_real: str = f'{file_name}_file'
        if hasattr(self, file_name_real):
            _file: TextIOWrapper = getattr(self, file_name_real)
        else:
            raise NameError(f'attr `file_name` must include in [record, ratio_record], but you give: {file_name}')
        return _file

    def _write(self, s: str, file_name: str = 'record'):
        _write_file = self._get_file(file_name)
        _write_file.write(s)
        _write_file.flush()

    def _seek(self, pos: int, file_name: str = 'record'):
        _seek_file = self._get_file(file_name)
        _seek_file.seek(pos)
        _seek_file.truncate()

    def _tell(self, file_name: str = 'record') -> int:
        _tell_file = self._get_file(file_name)
        return _tell_file.tell()
    
    def _write_reward_init(self):
        self._write('\nRewards: \n')
        self._write(self._get_line_string())
        self._write(self._get_attr_table_string('idx', 'value'))
        self._write(self._get_line_string())

    def _write_addparam_init(self):
        self._check_last_write_pos()
        self._seek(self._last_write_pos)
        self._write(self._get_line_string_with_name('AdditionalParams'))

    def _check_last_write_pos(self):
        if self._last_write_pos is None:
            raise Exception('logic have error! self._last_write_pos not should be None!')

    def _get_line_string(self) -> str:
        return SYMBOL.line * (self.config.max_name_length + self.config.max_value_length + self.config.bias) + '\n'
    
    def _get_line_string_with_name(self, name: str) -> str:
        total_length = self.config.max_name_length + self.config.max_value_length + self.config.bias
        return self._generate_end_string(name, total_length, SYMBOL.line) + '\n'

    def _get_attr_table_string(self, attr_name: str, attr_value: Union[List, Tuple, Dict, float, int, str]) -> str:
        # generate name's end-string
        attr_name_end_string = self._generate_end_string(attr_name, self.config.max_name_length, SYMBOL.space)
        
        # generate value's end-string
        attr_value_string: str = self._process_attr_value(attr_value, attr_name)
        attr_value_end_string = self._generate_end_string(attr_value_string, self.config.max_value_length, SYMBOL.space)
        return f'|{attr_name_end_string}|{attr_value_end_string}|\n'
    
    def _get_dictattr_table_string(self, attr_name: str, attr_value: Dict) -> str:
        attr_table_string: str = self._get_attr_table_string(attr_name, ' ')
        for idx, (key, value) in enumerate(attr_value.items()):
            sub_attr_name = f'{key}'
            if isinstance(value, Dict):
                attr_table_string += self._get_dictattr_table_string(sub_attr_name, value)
            else:
                attr_table_string += self._get_attr_table_string(sub_attr_name, value)
            # if idx != len(attr_value) - 1: attr_table_string += self._get_line_string()
        return attr_table_string

    def _get_config_modules_dynamic(self) -> Dict[str, Dict]:
        config_dict = dict()
        for key, value in CONFIG_DICT.items():
            key_splited = key.split('.')
            main_module = '.'.join(key_splited[0:-1])
            config_name = key_splited[-1]
            module_total_name = f'metadrive.custom2_version2.{main_module}'
            module = importlib.import_module(module_total_name)
            
            if not hasattr(module, config_name):
                raise NameError(f'Config name: {config_name} not Founded in {module_total_name}.')

            single_config_dict = dict()
            config_cls = getattr(module, config_name)
            config_cls_dict = asdict(config_cls()) # 实例化一下
            
            for attr_name in value:
                # if not hasattr(config_cls, attr_name):
                #     raise NameError(f'{attr_name} not Founded in {config_name}.')
                # attr_value = getattr(config_cls, attr_name)
                if attr_name not in config_cls_dict:
                    raise NameError(f'{attr_name} not Founded in {config_name}')
                attr_value = config_cls_dict.get(attr_name)
                single_config_dict[attr_name] = attr_value # 赋值
            config_dict[config_name] = single_config_dict
        return config_dict
    
    def _generate_end_string(self, s: str, max_length: int, symbol: str) -> str:
        res_length = max_length - len(s)
        split_string = (res_length // 2, res_length // 2 + int(res_length % 2 != 0))
        return f'{split_string[0] * symbol}{s}{split_string[1] * symbol}'

    def _process_attr_value(self, value: Union[List, Dict, Tuple, float, int, str, ], name: str, num: int = 4) -> str:
        if isinstance(value, float):
            return f'{round(value, num)}'
        elif isinstance(value, (int, List, Dict, Tuple, )):
            return str(value)
        elif isinstance(value, str):
            return value
        else:
            raise TypeError(f'the type of attr {name} is {type(value)}, not in [float, int, str, list, dict, tuple, ].')

if __name__ == '__main__':
    logger_config = LoggerConfig()
    logger = Logger(logger_config)
    logger.write_reward(1.2331, 3.213)
    time.sleep(2)
    logger.write_reward(2.23431, 3.213)
    time.sleep(2)
    logger.write_reward(3.213, 3.213)
    time.sleep(2)
    logger.close()
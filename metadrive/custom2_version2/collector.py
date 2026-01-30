from typing import List
import os
import numpy as np

class Collector:
    def __init__(self, name: str, freq: int, path_root: str):
        self._path_root: str = path_root
        self._name: str = name; self._freq = freq
        self._tmp_file_path: str = os.path.join(self._path_root, f'{self._name}.tmp')
        self._file_path: str = os.path.join(self._path_root, f'{self._name}.txt')
        self._tmp_file = open(self._tmp_file_path, mode='w', encoding='utf-8')
        self._buffer: List[float] = list()
        self._idx: int = 0

    def collect_data(self, value: float):
        if (self._idx + 1) % self._freq == 0: self._write_in_file()
        self._buffer.append(value)
        self._idx += 1
    
    def _write_in_file(self):
        while len(self._buffer) > 0:
            self._tmp_file.write(str(self._buffer.pop(0)) + '\n')
            self._tmp_file.flush()
    
    def _merge_result(self):
        self._tmp_file.close()
        os.replace(self._tmp_file_path, self._file_path)
        
    def close(self):
        self._write_in_file() # 把buffer中的条目全部写入
        self._merge_result() # 修改.tmp为.txt

if __name__ == '__main__':
    data = np.random.normal(0, 1, [20]).tolist()
    collector = Collector(name='value-1')
    
    for d in data:
        collector.collect_data(d)
    
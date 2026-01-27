from typing import List
import os
import numpy as np

class Collector:
    def __init__(self, name: str, freq: int, path_root: str):
        self._path_root: str = path_root
        self._name: str = name; self._freq = freq
        self._file_path: str = os.path.join(self._path_root, f'{self._name}.txt')
        self._file = open(self._file_path, mode='w', encoding='utf-8')
        self._buffer: List[float] = list()
        self._idx: int = 0

    def collect_data(self, value: float):
        if (self._idx + 1) % self._freq == 0: self._write_in_file()
        self._buffer.append(value)
        self._idx += 1
    
    def _write_in_file(self):
        while len(self._buffer) > 0:
            self._file.write(str(self._buffer.pop(0)) + '\n')
            self._file.flush()
    
    def close(self):
        self._write_in_file()
        self._file.close()


if __name__ == '__main__':
    data = np.random.normal(0, 1, [20]).tolist()
    collector = Collector(name='value-1')
    
    for d in data:
        collector.collect_data(d)
    
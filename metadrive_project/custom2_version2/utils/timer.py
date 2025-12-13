import time
from typing import Optional, Tuple, Dict

class Timer:
    def __init__(self):
        ...

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end()
        return False
    
    def start(self, name: Optional[str] = None):
        row_string = 'Start to time!'
        if name is not None: row_string = f'In the `{name}` processing..., ' + row_string.lower()
        print(row_string)
        self.start_time = time.time()
    
    def end(self) -> Dict[str, float]:
        end_time = time.time()
        duration = end_time - self.start_time
        dur_min, dur_s = self._second_to_minute(duration)
        print(f'Time out! Total Time is {dur_min:.3f} min {dur_s:.3f} s')
        return {'min': dur_min, 's': dur_s, }

    def _second_to_minute(self, duration: float):
        return duration // 60, duration % 60
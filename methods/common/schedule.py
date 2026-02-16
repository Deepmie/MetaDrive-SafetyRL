class ConstantSchedule:
    def __init__(self, val: float):
        self.val = val
    
    def __call__(self, _: float) -> float:
        return self.val
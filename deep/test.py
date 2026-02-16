from dataclasses import dataclass, field, fields

@dataclass
class Action:
    action_dim: int = 4

@dataclass
class A:
    method: str = 'aaaa'
    action_dim: int = 4
    action: Action = field(default_factory=Action)

@dataclass
class B(A):
    def __init__(self, action_dim: int = 2):
        super().__init__()
        self.action_dim = action_dim
        self.action.action_dim = action_dim

        for f in fields(self):
            print(f.name)


if __name__ == '__main__':
    a = A()
    b = B()

    print('a', a.action_dim)
    print('b', b.action, b.action.action_dim)

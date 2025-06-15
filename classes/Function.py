import numpy as np

class Function:
    def __init__(self, f, name = None):
        self.f = f
        self.name = name

    def evaluate(self, x):
        if isinstance(x, (list, np.ndarray)):
            return self.f(x[0], x[1])
        else:
            return self.f(x)
        
    def __mul__(self, other):
        if self.f.__code__.co_argcount == 1 and other.f.__code__.co_argcount == 1:
            return Function(lambda x: self.evaluate(x) * other.evaluate(x))
        else:
            return Function(lambda x, y: self.evaluate([x, y]) * other.evaluate([x, y]))
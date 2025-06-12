import numpy as np

class Function:
    def __init__(self, f, name = None):
        self.f = f
        self.name = name

    def evaluate(self, x):
        if isinstance(x, int | float):
            return self.f(x)
        else:
            return self.f(x[0], x[1])
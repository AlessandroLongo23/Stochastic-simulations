import random

class RNG:
    def __init__(self, seed: int = 42):
        self.seed = seed

class LCG(RNG):
    def __init__(self, a: int = 1664525, c: int = 1013904223, m: int = 2**32, builtin: bool = False):
        super().__init__()
        self.set_parameters(a, c, m)
        self.name = 'LCG'
        self.builtin = builtin

    def pdf(self, x: int) -> float:
        return 1 / self.m if 0 <= x < self.m else 0
    
    def cdf(self, x: int) -> float:
        return x / self.m if 0 <= x < self.m else 0 if x < 0 else 1

    def set_parameters(self, a: int, c: int, m: int):
        self.a = a
        self.c = c
        self.m = m

    def next(self) -> int:
        if self.builtin:
            self.seed = random.randint(0, self.m - 1)
            return self.seed

        self.seed = (self.a * self.seed + self.c) % self.m
        return self.seed
    
    def simulate(self, seed: int = None, n: int = 10000, normalize: bool = False, sort: bool = False, plot: bool = False, savepath: str = None) -> list[int]:
        if seed is None:
            seed = self.seed

        self.seed = seed
        data = []
        for _ in range(n):
            data.append(self.next())

        if normalize:
            data = self.normalize(data)

        if sort:
            data.sort()

        return data
    
    def normalize(self, data: list[int]) -> list[int]:
        return [x / self.m for x in data]

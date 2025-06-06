class RNG:
    def __init__(self, seed: int = 42):
        self.seed = seed

class LCG(RNG):
    def __init__(self, a: int = 1664525, c: int = 1013904223, m: int = 2**32):
        super().__init__()
        self.a = a
        self.c = c
        self.m = m

    def next(self) -> int:
        self.seed = (self.a * self.seed + self.c) % self.m
        # self.seed = random.randint(0, self.m - 1)
        return self.seed
    
    def simulate(self, seed: int, n: int, normalize: bool = False, sort: bool = False, plot: bool = False, savepath: str = None) -> list[int]:
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

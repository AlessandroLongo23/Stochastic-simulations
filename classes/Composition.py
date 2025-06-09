from classes.CRVG import CRVG
from classes.DRVG import DRVG
import matplotlib.pyplot as plt

class Composition:
    def __init__(self, rvgs: list[CRVG | DRVG], weights: list[float]):
        self.rvgs = rvgs
        self.weights = weights

    def sample(self):
        return sum(rvg.sample() * weight for rvg, weight in zip(self.rvgs, self.weights))
    
    def simulate(self, n, plot = True, savepath = False):
        data = []

        for _ in range(n):
            data.append(self.sample())

        if plot:
            plt.hist(data, bins=100, range=(0, 5))
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(f'Composition')
            if savepath:
                plt.savefig(savepath)
            plt.show()

        return data
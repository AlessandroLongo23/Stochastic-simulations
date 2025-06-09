from classes.CRVG import CRVG
from classes.DRVG import DRVG
from classes.Plotter import Plotter
import numpy as np

class Simulator:
    def __init__(self):
        pass

    def simulate(self, rvgs: DRVG | CRVG | list[DRVG | CRVG], n: int, plot: bool = True, savepath: str = None):
        if isinstance(rvgs, list):
            rvgs = rvgs
        else:
            rvgs = [rvgs]

        data = {}
        for generator in rvgs:
            [observed, x_range, label] = generator.simulate(n)
            data[generator.id] = {
                'generator': generator,
                'observed': observed,
                'x_range': x_range,
                'label': label,
            }

        return data
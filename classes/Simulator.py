from classes.CRVG import CRVG
from classes.DRVG import DRVG

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
            data = generator.simulate(n)
            data[generator.id] = {
                'generator': generator,
                'observed': data['observed'],
                'x_range': data['x_range'],
                'label': data['label'],
            }

        return data
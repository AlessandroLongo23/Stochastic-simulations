import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from classes.RNG import LCG
from classes.Plotter import Plotter
from classes.Evaluator import Evaluator

def main() -> None:
    lcg = LCG()
    data = lcg.simulate(seed = 325, n = 10000)

    plotter = Plotter()
    plotter.plot_histogram(data, classes = 10, savepath = 'histogram.png')
    plotter.plot_scatter(data, savepath = 'scatter.png')

    evaluator = Evaluator()

    evaluator.chi_square(generators = [lcg], n = 1000, simulations = 1000, savepath = 'chi_square.png')
    evaluator.kolmogorov_smirnov(generators = [lcg], n = 1000, simulations = 1000, savepath = 'kolmogorov_smirnov.png')

    evaluator.above_below(generators = [lcg], n = 1000, simulations = 1000, savepath = 'above_below.png')
    evaluator.up_down_knuth(generators = [lcg], n = 1000, simulations = 1000, savepath = 'up_down_knuth.png')
    evaluator.up_and_down(generators = [lcg], n = 1000, simulations = 1000, savepath = 'up_and_down.png')

    evaluator.estimated_correlation(generators = [lcg], n = 1000, simulations = 1000, gap = 1, savepath = 'estimated_correlation.png')

if __name__ == "__main__":
    main()
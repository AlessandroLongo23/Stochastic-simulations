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

    evaluator = Evaluator(lcg)

    evaluator.chi_square()
    evaluator.kolmogorov_smirnov(n = 1000, simulations = 1000, savepath = 'kolmogorov_smirnov.png')

    evaluator.above_below(n = 1000, simulations = 10000, savepath = 'above_below.png')
    evaluator.up_down_knuth(n = 1000, simulations = 10000, savepath = 'up_down_knuth.png')
    evaluator.up_and_down(n = 1000, simulations = 10000, savepath = 'up_and_down.png')

    evaluator.estimated_correlation(n = 10000, simulations = 1000, gap = 1, savepath = 'estimated_correlation.png')

if __name__ == "__main__":
    main()
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from classes.RNG import LCG
from classes.Plotter import Plotter
from classes.Evaluator import Evaluator

def part1() -> None:
    lcg = LCG()

    # 1.c. Repeat for different values of a, b and M
    params = [
        (503, 643, 2**13 - 1),
        (6217, 7867, 2**17 - 1),
        (21517, 102013, 2**19 - 1),
        (1664525, 1013904223, 2**31 - 1),
    ]
    
    for a, c, m in params:
        lcg.set_parameters(a, c, m)
        # 1.a. Generate 10000 numbers and plot histogram and scatter plot
        data = lcg.simulate(seed = 325, n = 10000)

        plotter = Plotter()
        plotter.plot_histogram(data, bins = 10, savepath = f'histogram_{a}_{c}_{m}.png')
        plotter.plot_scatter(data[:-1], data[1:], savepath = f'scatter_{a}_{c}_{m}.png')

        # 1.b. Run statistical tests
        evaluator = Evaluator()
        evaluator.run_GOF_tests(generators = [lcg], n = 10000, simulations = 100)


def part2() -> None:
    builtin_lcg = LCG(builtin = True)
    data = builtin_lcg.simulate()

    plotter = Plotter()
    plotter.plot_histogram(data, bins = 10, savepath = 'histogram.png')
    plotter.plot_scatter(data[:-1], data[1:], savepath = 'scatter.png')

    evaluator = Evaluator()
    evaluator.run_GOF_tests(generators = [builtin_lcg], n = 1000, simulations = 1000)

def part3() -> None:
    lcg = LCG()
    data = lcg.simulate(seed = 325, n = 10000)

    evaluator = Evaluator()

    evaluator.chi_square(generators = [lcg], n = 1000, simulations = 1000, savepath = 'chi_square.png')
    evaluator.kolmogorov_smirnov(generators = [lcg], n = 1000, simulations = 1000, savepath = 'kolmogorov_smirnov.png')

    evaluator.above_below(generators = [lcg], n = 1000, simulations = 1000, savepath = 'above_below.png')
    evaluator.up_down_knuth(generators = [lcg], n = 1000, simulations = 1000, savepath = 'up_down_knuth.png')
    evaluator.up_and_down(generators = [lcg], n = 1000, simulations = 1000, savepath = 'up_and_down.png')

    evaluator.estimated_correlation(generators = [lcg], n = 1000, simulations = 1000, gap = 1, savepath = 'estimated_correlation.png')

def part4() -> None:
    evaluator = Evaluator()
    evaluator.p_value(generator = LCG(), n = 1000, simulations = 10000, savepath = 'p_value.png')


if __name__ == "__main__":
    # part1()
    part2()
    # part3()
    # part4()
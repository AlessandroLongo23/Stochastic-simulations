from LCG import LCG
from Plotter import Plotter
from Evaluator import Evaluator

def main() -> None:
    lcg = LCG()
    data = lcg.simulate(seed = 325, n = 10000)

    plotter = Plotter()
    plotter.plot_histogram(data, classes = 10)
    plotter.plot_scatter(data)

    evaluator = Evaluator(lcg)

    evaluator.chi_square()
    evaluator.kolmogorov_smirnov(n = 1000, simulations = 1000)

    # run tests
    evaluator.above_below(n = 1000, simulations = 10000)
    evaluator.up_down_knuth(n = 1000, simulations = 10000)
    evaluator.up_and_down(n = 1000, simulations = 10000)

    evaluator.estimated_correlation(n = 10000, simulations = 1000, gap = 1)

if __name__ == "__main__":
    main()
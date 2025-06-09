import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import random
from classes.CRVG import Uniform, Exponential, Gaussian, Pareto
from classes.Evaluator import Evaluator
from classes.Simulator import Simulator
from classes.Plotter import Plotter
# from classes.Composition import Composition

def main() -> None:
    simulator = Simulator()
    plotter = Plotter()

    exponentials = [
        Exponential(lambda_ = 3),
        Exponential(lambda_ = 4),
        Exponential(lambda_ = 5),
        Exponential(lambda_ = 6),
        Exponential(lambda_ = 7),
    ]
    data = simulator.simulate(exponentials, 10000)
    plotter.plot_density(data, savepath='exponential comparison.png')

    gaussians = [
        Gaussian(mu = 0, sigma = 1),
        Gaussian(mu = 0, sigma = 2),
        Gaussian(mu = 0, sigma = 3),
        Gaussian(mu = 0, sigma = 4),
        Gaussian(mu = 0, sigma = 5),
    ]
    data = simulator.simulate(gaussians, 10000)
    plotter.plot_density(data, savepath='gaussian comparison.png')

    gaussian = gaussians[0]
    gaussian.generate_confidence_interval(observations = 10, simulations = 100, confidence_level = 0.95, savepath = 'gaussian_confidence_interval.png')

    paretos = [
        Pareto(k = 2.05, beta = 1),
        Pareto(k = 2.5, beta = 1),
        Pareto(k = 3, beta = 1),
        Pareto(k = 4, beta = 1)
    ]
    data = simulator.simulate(paretos, 10000)
    plotter.plot_density(data, savepath='pareto comparison.png', x_range = [1, 5])

    evaluator = Evaluator()
    evaluator.analyze_pareto(generator = pareto, n = 10000, simulations = 1000, savepath = 'pareto_analysis.png')

    composition = Composition(pareto, pareto)
    composition.simulate(n = 100000, savepath = 'pareto_composition.png')

if __name__ == "__main__":
    main()
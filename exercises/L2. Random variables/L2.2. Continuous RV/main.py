import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import math
from classes.CRVG import Uniform, Exponential, Gaussian, Pareto, Gamma, HyperExponential, Erlang, CustomDistribution
from classes.Evaluator import Evaluator
from classes.Simulator import Simulator
from classes.Plotter import Plotter
from classes.Composition import Composition
import time
import numpy as np

simulator = Simulator()
plotter = Plotter()
evaluator = Evaluator()

def part1() -> None:
    generators = [
        # Exponential(lambda_ = 3),
        # Gaussian(mu = 0, sigma = 1),
        Pareto(k = 2.05, beta = 1),
        Pareto(k = 2.5, beta = 1),
        Pareto(k = 3, beta = 1),
        Pareto(k = 4, beta = 1)
    ]

    for generator in generators:
        data = simulator.simulate(generator, n = 1000)
        plotter.plot_density(observed_plot_type="histogram", data=data, savepath = f'{generator.name}.png', x_range = [1, 8], classes = 25)
        evaluator.run_GOF_tests(generators = [generator], n = 1000, simulations = 10000, savepath = f'GOF_tests_{generator.name}.png')

def part2() -> None:
    pareto = Pareto(k = 2.5, beta = 1)
    evaluator.analyze_pareto(generator = pareto, n = 1000, simulations = 100, savepath = 'pareto_analysis.png')

def part3() -> None:
    simulator = Simulator()
    plotter = Plotter()
    evaluator = Evaluator()

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

def part4() -> None:
    mu = 4
    y_distribution = CustomDistribution(lambda y: mu * math.exp(-mu * y), support = (1, 5))
    data = [y_distribution.sample() for _ in range(1000)]
    plotter.plot_density(observed_plot_type='histogram', data=data, savepath='y_distribution.png', classes = 25, title = 'Y Distribution')

    x_data = []
    for y in data:
        x_distribution = CustomDistribution(lambda x: y * math.exp(-y * x), support = (1, 5))
        x_data.append(x_distribution.sample())

    plotter.plot_density(observed_plot_type='histogram', data=x_data, savepath='x_distribution.png', classes = 25, title = 'X Distribution')

if __name__ == "__main__":
    # part1()
    # part2()
    # part3()
    part4()
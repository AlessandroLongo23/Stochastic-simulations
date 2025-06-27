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

# def main() -> None:
    # simulator = Simulator()
    # plotter = Plotter()

    # ## UNIFORM

    # uniforms = [
    #     Uniform(a = 0, b = 1),
    #     Uniform(a = 1, b = 2),
    #     Uniform(a = 1, b = 3),
    #     Uniform(a = 3, b = 4),
    #     Uniform(a = 0, b = 5),
    # ]
    # data = simulator.simulate(uniforms, n=1000)
    # plotter.plot_density(data, savepath='uniform comparison.png')

    
    # ## POISSON

    # poissons = [
    #     Poisson(lambda_=1),
    #     Poisson(lambda_=5),
    #     Poisson(lambda_=10),
    # ]
    # data = simulator.simulate(poissons, n=1000)
    # plotter.plot_density(data, savepath='density.png', title=f'Poisson (λ={poissons[0].lambda_})')


    # ## PARETO

    # paretos = [
    #     Pareto(k = 2.05, beta = 1),
    #     Pareto(k = 2.5, beta = 1),
    #     Pareto(k = 3, beta = 1),
    #     Pareto(k = 4, beta = 1)
    # ]
    # data = simulator.simulate(paretos, 10000)
    # plotter.plot_density(data, savepath='pareto comparison.png', x_range = [1, 5])

    # evaluator = Evaluator()
    # evaluator.analyze_pareto(generator = paretos[0], n = 10000, simulations = 1000, savepath = 'pareto_analysis.png')


    # ## EXPONENTIAL

    # exponentials = [
    #     Exponential(lambda_ = 3),
    #     Exponential(lambda_ = 4),
    #     Exponential(lambda_ = 5),
    # ]

    # weights = [0.2, 0.3, 0.5]
    # composition = Composition(exponentials, weights)
    # composition.simulate(n = 100000, savepath = 'exponential_composition.png')

    # ## HYPER-EXPONENTIAL

    # hyper_exponentials = [
    #     HyperExponential(lambdas = [0.8333, 5], weights = [0.8, 0.2]),
    #     HyperExponential(lambdas = [3, 0.8333, 5], weights = [0.3, 0.2, 0.5]),
    # ]
    # data = simulator.simulate(hyper_exponentials, n=1000)
    # plotter.plot_density(data, savepath='hyper_exponential comparison.png')


    # ## GAMMA

    # gammas = [
    #     Gamma(k = 1.3, theta = 1),
    #     Gamma(k = 2.2, theta = 1),
    #     Gamma(k = 3.1, theta = 1),
    #     Gamma(k = 4.5, theta = 1),
    # ]
    # data = simulator.simulate(gammas, n=1000)
    # plotter.plot_density(data, savepath='gamma comparison.png')

    # ## ERLANG

    # erlangs = [
    #     Erlang(n = 1, lambda_ = 1),
    #     Erlang(n = 2, lambda_ = 1),
    #     Erlang(n = 3, lambda_ = 1),
    #     Erlang(n = 4, lambda_ = 1),
    # ]
    # data = simulator.simulate(erlangs, n=1000)
    # plotter.plot_density(data, savepath='erlang comparison.png')


    # # Exercise 4
    # mu = 4
    # y_distribution = CustomDistribution(lambda y: mu * math.exp(-mu * y), support = (1, 5))
    # data = [y_distribution.sample() for _ in range(1000)]
    # plotter.plot_density(data, savepath='y_distribution.png')

    # x_data = []
    # for y in data:
    #     x_distribution = CustomDistribution(lambda x: y * math.exp(-y * x), support = (1, 5))
    #     x_data.append(x_distribution.sample())

    # plotter.plot_density(x_data, savepath='x_distribution.png')

def part1() -> None:
    simulator = Simulator()
    plotter = Plotter()
    evaluator = Evaluator()
    
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
        plotter.plot_density(data, savepath = f'{generator.name}.png')
        evaluator.run_GOF_tests(generators = [generator], n = 1000, simulations = 1000, savepath = f'GOF_tests_{generator.name}.png')
    

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


if __name__ == "__main__":
    # part1()
    part3()
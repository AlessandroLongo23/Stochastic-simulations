import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from classes.MCMC import MCMC
from classes.Function import Function
import math
from classes.Plotter import Plotter
from classes.Estimator import Crude

def main():
    plotter = Plotter()

    # EX 1
    # A = 8
    # m = 10
    # unnormalized_density = Function(lambda x: A ** x / (math.gamma(x + 1)))
    # mcmc = MCMC(
    #     unnormalized_density = unnormalized_density,
    #     m = m
    # )
    # mcmc.run(n = 10000, burn_in = 100)
    # plotter.plot_histogram(mcmc.chain, mcmc.m + 1, title = 'Histogram of the chain', x_label = 'x', y_label = 'Frequency')

    # estimator = Crude()
    # integral_value = estimator.estimateIntegral(
    #     unnormalized_density,
    #     a = 0,
    #     b = 10,
    #     n = 10000,
    #     # strata = 1000
    # )

    # normalized_density = Function(lambda x: unnormalized_density.evaluate(x) / integral_value)

    # estimator.estimateIntegral(
    #     normalized_density,
    #     a = 0,
    #     b = 10,
    #     n = 10000,
    #     # strata = 1000
    # )

    # plotter.plot_function(normalized_density, title = 'Normalized Density', x_label = 'x', y_label = 'f(x)')


    # EX 2
    # A1, A2 = 4, 4
    # m = 10
    # unnormalized_density = Function(lambda x, y: (A1 ** x) / (math.factorial(x)) * (A2 ** y) / (math.factorial(y)))
    # mcmc = MCMC(
    #     unnormalized_density = unnormalized_density,
    #     m = m,
    # )
    # mcmc.run(n = 10000, burn_in = 100, method = 'gibbs')
    # plotter.plot_histogram(mcmc.chain, mcmc.m + 1, title = 'Histogram of the chain', x_label = 'x', y_label = 'Frequency')


    # EX 3

if __name__ == "__main__":
    main()
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import random
from classes.DRVG import Bernoulli, Binomial, Geometric, DiscreteBuiltin, DiscreteDirect, DiscreteRejection, DiscreteAlias
from classes.CRVG import Uniform
from classes.Evaluator import Evaluator

def main() -> None:
    # bernoulli = Bernoulli(0.3)
    # bernoulli.simulate(10000, savepath = 'bernoulli.png')

    # uniform = Uniform(1, 6)
    # uniform.simulate(10000, savepath = 'uniform.png')

    # binomial = Binomial(10, 0.3)
    # binomial.simulate(10000, savepath = 'binomial.png')

    # geometric = Geometric(0.3)
    # geometric.simulate(10000, savepath = 'geometric.png', builtin = False)

    # n = 100
    # p = [0] + sorted([random.uniform(0, 1) for i in range(n - 1)]) + [1]
    # p = [p[i + 1] - p[i] for i in range(n)]
    p = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
    generators = [
        DiscreteBuiltin(p),
        DiscreteDirect(p, 'inefficient'), # O(classes ^ 2)
        DiscreteDirect(p, 'linear'),
        DiscreteDirect(p, 'binary'),
        DiscreteRejection(p),
        DiscreteAlias(p)
    ]

    # for gen in generators:
    #     gen.simulate(n = 1000, plot = True, savepath = gen.name)

    evaluator = Evaluator()
    # evaluator.analyze_time(generators = generators, n = 1000, simulations = 100, savepath = 'time.png')
    # evaluator.analyze_time(generators = generators, n = 100, classes = np.linspace(5, 500, 10).astype(int), simulations = 250, savepath = 'time_over_classes.png')
    evaluator.chi_square(generators = generators, n = 1000, simulations = 1000, savepath = 'chi_square.png')


if __name__ == "__main__":
    main()
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
from classes.DRVG import Bernoulli, Binomial, Geometric, Discrete
from classes.CRVG import Uniform
from classes.Evaluator import Evaluator

def main() -> None:
    bernoulli = Bernoulli(0.3)
    bernoulli.simulate(10000, savepath = 'bernoulli.png')

    uniform = Uniform(1, 6)
    uniform.simulate(10000, savepath = 'uniform.png')

    binomial = Binomial(10, 0.3)
    binomial.simulate(10000, savepath = 'binomial.png')

    geometric = Geometric(0.3)
    geometric.simulate(10000, savepath = 'geometric.png', builtin = False)

    # p = [0] + sorted([random.uniform(0, 1) for i in range(99)]) + [1]
    # d = [p[i + 1] - p[i] for i in range(100)]
    # discrete = Discrete(d)
    discrete = Discrete([7/48, 5/48, 1/8, 1/16, 1/4, 5/16])
    discrete.simulate(10000, savepath = 'discrete.png', method = 'alias')

    evaluator = Evaluator(discrete)
    evaluator.analyze_time(n = 1000, methods = ['builtin', 'direct', 'rejection', 'alias'], simulations = 100, savepath = 'time.png')
    evaluator.chi_square(n = 1000, simulations = 1000, savepath = 'chi_square.png')


if __name__ == "__main__":
    main()
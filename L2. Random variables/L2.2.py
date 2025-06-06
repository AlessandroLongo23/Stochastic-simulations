import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
from classes.CRVG import Uniform, Exponential, Gaussian, Pareto
from classes.Evaluator import Evaluator
from classes.Composition import Composition

def main() -> None:
    # exponential = Exponential(lambda_ = 3)
    # exponential.simulate(10000, savepath = 'exponential.png')

    # gaussian = Gaussian()
    # gaussian.simulate(10000, savepath = 'gaussian.png')
    # gaussian.generate_confidence_interval(observations = 10, simulations = 100, confidence_level = 0.95, savepath = 'gaussian_confidence_interval.png')

    pareto = Pareto(k = 2.5, beta = 1)
    # pareto.simulate(100000, savepath = 'pareto.png')

    # evaluator = Evaluator(pareto)
    # evaluator.analyze_pareto(n = 10000, simulations = 1000, savepath = 'pareto_analysis.png')

    composition = Composition(pareto, pareto)
    composition.simulate(n = 100000, savepath = 'pareto_composition.png')

if __name__ == "__main__":
    main()
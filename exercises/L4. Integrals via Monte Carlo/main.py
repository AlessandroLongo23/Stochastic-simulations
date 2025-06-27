import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np

from classes.Estimator import Crude, Antithetic, Control, Stratified, ImportanceSampling
from classes.CRVG import Gaussian, CustomDistribution, Exponential
from classes.Function import Function
from classes.Plotter import Plotter

def main():
    target_function = Function(lambda x: np.exp(x), 'h(x) = e^x')
    a, b = 0, 1

    # EX 1
    print('EX 1')
    estimator = Crude()
    estimator.estimateIntegral(
        target_fn=target_function, 
        a=a, b=b, 
        n=20000
    )
    
    # EX 2
    print('EX 2')
    estimator = Antithetic()
    estimator.estimateIntegral(
        target_fn=target_function, 
        a=a, b=b,
        n=10000
    )

    # EX 3
    print('EX 3')
    estimator = Control()
    estimator.estimateIntegral(
        target_fn=target_function, 
        a=a, b=b, 
        n=10000
    )

    # EX 4
    print('EX 4')   
    estimator = Stratified()
    estimator.estimateIntegral(
        target_fn=target_function, 
        a=a, b=b, 
        n=100, 
        strata=200
    )

    # EX 7
    print('EX 7')   
    estimator = Crude()
    estimator.estimateProbability(
        target_fn=Gaussian(mu=0, sigma=1), 
        a=2, 
        n=10000
    )

    estimator = ImportanceSampling()
    estimator.estimateProbability(
        target_fn=Gaussian(mu=0, sigma=1), 
        importance_distribution=Gaussian(mu=a, sigma=1), 
        a=2, 
        n=10000
    )

    # EX 8
    print('EX 8')
    bins = 50
    lambdas = np.linspace(0.1, 10, bins)
    stds = []
    means = []
    for lambda_ in lambdas:
        estimator = ImportanceSampling()
        mean, std, _ = estimator.estimateIntegral(
            target_fn=CustomDistribution(lambda x: np.exp(x), support=(0, 10), name = 'f(x) = e^x'), 
            importance_distribution=CustomDistribution(lambda x: lambda_ * np.exp(-lambda_ * x), support=(0, 10), name = 'f(x) = λe^(-λx)'), 
            a=0, b=1, 
            n=10000
        )
        means.append(mean)
        stds.append(std)

    plotter = Plotter()
    plotter.plot_line(x = lambdas, y = stds, x_label = 'Lambda', y_label = 'Standard deviation', title = 'Standard deviation over lambda', savepath = 'standard_deviation_over_lambda.png')

    min_std = min(stds)
    min_lambda = lambdas[stds.index(min_std)]
    print("Estimate of integral: ", means[stds.index(min_std)])
    print(f'Lambda that minimizes std: {min_lambda}. Min std: {min_std}')


if __name__ == "__main__":
    main()

    
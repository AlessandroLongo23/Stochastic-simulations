import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math

from classes.Estimator import Crude, Antithetic, Control, Stratified, ImportanceSampling
from classes.CRVG import Gaussian, CustomDistribution, Exponential
from classes.Function import Function

def main():
    # # Regular Monte Carlo Integration Examples
    target_function = Function(lambda x: np.exp(x), 'h(x) = e^x')
    a, b = 0, 1

    # EX 1
    estimator = Crude()
    estimator.estimateIntegral(
        target_fn=target_function, 
        a=a, b=b, 
        n=20000
    )
    
    # EX 2
    estimator = Antithetic()
    estimator.estimateIntegral(
        target_fn=target_function, 
        a=a, b=b, 
        n=10000
    )

    # EX 3
    estimator = Control()
    estimator.estimateIntegral(
        target_fn=target_function, 
        a=a, b=b, 
        n=10000
    )

    # EX 4
    estimator = Stratified()
    estimator.estimateIntegral(
        target_fn=target_function, 
        a=a, b=b, 
        n=100, 
        strata=200
    )

    # EX 7
    estimator = ImportanceSampling()
    estimator.estimateProbability(
        target_fn=Gaussian(mu=0, sigma=1), 
        importance_distribution=Gaussian(mu=a, sigma=1), 
        a=2, 
        n=10000
    )

    # EX 8
    lambda_ = 1
    estimator = ImportanceSampling()
    estimator.estimateIntegral(
        target_fn=CustomDistribution(lambda x: np.exp(x), support=(0, 10), name = 'f(x) = e^x'), 
        importance_distribution=CustomDistribution(lambda x: lambda_ * np.exp(-lambda_ * x), support=(0, 10), name = 'f(x) = λe^(-λx)'), 
        a=0, b=1, 
        n=10000
    )

    estimator.estimateIntegral(
        target_fn=CustomDistribution(lambda x: 1 / x, support=(0.00001, 10), name = 'f(x) = 1/x'), 
        importance_distribution=Exponential(lambda_=1), 
        a=1, b=2,
        n=10000
    )

if __name__ == "__main__":
    main()
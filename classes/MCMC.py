from classes.Function import Function
from classes.DRVG import CustomDRVG

import numpy as np

class MCMC:
    def __init__(self, unnormalized_density: Function, m = 10):
        self.unnormalized_density = unnormalized_density
        self.m = m
        self.chain = []

    def run(self, n = 1000, burn_in = 100, method = 'gibbs'):
        self.chain = []
        if method == 'gibbs':
            self.gibbs(n)
        else:
            self.metropolis_hastings(n, method)
        self.burn_in(n = burn_in)

        return self.chain
    
    def gibbs(self, n):
        vars_ = self.unnormalized_density.f.__code__.co_argcount
        X = np.random.randint(0, self.m + 1, size = vars_)

        for _ in range(n):
            for i in range(vars_):
                X = self.gibbs_next(X, i)

            self.chain.append(X)

    def gibbs_next(self, X, i):
        if i == 0:
            slice_ = [[int(X[i]), j] for j in range(self.m + 1)]
        else:
            slice_ = [[j, int(X[i])] for j in range(self.m + 1)]

        distribution_function = Function(lambda x: self.unnormalized_density.evaluate(slice_[x]))
        conditional_density = CustomDRVG(equation = distribution_function, support = (0, self.m + 1))

        if i == 0:
            return [X[i], conditional_density.sample()]
        else:
            return [conditional_density.sample(), X[i]]


    def metropolis_hastings(self, n, method):
        vars_ = self.unnormalized_density.f.__code__.co_argcount
        if vars_ == 1:
            X = np.random.randint(0, self.m + 1)
        else:
            X = np.random.randint(0, self.m + 1, size = vars_)

        for _ in range(n):
            proposal = self.get_proposal(X, method)
            accept_reject = self.accept_reject(X, proposal)
            X = self.update_chain(X, proposal, accept_reject)
            self.chain.append(X)

    def get_proposal(self, X: float | list[float], method = 'mh_ordinary'):
        if isinstance(X, int | float):
            delta = np.random.randint(-1, 2)
            return min(self.m, max(0, X + delta))
        else:
            if method == 'mh_ordinary':
                delta = np.random.randint(-1, 2, size = len(X))
            elif method == 'mh_coordinate-wise':
                delta = [0] * len(X)
                delta[np.random.randint(0, len(X))] = np.random.randint(-1, 2)
            return [min(self.m, max(0, x + d)) for x, d in zip(X, delta)]
    
    def accept_reject(self, X: float | list[float], proposal: float | list[float]):
        return min(1, self.unnormalized_density.evaluate(proposal) / self.unnormalized_density.evaluate(X))
    
    def update_chain(self, X: float | list[float], proposal: float | list[float], accept_reject: float):
        if np.random.uniform(0, 1) < accept_reject:
            return proposal
        return X
    
    def burn_in(self, n = 100):
        self.chain = self.chain[n:]
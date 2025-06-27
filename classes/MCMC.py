from classes.Function import Function
from classes.DRVG import CustomDRVG

import numpy as np
import math
from typing import Callable

class MCMC:
    def __init__(self, unnormalized_density: Function, m = 10):
        self.unnormalized_density = unnormalized_density
        self.m = m
        self.chain = []
        self.epsilon = 1e-6

    def run(self, n = 1000, burn_in = 100, method = 'mh_ordinary', domain = 'discrete', verbose = False, condition: Callable = lambda x: True):
        self.chain = []
        if method == 'gibbs':
            self.gibbs(n, domain)
        else:
            self.metropolis_hastings(n, method, domain, verbose, condition)
        self.burn_in(burn_in = burn_in)

        return self.chain
    
    def thin(self, gap = 10):
        self.chain = self.chain[::gap]
            
    def gibbs(self, n, domain):
        vars_ = self.unnormalized_density.f.__code__.co_argcount
        if domain == 'discrete':
            X = np.random.randint(0, self.m + 1, size = vars_)
        else:
            X = np.random.uniform(self.epsilon, 100, size = vars_)

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


    def metropolis_hastings(self, n, method, domain = 'discrete', verbose = False, condition = lambda x: True):
        vars_ = self.unnormalized_density.f.__code__.co_argcount
        X = None
        while True:
            if domain == 'discrete':
                if vars_ == 1:
                    X = np.random.randint(0, self.m + 1)
                else:
                    X = np.random.randint(0, self.m + 1, size = vars_)
            else:
                if vars_ == 1:
                    X = np.random.uniform(self.epsilon, 1)
                else:
                    X = np.random.uniform(self.epsilon, 1, size = vars_)

            if condition(X):
                break

        for _ in range(n):
            proposal = self.get_proposal(X, method, domain, condition)
            accept_reject = self.accept_reject(X, proposal)
            X = self.update_chain(X, proposal, accept_reject)
            self.chain.append(X)

            if verbose:
                print(f"Iteration {_ + 1}/{n}, Acceptance rate: {self.acceptance_count / self.total_proposals:.3f}, Step size: {self.step_sizer.get_step_size():.6f}")

    def get_proposal(self, X: float | list[float], method = 'mh_ordinary', domain = 'discrete', condition: Callable = lambda x: True):
        if domain == 'discrete':
            if isinstance(X, int | float):
                delta = np.random.randint(-1, 2)
                return min(self.m, max(0, X + delta))
            else:
                if method == 'mh_ordinary':
                    delta = np.random.randint(-1, 2, size = len(X))
                elif method == 'mh_coordinate_wise':
                    delta = [0] * len(X)
                    delta[np.random.randint(0, len(X))] = np.random.randint(-1, 2)

                next_ = [min(self.m, max(0, x + d)) for x, d in zip(X, delta)]
                while True:
                    delta = np.random.randint(-1, 2, size = len(X))
                    next_ = [min(self.m, max(0, x + d)) for x, d in zip(X, delta)]
                    if condition(next_):
                        break


                return next_
        else:
            if isinstance(X, float):
                delta = np.random.normal(0, 1)
                return min(1, max(self.epsilon, X + delta))
            else:
                if method == 'mh_ordinary':
                    delta = np.random.normal(0, 1, size = len(X))
                elif method == 'mh_coordinate_wise':
                    delta = [0] * len(X)
                    delta[np.random.randint(0, len(X))] = np.random.normal(0, 1)

                return [float(max(self.epsilon, x + d)) for x, d in zip(X, delta)]
        
    def accept_reject(self, X: float | list[float], proposal: float | list[float]):
        return min(1, self.unnormalized_density.evaluate(proposal) / max(self.unnormalized_density.evaluate(X), 1e-500))
    
    def update_chain(self, X: float | list[float], proposal: float | list[float], accept_reject: float):
        if np.random.uniform(0, 1) < accept_reject:
            return proposal
        return X
    
    def burn_in(self, burn_in = 100):
        self.chain = self.chain[burn_in:]


class AdaptiveStepSizer:
    def __init__(self, initial_step_size=0.1, target_acceptance=0.44, adaptation_window=50):
        self.step_size = initial_step_size
        self.target_acceptance = target_acceptance
        self.adaptation_window = adaptation_window
        self.acceptance_history = []
        
    def update(self, accepted):
        self.acceptance_history.append(accepted)
        
        if len(self.acceptance_history) >= self.adaptation_window:
            recent_acceptance = np.mean(self.acceptance_history[-self.adaptation_window:])
            
            if recent_acceptance > self.target_acceptance:
                self.step_size *= 1.05
            else:
                self.step_size *= 0.95
                
            self.step_size = np.clip(self.step_size, 1e-6, 2.0)
    
    def get_step_size(self):
        return self.step_size


class LogPosteriorFunction:
    def __init__(self, log_posterior_func):
        self.log_posterior_func = log_posterior_func
    
    def log_evaluate(self, x, y=None):
        if isinstance(x, list) and len(x) == 2:
            return self.log_posterior_func(x[0], x[1])
        elif y is not None:
            return self.log_posterior_func(x, y)
        else:
            raise ValueError("Invalid input format")
    
    def evaluate(self, x, y=None):
        log_val = self.log_evaluate(x, y)
        if log_val < -500:
            return 1e-300
        return math.exp(log_val)


class CustomMCMC(MCMC):
    def __init__(self, unnormalized_density: Function, log_posterior_func=None, m=10):
        super().__init__(unnormalized_density, m)
        self.log_posterior_func = log_posterior_func
        self.step_sizer = AdaptiveStepSizer(initial_step_size=0.01, target_acceptance=0.44)
        self.acceptance_count = 0
        self.total_proposals = 0
    
    def get_proposal(self, X, method='mh_ordinary', domain='continuous'):
        if domain == 'continuous':
            step_size = self.step_sizer.get_step_size()
            
            if isinstance(X, float):
                delta = np.random.normal(0, step_size)
                return max(self.epsilon, X + delta)
            else:
                if method == 'mh_ordinary':
                    delta = np.random.normal(0, step_size, size=len(X))
                elif method == 'mh_coordinate_wise':
                    delta = [0] * len(X)
                    delta[np.random.randint(0, len(X))] = np.random.normal(0, step_size)
                return [max(self.epsilon, x + d) for x, d in zip(X, delta)]
        else:
            return super().get_proposal(X, method, domain)
    
    def accept_reject(self, X, proposal):
        if self.log_posterior_func is None:
            return super().accept_reject(X, proposal)
        
        try:
            if isinstance(X, list):
                current_log_density = self.log_posterior_func(X[0], X[1])
                proposal_log_density = self.log_posterior_func(proposal[0], proposal[1])
            else:
                current_log_density = self.log_posterior_func(X)
                proposal_log_density = self.log_posterior_func(proposal)
            
            if not np.isfinite(current_log_density):
                current_log_density = -np.inf
            if not np.isfinite(proposal_log_density):
                proposal_log_density = -np.inf
            
            log_ratio = proposal_log_density - current_log_density
            
            if log_ratio >= 0:
                acceptance_prob = 1.0
            elif log_ratio < -500:
                acceptance_prob = 0.0
            else:
                acceptance_prob = math.exp(log_ratio)
                
            return acceptance_prob
            
        except (OverflowError, ValueError, TypeError) as e:
            return 0.0
    
    def update_chain(self, X, proposal, accept_reject):
        self.total_proposals += 1
        accepted = np.random.uniform(0, 1) < accept_reject
        
        if accepted:
            self.acceptance_count += 1
            result = proposal
        else:
            result = X
            
        self.step_sizer.update(accepted)
        
        return result
    
    def metropolis_hastings(self, n, method, domain='discrete', verbose = False):
        vars_ = self.unnormalized_density.f.__code__.co_argcount
        
        if domain == 'discrete':
            if vars_ == 1:
                X = np.random.randint(0, self.m + 1)
            else:
                X = np.random.randint(0, self.m + 1, size=vars_)
        else:
            if vars_ == 1:
                X = max(self.epsilon, 1.0)
            else:
                X = [max(self.epsilon, 1.0), max(self.epsilon, 1.0)]

        for i in range(n):
            proposal = self.get_proposal(X, method, domain)
            accept_prob = self.accept_reject(X, proposal)
            X = self.update_chain(X, proposal, accept_prob)
            self.chain.append(X)
            
            if verbose and (i + 1) % 1000 == 0:
                current_acceptance = self.acceptance_count / self.total_proposals if self.total_proposals > 0 else 0
                print(f"Iteration {i+1}/{n}, Acceptance rate: {current_acceptance:.3f}, Step size: {self.step_sizer.get_step_size():.6f}")
    
    def get_diagnostics(self):
        acceptance_rate = self.acceptance_count / self.total_proposals if self.total_proposals > 0 else 0
        return {
            'acceptance_rate': acceptance_rate,
            'final_step_size': self.step_sizer.get_step_size(),
            'total_proposals': self.total_proposals,
            'chain_length': len(self.chain)
        }
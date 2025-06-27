import numpy as np
from classes.Function import Function
import time

class MonteCarloEstimator():
    def __init__(self):
        pass

    def simulate(self, n: int, simulations: int, args = {}) -> list:
        means = []
        for _ in range(simulations):
            f_values = self.estimateExpected(n, **args)
            means.append(np.mean(f_values))

        return means
    
    def calc_statistics(self, values: list, n: int) -> tuple:
        mean = np.mean(values)
        std = np.std(values)
        confidence_interval = (
            float(mean - 1.96 * std / np.sqrt(n)), 
            float(mean + 1.96 * std / np.sqrt(n))
        )

        return mean, std, confidence_interval
      
    def estimateIntegral(self, target_fn: Function, a = 0, b = 1, n = 10, verbose = True, **args) -> float:
        f_values = self.estimateExpected(target_fn, a, b, n, **args)

        if verbose:
            print(f'Using {self.name} method\nFunction: {target_fn.name}')

        integral_mean = np.mean(f_values) * (b - a)
        integral_std = np.std(f_values) * (b - a)
        confidence_interval = (
            float(integral_mean - 1.96 * integral_std / np.sqrt(n)), 
            float(integral_mean + 1.96 * integral_std / np.sqrt(n))
        )

        if verbose:
            print(f'Integral estimate ({a}, {b}): {integral_mean}\nVariance: {integral_std ** 2}\nConfidence interval: {confidence_interval}\n\n')

        return integral_mean

    def estimateProbability(self, target_fn: Function, a = 0, n = 10, verbose = True) -> None:
        self.target_fn = target_fn
        f_values = self.estimateIntegral(target_fn, a, b = 10, n = n, verbose = False)

        mean, std, confidence_interval = self.calc_statistics(f_values, n)

        if verbose:
            print(f'P(X > {a}) where X ~ {self.target_fn.name}')
            print(f'Probability estimate: {mean}')
            print(f'Variance: {std ** 2}')
            print(f'Confidence interval: {confidence_interval}')

class Crude(MonteCarloEstimator):
    def __init__(self):
        self.name = 'Crude'
        pass

    def estimateExpected(self, target_fn: Function, a = 0, b = 1, n = 10) -> np.ndarray:
        time_start = time.time()
        U = np.random.uniform(a, b, n)
        values = [target_fn.evaluate(u) for u in U]
        time_end = time.time()
        print(f"Time taken: {time_end - time_start}")
        return values

class Antithetic(MonteCarloEstimator):
    def __init__(self):
        self.name = 'Antithetic'
        pass

    def estimateExpected(self, target_fn: Function, a = 0, b = 1, n = 10, **args) -> np.ndarray:
        time_start = time.time()
        U = np.random.uniform(a, b, n)
        values = [(target_fn.evaluate(u) + target_fn.evaluate(1 - u)) / 2 for u in U]
        time_end = time.time()
        print(f"Time taken: {time_end - time_start}")
        return values

class Control(MonteCarloEstimator):
    def __init__(self):
        self.name = 'Control'
        pass

    def estimateExpected(self, target_fn: Function, a = 0, b = 1, n = 10, **args) -> np.ndarray:
        time_start = time.time()
        U = np.random.uniform(a, b, n * 2).reshape(n, 2)
        U1, U2 = U[:, 0], U[:, 1]

        f_U1 = [target_fn.evaluate(u) for u in U1]
        cov = np.mean(U1 * f_U1) - np.mean(U1) * np.mean(f_U1)
        c = -cov * 12

        f_U2 = [target_fn.evaluate(u) for u in U2]
        values = [f_U2[i] + c * (U2[i] - 0.5) for i in range(n)]
        time_end = time.time()
        print(f"Time taken: {time_end - time_start}")
        return values

class Stratified(MonteCarloEstimator):
    def __init__(self):
        self.name = 'Stratified'
        pass

    def estimateExpected(self, target_fn: Function, a = 0, b = 1, n = 10, **args) -> np.ndarray:
        strata = args.get('strata', 10)
        time_start = time.time()
        U = np.random.uniform(a, b, n * strata).reshape(n, strata)
        f_values = np.zeros(n)
        for i in range(strata):
            f_values += [target_fn.evaluate((i + U[j, i]) / strata) for j in range(n)]
        f_values /= strata
        time_end = time.time()
        print(f"Time taken: {time_end - time_start}")

        return f_values

class ImportanceSampling(MonteCarloEstimator):
    def __init__(self):
        self.name = 'Importance Sampling'
        pass

    def estimateExpected(self, target_fn: Function, importance_distribution: Function, a = 0, b = 1, n = 10) -> np.ndarray:
        Y_samples = np.array([importance_distribution.sample() for _ in range(n)])

        h_values = target_fn.evaluate(Y_samples)

        g_values = importance_distribution.evaluate(Y_samples) 
        g_values = np.maximum(g_values, 1e-10)
        
        if b == float('inf'):
            f_values = (Y_samples > a).astype(float)
            importance_weights = f_values * h_values / g_values
        else:
            within_bounds = (Y_samples >= a) & (Y_samples <= b)
            importance_weights = np.where(within_bounds, h_values / g_values, 0.0)
        
        return importance_weights
    
    def estimateIntegral(self, target_fn: Function, importance_distribution: Function, a = 0, b = 1, n = 10) -> tuple:
        f_values = self.estimateExpected(target_fn, importance_distribution=importance_distribution, a = a, b = b, n = n)

        mean, std, confidence_interval = self.calc_statistics(f_values, n)

        return mean, std, confidence_interval

        # print(f'{target_fn.name}')
        # print(f'Integral estimate ({a}, {b}): {mean}')
        # print(f'Variance: {std ** 2}')
        # print(f'Confidence interval: {confidence_interval}\n\n')

    def estimateProbability(self, target_fn: Function, importance_distribution: Function, a = 0, n = 10) -> None:
        f_values = self.estimateExpected(target_fn, importance_distribution=importance_distribution, a = a, b = float('inf'), n = n)

        mean, std, confidence_interval = self.calc_statistics(f_values, n)

        print(f'P(X > {a}) where X ~ {target_fn.name}')
        print(f'Probability estimate: {mean}')
        print(f'Variance: {std ** 2}')
        print(f'Confidence interval: {confidence_interval}\n\n')
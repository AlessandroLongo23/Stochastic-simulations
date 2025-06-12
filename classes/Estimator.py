import numpy as np

class MonteCarloEstimator():
    def __init__(self):
        pass

    def simulate(self, n, simulations, args = {}):
        means = []
        for _ in range(simulations):
            f_values = self.estimateExpected(n, **args)
            means.append(np.mean(f_values))

        return means
    
    def calc_statistics(self, values, n):
        mean = np.mean(values)
        std = np.std(values)
        confidence_interval = (
            float(mean - 1.96 * std / np.sqrt(n)), 
            float(mean + 1.96 * std / np.sqrt(n))
        )

        return mean, std, confidence_interval
      
    def estimateIntegral(self, target_fn, a = 0, b = 1, n = 10, **args):
        f_values = self.estimateExpected(target_fn, a, b, n, **args)

        print(f'Using {self.name} method\nFunction: {target_fn.name}')

        integral_mean = np.mean(f_values) * (b - a)
        integral_std = np.std(f_values) * (b - a)
        confidence_interval = (
            float(integral_mean - 1.96 * integral_std / np.sqrt(n)), 
            float(integral_mean + 1.96 * integral_std / np.sqrt(n))
        )

        print(f'Integral estimate ({a}, {b}): {integral_mean}\nVariance: {integral_std ** 2}\nConfidence interval: {confidence_interval}\n\n')

        return integral_mean

    def estimateProbability(self, target_fn, a = 0, n = 10):
        f_values = self.estimateExpected(target_fn, a, b = 1e6, n = n)

        mean, std, confidence_interval = self.calc_statistics(f_values, n)

        print(f'P(X > {a}) where X ~ {self.target_fn.name}')
        print(f'Probability estimate: {mean}')
        print(f'Variance: {std ** 2}')
        print(f'Confidence interval: {confidence_interval}')

class Crude(MonteCarloEstimator):
    def __init__(self):
        self.name = 'Crude'
        pass

    def estimateExpected(self, target_fn, a = 0, b = 1, n = 10):
        U = np.random.uniform(a, b, n)
        return target_fn.evaluate(U)

class Antithetic(MonteCarloEstimator):
    def __init__(self):
        self.name = 'Antithetic'
        pass

    def estimateExpected(self, target_fn, a = 0, b = 1, n = 10, **args):
        U = np.random.uniform(a, b, n)
        return (target_fn.evaluate(U) + target_fn.evaluate(1 - U)) / 2

class Control(MonteCarloEstimator):
    def __init__(self):
        self.name = 'Control'
        pass

    def estimateExpected(self, target_fn, a = 0, b = 1, n = 10, **args):
        U = np.random.uniform(a, b, n * 2).reshape(n, 2)
        U1, U2 = U[:, 0], U[:, 1]

        cov = np.mean(U1 * target_fn.evaluate(U1)) - np.mean(U1) * np.mean(target_fn.evaluate(U1))
        c = -cov * 12

        return target_fn.evaluate(U2) + c * (U2 - 0.5)

class Stratified(MonteCarloEstimator):
    def __init__(self):
        self.name = 'Stratified'
        pass

    def estimateExpected(self, target_fn, a = 0, b = 1, n = 10, **args):
        strata = args.get('strata', 10)
        U = np.random.uniform(a, b, n * strata).reshape(n, strata)
        f_values = np.zeros(n)
        for i in range(strata):
            f_values += target_fn.evaluate((i + U[:, i]) / strata)
        f_values /= strata

        return f_values

class ImportanceSampling(MonteCarloEstimator):
    def __init__(self):
        self.name = 'Importance Sampling'
        pass

    def estimateExpected(self, target_fn=None, importance_distribution=None, a=0, b=1, n=10):
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
    
    def estimateIntegral(self, target_fn, importance_distribution, a = 0, b = 1, n = 10):
        f_values = self.estimateExpected(target_fn, importance_distribution=importance_distribution, a = a, b = b, n = n)

        mean, std, confidence_interval = self.calc_statistics(f_values, n)

        print(f'{target_fn.name}')
        print(f'Integral estimate ({a}, {b}): {mean}')
        print(f'Variance: {std ** 2}')
        print(f'Confidence interval: {confidence_interval}\n\n')

    def estimateProbability(self, target_fn, importance_distribution, a = 0, n = 10):
        f_values = self.estimateExpected(target_fn, importance_distribution=importance_distribution, a = a, b = float('inf'), n = n)

        mean, std, confidence_interval = self.calc_statistics(f_values, n)

        print(f'P(X > {a}) where X ~ {target_fn.name}')
        print(f'Probability estimate: {mean}')
        print(f'Variance: {std ** 2}')
        print(f'Confidence interval: {confidence_interval}\n\n')
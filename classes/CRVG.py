import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random

class CRVG:
    def __init__(self):
        pass

    def evaluate(self, x):
        return self.theoretical_density(x)

    def generate_confidence_interval(self, observations: int = 10, simulations: int = 100, confidence_level: float = 0.95, savepath: str = None) -> tuple:
        import scipy.stats as stats
        
        means = []
        lower_bounds_mean = []
        upper_bounds_mean = []
        
        variances = []
        lower_bounds_var = []
        upper_bounds_var = []
        
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha / 2)
        
        df = observations - 1
        chi2_lower_critical = stats.chi2.ppf(alpha / 2, df)
        chi2_upper_critical = stats.chi2.ppf(1 - alpha / 2, df)
        
        for _ in range(simulations):
            sim_result = self.simulate(n=observations, plot=False, savepath=False)
            if isinstance(sim_result, dict):
                sample_data = sim_result['observed']
            else:
                sample_data = sim_result[0]
            
            sample_mean = np.mean(sample_data)
            sample_variance = np.var(sample_data, ddof=1)
            
            standard_error = math.sqrt(sample_variance / observations)
            margin_of_error = z_critical * standard_error
            
            means.append(sample_mean)
            lower_bounds_mean.append(sample_mean - margin_of_error)
            upper_bounds_mean.append(sample_mean + margin_of_error)
            
            variances.append(sample_variance)
            if chi2_upper_critical > 0:
                lower_bounds_var.append((df * sample_variance) / chi2_upper_critical)
            else:
                lower_bounds_var.append(0)

            if chi2_lower_critical > 0:
                upper_bounds_var.append((df * sample_variance) / chi2_lower_critical)
            else:
                upper_bounds_var.append(float('inf'))

        if savepath or True:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            theoretical_mean = self.mean() if hasattr(self, 'mean') else None
            theoretical_variance = self.variance() if hasattr(self, 'variance') else None
            
            # Mean Confidence Intervals
            covered_count_mean = 0
            for i in range(simulations):
                color = 'blue'
                if theoretical_mean is not None and not (lower_bounds_mean[i] <= theoretical_mean <= upper_bounds_mean[i]):
                    color = 'red'
                elif theoretical_mean is not None:
                    covered_count_mean += 1
                ax1.vlines(i + 1, lower_bounds_mean[i], upper_bounds_mean[i], color=color, alpha=0.7)

            if theoretical_mean is not None:
                ax1.axhline(theoretical_mean, color='red', linestyle='--', linewidth=2, label=f'Theoretical Mean = {theoretical_mean:.3f}')
                coverage_rate_mean = covered_count_mean / simulations
                ax1.set_title(f'{confidence_level*100}% Confidence Intervals for Mean (Coverage: {coverage_rate_mean:.2f})')
                ax1.legend()
            else:
                ax1.set_title(f'{confidence_level*100}% Confidence Intervals for Mean')
            ax1.set_ylabel('Value')
            ax1.grid(True, alpha=0.3)

            # Variance Confidence Intervals
            covered_count_var = 0
            for i in range(simulations):
                color = 'blue'
                if theoretical_variance is not None and not (lower_bounds_var[i] <= theoretical_variance <= upper_bounds_var[i]):
                    color = 'red'
                elif theoretical_variance is not None:
                    covered_count_var += 1
                ax2.vlines(i + 1, lower_bounds_var[i], upper_bounds_var[i], color=color, alpha=0.7)

            if theoretical_variance is not None:
                ax2.axhline(theoretical_variance, color='red', linestyle='--', linewidth=2, label=f'Theoretical Variance = {theoretical_variance:.3f}')
                coverage_rate_var = covered_count_var / simulations
                ax2.set_title(f'{confidence_level*100}% Confidence Intervals for Variance (Coverage: {coverage_rate_var:.2f})')
                ax2.legend()
            else:
                ax2.set_title(f'{confidence_level*100}% Confidence Intervals for Variance')
            
            ax2.set_xlabel('Simulation')
            ax2.set_ylabel('Value')
            ax2.grid(True, alpha=0.3)

            fig.suptitle(f'Confidence Intervals for {self.name}\n({simulations} simulations, {observations} observations each)', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', savepath), dpi=300, bbox_inches='tight')
            plt.show()

class Uniform(CRVG):
    def __init__(self, a, b):
        self.name = 'Uniform'
        self.a = a
        self.b = b
        self.id = f'uniform_{self.a}_{self.b}'
    
    def pdf(self, x):
        if self.a <= x <= self.b:
            return 1 / (self.b - self.a)
        return 0
    
    def cdf(self, x):
        if self.a <= x <= self.b:
            return (x - self.a) / (self.b - self.a)
        if x < self.a:
            return 0
        if x > self.b:
            return 1
    
    def mean(self):
        return (self.a + self.b) / 2
    
    def variance(self):
        return (self.b - self.a) ** 2 / 12

    def sample(self, a: int = None, b: int = None, builtin = True):
        a = self.a if a is None else a
        b = self.b if b is None else b

        if builtin:
            return random.uniform(a, b)

        U = random.uniform(0, 1)
        return U * (b - a) + a
    
    def theoretical_density(self, x):
        if self.a <= x <= self.b:
            return 1 / (self.b - self.a)
        return 0
    
    def simulate(self, n, plot = False, savepath = False, builtin = True, seed = None, normalize = False, sort = False):
        data = [self.sample(builtin = builtin) for _ in range(n)]

        x_range = np.linspace(self.a, self.b + 1, 100)
        label = f'Uniform (a={self.a}, b={self.b})'

        return {'observed': data, 'x_range': x_range, 'label': label}
    

class Constant(Uniform):
    def __init__(self, value):
        super().__init__(a = value, b = value)

        self.name = 'Constant'
        self.id = f'constant_{value}'
        self.value = value

    def pdf(self, x):
        if x == self.value:
            return 1
        return 0
    
    def cdf(self, x):
        if x >= self.value:
            return 1
        return 0
    
    def mean(self):
        return self.value
    
    def variance(self):
        return 0

    def simulate(self, n, plot = False, savepath = False, builtin = True, seed = None, normalize = False, sort = False):
        data = [self.sample(builtin = builtin) for _ in range(n)]

        x_range = np.linspace(self.a, self.b + 1, 100)
        label = f'Constant (value={self.value})'

        return {'observed': data, 'x_range': x_range, 'label': label}
    
    def generate_confidence_interval(self, observations: int = 10, simulations: int = 100, confidence_level: float = 0.95, savepath: str = None) -> tuple:
        import scipy.stats as stats
        
        means = []
        lower_bounds = []
        upper_bounds = []
        
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        for _ in range(simulations):
            sample_data = self.simulate(n=observations, plot=False, savepath=False)['observed']
            
            sample_mean = sum(sample_data) / len(sample_data)
            sample_variance = sum((x - sample_mean) ** 2 for x in sample_data) / (len(sample_data) - 1)
            standard_error = math.sqrt(sample_variance / observations)
            
            margin_of_error = z_critical * standard_error
            lower_bound = sample_mean - margin_of_error
            upper_bound = sample_mean + margin_of_error
            
            means.append(sample_mean)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        
        if savepath or True:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            theoretical_mean = None
            if hasattr(self, 'k') and hasattr(self, 'beta') and self.k > 1:
                theoretical_mean = self.beta * self.k / (self.k - 1)
            elif hasattr(self, 'lambda_'):
                theoretical_mean = 1 / self.lambda_
            elif hasattr(self, 'mu'):
                theoretical_mean = self.mu

            covered_count = 0
            for i in range(simulations):
                color = 'blue'
                if theoretical_mean is not None:
                    if not (lower_bounds[i] <= theoretical_mean <= upper_bounds[i]):
                        color = 'red'
                    else:
                        covered_count +=1
                
                ax.vlines(i + 1, lower_bounds[i], upper_bounds[i], color=color, alpha=0.7)

            if theoretical_mean is not None:
                ax.axhline(theoretical_mean, color='red', linestyle='--', linewidth=2, 
                          label=f'Theoretical Mean = {theoretical_mean:.3f}')
                
                coverage_rate = covered_count / simulations
                ax.set_title(f'{confidence_level*100}% Confidence Intervals\n'
                            f'({simulations} simulations, {observations} observations each)\n'
                            f'Coverage Rate: {coverage_rate:.2f}')
                ax.legend()

            else:
                ax.set_title(f'{confidence_level*100}% Confidence Intervals\n'
                        f'({simulations} simulations, {observations} observations each)')
            
            ax.set_xlabel('Simulation')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', savepath), dpi=300, bbox_inches='tight')
            plt.show()
        
class Exponential(CRVG):
    def __init__(self, lambda_):
        self.name = 'Exponential'
        self.id = f'exponential_{lambda_}'
        self.lambda_ = lambda_

    def pdf(self, x):
        return self.lambda_ * math.exp(-self.lambda_ * x)
    
    def cdf(self, x):
        return 1 - math.exp(-self.lambda_ * x)

    def mean(self):
        return 1 / self.lambda_
    
    def variance(self):
        return 1 / (self.lambda_ ** 2)

    def sample(self, lambda_ = None, builtin = True):
        lambda_ = self.lambda_ if lambda_ is None else lambda_

        if builtin:
            return random.expovariate(lambda_)
        
        U = random.uniform(0, 1)
        return -math.log(U) / lambda_
    
    def theoretical_density(self, x):
        if isinstance(x, np.ndarray):
            return np.array([self.theoretical_density(x_i) for x_i in x])
        return self.lambda_ * math.exp(-self.lambda_ * x)
    
    def simulate(self, n: int, plot = False, savepath = False, builtin = True, seed = None, normalize = False, sort = False, U1 = None, U2 = None):
        if U2 is None:
            data = [self.sample(builtin = builtin) for i in range(n)]
        else:
            data = [-math.log(U2[i]) / self.lambda_ for i in range(n)]

        x_range = np.linspace(min(data), max(data), 100)
        label = f'Exponential (λ={self.lambda_})'

        return {'observed': data, 'x_range': x_range, 'label': label}
    
class HyperExponential(CRVG):
    def __init__(self, lambdas, weights):
        self.name = 'HyperExponential'
        self.id = f'hyper_exponential_{lambdas}_{weights}'
        self.lambdas = lambdas
        self.weights = weights
        self.exponentials = [Exponential(lambda_ = lambda_) for lambda_ in lambdas]

    def mean(self):
        return sum(self.weights[i] * (1 / self.lambdas[i]) for i in range(len(self.lambdas)))
    
    def variance(self):
        mean_val = self.mean()
        second_moment = sum(self.weights[i] * (2 / (self.lambdas[i] ** 2)) for i in range(len(self.lambdas)))
        return second_moment - mean_val ** 2

    def sample(self, builtin = True):
        if builtin:
            return random.choices(self.exponentials, weights = self.weights)[0].sample()

        U = random.uniform(0, 1)
        for i, exponential in enumerate(self.exponentials):
            if U < sum(self.weights[:i + 1]):
                return exponential.sample()
        return self.exponentials[-1].sample()
    
    def simulate(self, n, plot = False, savepath = False, builtin = True, seed = None, normalize = False, sort = False, U1 = None, U2 = None):
        if U1 is None and U2 is None:
            data = [self.sample(builtin = builtin) for i in range(n)]
        else:
            # exercise
            choices = [0 if U1[i] < self.weights[0] else 1 for i in range(n)]
            data = [-math.log(U2[i]) / self.lambdas[choices[i]] for i in range(n)]

        x_range = np.linspace(min(data), max(data), 100)
        label = f'HyperExponential (λ={self.lambdas}, w={self.weights})'

        return {'observed': data, 'x_range': x_range, 'label': label}
    
    def theoretical_density(self, x):
        if isinstance(x, np.ndarray):
            return np.array([self.theoretical_density(x_i) for x_i in x])
        return sum(self.weights[i] * self.exponentials[i].theoretical_density(x) for i in range(len(self.exponentials)))
    
    
class Gaussian(CRVG):
    def __init__(self, mu = 0, sigma = 1):
        self.name = 'Gaussian'
        self.id = f'gaussian_{mu}_{sigma}'
        self.mu = mu
        self.sigma = sigma

    def mean(self):
        return self.mu
    
    def variance(self):
        return self.sigma ** 2

    def pdf(self, x):
        return math.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2)) / (self.sigma * math.sqrt(2 * math.pi))
    
    def cdf(self, x):
        return 0.5 * (1 + math.erf((x - self.mu) / (self.sigma * math.sqrt(2))))
    
    def sample(self, mu = None, sigma = None, builtin = True):
        if builtin:
            return random.gauss(self.mu, self.sigma)
        
        mu = self.mu if mu is None else mu
        sigma = self.sigma if sigma is None else sigma

        U = [random.uniform(0, 1) for _ in range(2)]

        r = math.sqrt(-2 * math.log(U[0]))
        theta = 2 * math.pi * U[1]

        return (r * math.cos(theta)) * self.sigma + self.mu
    
    def theoretical_density(self, x):
        if isinstance(x, np.ndarray):
            return np.array([self.theoretical_density(x_i) for x_i in x])
        return math.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2)) / (self.sigma * math.sqrt(2 * math.pi))
    
    def simulate(self, n, plot = False, savepath = False, builtin = True, seed = None, normalize = False, sort = False):
        data = [self.sample(builtin = builtin) for _ in range(n)]
        
        x_range = np.linspace(min(data), max(data), 100)
        label = f'Gaussian (μ={self.mu}, σ={self.sigma})'

        return {'observed': data, 'x_range': x_range, 'label': label}
    
class Pareto(CRVG):
    def __init__(self, k = None, beta = None, mean = None):
        self.name = 'Pareto'
        self.id = f'pareto_{k}_{beta}'
        
        self.k = k if k is not None else -beta / (mean - beta)
        self.beta = beta if beta is not None else mean * (k - 1) / k

    def pdf(self, x):
        return self.k * (self.beta ** self.k) / (x ** (self.k + 1))
    
    def cdf(self, x):
        if x == 0:
            return 0
        return 1 - (self.beta / x) ** self.k

    def mean(self):
        return self.beta * self.k / (self.k - 1)
    
    def variance(self):
        return self.beta ** 2 * self.k / ((self.k - 1) ** 2 * (self.k - 2))
    
    def median(self):
        return self.beta * (2 ** (1 / self.k))

    def sample(self, k = None, beta = None, builtin = True):
        k = self.k if k is None else k
        beta = self.beta if beta is None else beta

        U = random.uniform(0, 1)
        return beta * (U ** (-1 / k))
    
    def theoretical_density(self, x):
        if isinstance(x, np.ndarray):
            return np.array([self.theoretical_density(x_i) for x_i in x])
        return self.k * (self.beta ** self.k) / (x ** (self.k + 1))
    
    def simulate(self, n, plot = False, savepath = False, builtin = True, seed = None, normalize = False, sort = False):
        data = [self.sample(builtin = builtin) for _ in range(n)]

        x_range = np.linspace(self.beta, max(data), 100)
        label = f'Pareto (k={self.k}, β={self.beta})'

        return {'observed': data, 'x_range': x_range, 'label': label}

class Gamma(CRVG):
    def __init__(self, k = 2, theta = 1):
        self.name = 'Gamma'
        self.id = f'gamma_{k}_{theta}'
        self.k = k
        self.theta = theta

    def pdf(self, x):
        return (1 / (math.gamma(self.k) * (self.theta ** self.k))) * (x ** (self.k - 1)) * math.exp(-x / self.theta)
    
    def cdf(self, x):
        return math.gammainc(self.k, x / self.theta)

    def mean(self):
        return self.k * self.theta
    
    def variance(self):
        return self.k * (self.theta ** 2)
    
    def theoretical_density(self, x):
        if x <= 0:
            return 0
        return (1 / (math.gamma(self.k) * (self.theta ** self.k))) * (x ** (self.k - 1)) * math.exp(-x / self.theta)
    
    def sample(self, k = None, theta = None):
        k = self.k if k is None else k
        theta = self.theta if theta is None else theta
        
        return random.gammavariate(k, theta)
    
    def simulate(self, n, plot = False, savepath = False, builtin = True, seed = None, normalize = False, sort = False):
        data = [self.sample(builtin = builtin) for _ in range(n)]
        
        x_range = np.linspace(0, max(data), 100)
        label = f'Gamma (k={self.k}, θ={self.theta})'

        return {'observed': data, 'x_range': x_range, 'label': label}


class Erlang(CRVG):
    def __init__(self, k = 2, lambda_ = 1):
        if k != int(k):
            raise ValueError('k must be an integer')

        self.name = 'Erlang'
        self.id = f'erlang_{int(k)}_{lambda_}'
        self.k = k
        self.lambda_ = lambda_

    def pdf(self, x):
        return (self.lambda_ ** self.k) * (x ** (self.k - 1)) * math.exp(-self.lambda_ * x) / math.gamma(self.k)
    
    def cdf(self, x):
        return math.gammainc(self.k, self.lambda_ * x)

    def mean(self):
        return self.k / self.lambda_
    
    def variance(self):
        return self.k / (self.lambda_ ** 2)
    
    def sample(self, k = None, lambda_ = None, builtin = True):
        k = self.k if k is None else k
        lambda_ = self.lambda_ if lambda_ is None else lambda_

        if builtin:
            return random.gammavariate(k, 1 / lambda_)
        
        U = [random.uniform(0, 1) for _ in range(k)]
        return -math.log(np.prod(U)) / lambda_
    
    def theoretical_density(self, x):
        if isinstance(x, np.ndarray):
            return np.array([self.theoretical_density(x_i) for x_i in x])
        return (self.lambda_ ** self.k) * (x ** (self.k - 1)) * math.exp(-self.lambda_ * x) / math.gamma(self.k)
    
    def simulate(self, n, plot = False, savepath = False, builtin = True, seed = None, normalize = False, sort = False):
        data = [self.sample(builtin = builtin) for _ in range(n)]
        
        x_range = np.linspace(0, max(data), 100)
        label = f'Erlang (k={self.k}, λ={self.lambda_})'
        
        return {'observed': data, 'x_range': x_range, 'label': label}

# class Poisson(CRVG):
#     def __init__(self, lambda_):
#         self.name = 'Poisson'
#         self.id = f'poisson_{lambda_}'
#         self.lambda_ = lambda_

#     def mean(self):
#         return self.lambda_
    
#     def variance(self):
#         return self.lambda_

#     def sample(self, lambda_ = None):
#         return np.random.poisson(self.lambda_)

#     def theoretical_density(self, x):
#         if isinstance(x, np.ndarray):
#             return np.array([self.theoretical_density(x_i) for x_i in x])
#         return math.exp(-self.lambda_) * (self.lambda_ ** x) / math.gamma(x + 1)
    
#     def simulate(self, n, plot = False, savepath = False, builtin = True, seed = None, normalize = False, sort = False):
#         data = [self.sample(builtin = builtin) for _ in range(n)]
        
#         x_range = np.linspace(0, max(data), 100)
#         label = f'Poisson (λ={self.lambda_})'

#         return data, x_range, label
    
    # def compose(self, other: CRVG):
    #     lambda_ = other.sample()
    #     return self.sample(lambda_ = lambda_)

class CustomDistribution(CRVG):
    def __init__(self, distribution_function, support=(-5, 5), max_value=None, name=None):
        self.name = name if name is not None else 'CustomDistribution'
        self.id = f'custom_distribution_{distribution_function}'
        self.distribution_function = distribution_function
        self.support = support
        self.max_value = max_value
        
        if self.max_value is None:
            self._find_max_value()
    
    def _find_max_value(self):
        import numpy as np
        
        x_test = np.linspace(self.support[0], self.support[1], 10000)
        try:
            y_test = [self.distribution_function(x) for x in x_test]
            self.max_value = (max(y_test) + 1) / 2
        except:
            self.max_value = 1.0
            print("Warning: Could not determine maximum value, using 1.0")
    
    def theoretical_density(self, x):
        if isinstance(x, np.ndarray):
            result = np.zeros_like(x, dtype=float)
            mask = (x >= self.support[0]) & (x <= self.support[1])
            try:
                result[mask] = np.array([self.distribution_function(xi) for xi in x[mask]])
            except:
                pass
            return result
        else:
            if self.support[0] <= x <= self.support[1]:
                try:
                    return self.distribution_function(x)
                except:
                    return 0.0
            return 0.0
    
    def sample(self):
        import numpy as np
        
        max_iterations = 10000
        iterations = 0
        
        while iterations < max_iterations:
            x = random.uniform(self.support[0], self.support[1])
            
            y = random.uniform(0, self.max_value)
            
            try:
                fx = self.distribution_function(x)
                if y <= fx:
                    return x
            except:
                pass
                
            iterations += 1
        
        print(f"Warning: Rejection sampling failed after {max_iterations} iterations")
        return random.uniform(self.support[0], self.support[1])
    
    def simulate(self, n, plot = False, savepath = False, builtin = True, seed = None, normalize = False, sort = False):
        data = [self.sample(builtin = builtin) for _ in range(n)]
        
        x_range = np.linspace(min(data), max(data), 100)
        label = f'Custom Distribution ({self.distribution_function.__name__})'

        return {'observed': data, 'x_range': x_range, 'label': label}
    
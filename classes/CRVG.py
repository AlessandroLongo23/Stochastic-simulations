import numpy as np
import matplotlib.pyplot as plt
import math
import os

class CRVG:
    def __init__(self):
        pass

    def evaluate(self, x):
        return self.theoretical_density(x)

class Uniform(CRVG):
    def __init__(self, a, b):
        self.name = 'Uniform'
        self.a = a
        self.b = b
        self.id = f'uniform_{self.a}_{self.b}'
        # self.p = [1 / (b - a + 1) for _ in range(b - a + 1)]
    
    def mean(self):
        return (self.a + self.b) / 2
    
    def variance(self):
        return (self.b - self.a) ** 2 / 12

    def sample(self, a: int = None, b: int = None):
        a = self.a if a is None else a
        b = self.b if b is None else b

        U = np.random.uniform()
        return U * (b - a) + a
    
    def theoretical_density(self, x):
        return 1 / (self.b - self.a + 1)
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]

        x_range = np.linspace(self.a, self.b + 1, 100)
        label = f'Uniform (a={self.a}, b={self.b})'

        return data, x_range, label
    

class Constant(Uniform):
    def __init__(self, value):
        super().__init__(a = value, b = value)

        self.name = 'Constant'
        self.id = f'constant_{value}'
        self.value = value
    
    def mean(self):
        return self.value
    
    def variance(self):
        return 0

class Exponential(CRVG):
    def __init__(self, lambda_):
        self.name = 'Exponential'
        self.id = f'exponential_{lambda_}'
        self.lambda_ = lambda_
        # self.p = [math.exp(-lambda_ * x) for x in range(100)]

    def mean(self):
        return 1 / self.lambda_
    
    def variance(self):
        return 1 / (self.lambda_ ** 2)

    def sample(self, lambda_ = None):
        lambda_ = self.lambda_ if lambda_ is None else lambda_

        U = np.random.uniform()
        return -math.log(U) / lambda_
    
    def theoretical_density(self, x):
        if isinstance(x, np.ndarray):
            return np.array([self.theoretical_density(x_i) for x_i in x])
        return self.lambda_ * math.exp(-self.lambda_ * x)
    
    def simulate(self, n: int):
        data = [self.sample() for _ in range(n)]

        x_range = np.linspace(min(data), max(data), 100)
        label = f'Exponential (λ={self.lambda_})'

        return data, x_range, label
    
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

    def sample(self):
        U = np.random.uniform()
        for i, exponential in enumerate(self.exponentials):
            if U < sum(self.weights[:i + 1]):
                return exponential.sample()
        return self.exponentials[-1].sample()
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]
        data = sorted(data)

        x_range = np.linspace(min(data), max(data), 100)
        label = f'HyperExponential (λ={self.lambdas}, w={self.weights})'

        return data, x_range, label
    
    def theoretical_density(self, x):
        if isinstance(x, np.ndarray):
            return np.array([self.theoretical_density(x_i) for x_i in x])
        return sum(self.weights[i] * self.exponentials[i].theoretical_density(x) for i in range(len(self.exponentials)))
    
    
class Gaussian(CRVG):
    def __init__(self, mu = 0, sigma = 1):
        self.name = 'Gaussian'
        self.id = f'gaussian_{mu}_{sigma}'
        self.formula = f''
        self.mu = mu
        self.sigma = sigma
    
    def mean(self):
        return self.mu
    
    def variance(self):
        return self.sigma ** 2
    
    def sample(self, mu = None, sigma = None, builtin = True):
        if builtin:
            return np.random.normal(self.mu, self.sigma)
        
        mu = self.mu if mu is None else mu
        sigma = self.sigma if sigma is None else sigma

        U = np.random.uniform(size = 2)

        r = math.sqrt(-2 * math.log(U[0]))
        theta = 2 * math.pi * U[1]

        return (r * math.cos(theta)) * self.sigma + self.mu
    
    def theoretical_density(self, x):
        if isinstance(x, np.ndarray):
            return np.array([self.theoretical_density(x_i) for x_i in x])
        return math.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2)) / (self.sigma * math.sqrt(2 * math.pi))
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]
        data = sorted(data)
        
        x_range = np.linspace(min(data), max(data), 100)
        label = f'Gaussian (μ={self.mu}, σ={self.sigma})'

        return data, x_range, label
    
    def generate_confidence_interval(self, observations: int = 10, simulations: int = 100, confidence_level: float = 0.95, savepath: str = None) -> tuple:
        import scipy.stats as stats
        
        means = []
        lower_bounds = []
        upper_bounds = []
        
        # Calculate critical value for the confidence level
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        for _ in range(simulations):
            sample_data = self.simulate(n=observations, plot=False, savepath=False)[0]
            
            sample_mean = sum(sample_data) / len(sample_data)
            sample_variance = sum((x - sample_mean) ** 2 for x in sample_data) / (len(sample_data) - 1)  # Sample variance
            standard_error = math.sqrt(sample_variance / observations)
            
            # Calculate confidence interval bounds
            margin_of_error = z_critical * standard_error
            lower_bound = sample_mean - margin_of_error
            upper_bound = sample_mean + margin_of_error
            
            means.append(sample_mean)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        
        # Create box plot
        if savepath or True:  # Always show plot for now
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data for box plot
            box_data = [means, lower_bounds, upper_bounds]
            labels = ['Sample Means', 'Lower CI Bounds', 'Upper CI Bounds']
            
            # Create box plot
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            
            # Customize colors
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add theoretical mean line if available
            if hasattr(self, 'k') and hasattr(self, 'beta') and self.k > 1:
                theoretical_mean = self.beta * self.k / (self.k - 1)
                ax.axhline(theoretical_mean, color='red', linestyle='--', linewidth=2, 
                          label=f'Theoretical Mean = {theoretical_mean:.3f}')
                ax.legend()
            elif hasattr(self, 'lambda_'):
                theoretical_mean = 1 / self.lambda_
                ax.axhline(theoretical_mean, color='red', linestyle='--', linewidth=2, 
                          label=f'Theoretical Mean = {theoretical_mean:.3f}')
                ax.legend()
            elif hasattr(self, 'mu'):
                ax.axhline(self.mu, color='red', linestyle='--', linewidth=2, 
                          label=f'Theoretical Mean = {self.mu:.3f}')
                ax.legend()
            
            ax.set_title(f'Box Plot: Means and {confidence_level*100}% Confidence Intervals\n'
                        f'({simulations} simulations, {observations} observations each)')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', savepath), dpi=300, bbox_inches='tight')
            plt.show()
        
        return means, lower_bounds, upper_bounds

class Pareto(CRVG):
    def __init__(self, k = 2, beta = 1):
        self.name = 'Pareto'
        self.id = f'pareto_{k}_{beta}'
        self.k = k
        self.beta = beta
        # self.p = [1 - (self.beta / x) ** self.k for x in range(1, 30)]

    def mean(self):
        return self.beta * self.k / (self.k - 1)
    
    def variance(self):
        return self.beta ** 2 * self.k / ((self.k - 1) ** 2 * (self.k - 2))

    def sample(self, k = None, beta = None):
        k = self.k if k is None else k
        beta = self.beta if beta is None else beta

        U = np.random.uniform()
        return beta * (U ** (-1 / k))
    
    def theoretical_density(self, x):
        if isinstance(x, np.ndarray):
            return np.array([self.theoretical_density(x_i) for x_i in x])
        return self.k * self.beta ** self.k / x ** (self.k + 1)
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]
        data = sorted(data)

        x_range = np.linspace(self.beta, max(data), 100)
        label = f'Pareto (k={self.k}, β={self.beta})'

        return data, x_range, label

class Gamma(CRVG):
    def __init__(self, k = 2, theta = 1):
        self.name = 'Gamma'
        self.id = f'gamma_{k}_{theta}'
        self.k = k  # shape parameter (alpha)
        self.theta = theta  # scale parameter
        # self.p = [self.theoretical_density(x) for x in np.linspace(0.1, 10, 100)]
    
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
        
        return np.random.gamma(k, theta)
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]
        data = sorted(data)
        
        x_range = np.linspace(0, max(data), 100)
        label = f'Gamma (k={self.k}, θ={self.theta})'

        return data, x_range, label


class Erlang(Gamma):
    def __init__(self, n = 2, lambda_ = 1):
        if n != int(n):
            raise ValueError('n must be an integer')
        
        super().__init__(k = n, theta = 1 / lambda_)
        self.name = 'Erlang'
        self.id = f'erlang_{int(n)}_{lambda_}'
    

class Poisson(CRVG):
    def __init__(self, lambda_):
        self.name = 'Poisson'
        self.id = f'poisson_{lambda_}'
        self.lambda_ = lambda_
        # self.p = [math.exp(-lambda_) * (lambda_ ** x) / math.gamma(x + 1) for x in range(100)]

    def mean(self):
        return self.lambda_
    
    def variance(self):
        return self.lambda_

    def sample(self, lambda_ = None):
        return np.random.poisson(self.lambda_)

    def theoretical_density(self, x):
        if isinstance(x, np.ndarray):
            return np.array([self.theoretical_density(x_i) for x_i in x])
        return math.exp(-self.lambda_) * (self.lambda_ ** x) / math.gamma(x + 1)
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]
        
        x_range = np.linspace(0, max(data), 100)
        label = f'Poisson (λ={self.lambda_})'

        return data, x_range, label
    
    # def compose(self, other: CRVG):
    #     lambda_ = other.sample()
    #     return self.sample(lambda_ = lambda_)

class CustomDistribution(CRVG):
    def __init__(self, distribution_function, support=(-5, 5), max_value=None, name=None):
        self.name = name if name is not None else 'CustomDistribution'
        self.id = f'custom_distribution_{distribution_function}'
        self.distribution_function = distribution_function
        self.support = support  # (min, max) tuple defining where the f functions non-zero
        self.max_value = max_value  # Maximum value of the PDF (for efficiency)
        
        # Find maximum value if not provided
        if self.max_value is None:
            self._find_max_value()
    
    def _find_max_value(self):
        """Find the maximum value of the distribution function for rejection sampling"""
        import numpy as np
        
        # Sample many points and find the maximum
        x_test = np.linspace(self.support[0], self.support[1], 10000)
        try:
            y_test = [self.distribution_function(x) for x in x_test]
            self.max_value = (max(y_test) + 1) / 2
        except:
            # If function evaluation fails, use a conservative upper bound
            self.max_value = 1.0
            print("Warning: Could not determine maximum value, using 1.0")
    
    def theoretical_density(self, x):
        """Return the theoretical density at point x"""
        if isinstance(x, np.ndarray):
            # Handle array input
            result = np.zeros_like(x, dtype=float)
            mask = (x >= self.support[0]) & (x <= self.support[1])
            try:
                # Apply the distribution function only to values within support
                result[mask] = np.array([self.distribution_function(xi) for xi in x[mask]])
            except:
                # If function evaluation fails, return zeros
                pass
            return result
        else:
            # Handle single value input
            if self.support[0] <= x <= self.support[1]:
                try:
                    return self.distribution_function(x)
                except:
                    return 0.0
            return 0.0
    
    def sample(self):
        """Sample from the custom distribution using rejection sampling"""
        import numpy as np
        
        max_iterations = 10000  # Prevent infinite loops
        iterations = 0
        
        while iterations < max_iterations:
            # Sample x uniformly from the support
            x = np.random.uniform(self.support[0], self.support[1])
            
            # Sample y uniformly from [0, max_value]
            y = np.random.uniform(0, self.max_value)
            
            # Evaluate the distribution function at x
            try:
                fx = self.distribution_function(x)
                
                # Accept if y <= f(x)
                if y <= fx:
                    return x
                    
            except:
                # If function evaluation fails, skip this sample
                pass
                
            iterations += 1
        
        # If we reach here, something went wrong
        print(f"Warning: Rejection sampling failed after {max_iterations} iterations")
        return np.random.uniform(self.support[0], self.support[1])
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]
        data = sorted(data)
        
        x_range = np.linspace(min(data), max(data), 100)
        label = f'Custom Distribution ({self.distribution_function.__name__})'

        return data, x_range, label
    
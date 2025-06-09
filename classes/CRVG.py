import numpy as np
import matplotlib.pyplot as plt
import math
import os
from classes.Plotter import Plotter

class CRVG:
    def __init__(self):
        pass

class Uniform(CRVG):
    def __init__(self, a, b):
        self.name = 'Uniform'
        self.a = a
        self.b = b
        self.p = [1 / (b - a + 1) for _ in range(b - a + 1)]

    def sample(self, a: int = None, b: int = None):
        a = self.a if a is None else a
        b = self.b if b is None else b

        U = np.random.uniform()
        return math.floor(U * (b - a + 1)) + a
    
    def theoretical_density(self, x):
        return 1 / (self.b - self.a + 1)
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]

        # if plot:
        #     plotter = Plotter()
        #     plotter.plot_histogram(
        #         data, 
        #         classes = self.b - self.a + 1,
        #         theoretical_density = [self.theoretical_density(x) for x in range(self.a, self.b + 1)],
        #         x_label = 'Outcome',
        #         y_label = 'Frequency',
        #         title = f'Uniform Distribution (a={self.a}, b={self.b})',
        #         savepath = f'uniform_{self.a}_{self.b}.png'
        #     )

        x_range = np.linspace(self.a, self.b + 1, 100)
        label = f'Uniform (a={self.a}, b={self.b})'

        return data, x_range, label
    
    # def compose(self, other: CRVG):
    #     a = other.sample()
    #     b = other.sample()
    #     return self.sample(a, b)

class Exponential(CRVG):
    def __init__(self, lambda_):
        self.name = 'Exponential'
        self.id = f'exponential_{lambda_}'
        self.lambda_ = lambda_
        self.p = [math.exp(-lambda_ * x) for x in range(100)]

    def sample(self, lambda_ = None):
        lambda_ = self.lambda_ if lambda_ is None else lambda_

        U = np.random.uniform()
        return -math.log(U) / lambda_
    
    def theoretical_density(self, x):
        return self.lambda_ * math.exp(-self.lambda_ * x)
    
    def simulate(self, n: int):
        data = [self.sample() for _ in range(n)]
        # data = sorted(data)
        
        # if plot:
        #     x_range = np.linspace(min(data), max(data), 200)
        #     plotter = Plotter()
        #     plotter.plot_density(
        #         data, 
        #         x_range = x_range,
        #         theoretical_density = [self.theoretical_density(x) for x in x_range],
        #         x_label = 'Value',
        #         y_label = 'Density',
        #         title = f'Exponential Distribution Density (λ={self.lambda_})',
        #         savepath = f'exponential_{self.lambda_}.png'
        #     )

        x_range = np.linspace(min(data), max(data), 100)
        label = f'Exponential (λ={self.lambda_})'

        return data, x_range, label
    
    # def compose(self, other: CRVG):
    #     lambda_ = other.sample()
    #     return self.sample(lambda_ = lambda_)
    
class Gaussian(CRVG):
    def __init__(self, mu = 0, sigma = 1):
        self.name = 'Gaussian'
        self.id = f'gaussian_{mu}_{sigma}'
        self.mu = mu
        self.sigma = sigma
        self.p = [math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi)) for x in range(100)]
    
    def sample(self, mu = None, sigma = None):
        mu = self.mu if mu is None else mu
        sigma = self.sigma if sigma is None else sigma

        U = np.random.uniform(size = 2)

        r = math.sqrt(-2 * math.log(U[0]))
        theta = 2 * math.pi * U[1]

        return (r * math.cos(theta)) * self.sigma + self.mu
    
    def theoretical_density(self, x):
        return math.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2)) / (self.sigma * math.sqrt(2 * math.pi))
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]
        data = sorted(data)
        
        # if plot:
        #     x_range = np.linspace(min(data), max(data), 200)
        #     plotter = Plotter()
        #     plotter.plot_density(
        #         data, 
        #         x_range = x_range,
        #         theoretical_density = [self.theoretical_density(x) for x in x_range],
        #         x_label = 'Value',
        #         y_label = 'Density',
        #         title = f'Gaussian Distribution Density (μ={self.mu}, σ={self.sigma})',
        #         savepath = f'gaussian_{self.mu}_{self.sigma}.png'
        #     )

        x_range = np.linspace(min(data), max(data), 100)
        label = f'Gaussian (μ={self.mu}, σ={self.sigma})'

        return data, x_range, label
    
    # def compose(self, other: CRVG):
    #     mu = other.sample()
    #     sigma = other.sample()
    #     return self.sample(mu = mu, sigma = sigma)
    
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
        self.p = [1 - (self.beta / x) ** self.k for x in range(1, 30)]

    def sample(self, k = None, beta = None):
        k = self.k if k is None else k
        beta = self.beta if beta is None else beta

        U = np.random.uniform()
        return beta * (U ** (-1 / k))
    
    def theoretical_density(self, x):
        return self.k * self.beta ** self.k / x ** (self.k + 1)
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]
        data = sorted(data)

        # if plot:
        #     x_range = np.linspace(self.beta, 10, 100)
        #     plotter = Plotter()
        #     plotter.plot_density(
        #         data, 
        #         x_range = x_range,
        #         theoretical_density = [self.theoretical_density(x) for x in x_range],
        #         x_label = 'Value',
        #         y_label = 'Density',
        #         title = f'Pareto Distribution Density (k={self.k}, β={self.beta})',
        #         savepath = f'pareto_{self.k}_{self.beta}.png'
        #     )

        x_range = np.linspace(self.beta, max(data), 100)
        label = f'Pareto (k={self.k}, β={self.beta})'

        return data, x_range, label
    
    # def compose(self, other: CRVG):
    #     k = other.sample()
    #     beta = other.sample()
    #     return self.sample(k = k, beta = beta)


class Gamma(CRVG):
    def __init__(self, k = 2, theta = 1):
        self.name = 'Gamma'
        self.id = f'gamma_{k}_{theta}'
        self.k = k  # shape parameter (alpha)
        self.theta = theta  # scale parameter
        # Theoretical density values for plotting
        self.p = [self._pdf(x) for x in np.linspace(0.1, 10, 100)]
    
    def _pdf(self, x):
        if x <= 0:
            return 0
        return (1 / (math.gamma(self.k) * (self.theta ** self.k))) * (x ** (self.k - 1)) * math.exp(-x / self.theta)
    
    def sample(self, k = None, theta = None):
        k = self.k if k is None else k
        theta = self.theta if theta is None else theta
        
        if k == 1:
            # Gamma(1, theta) is Exponential(1/theta)
            U = np.random.uniform()
            return -theta * math.log(U)
        elif k == int(k) and k > 1:
            # For integer k, use sum of k exponential random variables
            # Gamma(k, theta) = sum of k Exponential(1/theta) random variables
            sum_exp = 0
            for _ in range(int(k)):
                U = np.random.uniform()
                sum_exp += -math.log(U)
            return theta * sum_exp
        else:
            # For non-integer k, use Ahrens-Dieter acceptance-rejection method
            return self._ahrens_dieter_sample(k, theta)
    
    def _ahrens_dieter_sample(self, k, theta):
        """Ahrens-Dieter acceptance-rejection method for non-integer shape parameter"""
        if k < 1:
            # For k < 1, use the method: Gamma(k) = Gamma(k+1) * U^(1/k)
            gamma_k_plus_1 = self._ahrens_dieter_sample(k + 1, 1)
            U = np.random.uniform()
            return theta * gamma_k_plus_1 * (U ** (1/k))
        
        # For k >= 1, use Ahrens-Dieter method
        d = k - 1/3
        c = 1 / math.sqrt(9 * d)
        
        while True:
            # Generate normal random variable
            U1, U2 = np.random.uniform(size=2)
            Z = math.sqrt(-2 * math.log(U1)) * math.cos(2 * math.pi * U2)  # Standard normal
            
            if Z > -1/c:
                V = (1 + c * Z) ** 3
                U = np.random.uniform()
                
                if math.log(U) < 0.5 * Z * Z + d * (1 - V + math.log(V)):
                    return theta * d * V
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]
        data = sorted(data)
        
        if plot:
            from scipy import stats
            density = stats.gaussian_kde(data)
            x_range = np.linspace(0.01, max(data) * 1.2, 200)
            plt.plot(x_range, density(x_range), 'b-', linewidth=2, label='Estimated Density')
            
            # Theoretical density
            theoretical_density = [self._pdf(x) for x in x_range]
            plt.plot(x_range, theoretical_density, 'r--', linewidth=2, label='Theoretical Density')
            
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'Gamma Distribution Density (k={self.k}, θ={self.theta})')
            plt.legend()
            
            if savepath:
                plt.savefig(savepath)
            plt.show()

        return data
    
    # def compose(self, other: CRVG):
    #     k = other.sample()
    #     theta = other.sample()
    #     return self.sample(k = k, theta = theta)


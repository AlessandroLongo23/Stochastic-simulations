import numpy as np
import matplotlib.pyplot as plt
import math
import os

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
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]

        if plot:
            plt.hist(data, bins=self.b - self.a + 1, range=(self.a, self.b))
            plt.xlabel('Outcome')
            plt.ylabel('Frequency')
            plt.title(f'Uniform Distribution (a={self.a}, b={self.b})')
            if savepath:
                plt.savefig(savepath)
            plt.show()

        return data
    
    def compose(self, other: CRVG):
        a = other.sample()
        b = other.sample()
        return self.sample(a, b)

class Exponential(CRVG):
    def __init__(self, lambda_):
        self.name = 'Exponential'
        self.lambda_ = lambda_
        self.p = [math.exp(-lambda_ * x) for x in range(100)]

    def sample(self, lambda_ = None):
        lambda_ = self.lambda_ if lambda_ is None else lambda_

        U = np.random.uniform()
        return -math.log(U) / lambda_
    
    def simulate(self, n, plot = True, savepath = False, plot_type = 'density'):
        data = [self.sample() for _ in range(n)]
        data = sorted(data)
        
        if plot:
            from scipy import stats
            density = stats.gaussian_kde(data)
            x_range = np.linspace(min(data), max(data), 200)
            plt.plot(x_range, density(x_range), 'b-', linewidth=2, label='Estimated Density')
            
            theoretical_density = [self.lambda_ * math.exp(-self.lambda_ * x) for x in x_range]
            plt.plot(x_range, theoretical_density, 'r--', linewidth=2, label='Theoretical Density')
            
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'Exponential Distribution Density (λ={self.lambda_})')
            plt.legend()
                
            if savepath:
                plt.savefig(savepath)
            plt.show()

        return data
    
    def compose(self, other: CRVG):
        lambda_ = other.sample()
        return self.sample(lambda_ = lambda_)
    
class Gaussian(CRVG):
    def __init__(self, mu = 0, sigma = 1):
        self.name = 'Gaussian'
        self.mu = mu
        self.sigma = sigma
        self.p = [math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi)) for x in range(100)]
    
    def sample(self, mu = None, sigma = None):
        mu = self.mu if mu is None else mu
        sigma = self.sigma if sigma is None else sigma

        U1 = np.random.uniform()
        U2 = np.random.uniform()

        r = math.sqrt(-2 * math.log(U1))
        theta = 2 * math.pi * U2

        return (r * math.cos(theta)) * self.sigma + self.mu
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]
        data = sorted(data)
        
        if plot:
            from scipy import stats
            density = stats.gaussian_kde(data)
            x_range = np.linspace(min(data), max(data), 200)
            plt.plot(x_range, density(x_range), 'b-', linewidth=2, label='Estimated Density')

            theoretical_density = [math.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2)) / (self.sigma * math.sqrt(2 * math.pi)) for x in x_range]
            plt.plot(x_range, theoretical_density, 'r--', linewidth=2, label='Theoretical Density')

            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'Gaussian Distribution Density (μ={self.mu}, σ={self.sigma})')
            plt.legend()
            
            if savepath:
                plt.savefig(savepath)
            plt.show()

        return data
    
    def compose(self, other: CRVG):
        mu = other.sample()
        sigma = other.sample()
        return self.sample(mu = mu, sigma = sigma)
    
    def generate_confidence_interval(self, observations: int = 10, simulations: int = 100, confidence_level: float = 0.95, savepath: str = None) -> tuple:
        import scipy.stats as stats
        
        means = []
        lower_bounds = []
        upper_bounds = []
        
        # Calculate critical value for the confidence level
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        for _ in range(simulations):
            sample_data = self.simulate(n=observations, plot=False, savepath=False)
            
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
        self.k = k
        self.beta = beta
        self.p = [1 - (self.beta / x) ** self.k for x in range(1, 30)]

    def sample(self, k = None, beta = None):
        k = self.k if k is None else k
        beta = self.beta if beta is None else beta

        U = np.random.uniform()
        return beta * (U ** (-1 / k))
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]
        data = sorted(data)

        if plot:
            from scipy import stats
            density = stats.gaussian_kde(data)
            x_range = np.linspace(self.beta, 30, 100)
            plt.plot(x_range, density(x_range), 'b-', linewidth=2, label='Estimated Density')
            
            theoretical_density = [self.k * self.beta ** self.k / x ** (self.k + 1) for x in x_range]
            plt.plot(x_range, theoretical_density, 'r--', linewidth=2, label='Theoretical Density')
            
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'Pareto Distribution Density (k={self.k}, β={self.beta})')
            plt.legend()

            if savepath:
                plt.savefig(savepath)
            plt.show()

        return data
    
    def compose(self, other: CRVG):
        k = other.sample()
        beta = other.sample()
        return self.sample(k = k, beta = beta)

import math
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time

from classes.RNG import RNG
from classes.CRVG import CRVG
from classes.DRVG import DRVG

class Evaluator:
    def __init__(self, generator: RNG | CRVG | DRVG):
        self.generator = generator
        pass

    def chi_square(self, n: int = 10000, simulations: int = 100, classes: int = 10, savepath = None) -> float:
        Ts = []

        if isinstance(self.generator, RNG | CRVG):
            p = [1 / classes for _ in range(classes)]
        else:
            p = self.generator.p

        for _ in range(simulations):
            data = self.generator.simulate(seed = random.randint(1, 10000), n = n, normalize = True, plot = False, savepath = False)
            
            T = 0
            cumulative_probability = 0
            epsilon = 1e-6
            for i in range(len(p)):
                if isinstance(self.generator, RNG | CRVG):
                    observed = len([x for x in data if x >= cumulative_probability and x < cumulative_probability + p[i]])
                else:
                    observed = len([x for x in data if math.fabs(x - i / len(p)) < epsilon])
                cumulative_probability += p[i]
                expected = p[i] * n
                T += ((observed - expected) ** 2) / expected

            Ts.append(T)

        plt.hist(Ts, bins = 50)
        plt.title('Chi-square')
        plt.xlabel('T')
        plt.ylabel('Frequency')
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()

        return sum(Ts) / simulations

    def kolmogorov_smirnov(self, n: int = 10000, simulations: int = 100, savepath = None) -> float:
        D = []

        for _ in range(simulations):
            data = self.generator.simulate(seed = random.randint(1, 10000), n = n, normalize = True, sort = True)

            F_n = [i / n for i in range(n)]
            F_n_data = [sum(1 for x in data if x < i / n) / n for i in range(n)]

            D.append(max(abs(F_n_data[i] - F_n[i]) for i in range(n)))

        Z = [(math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * D[i] for i in range(len(D))]

        plt.hist(Z, bins = 50)
        plt.title('Kolmogorov-Smirnov')
        plt.xlabel('Z')
        plt.ylabel('Frequency')
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()


    def above_below(self, n: int = 10000, simulations: int = 100, savepath = None) -> float:
        number_of_runs = []
        for _ in range(simulations):
            data = self.generator.simulate(seed = random.randint(1, 10000), n = n)
            median = np.median(data)

            # n1 = len([x for x in data if x > median])
            # n2 = len([x for x in data if x < median])

            runs = 0
            is_above = data[0] > median

            for i in range(1, len(data)):
                if data[i] > median and not is_above:
                    runs += 1
                    is_above = True
                elif data[i] < median and is_above:
                    runs += 1
                    is_above = False

            runs += 1
            number_of_runs.append(runs)
            # mean = 2 * (n1 * n2) / (n1 + n2) + 1
            # variance = 2 * (n1 * n2) * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))

        plt.hist(number_of_runs, bins = 50)
        plt.title('Above and below')
        plt.xlabel('Number of runs')
        plt.ylabel('Frequency')
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()

        # return runs, mean, variance

    def up_down_knuth(self, n: int = 10000, simulations: int = 100, savepath = None) -> float:
        Z = []
        for _ in range(simulations):
            data = self.generator.simulate(seed = random.randint(1, 10000), n = n)

            runs = []
            count = 1
            for i in range(1, len(data)):
                if data[i] > data[i - 1]:
                    count += 1
                else:
                    runs.append(count)
                    count = 1

            runs.append(count)

            R = [
                runs.count(1),
                runs.count(2),
                runs.count(3),
                runs.count(4),
                runs.count(5),
            ]
            temp = sum(R)
            R.append(len(runs) - temp)

            A = np.array([
                [4529.4, 9044.9, 13568, 18091, 22615, 27892],
                [9044.9, 18097, 27139, 36187, 45234, 55789],
                [13568, 27139, 40721, 54281, 67852, 83685],
                [18091, 36187, 54281, 72414, 90470, 111580],
                [22615, 45234, 67852, 90470, 113262, 139476],
                [27892, 55789, 83685, 111580, 139476, 172860]
            ])

            B = np.array([1/6, 5/24, 11/120, 19/720, 29/5040, 1/840])
            
            n = len(data)
            R = np.array(R)
            
            # Calculate Z statistic according to the formula
            diff = R - n * B
            Z.append((1 / (n - 6)) * diff.T @ A @ diff)
        
        plt.hist(Z, bins = 50)
        plt.title('Up and down Knuth')
        plt.xlabel('Z')
        plt.ylabel('Frequency')
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()

    def up_and_down(self, n: int = 10000, simulations: int = 100, savepath = None) -> float:
        Z = []
        for _ in range(simulations):
            data = self.generator.simulate(seed = random.randint(1, 10000), n = n)

            comparisons = []
            for i in range(1, len(data)):
                if data[i] > data[i - 1]:
                    comparisons.append(1)
                else:
                    comparisons.append(-1)

            runs = []
            count = 1
            for i in range(1, len(comparisons)):
                if comparisons[i] == comparisons[i - 1]:
                    count += 1
                else:
                    runs.append(count)
                    count = 1

            runs.append(count)

            X, n = len(runs), len(data)

            Z.append((X - (2 * n - 1) / 3) / math.sqrt((16 * n - 29) / 90))

        plt.hist(Z, bins = 50)
        plt.title('Up and down')
        plt.xlabel('Z')
        plt.ylabel('Frequency')
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()


    def estimated_correlation(self, n: int = 10000, simulations: int = 1000, gap: int = 1, savepath = None) -> float:
        c = []
        for _ in range(simulations):
            data = self.generator.simulate(seed = random.randint(1, 10000), n = n, normalize = True)
        
            n = len(data)
            sum_of_products = 0

            for i in range(n - gap):
                sum_of_products += data[i] * data[i + gap]

            c.append(1 / (n - gap) * sum_of_products)

        plt.hist(c, bins = 50)
        plt.title('Estimated correlation')
        plt.xlabel('c')
        plt.ylabel('Frequency')
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()

    def analyze_time(self, n: int, methods: list[str], simulations: int = 1000, savepath = None) -> dict:
        data = {
            method: {
                'time': [],
                'mean': 0,
                'std': 0
            } for method in methods
        }
        for method in methods:
            for _ in range(simulations):
                start_time = time.time()
                _ = self.generator.simulate(n = n, plot = False, savepath = False, method = method)
                end_time = time.time()
                data[method]['time'].append(end_time - start_time)
            
            data[method]['mean'] = sum(data[method]['time']) / simulations
            data[method]['std'] = math.sqrt(sum((x - data[method]['mean']) ** 2 for x in data[method]['time']) / simulations)

        plt.bar(methods, [data[method]['mean'] for method in methods], yerr = [data[method]['std'] for method in methods], capsize = 5)
        plt.title('Execution Time Comparison')
        plt.xlabel('Method')
        plt.ylabel('Time (seconds)')
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()
        
        return data
    
    def analyze_pareto(self, n: int = 10000, simulations: int = 100, savepath = None) -> tuple:
        observed_means = []
        observed_variances = []
        for _ in range(simulations):
            data = self.generator.simulate(n = n, plot = False, savepath = False)

            observed_mean = sum(data) / len(data)
            observed_variance = sum((x - observed_mean) ** 2 for x in data) / len(data)

            observed_means.append(observed_mean)
            observed_variances.append(observed_variance)

        # Calculate theoretical values
        if self.generator.k > 1:
            theoretical_mean = self.generator.beta * self.generator.k / (self.generator.k - 1)
        else:
            theoretical_mean = float('inf')
            print('k must be greater than 1 for finite mean')
        
        if self.generator.k > 2:
            theoretical_variance = self.generator.beta ** 2 * self.generator.k / ((self.generator.k - 1) ** 2 * (self.generator.k - 2))
        else:
            theoretical_variance = float('inf')
            print('k must be greater than 2 for finite variance')

        # Create subplot figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot observed means
        ax1.hist(observed_means, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        if theoretical_mean != float('inf'):
            ax1.axvline(theoretical_mean, color='red', linestyle=':', linewidth=2, 
                       label=f'Theoretical Mean = {theoretical_mean:.3f}')
        ax1.set_title('Distribution of Observed Means')
        ax1.set_xlabel('Mean')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot observed variances
        ax2.hist(observed_variances, bins=400, alpha=0.7, color='lightcoral', edgecolor='black')
        if theoretical_variance != float('inf'):
            ax2.axvline(theoretical_variance, color='red', linestyle=':', linewidth=2, 
                       label=f'Theoretical Variance = {theoretical_variance:.3f}')
        ax2.set_title('Distribution of Observed Variances')
        ax2.set_xlabel('Variance')
        ax2.set_ylabel('Frequency')
        ax2.set_xlim(0, 10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Pareto Distribution Analysis (k={self.generator.k}, β={self.generator.beta})')
        plt.tight_layout()
        
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath), dpi=300, bbox_inches='tight')
        plt.show()

        return observed_means, observed_variances, theoretical_mean, theoretical_variance

import math
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time
from scipy.stats import ttest_ind

from classes.RNG import RNG, LCG
from classes.CRVG import CRVG, Pareto
from classes.DRVG import DRVG, Discrete, Geometric
from classes.Plotter import Plotter

class Evaluator:
    def __init__(self):
        pass

    def p_value(self, generator: RNG | CRVG | DRVG, n: int = 10000, simulations: int = 100, savepath = None) -> dict:
        p_values = []

        for _ in range(simulations):
            data = generator.simulate(seed = random.randint(1, 10000), n = n, normalize = True, plot = False, savepath = False)
            if isinstance(data, dict):
                data1 = data['observed']
            else:
                data1 = data

            data2 = generator.simulate(seed = random.randint(1, 10000), n = n, normalize = True, plot = False, savepath = False)
            if isinstance(data2, dict):
                data2 = data2['observed']
            else:
                data2 = data2

            p_values.append(ttest_ind(data1, data2).pvalue)

        plotter = Plotter()
        plotter.plot_histogram(p_values, title = f'P-value - {generator.name}', x_label = 'P-value', y_label = 'Frequency', savepath = savepath)
            
    def chi_square(self, generators: list[RNG | CRVG | DRVG], n: int = 10000, simulations: int = 100, savepath = None, classes: int = 10) -> dict:
        results = {}
        
        for idx, generator in enumerate(generators):
            Ts = []

            if isinstance(generator, LCG):
                points = [i / classes * generator.m for i in range(classes + 1)]
                p = [1 / classes for i in range(classes)]
            if isinstance(generator, CRVG):
                if isinstance(generator, Pareto):
                    points = np.linspace(1, 50, 100).tolist()
                else:
                    points = np.linspace(0, 50, 100).tolist()
                p = [generator.cdf(points[i + 1]) - generator.cdf(points[i]) for i in range(len(points) - 1)]
            elif isinstance(generator, DRVG):
                if isinstance(generator, Geometric):
                    points = [i for i in range(1, classes + 1)]
                    p = [generator.pdf(i) for i in range(1, classes)]
                elif isinstance(generator, Discrete):
                    classes = len(generator.p)
                    points = [i for i in range(classes + 1)]
                    p = [generator.pdf(i) for i in range(classes)]

            for i in range(len(p) - 1, -1, -1):
                if p[i] * n < 5:
                    p.pop(i)
                    points.pop(i + 1)

            for _ in range(simulations):
                data = generator.simulate(seed = random.randint(1, 10000), n = n, normalize = False, plot = False, savepath = False)
                if isinstance(data, dict):
                    data = data['observed']

                observed_frequencies = [len([x for x in data if x >= points[i] and x < points[i + 1]]) for i in range(len(points) - 1)]
                expected_frequencies = [p[i] * n for i in range(len(p))]

                T = 0
                for i in range(len(p)):
                    T += ((observed_frequencies[i] - expected_frequencies[i]) ** 2) / expected_frequencies[i]

                Ts.append(T)

            mean_T = sum(Ts) / len(Ts)
            std_T = math.sqrt(sum((x - mean_T) ** 2 for x in Ts) / len(Ts))
            
            results[f'Generator_{idx}'] = {
                'values': Ts,
                'mean': mean_T,
                'std': std_T
            }

            plt.figure(figsize=(10, 6))
            plt.hist(Ts, bins = 50, alpha=0.7)
            plt.title(f'Chi-square - {generator.name}')
            plt.xlabel('T')
            plt.ylabel('Frequency')
            
            plt.axvline(mean_T, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_T:.3f}')
            plt.axvline(mean_T - std_T, color='orange', linestyle='--', linewidth=1, label=f'μ - σ = {mean_T - std_T:.3f}')
            plt.axvline(mean_T + std_T, color='orange', linestyle='--', linewidth=1, label=f'μ + σ = {mean_T + std_T:.3f}')
            plt.legend()
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', f'{savepath}_{generator.name}.png'))
            plt.show()

        if len(generators) > 1:
            plt.figure(figsize=(10, 6))
            generator_names = [gen.name for gen in generators]
            means = [results[key]['mean'] for key in results.keys()]
            stds = [results[key]['std'] for key in results.keys()]
            
            plt.bar(generator_names, means, yerr=stds, capsize=5, alpha=0.7)
            plt.title('Chi-square Test - Comparison')
            plt.xlabel('Generator')
            plt.ylabel('Mean T Value')
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', f'{savepath}_comparison.png'))
            plt.show()

        return results

    def kolmogorov_smirnov(self, generators: list[RNG | CRVG | DRVG], n: int = 10000, simulations: int = 100, savepath = None) -> dict:
        results = {}
        
        for idx, generator in enumerate(generators):
            D = []

            for _ in range(simulations):
                data = generator.simulate(seed = random.randint(1, 10000), n = n, normalize = False, sort = True)
                if isinstance(data, dict):
                    data = data['observed']

                Fn_theoretical = [generator.cdf(data[i]) for i in range(n)]
                Fn_observed = [len([x for x in data if x <= data[i]]) / n for i in range(n)]

                D.append(max([abs(Fn_observed[i] - Fn_theoretical[i]) for i in range(n)]))

            Z = [(math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * D[i] for i in range(len(D))]

            mean_Z = sum(Z) / len(Z)
            std_Z = math.sqrt(sum((x - mean_Z) ** 2 for x in Z) / len(Z))
            
            results[f'Generator_{idx}'] = {
                'values': Z,
                'mean': mean_Z,
                'std': std_Z
            }

            plt.figure(figsize=(10, 6))
            plt.hist(Z, bins = 50, alpha=0.7)
            plt.title(f'Kolmogorov-Smirnov - {generator.name}')
            plt.xlabel('Z')
            plt.ylabel('Frequency')
            
            plt.axvline(mean_Z, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_Z:.3f}')
            plt.axvline(mean_Z - std_Z, color='orange', linestyle='--', linewidth=1, label=f'μ - σ = {mean_Z - std_Z:.3f}')
            plt.axvline(mean_Z + std_Z, color='orange', linestyle='--', linewidth=1, label=f'μ + σ = {mean_Z + std_Z:.3f}')
            plt.legend()
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', f'{savepath}_{generator.name}.png'))
            plt.show()

        if len(generators) > 1:
            plt.figure(figsize=(10, 6))
            generator_names = [gen.name for gen in generators]
            means = [results[key]['mean'] for key in results.keys()]
            stds = [results[key]['std'] for key in results.keys()]
            
            plt.bar(generator_names, means, yerr=stds, capsize=5, alpha=0.7)
            plt.title('Kolmogorov-Smirnov Test - Comparison')
            plt.xlabel('Generator')
            plt.ylabel('Mean Z Value')
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', f'{savepath}_comparison.png'))
            plt.show()

        return results

    def above_below(self, generators: list[RNG | CRVG | DRVG], n: int = 10000, simulations: int = 100, savepath = None) -> dict:
        results = {}
        
        for idx, generator in enumerate(generators):
            number_of_runs = []
            for _ in range(simulations):
                data = generator.simulate(seed = random.randint(1, 10000), n = n)
                if isinstance(data, dict):
                    data = data['observed']

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

            mean_runs = sum(number_of_runs) / len(number_of_runs)
            std_runs = math.sqrt(sum((x - mean_runs) ** 2 for x in number_of_runs) / len(number_of_runs))
            
            results[f'Generator_{idx}'] = {
                'values': number_of_runs,
                'mean': mean_runs,
                'std': std_runs
            }

            plt.figure(figsize=(10, 6))
            plt.hist(number_of_runs, bins = 50, alpha=0.7)
            plt.title(f'Above and below - {generator.name}')
            plt.xlabel('Number of runs')
            plt.ylabel('Frequency')
            
            plt.axvline(mean_runs, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_runs:.3f}')
            plt.axvline(mean_runs - std_runs, color='orange', linestyle='--', linewidth=1, label=f'μ - σ = {mean_runs - std_runs:.3f}')
            plt.axvline(mean_runs + std_runs, color='orange', linestyle='--', linewidth=1, label=f'μ + σ = {mean_runs + std_runs:.3f}')
            plt.legend()
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', f'{savepath}_{generator.name}.png'))
            plt.show()

        if len(generators) > 1:
            plt.figure(figsize=(10, 6))
            generator_names = [gen.name for gen in generators]
            means = [results[key]['mean'] for key in results.keys()]
            stds = [results[key]['std'] for key in results.keys()]
            
            plt.bar(generator_names, means, yerr=stds, capsize=5, alpha=0.7)
            plt.title('Above and Below Test - Comparison')
            plt.xlabel('Generator')
            plt.ylabel('Mean Number of Runs')
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', f'{savepath}_comparison.png'))
            plt.show()

        return results

    def up_down_knuth(self, generators: list[RNG | CRVG | DRVG], n: int = 10000, simulations: int = 100, savepath = None) -> dict:
        results = {}
        
        for idx, generator in enumerate(generators):
            Z = []
            for _ in range(simulations):
                data = generator.simulate(seed = random.randint(1, 10000), n = n)
                if isinstance(data, dict):
                    data = data['observed']

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
                
                diff = R - n * B
                Z.append((1 / (n - 6)) * diff.T @ A @ diff)
            
            mean_Z = sum(Z) / len(Z)
            std_Z = math.sqrt(sum((x - mean_Z) ** 2 for x in Z) / len(Z))
            
            results[f'Generator_{idx}'] = {
                'values': Z,
                'mean': mean_Z,
                'std': std_Z
            }
            
            plt.figure(figsize=(10, 6))
            plt.hist(Z, bins = 50, alpha=0.7)
            plt.title(f'Up and down Knuth - {generator.name}')
            plt.xlabel('Z')
            plt.ylabel('Frequency')
            
            plt.axvline(mean_Z, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_Z:.3f}')
            plt.axvline(mean_Z - std_Z, color='orange', linestyle='--', linewidth=1, label=f'μ - σ = {mean_Z - std_Z:.3f}')
            plt.axvline(mean_Z + std_Z, color='orange', linestyle='--', linewidth=1, label=f'μ + σ = {mean_Z + std_Z:.3f}')
            plt.legend()
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', f'{savepath}_{generator.name}.png'))
            plt.show()

        if len(generators) > 1:
            plt.figure(figsize=(10, 6))
            generator_names = [gen.name for gen in generators]
            means = [results[key]['mean'] for key in results.keys()]
            stds = [results[key]['std'] for key in results.keys()]
            
            plt.bar(generator_names, means, yerr=stds, capsize=5, alpha=0.7)
            plt.title('Up and Down Knuth Test - Comparison')
            plt.xlabel('Generator')
            plt.ylabel('Mean Z Value')
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', f'{savepath}_comparison.png'))
            plt.show()

        return results

    def up_and_down(self, generators: list[RNG | CRVG | DRVG], n: int = 10000, simulations: int = 100, savepath = None) -> dict:
        results = {}
        
        for idx, generator in enumerate(generators):
            Z = []
            for _ in range(simulations):
                data = generator.simulate(seed = random.randint(1, 10000), n = n)
                if isinstance(data, dict):
                    data = data['observed']

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

            mean_Z = sum(Z) / len(Z)
            std_Z = math.sqrt(sum((x - mean_Z) ** 2 for x in Z) / len(Z))
            
            results[f'Generator_{idx}'] = {
                'values': Z,
                'mean': mean_Z,
                'std': std_Z
            }

            plt.figure(figsize=(10, 6))
            plt.hist(Z, bins = 50, alpha=0.7)
            plt.title(f'Up and down - {generator.name}')
            plt.xlabel('Z')
            plt.ylabel('Frequency')
            
            plt.axvline(mean_Z, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_Z:.3f}')
            plt.axvline(mean_Z - std_Z, color='orange', linestyle='--', linewidth=1, label=f'μ - σ = {mean_Z - std_Z:.3f}')
            plt.axvline(mean_Z + std_Z, color='orange', linestyle='--', linewidth=1, label=f'μ + σ = {mean_Z + std_Z:.3f}')
            plt.legend()
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', f'{savepath}_{generator.name}.png'))
            plt.show()

        if len(generators) > 1:
            plt.figure(figsize=(10, 6))
            generator_names = [gen.name for gen in generators]
            means = [results[key]['mean'] for key in results.keys()]
            stds = [results[key]['std'] for key in results.keys()]
            
            plt.bar(generator_names, means, yerr=stds, capsize=5, alpha=0.7)
            plt.title('Up and Down Test - Comparison')
            plt.xlabel('Generator')
            plt.ylabel('Mean Z Value')
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', f'{savepath}_comparison.png'))
            plt.show()

        return results

    def estimated_correlation(self, generators: list[RNG | CRVG | DRVG], n: int = 10000, simulations: int = 1000, gap: int = 1, savepath = None) -> dict:
        results = {}
        
        for idx, generator in enumerate(generators):
            c = []
            for _ in range(simulations):
                data = generator.simulate(seed = random.randint(1, 10000), n = n, normalize = True)
                if isinstance(data, dict):
                    data = data['observed']

                n = len(data)
                sum_of_products = 0

                for i in range(n - gap):
                    sum_of_products += data[i] * data[i + gap]

                c.append(1 / (n - gap) * sum_of_products)

            mean_c = sum(c) / len(c)
            std_c = math.sqrt(sum((x - mean_c) ** 2 for x in c) / len(c))
            
            results[f'Generator_{idx}'] = {
                'values': c,
                'mean': mean_c,
                'std': std_c
            }

            plt.figure(figsize=(10, 6))
            plt.hist(c, bins = 50, alpha=0.7)
            plt.title(f'Estimated correlation - {generator.name}')
            plt.xlabel('c')
            plt.ylabel('Frequency')
            
            plt.axvline(mean_c, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_c:.3f}')
            plt.axvline(mean_c - std_c, color='orange', linestyle='--', linewidth=1, label=f'μ - σ = {mean_c - std_c:.3f}')
            plt.axvline(mean_c + std_c, color='orange', linestyle='--', linewidth=1, label=f'μ + σ = {mean_c + std_c:.3f}')
            plt.legend()
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', f'{savepath}_{generator.name}.png'))
            plt.show()

        if len(generators) > 1:
            plt.figure(figsize=(10, 6))
            generator_names = [gen.name for gen in generators]
            means = [results[key]['mean'] for key in results.keys()]
            stds = [results[key]['std'] for key in results.keys()]
            
            plt.bar(generator_names, means, yerr=stds, capsize=5, alpha=0.7)
            plt.title('Estimated Correlation Test - Comparison')
            plt.xlabel('Generator')
            plt.ylabel('Mean Correlation')
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', f'{savepath}_comparison.png'))
            plt.show()

        return results
    
    def run_all_tests(self, generators: list[RNG | CRVG | DRVG], n: int = 10000, simulations: int = 100, savepath = None, classes: int = 10) -> dict:
        self.run_GOF_tests(generators = generators, n = n, simulations = simulations, savepath = savepath, classes = classes)
        self.run_runs_tests(generators = generators, n = n, simulations = simulations, savepath = savepath)
        self.estimated_correlation(generators = generators, n = n, simulations = simulations, gap = 1, savepath = savepath)

    def run_GOF_tests(self, generators: list[RNG | CRVG | DRVG], n: int = 10000, simulations: int = 100, savepath = None, classes: int = 10) -> dict:
        self.chi_square(generators = generators, n = n, simulations = simulations, savepath = savepath, classes = classes)
        self.kolmogorov_smirnov(generators = generators, n = n, simulations = simulations, savepath = savepath)

    def run_runs_tests(self, generators: list[RNG | CRVG | DRVG], n: int = 10000, simulations: int = 100, savepath = None) -> dict:
        self.above_below(generators = generators, n = n, simulations = simulations, savepath = savepath)
        self.up_down_knuth(generators = generators, n = n, simulations = simulations, savepath = savepath)
        self.up_and_down(generators = generators, n = n, simulations = simulations, savepath = savepath)

    def analyze_time(self, generators: list[RNG | CRVG | DRVG], n: int, classes: list[int] = None, simulations: int = 1000, savepath = None) -> dict:
        if classes is None:
            data = {
                f'Generator_{idx}': {
                    'time': [],
                    'mean': 0,
                    'std': 0
                } for idx, generator in enumerate(generators)
            }
            for idx, generator in enumerate(generators):
                generator_key = f'Generator_{idx}'
                for _ in range(simulations):
                    start_time = time.time()
                    _ = generator.simulate(n = n, plot = False, savepath = False)
                    end_time = time.time()
                    data[generator_key]['time'].append(end_time - start_time)
                
                data[generator_key]['mean'] = sum(data[generator_key]['time']) / simulations
                data[generator_key]['std'] = math.sqrt(sum((x - data[generator_key]['mean']) ** 2 for x in data[generator_key]['time']) / simulations)

            generator_names = [gen.name for gen in generators]
            plt.figure(figsize=(10, 6))
            plt.bar(generator_names, [data[f'Generator_{i}']['mean'] for i in range(len(generators))], yerr = [data[f'Generator_{i}']['std'] for i in range(len(generators))], capsize = 5)
            plt.title(f'Execution Time Comparison (n = {n}, classes = {len(generators[0].p)})')
            plt.xlabel('Generator')
            plt.ylabel('Time (seconds)')
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', savepath))
            plt.show()
            
        else:
            data = {
                f'Generator_{idx}': {
                    'n_values': n,
                    'mean_times': [],
                    'std_times': []
                } for idx, generator in enumerate(generators)
            }
            
            for idx, generator in enumerate(generators):
                generator_key = f'Generator_{idx}'
                
                for class_val in classes:
                    p = [0] + sorted([random.uniform(0, 1) for i in range(class_val - 1)]) + [1]
                    p = [p[i + 1] - p[i] for i in range(class_val)]
                    generator.setProbabilities(p)

                    times_for_class = []
                    for _ in range(simulations):
                        start_time = time.time()
                        _ = generator.simulate(n = n, plot = False, savepath = False)
                        end_time = time.time()
                        times_for_class.append(end_time - start_time)
                    
                    mean_time = sum(times_for_class) / len(times_for_class)
                    std_time = math.sqrt(sum((x - mean_time) ** 2 for x in times_for_class) / len(times_for_class))
                    
                    data[generator_key]['mean_times'].append(mean_time)
                    data[generator_key]['std_times'].append(std_time)
            
            plt.figure(figsize=(12, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, len(generators)))
            
            for idx, generator in enumerate(generators):
                generator_key = f'Generator_{idx}'
                means = data[generator_key]['mean_times']
                stds = data[generator_key]['std_times']
                
                plt.plot(classes, means, color=colors[idx], label=generator.name, linewidth=2, marker='o')
                
                upper_bound = [means[i] + stds[i] for i in range(len(means))]
                lower_bound = [means[i] - stds[i] for i in range(len(means))]
                plt.fill_between(classes, lower_bound, upper_bound, color=colors[idx], alpha=0.1)
            
            plt.title(f'Execution Time vs Sample Size (n = {n})')
            plt.xlabel('n (Sample Size)')
            plt.ylabel('Time (seconds)')
            plt.legend()
            plt.ylim(bottom = 0, top = 0.0008)
            plt.grid(True, alpha=0.3)
            
            if savepath:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.savefig(os.path.join('plots', savepath))
            plt.show()
        
        return data
    
    def analyze_pareto(self, generator: RNG | CRVG | DRVG, n: int = 10000, simulations: int = 100, savepath = None) -> tuple:
        if not isinstance(generator, Pareto):
            raise ValueError('Generator must be a Pareto distribution')
        
        observed_means = []
        observed_variances = []
        for _ in range(simulations):
            data = generator.simulate(n = n, plot = False, savepath = False)
            if isinstance(data, dict):
                data = data['observed']

            observed_mean = sum(data) / len(data)
            observed_variance = sum((x - observed_mean) ** 2 for x in data) / len(data)

            observed_means.append(observed_mean)
            observed_variances.append(observed_variance)

        mean_of_means = np.mean(observed_means)
        mean_of_variances = np.mean(observed_variances)

        if generator.k > 1:
            theoretical_mean = generator.beta * generator.k / (generator.k - 1)
        else:
            theoretical_mean = float('inf')
            print('k must be greater than 1 for finite mean')
        
        if generator.k > 2:
            theoretical_variance = generator.beta ** 2 * generator.k / ((generator.k - 1) ** 2 * (generator.k - 2))
        else:
            theoretical_variance = float('inf')
            print('k must be greater than 2 for finite variance')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(observed_means, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        if theoretical_mean != float('inf'):
            ax1.axvline(theoretical_mean, color='red', linestyle=':', linewidth=2, 
                       label=f'Theoretical Mean = {theoretical_mean:.3f}')
        ax1.axvline(mean_of_means, color='blue', linestyle='--', linewidth=2,
                    label=f'Observed Mean = {mean_of_means:.3f}')
        ax1.set_title('Distribution of Observed Means')
        ax1.set_xlabel('Mean')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(observed_variances, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        if theoretical_variance != float('inf'):
            ax2.axvline(theoretical_variance, color='red', linestyle=':', linewidth=2, 
                       label=f'Theoretical Variance = {theoretical_variance:.3f}')
        ax2.axvline(mean_of_variances, color='blue', linestyle='--', linewidth=2,
                    label=f'Observed Mean of Variances = {mean_of_variances:.3f}')
        ax2.set_title('Distribution of Observed Variances')
        ax2.set_xlabel('Variance')
        ax2.set_ylabel('Frequency')
        ax2.set_xlim(0, 10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Pareto Distribution Analysis (k={generator.k}, β={generator.beta})')
        plt.tight_layout()
        
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath), dpi=300, bbox_inches='tight')
        plt.show()

        return observed_means, observed_variances, theoretical_mean, theoretical_variance

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from classes.Bootstrap import Bootstrap
from classes.CRVG import Gaussian, Pareto
import numpy as np
from classes.Plotter import Plotter
import math

if __name__ == "__main__":
    bootstrap = Bootstrap()

    # EX 1
    n = 10
    X_i = [56, 101, 78, 67, 93, 87, 64, 72, 80, 69]
    a, b = -5, 5

    n_bootstrap = 100000
    means, medians, stds = bootstrap.estimate(X_i, n_bootstrap=n_bootstrap)
    sample_mean = sum(X_i) / n
    Ys = [sample_mean - means[i] for i in range(n_bootstrap)]

    plotter = Plotter()
    plotter.plot_histogram(Ys, classes=50, title="Histogram of Ys", x_label="Y", y_label="Frequency", savepath="Ys_histogram.png")

    gaussian = Gaussian(np.mean(Ys), np.std(Ys))
    print(f"P(a <= X <= b) = {gaussian.cdf(b) - gaussian.cdf(a)}")


    # EX 2
    n = 15
    X_i = [5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 16, 8]

    means, medians, stds = bootstrap.estimate(X_i, n_bootstrap=n_bootstrap)

    Ss_squared = [np.sum(X_i - means[i]) ** 2 / (n - 1) for i in range(n_bootstrap)]
    var_Ss_squared = np.var(Ss_squared)

    print(f"Mean: {np.mean(Ss_squared)}, Var: {var_Ss_squared}")


    # EX 3
    n = 200
    pareto = Pareto(k = 1.05, beta = 1)
    X_i = [pareto.sample() for _ in range(n)]
    means, medians, stds = bootstrap.estimate(X_i, n_bootstrap=100)

    theoretical_mean = pareto.mean()
    theoretical_median = pareto.median()

    plotter = Plotter()
    plotter.plot_histogram(X_i, title="Samples of Pareto distribution", x_label="Sample", y_label="Frequency", savepath="pareto_samples.png")

    print(f'Mean (sample): {np.mean(X_i)}\nMedian (sample): {np.median(X_i)}')

    print(f'Variance of the mean (bootstrap): {np.var(means)}')

    print(f"Mean of the medians (bootstrap): {np.mean(medians)}\nStd of the medians (bootstrap): {np.std(medians)}")

    mean_precision = 1 - math.fabs(np.mean(means) - theoretical_mean) / theoretical_mean
    median_precision = 1 - math.fabs(np.mean(medians) - theoretical_median) / theoretical_median
    print(f'Precision of the mean: {(mean_precision * 100):2f}%\nPrecision of the median: {(median_precision * 100):2f}%')

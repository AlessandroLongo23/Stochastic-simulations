import numpy as np

class Bootstrap:
    def __init__(self):
        pass

    def estimate(self, X_i, n_bootstrap=100):
        n = len(X_i)

        means = []
        medians = []
        stds = []
        for _ in range(n_bootstrap):
            X_i_bootstrap = np.random.choice(X_i, size=n, replace=True)
            means.append(np.mean(X_i_bootstrap))
            medians.append(np.median(X_i_bootstrap))
            stds.append(np.std(X_i_bootstrap))

        return means, medians, stds
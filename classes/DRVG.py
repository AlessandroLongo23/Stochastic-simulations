import numpy as np
import matplotlib.pyplot as plt
import math

class DRVG:
    def __init__(self):
        pass

class Bernoulli(DRVG):
    def __init__(self, p):
        self.p = [p, 1 - p]

    def sample(self):
        return np.random.binomial(1, self.p[0])
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]
        if plot:
            plt.hist(data, bins=2)
            plt.xlabel('Outcome')
            plt.ylabel('Frequency')
            plt.title(f'Bernoulli Distribution (p={self.p[0]})')
            if savepath:
                plt.savefig(savepath)
            plt.show()

        return data

class Binomial(DRVG):
    def __init__(self, n, p):
        self.n = n
        self.pp = p
        self.p = [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]

    def sample(self):
        return np.random.binomial(self.n, self.pp)

    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]

        if plot:
            plt.hist(data, bins=self.n+1, range=(0, self.n))
            plt.xlabel('Number of successes')
            plt.ylabel('Frequency')
            plt.title(f'Binomial Distribution (n={self.n}, p={self.pp})')
            if savepath:
                plt.savefig(savepath)
            plt.show()

class Geometric(DRVG):
    def __init__(self, p):
        self.p = [(1 - p) ** k * p for k in range(100)]

    def sample(self):
        U = np.random.uniform()
        return math.floor(np.log(U) / np.log(1 - self.p[0])) + 1
    
    def simulate(self, n, plot = True, savepath = False, builtin = True):
        if builtin:
            data = np.random.geometric(self.p, n)
        else:
            data = [self.sample() for _ in range(n)]

        if plot:
            max_value = max(data)
            plt.hist(data, bins=max_value + 1, range=(0, max_value))
            plt.xlabel('Number of trials')
            plt.ylabel('Frequency')
            plt.title(f'Geometric Distribution (p={self.p[0]})')
            if savepath:
                plt.savefig(savepath)
            plt.show()

        return data

class Discrete(DRVG):
    def __init__(self, p):
        self.p = p
        self.F, self.L = self.generate_alias_vectors()
        if sum(p) != 1:
            raise ValueError("The sum of probabilities must be 1")

    def direct(self):
        U = np.random.uniform()
        for i in range(len(self.p)):
            if U < sum(self.p[:i+1]):
                return i
            
    def rejection(self):
        c = max(self.p)
        while True:
            U1 = np.random.uniform()
            I = math.floor(U1 * len(self.p))
            U2 = np.random.uniform()
            if U2 <= self.p[I] / c:
                return I
            
    def generate_alias_vectors(self):
        eps = 1e-6

        L = [i + 1 for i in range(len(self.p))]
        F = [self.p[i] * len(self.p) for i in range(len(self.p))]

        G = [i for i in range(len(self.p)) if F[i] >= 1]
        S = [i for i in range(len(self.p)) if F[i] <= 1]

        while len(S) > 0 and len(G) > 0:
            i = G[0]
            j = S[0]
            L[j] = i
            F[i] = F[i] - (1 - F[j])
            if F[i] < 1 - eps:
                temp = G.pop(0)
                S.append(temp)
            
            S.pop(0)

        return F, L
        
    def alias(self, F, L):
        U1 = np.random.uniform()
        I = math.floor(U1 * len(self.p))
        U2 = np.random.uniform()
        if U2 <= F[I]:
            return I
        else:
            return L[I]
    
    def simulate(self, n: int, plot: bool = True, savepath: str = False, method: str = 'builtin', seed: int = None, normalize: bool = False, sort: bool = False):
        if seed:
            np.random.seed(seed)

        if method == 'builtin':
            data = np.random.choice(len(self.p), n, p=self.p)
        elif method == 'direct':
            data = [self.direct() for _ in range(n)]
        elif method == 'rejection':
            data = [self.rejection() for _ in range(n)]
        elif method == 'alias':
            data = [self.alias(self.F, self.L) for _ in range(n)]
        else:
            raise ValueError("Invalid method")
        
        if normalize:
            data = [x / len(self.p) for x in data]

        if sort:
            data.sort()

        if plot:
            plt.hist(data, bins=len(self.p), range=(0, len(self.p)))
            plt.xlabel('Outcome')
            plt.ylabel('Frequency')
            plt.title(f'Discrete Distribution (p={[float(f"{p:.3f}") for p in self.p]})')
            if savepath:
                plt.savefig(savepath)
            plt.show()

        return data

import numpy as np
import matplotlib.pyplot as plt
import math

from classes.Function import Function

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
        self.setProbabilities(p)
        
    def setProbabilities(self, p):
        self.p = p
        if sum(p) != 1:
            # raise ValueError("The sum of probabilities must be 1")
            self.p = [x / sum(p) for x in p]

    def simulate(self, n: int, plot: bool = True, savepath: str = False, seed: int = None, normalize: bool = False, sort: bool = False):
        if seed:
            np.random.seed(seed)

        data = [self.sample() for _ in range(n)]
        
        if normalize:
            data = [x / len(self.p) for x in data]

        if sort:
            data.sort()

        if plot:
            plt.hist(data, bins=len(self.p), range=(0, len(self.p)))
            plt.xlabel('Outcome')
            plt.ylabel('Frequency')
            plt.title(f'Discrete Distribution using {self.name} method')
            if savepath:
                plt.savefig(f'{savepath}.png')
            plt.show()

        return data

        
class DiscreteBuiltin(Discrete):
    def __init__(self, p):
        super().__init__(p)
        self.name = 'built-in'

    def simulate(self, n, plot = True, savepath = False, seed = None, normalize = False, sort = False):
        if seed:
            np.random.seed(seed)
        data = np.random.choice(len(self.p), p=self.p, size = n)
        if normalize:
            data = [x / len(self.p) for x in data]
        if sort:
            data.sort()
        if plot:
            plt.hist(data, bins=len(self.p), range=(0, len(self.p)))
            plt.xlabel('Outcome')
            plt.ylabel('Frequency')
            plt.title(f'Discrete Distribution using {self.name} method')
            if savepath:
                plt.savefig(f'{savepath}.png')
            plt.show()
        return data

class DiscreteDirect(Discrete):
    def __init__(self, p, method):
        super().__init__(p)
        self.name = f'direct_{method}'
        self.method = method

    def setProbabilities(self, p):
        super().setProbabilities(p)
        self.cdf = [sum(self.p[:i+1]) for i in range(len(self.p))]

    def sample(self):
        if self.method == 'inefficient':
            return self.sample_inefficient()
        elif self.method == 'linear':
            return self.sample_linear()
        elif self.method == 'binary':
            return self.sample_binary()
        
    def sample_inefficient(self):
        U = np.random.uniform()
        for i in range(len(self.p)):
            if U < sum(self.p[:i+1]):
                return i
        return len(self.p) - 1

    def sample_linear(self):
        U = np.random.uniform()
        for i in range(len(self.p)):
            if U < self.cdf[i]:
                return i
        return len(self.p) - 1
        
    def sample_binary(self):
        U = np.random.uniform()
        left, right = 0, len(self.p) - 1
        while left < right:
            mid = (left + right) // 2
            if U < self.cdf[mid]:
                right = mid
            else:
                left = mid + 1
        return left

class DiscreteRejection(Discrete):
    def __init__(self, p):
        super().__init__(p)
        self.name = 'rejection'

    def setProbabilities(self, p):
        super().setProbabilities(p)
        self.c = max(self.p)

    def sample(self):
        while True:
            U = np.random.uniform(size = 2)
            I = math.floor(U[0] * len(self.p))
            U2 = U[1]
            if U2 <= self.p[I] / self.c:
                return I

class DiscreteAlias(Discrete):
    def __init__(self, p):
        super().__init__(p)
        self.name = 'alias'

    def setProbabilities(self, p):
        super().setProbabilities(p)
        self.F, self.L = self.generate_alias_vectors()

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
        U = np.random.uniform(size = 2)
        I = math.floor(U[0] * len(self.p))
        U2 = U[1]
        if U2 <= F[I]:
            return I
        else:
            return L[I]
        
    def sample(self):
        return self.alias(self.F, self.L)
    

class CustomDRVG(DRVG):
    def __init__(self, equation: Function, support: int | tuple, name: str = None):
        if isinstance(support, int):
            self.support = (0, support)
        else:
            self.support = support

        self.p = [equation.evaluate(i) for i in range(self.support[0], self.support[1])]
        self.p = [x / sum(self.p) for x in self.p]
        self.name = name

    def sample(self):
        return np.random.choice(len(self.p), p=self.p)
    
    def simulate(self, n, plot = True, savepath = False):
        data = [self.sample() for _ in range(n)]
        if plot:
            plt.hist(data, bins=len(self.p), range=(self.support[0], self.support[1]))
            plt.xlabel('Outcome')
            plt.ylabel('Frequency')
            plt.title(f'Custom Distribution {self.name}')
            if savepath:
                plt.savefig(f'{savepath}.png')
            plt.show()

        return data
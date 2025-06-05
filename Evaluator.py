import math
import numpy as np
from RNG import RNG
import random
import matplotlib.pyplot as plt
class Evaluator:
    def __init__(self, rng: RNG):
        self.rng = rng
        pass

    def chi_square(self, n: int = 10000, simulations: int = 100, classes: int = 10) -> float:
        T = 0
        for _ in range(simulations):
            data = self.rng.simulate(random.randint(1, 10000), n, normalize = True, sort = True)
            
            for i in range(classes):
                data_in_class = [x for x in data if x >= i / classes and x < (i + 1) / classes]
                T += (len(data_in_class) - n / classes) ** 2 / (n / classes)

        return T / simulations

    def kolmogorov_smirnov(self, n: int = 10000, simulations: int = 100) -> float:
        D = []

        for _ in range(simulations):
            data = self.rng.simulate(random.randint(1, 10000), n, normalize = True, sort = True)

            F_n = [i / n for i in range(n)]
            F_n_data = [sum(1 for x in data if x < i / n) / n for i in range(n)]

            D.append(max(abs(F_n_data[i] - F_n[i]) for i in range(n)))

        Z = [(math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * D[i] for i in range(len(D))]

        plt.hist(Z, bins = 50)
        plt.show()


    def above_below(self, n: int = 10000, simulations: int = 100) -> float:
        number_of_runs = []
        for _ in range(simulations):
            data = self.rng.simulate(random.randint(1, 10000), n)
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
        plt.show()

        # return runs, mean, variance

    def up_down_knuth(self, n: int = 10000, simulations: int = 100) -> float:
        Z = []
        for _ in range(simulations):
            data = self.rng.simulate(random.randint(1, 10000), n)

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
        plt.show()

    def up_and_down(self, n: int = 10000, simulations: int = 100) -> float:
        Z = []
        for _ in range(simulations):
            data = self.rng.simulate(random.randint(1, 10000), n)

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
        plt.show()


    def estimated_correlation(self, n: int = 10000, simulations: int = 1000, gap: int = 1) -> float:
        c = []
        for _ in range(simulations):
            data = self.rng.simulate(random.randint(1, 10000), n, normalize = True)
        
            n = len(data)
            sum_of_products = 0

            for i in range(n - gap):
                sum_of_products += data[i] * data[i + gap]

            c.append(1 / (n - gap) * sum_of_products)

        plt.hist(c, bins = 50)
        plt.show()


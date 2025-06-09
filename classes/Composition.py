# from classes.CRVG import CRVG
# import matplotlib.pyplot as plt

# class Composition:
#     def __init__(self, a: CRVG, b: CRVG):
#         self.a = a
#         self.b = b

#     def sample(self):
#         return self.a.compose(self.b)
    
#     def simulate(self, n, plot = True, savepath = False):
#         data = []

#         for _ in range(n):
#             data.append(self.sample())

#         if plot:
#             plt.hist(data, bins=100, range=(self.a.beta, 30))
#             plt.xlabel('Value')
#             plt.ylabel('Frequency')
#             plt.title(f'Composition of {self.a.name} and {self.b.name}')
#             if savepath:
#                 plt.savefig(savepath)
#             plt.show()

#         return data
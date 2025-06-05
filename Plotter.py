import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        pass

    def plot_histogram(self, data, classes):
        plt.hist(data, bins=classes, edgecolor='black')
        plt.show()

    def plot_scatter(self, data):
        plt.scatter(data[:-1], data[1:], s=0.1)
        plt.show()

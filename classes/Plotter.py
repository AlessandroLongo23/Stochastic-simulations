import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self):
        pass

    def plot_histogram(self, data, classes, savepath = None):
        plt.hist(data, bins=classes, edgecolor='black')
        plt.title('Histogram of the data')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()

    def plot_scatter(self, data, savepath = None):
        plt.scatter(data[:-1], data[1:], s=0.1)
        plt.title('Scatter plot of the data')
        plt.xlabel('x[i]')
        plt.ylabel('x[i+1]')
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()

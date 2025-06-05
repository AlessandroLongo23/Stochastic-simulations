import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        pass

    def plot_histogram(self, data, classes, savepath = None):
        plt.hist(data, bins=classes, edgecolor='black')
        if savepath:
            plt.savefig(savepath)
        plt.show()

    def plot_scatter(self, data, savepath = None):
        plt.scatter(data[:-1], data[1:], s=0.1)
        if savepath:
            plt.savefig(savepath)
        plt.show()

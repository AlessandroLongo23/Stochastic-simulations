import matplotlib.pyplot as plt
import os
from scipy import stats
import numpy as np

class Plotter:
    def __init__(self):
        pass

    def plot_scatter(self, data, x_label = 'x[i]', y_label = 'x[i+1]', title = 'Scatter plot of the data', savepath = None):
        plt.scatter(data[:-1], data[1:], s=0.1)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()

    def plot_histogram(self, data, classes, theoretical_density = None, x_label = 'Value', y_label = 'Frequency', title = 'Histogram of the data', savepath = None):
        plt.hist(data, bins=classes, edgecolor='black')
        if theoretical_density:
            plt.plot(theoretical_density, 'r--', linewidth=2, label='Theoretical Density')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()

    def plot_density(self, data, x_label='Value', y_label='Density', title='Density of the data', savepath=None, x_range=None):
        colors = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-']
        
        if x_range:
            combined_x_range = np.linspace(x_range[0], x_range[1], 100)
        else:
            all_data_flat = []
            for dataset_info in data.values():
                if 'observed' in dataset_info:
                    all_data_flat.extend(dataset_info['observed'])
            
            if not all_data_flat:
                raise ValueError("No observed data found in the provided datasets")
            
            x_min, x_max = min(all_data_flat), max(all_data_flat)
            combined_x_range = np.linspace(x_min, x_max, 100)
        
        for i, dataset_info in enumerate(data.values()):
            if 'observed' not in dataset_info:
                continue
                
            density = stats.gaussian_kde(dataset_info['observed'])
            color = colors[i % len(colors)]
            label_estimated = f'{dataset_info['label']} - Estimated'
            
            plt.plot(
                    combined_x_range, 
                    density(combined_x_range), 
                    color[0] + '-', 
                    linewidth=2, 
                    label=label_estimated,
                )
            plt.xlim([combined_x_range[0], combined_x_range[-1]])
            
            label_theoretical = f'{dataset_info['label']} - Theoretical'
            theoretical = [dataset_info['generator'].theoretical_density(x) for x in combined_x_range]
            plt.plot(
                combined_x_range, 
                theoretical, 
                color[0] + '--', 
                linewidth=2, 
                label=label_theoretical,
            )
            plt.xlim([combined_x_range[0], combined_x_range[-1]])

        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()
        
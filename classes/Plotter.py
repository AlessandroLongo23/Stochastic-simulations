import matplotlib.pyplot as plt
import os
from scipy import stats
import numpy as np

class Plotter:
    def __init__(self):
        pass

    def plot_scatter(self, x_data, y_data, x_label = 'x[i]', y_label = 'x[i+1]', title = 'Scatter plot of the data', savepath = None):
        plt.scatter(x_data, y_data, s=0.1)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()

    def plot_line(self, x, y, x_label = 'x', y_label = 'y', title = 'Line plot of the data', savepath = None):
        plt.plot(x, y, 'b-', linewidth=2, label='Data')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()

    def plot_function(self, function, x_label = 'x', y_label = 'f(x)', title = 'Function', savepath = None):
        plt.plot([function.evaluate(x) for x in np.linspace(0, 10, 100)], 'r--', linewidth=2, label='Function')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()

    def plot_histogram(self, data, classes = 10, theoretical_density = None, x_label = 'Value', y_label = 'Frequency', title = 'Histogram of the data', savepath = None):
        # Handle different input formats and determine if data is 1D or 2D
        if isinstance(data, (list, tuple)) and len(data) == 2 and all(isinstance(d, (list, tuple, np.ndarray)) for d in data):
            # Case: data is ([x_values], [y_values])
            x_data, y_data = data
            is_2d = True
        else:
            # Convert to numpy array for shape checking
            data_array = np.array(data)
            
            if data_array.ndim == 1:
                # 1D data
                is_2d = False
            elif data_array.ndim == 2 and data_array.shape[1] == 2:
                # 2D data with shape (n, 2)
                x_data, y_data = data_array[:, 0], data_array[:, 1]
                is_2d = True
            else:
                raise ValueError(f"Unsupported data shape: {data_array.shape}. Expected 1D array, 2D array with 2 columns, or tuple/list of two 1D arrays.")
        
        if is_2d:
            # 2D histogram
            # If classes is a single number, use it for both dimensions
            if isinstance(classes, (int, float)):
                bins = [classes, classes]
            else:
                bins = classes
            
            plt.hist2d(x_data, y_data, bins=bins)
            plt.colorbar(label='Frequency')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            # Note: theoretical_density is not typically used for 2D histograms
            if theoretical_density:
                print("Warning: theoretical_density parameter is ignored for 2D histograms")
        else:
            # 1D histogram (existing behavior)
            plt.hist(data, bins=classes, edgecolor='black')
            if theoretical_density:
                plt.plot(theoretical_density, 'r--', linewidth=2, label='Theoretical Density')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        
        plt.title(title)
        if savepath:
            plt.savefig(savepath)
        plt.show()

    def plot_density(self, data, x_label='Value', y_label='Density', title='Density of the data', savepath=None, x_range=None):
        colors = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-']
        
        # Handle numpy array input - convert to simple format
        if isinstance(data, np.ndarray):
            # Convert numpy array to list for processing
            data_list = data.tolist() if hasattr(data, 'tolist') else list(data)
            
            if x_range:
                combined_x_range = np.linspace(x_range[0], x_range[1], 100)
            else:
                x_min, x_max = min(data_list), max(data_list)
                combined_x_range = np.linspace(x_min, x_max, 100)
            
            # Create density plot for numpy array
            density = stats.gaussian_kde(data_list)
            plt.plot(combined_x_range, density(combined_x_range), 'b-', linewidth=2, label='Estimated Density')
            plt.xlim([combined_x_range[0], combined_x_range[-1]])
            
        # Handle dictionary input (original format)
        elif isinstance(data, dict):
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
                label_estimated = f'{dataset_info["label"]} - Estimated'
                
                plt.plot(
                        combined_x_range, 
                        density(combined_x_range), 
                        color[0] + '-', 
                        linewidth=2, 
                        label=label_estimated,
                    )
                plt.xlim([combined_x_range[0], combined_x_range[-1]])
                
                label_theoretical = f'{dataset_info["label"]} - Theoretical'
                theoretical = [dataset_info['generator'].theoretical_density(x) for x in combined_x_range]
                plt.plot(
                    combined_x_range, 
                    theoretical, 
                    color[0] + '--', 
                    linewidth=2, 
                    label=label_theoretical,
                )
                plt.xlim([combined_x_range[0], combined_x_range[-1]])
        
        # Handle list input
        elif isinstance(data, (list, tuple)):
            if x_range:
                combined_x_range = np.linspace(x_range[0], x_range[1], 100)
            else:
                x_min, x_max = min(data), max(data)
                combined_x_range = np.linspace(x_min, x_max, 100)
            
            # Create density plot for list/tuple
            density = stats.gaussian_kde(data)
            plt.plot(combined_x_range, density(combined_x_range), 'b-', linewidth=2, label='Estimated Density')
            plt.xlim([combined_x_range[0], combined_x_range[-1]])
        
        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Expected numpy.ndarray, dict, list, or tuple.")

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        if savepath:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(os.path.join('plots', savepath))
        plt.show()
        
    def plot_path(self, coordinates, path, title = 'Path', savepath = None):
        plt.scatter(coordinates[:, 0], coordinates[:, 1])
        for i in range(len(path)):
            plt.plot([coordinates[path[i], 0], coordinates[path[(i + 1) % len(path)], 0]], [coordinates[path[i], 1], coordinates[path[(i + 1) % len(path)], 1]])
        plt.title(title)
        if savepath:
            if not os.path.exists(os.path.dirname(savepath)):
                os.makedirs(os.path.dirname(savepath))
            plt.savefig(savepath)
        plt.show()
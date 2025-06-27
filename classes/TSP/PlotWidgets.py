import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


class PlotWidget(ABC):
    def __init__(self, title: str = "", size: Tuple[int, int] = (1, 1)):
        self.title = title
        self.size = size
    
    @abstractmethod
    def render(self, ax: plt.Axes, data: Dict[str, Any]) -> None:
        pass


class CostEvolutionWidget(PlotWidget):
    def __init__(self, show_current: bool = True, show_best: bool = True, 
                 rolling_window: int = 500, title: str = "Cost Evolution"):
        super().__init__(title, (1, 1))
        self.show_current = show_current
        self.show_best = show_best
        self.rolling_window = rolling_window
    
    def render(self, ax: plt.Axes, data: Dict[str, Any]) -> None:
        cost_history = data.get('cost_history', [])
        best_cost_history = data.get('best_cost_history', [])
        
        iterations = range(len(cost_history))
        
        if self.show_current and cost_history:
            ax.plot(iterations, cost_history, 'b-', alpha=0.3, linewidth=0.5, label='Current Cost')
            if len(cost_history) > self.rolling_window:
                rolling_avg = self._rolling_average(cost_history, self.rolling_window)
                rolling_iter = range(self.rolling_window-1, len(cost_history))
                ax.plot(rolling_iter, rolling_avg, 'b-', linewidth=2, label='Current (Rolling Avg)')
        
        if self.show_best and best_cost_history:
            ax.plot(iterations, best_cost_history, 'r-', linewidth=2, label='Best Cost')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_title(self.title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _rolling_average(self, data: List[float], window_size: int) -> List[float]:
        if len(data) < window_size:
            return data
        return [sum(data[i:i+window_size])/window_size 
                for i in range(len(data)-window_size+1)]


class TemperatureWidget(PlotWidget):
    def __init__(self, log_scale: bool = True, title: str = "Temperature Evolution"):
        super().__init__(title, (1, 1))
        self.log_scale = log_scale
    
    def render(self, ax: plt.Axes, data: Dict[str, Any]) -> None:
        temperature_history = data.get('temperature_history', [])
        
        if temperature_history:
            iterations = range(len(temperature_history))
            ax.plot(iterations, temperature_history, 'orange', linewidth=2)
            
            if self.log_scale:
                ax.set_yscale('log')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Temperature')
            ax.set_title(self.title)
            ax.grid(True, alpha=0.3)


class AcceptanceRateWidget(PlotWidget):
    def __init__(self, window_size: int = 500, title: str = "Acceptance Rate"):
        super().__init__(title, (1, 1))
        self.window_size = window_size
    
    def render(self, ax: plt.Axes, data: Dict[str, Any]) -> None:
        acceptance_history = data.get('acceptance_history', [])
        
        if len(acceptance_history) > self.window_size:
            acceptance_rate = self._rolling_average(acceptance_history, self.window_size)
            iterations = range(self.window_size-1, len(acceptance_history))
            ax.plot(iterations, acceptance_rate, 'green', linewidth=2)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Acceptance Rate')
            ax.set_title(self.title)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
    
    def _rolling_average(self, data: List[bool], window_size: int) -> List[float]:
        return [sum(data[i:i+window_size])/window_size 
                for i in range(len(data)-window_size+1)]


class SolutionPathWidget(PlotWidget):
    def __init__(self, show_cities: bool = True, show_labels: bool = False, 
                 title: str = "Solution Path"):
        super().__init__(title, (1, 1))
        self.show_cities = show_cities
        self.show_labels = show_labels
    
    def render(self, ax: plt.Axes, data: Dict[str, Any]) -> None:
        coordinates = data.get('coordinates')
        solution = data.get('solution')
        cost = data.get('cost', 0)
        
        if coordinates is not None and solution is not None:
            if self.show_cities:
                ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                          c='red', s=50, zorder=3, alpha=0.7)
            
            for i in range(len(solution)):
                start = coordinates[solution[i]]
                end = coordinates[solution[(i + 1) % len(solution)]]
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       'b-', linewidth=1, alpha=0.7)
            
            if self.show_labels:
                for i, coord in enumerate(coordinates):
                    ax.annotate(str(i), coord, xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
            
            ax.set_title(f"{self.title} (Cost: {cost:.2f})")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)


class StatisticsWidget(PlotWidget):
    def __init__(self, title: str = "Algorithm Statistics"):
        super().__init__(title, (1, 1))
    
    def render(self, ax: plt.Axes, data: Dict[str, Any]) -> None:
        stats = data.get('statistics', {})
        
        ax.axis('off')
        
        text_lines = []
        text_lines.append(f"Initial Cost: {stats.get('initial_cost', 'N/A'):.2f}")
        text_lines.append(f"Final Cost: {stats.get('final_best_cost', 'N/A'):.2f}")
        text_lines.append(f"Improvement: {stats.get('improvement_percentage', 'N/A'):.1f}%")
        text_lines.append(f"Total Iterations: {stats.get('total_iterations', 'N/A'):,}")
        text_lines.append(f"Convergence Iter: {stats.get('convergence_iteration', 'N/A'):,}")
        text_lines.append(f"Acceptance Rate: {stats.get('overall_acceptance_rate', 'N/A'):.1f}%")
        
        text = '\n'.join(text_lines)
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax.set_title(self.title)


class ComparisonWidget(PlotWidget):
    def __init__(self, metric: str = 'best_cost_history', title: str = "Algorithm Comparison"):
        super().__init__(title, (1, 1))
        self.metric = metric
    
    def render(self, ax: plt.Axes, data: Dict[str, Any]) -> None:
        comparison_data = data.get('comparison_results', {})
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_data)))
        
        for i, (name, result) in enumerate(comparison_data.items()):
            sa_instance = result.get('sa_instance')
            if sa_instance and hasattr(sa_instance, self.metric):
                metric_data = getattr(sa_instance, self.metric)
                iterations = range(len(metric_data))
                ax.plot(iterations, metric_data, color=colors[i], 
                       linewidth=2, label=name)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel(self.metric.replace('_', ' ').title())
        ax.set_title(self.title)
        ax.legend()
        ax.grid(True, alpha=0.3)


class HeatmapWidget(PlotWidget):
    def __init__(self, title: str = "Heatmap"):
        super().__init__(title, (1, 1))
    
    def render(self, ax: plt.Axes, data: Dict[str, Any]) -> None:
        matrix = data.get('matrix')
        
        if matrix is not None:
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            ax.set_title(self.title)
            plt.colorbar(im, ax=ax)


class ComposablePlotter:
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.widgets = []
        self.figsize = figsize
        self.layout = None
    
    def add_widget(self, widget: PlotWidget, position: Optional[Tuple[int, int]] = None) -> None:
        self.widgets.append((widget, position))
    
    def set_layout(self, rows: int, cols: int) -> None:
        self.layout = (rows, cols)
    
    def create_plot(self, data: Dict[str, Any], save_path: Optional[str] = None, 
                   show: bool = True) -> plt.Figure:
        if self.layout is None:
            n_widgets = len(self.widgets)
            cols = min(3, n_widgets)
            rows = (n_widgets + cols - 1) // cols
            self.layout = (rows, cols)
        
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(self.layout[0], self.layout[1], figure=fig)
        
        for i, (widget, position) in enumerate(self.widgets):
            if position is not None:
                row, col = position
                ax = fig.add_subplot(gs[row, col])
            else:
                row = i // self.layout[1]
                col = i % self.layout[1]
                ax = fig.add_subplot(gs[row, col])
            
            try:
                widget.render(ax, data)
            except Exception as e:
                print(f"Error rendering widget {widget.__class__.__name__}: {e}")
                ax.text(0.5, 0.5, f"Error: {str(e)}", 
                       transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def create_dashboard(self, sa_instance, coordinates: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
        self.widgets = []
        
        self.set_layout(2, 2)
        self.add_widget(CostEvolutionWidget())
        self.add_widget(TemperatureWidget())
        self.add_widget(AcceptanceRateWidget())
        # self.add_widget(StatisticsWidget())
        
        if coordinates is not None:
            self.add_widget(SolutionPathWidget())
        
        data = {
            'cost_history': getattr(sa_instance, 'cost_history', []),
            'best_cost_history': getattr(sa_instance, 'best_cost_history', []),
            'temperature_history': getattr(sa_instance, 'temperature_history', []),
            'acceptance_history': getattr(sa_instance, 'acceptance_history', []),
            'coordinates': coordinates,
            'solution': getattr(sa_instance, 'best_solution', None),
            'cost': getattr(sa_instance, 'best_cost', 0),
            'statistics': sa_instance.get_statistics() if hasattr(sa_instance, 'get_statistics') else {}
        }
        
        return self.create_plot(data, save_path)
    
    def create_comparison_plot(self, comparison_results: Dict[str, Any], 
                             save_path: Optional[str] = None) -> plt.Figure:
        self.widgets = []
        
        self.add_widget(ComparisonWidget('best_cost_history', 'Cost Evolution Comparison'))
        self.add_widget(ComparisonWidget('temperature_history', 'Temperature Comparison'))
        self.add_widget(ComparisonWidget('acceptance_history', 'Acceptance Rate Comparison'))
        
        data = {
            'comparison_results': comparison_results
        }
        
        return self.create_plot(data, save_path) 
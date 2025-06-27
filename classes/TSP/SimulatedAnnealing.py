import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Tuple, List, Optional
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

class SimulatedAnnealing:
    def __init__(self, track_history: bool = True, save_frequency: int = 100, 
                 neighborhood_type: str = "2-opt", initialization: str = "nearest_neighbor"):
        self.current_solution = None
        self.current_cost = float('inf')
        self.cost_matrix = None
        self.T_function = None
        self.T = 0
        self.k = 0
        
        self.best_solution = None
        self.best_cost = float('inf')
        
        self.neighborhood_type = neighborhood_type
        self.initialization = initialization

        self.track_history = track_history
        self.save_frequency = save_frequency
        
        if self.track_history:
            self.cost_history = []
            self.best_cost_history = []
            self.temperature_history = []
            self.acceptance_history = []
            self.best_solution_snapshots = {}
            self.iteration_snapshots = []
    
    def run(self, cost_matrix: np.ndarray, n: int, T_function: Callable[[int], float], 
            coordinates: Optional[np.ndarray] = None) -> np.ndarray:

        self.cost_matrix = cost_matrix
        self.T_function = T_function
        self.coordinates = coordinates
        self.total_iterations = n
        
        self.initialize()
        
        print(f"Starting Enhanced Simulated Annealing with {n} iterations...")
        print(f"Neighborhood: {self.neighborhood_type}, Initialization: {self.initialization}")
        print(f"Initial cost: {self.current_cost:.2f}")
        
        for i in range(n):
            self.step()
            
            if (i + 1) % (n // 10) == 0:
                progress = (i + 1) / n * 100
                improvement = ((self.cost_history[0] - self.best_cost) / self.cost_history[0] * 100)
                print(f"Progress: {progress:.1f}% - Current: {self.current_cost:.2f}, Best: {self.best_cost:.2f} ({improvement:.1f}% improvement), T: {self.T:.4f}")
        
        print(f"Optimization complete! Final best cost: {self.best_cost:.2f}")
        return self.best_solution.copy()
    
    def initialize(self):
        self.k = 0
        n_cities = self.cost_matrix.shape[0]
        
        if self.initialization == "nearest_neighbor":
            self.current_solution = self.nearest_neighbor_heuristic()
        elif self.initialization == "greedy":
            self.current_solution = self.greedy_heuristic()
        else:
            self.current_solution = np.random.permutation(n_cities)
            
        self.current_cost = self.evaluate_cost(self.current_solution)
        
        self.best_solution = self.current_solution.copy()
        self.best_cost = self.current_cost
        
        if self.track_history:
            self.cost_history = [self.current_cost]
            self.best_cost_history = [self.best_cost]
            self.temperature_history = [self.T_function(0)]
            self.acceptance_history = []
            self.best_solution_snapshots = {0: self.best_solution.copy()}
            self.iteration_snapshots = [0]
    
    def nearest_neighbor_heuristic(self) -> np.ndarray:
        n_cities = self.cost_matrix.shape[0]
        unvisited = set(range(1, n_cities))
        current = 0
        tour = [current]
        
        while unvisited:
            nearest = min(unvisited, key=lambda city: self.cost_matrix[current][city])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        return np.array(tour)
    
    def greedy_heuristic(self) -> np.ndarray:
        n_cities = self.cost_matrix.shape[0]
        edges = []
        
        for i in range(n_cities):
            for j in range(i + 1, n_cities):
                edges.append((self.cost_matrix[i][j], i, j))
        
        edges.sort()
        
        degree = [0] * n_cities
        tour_edges = []
        
        for cost, i, j in edges:
            if degree[i] < 2 and degree[j] < 2:
                if len(tour_edges) < n_cities - 1 or (degree[i] == 1 and degree[j] == 1):
                    tour_edges.append((i, j))
                    degree[i] += 1
                    degree[j] += 1
                    
                    if len(tour_edges) == n_cities:
                        break
        
        return self.nearest_neighbor_heuristic()
    
    def step(self):
        self.k += 1
        self.T = self.T_function(self.k)
        
        if self.neighborhood_type == "2-opt":
            neighbor = self.two_opt_neighbor()
        elif self.neighborhood_type == "3-opt":
            neighbor = self.three_opt_neighbor()
        elif self.neighborhood_type == "or-opt":
            neighbor = self.or_opt_neighbor()
        else:
            neighbor, _, _ = self.generate_neighbor()
        
        neighbor_cost = self.evaluate_cost(neighbor)
        
        delta_cost = neighbor_cost - self.current_cost
        
        accepted = False
        if delta_cost < 0:
            accepted = True
        else:
            if self.T > 0:
                acceptance_prob = np.exp(-delta_cost / self.T)
                accepted = np.random.rand() < acceptance_prob
        
        if accepted:
            self.current_solution = neighbor
            self.current_cost = neighbor_cost
            
            if neighbor_cost < self.best_cost:
                self.best_solution = neighbor.copy()
                self.best_cost = neighbor_cost
        
        if self.track_history:
            self.cost_history.append(self.current_cost)
            self.best_cost_history.append(self.best_cost)
            self.temperature_history.append(self.T)
            self.acceptance_history.append(accepted)
            
            if self.k % self.save_frequency == 0 or self.current_cost == self.best_cost:
                self.best_solution_snapshots[self.k] = self.best_solution.copy()
                self.iteration_snapshots.append(self.k)
    
    def two_opt_neighbor(self) -> np.ndarray:
        neighbor = self.current_solution.copy()
        n = len(neighbor)
        
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        
        if i > j:
            i, j = j, i
        if j - i < 2:
            j = (i + 2) % n
            if j < i:
                i, j = j, i
        
        neighbor[i+1:j+1] = neighbor[i+1:j+1][::-1]
        
        return neighbor
    
    def three_opt_neighbor(self) -> np.ndarray:
        neighbor = self.current_solution.copy()
        n = len(neighbor)
        
        points = sorted(np.random.choice(n, 3, replace=False))
        i, j, k = points
        
        reconnection_type = np.random.randint(0, 4)
        
        if reconnection_type == 0:
            neighbor[i:j] = neighbor[i:j][::-1]
        elif reconnection_type == 1:
            neighbor[j:k] = neighbor[j:k][::-1]
        elif reconnection_type == 2:
            neighbor[i:j] = neighbor[i:j][::-1]
            neighbor[j:k] = neighbor[j:k][::-1]
        else:
            segment1 = neighbor[i:j]
            segment2 = neighbor[j:k]
            neighbor[i:k] = np.concatenate([segment2, segment1])
        
        return neighbor
    
    def or_opt_neighbor(self) -> np.ndarray:
        neighbor = self.current_solution.copy()
        n = len(neighbor)
        
        seq_length = np.random.choice([1, 2, 3])
        seq_length = min(seq_length, n - 1)
        
        seq_start = np.random.randint(0, n - seq_length + 1)
        
        valid_positions = list(range(0, seq_start)) + list(range(seq_start + seq_length, n + 1))
        if not valid_positions:
            return neighbor
            
        insert_pos = np.random.choice(valid_positions)
        
        sequence = neighbor[seq_start:seq_start + seq_length]
        
        remaining = np.concatenate([neighbor[:seq_start], neighbor[seq_start + seq_length:]])
        
        if insert_pos <= seq_start:
            new_neighbor = np.concatenate([remaining[:insert_pos], sequence, remaining[insert_pos:]])
        else:
            adj_insert_pos = insert_pos - seq_length
            new_neighbor = np.concatenate([remaining[:adj_insert_pos], sequence, remaining[adj_insert_pos:]])
        
        return new_neighbor
    
    def generate_neighbor(self) -> Tuple[np.ndarray, int, int]:
        neighbor = self.current_solution.copy()
        
        n1, n2 = np.random.choice(len(neighbor), 2, replace=False)
        
        neighbor[n1], neighbor[n2] = neighbor[n2], neighbor[n1]
        
        return neighbor, n1, n2
    
    def evaluate_cost(self, solution: np.ndarray) -> float:
        cost = 0
        for i in range(len(solution)):
            cost += self.cost_matrix[solution[i], solution[(i + 1) % len(solution)]]
        return cost
    
    def plot_optimization_progress(self, figsize: Tuple[int, int] = (15, 12), 
                                 save_path: Optional[str] = None):
        if not self.track_history:
            print("History tracking is disabled. Enable it to plot progress.")
            return
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :2])
        iterations = range(len(self.cost_history))
        
        ax1.plot(iterations, self.cost_history, 'lightblue', alpha=0.7, linewidth=1, label='Current Cost')
        ax1.plot(iterations, self.best_cost_history, 'darkred', linewidth=2, label='Best Cost')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.set_title('Cost Evolution During Optimization', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        improvements = []
        for i in range(1, len(self.best_cost_history)):
            if self.best_cost_history[i] < self.best_cost_history[i-1]:
                improvements.append(i)
        
        if improvements:
            ax1.scatter(improvements, [self.best_cost_history[i] for i in improvements], 
                       color='gold', s=50, marker='*', zorder=5, label='Improvements')
            ax1.legend()
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(iterations, self.temperature_history, 'orange', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Temperature')
        ax2.set_title('Temperature Schedule', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        ax3 = fig.add_subplot(gs[1, :2])
        window_size = max(100, len(self.acceptance_history) // 50)
        acceptance_rate = self._rolling_average(self.acceptance_history, window_size)
        ax3.plot(range(len(acceptance_rate)), acceptance_rate, 'green', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Acceptance Rate')
        ax3.set_title(f'Acceptance Rate (Rolling Average, Window: {window_size})', 
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.hist(self.cost_history, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(self.best_cost, color='red', linestyle='--', linewidth=2, label=f'Best: {self.best_cost:.2f}')
        ax4.set_xlabel('Cost')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Cost Distribution', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        if self.coordinates is not None:
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_solution_path(ax5, self.best_solution, self.coordinates, 
                                   title=f'Best Solution Path (Cost: {self.best_cost:.2f}) - {self.neighborhood_type}')
        else:
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_algorithm_stats(ax5)
        
        plt.suptitle(f'Enhanced Simulated Annealing Results ({self.neighborhood_type})', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_solution_evolution(self, num_snapshots: int = 6, figsize: Tuple[int, int] = (18, 12),
                              save_path: Optional[str] = None):
        if self.coordinates is None:
            print("Coordinates not provided. Cannot plot solution evolution.")
            return
        
        if not self.track_history or not self.iteration_snapshots:
            print("No solution snapshots available.")
            return
        
        snapshot_indices = np.linspace(0, len(self.iteration_snapshots)-1, 
                                     min(num_snapshots, len(self.iteration_snapshots)), 
                                     dtype=int)
        
        cols = 3
        rows = (len(snapshot_indices) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, snap_idx in enumerate(snapshot_indices):
            iteration = self.iteration_snapshots[snap_idx]
            solution = self.best_solution_snapshots[iteration]
            cost = self.evaluate_cost(solution)
            
            ax = axes[i]
            self._plot_solution_path(ax, solution, self.coordinates, 
                                   title=f'Iteration {iteration}\nCost: {cost:.2f}')
        
        for i in range(len(snapshot_indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Solution Evolution Over Time ({self.neighborhood_type})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evolution plot saved to {save_path}")
        
        plt.show()
    
    def _plot_solution_path(self, ax, solution: np.ndarray, coordinates: np.ndarray, 
                          title: str = "Solution Path"):
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c='red', s=100, zorder=5)
        
        for i in range(len(solution)):
            start_city = solution[i]
            end_city = solution[(i + 1) % len(solution)]
            
            start_pos = coordinates[start_city]
            end_pos = coordinates[end_city]
            
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   'b-', alpha=0.7, linewidth=2)
        
        for i, (x, y) in enumerate(coordinates):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left')
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_algorithm_stats(self, ax):
        stats_text = f"""
        Algorithm Statistics:
        
        Total Iterations: {len(self.cost_history):,}
        Initial Cost: {self.cost_history[0]:.2f}
        Final Best Cost: {self.best_cost:.2f}
        Improvement: {((self.cost_history[0] - self.best_cost) / self.cost_history[0] * 100):.1f}%
        
        Neighborhood: {self.neighborhood_type}
        Initialization: {self.initialization}
        
        Total Accepted Moves: {sum(self.acceptance_history):,}
        Overall Acceptance Rate: {(sum(self.acceptance_history) / len(self.acceptance_history) * 100):.1f}%
        
        Final Temperature: {self.temperature_history[-1]:.6f}
        """
        
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Algorithm Summary', fontweight='bold')
    
    def _rolling_average(self, data: List[float], window_size: int) -> List[float]:
        if len(data) < window_size:
            return [np.mean(data[:i+1]) for i in range(len(data))]
        
        rolling_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            rolling_avg.append(np.mean(data[start_idx:i+1]))
        return rolling_avg
    
    def get_statistics(self) -> dict:
        if not self.track_history:
            return {"error": "History tracking is disabled"}
        
        return {
            "initial_cost": self.cost_history[0],
            "final_best_cost": self.best_cost,
            "improvement_percentage": ((self.cost_history[0] - self.best_cost) / self.cost_history[0] * 100),
            "total_iterations": len(self.cost_history),
            "total_accepted_moves": sum(self.acceptance_history),
            "overall_acceptance_rate": (sum(self.acceptance_history) / len(self.acceptance_history) * 100),
            "final_temperature": self.temperature_history[-1],
            "best_solution": self.best_solution.tolist(),
            "convergence_iteration": next((i for i, cost in enumerate(self.best_cost_history) 
                                         if cost == self.best_cost), len(self.best_cost_history)-1),
            "neighborhood_type": self.neighborhood_type,
            "initialization": self.initialization
        }
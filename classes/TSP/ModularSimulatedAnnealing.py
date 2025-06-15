"""
Modular Simulated Annealing implementation that uses component-based architecture.
This class composes different strategies for initialization, temperature scheduling,
and proposal generation.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from .InitializationStrategies import InitializationStrategy, RandomInitialization
from .TemperatureSchedules import TemperatureSchedule, ExponentialCooling
from .ProposalGenerators import ProposalGenerator, TwoOptProposal


class ModularSimulatedAnnealing:
    """
    Modular Simulated Annealing implementation using strategy pattern
    """
    
    def __init__(self, 
                 initialization_strategy: InitializationStrategy = None,
                 temperature_schedule: TemperatureSchedule = None,
                 proposal_generator: ProposalGenerator = None,
                 track_history: bool = True,
                 save_frequency: int = 100,
                 name: str = "SA"):
        
        # Components (use defaults if not provided)
        self.initialization_strategy = initialization_strategy or RandomInitialization()
        self.temperature_schedule = temperature_schedule or ExponentialCooling()
        self.proposal_generator = proposal_generator or TwoOptProposal()
        
        # Configuration
        self.track_history = track_history
        self.save_frequency = save_frequency
        self.name = name
        
        # State variables
        self.current_solution = None
        self.current_cost = float('inf')
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_matrix = None
        self.coordinates = None
        self.iteration = 0
        self.total_iterations = 0
        
        # History tracking
        if self.track_history:
            self.cost_history = []
            self.best_cost_history = []
            self.temperature_history = []
            self.acceptance_history = []
            self.best_solution_snapshots = {}
            self.iteration_snapshots = []
    
    def run(self, cost_matrix: np.ndarray, n_iterations: int, 
            coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run the simulated annealing algorithm
        
        Args:
            cost_matrix: Distance/cost matrix between cities
            n_iterations: Number of iterations to run
            coordinates: Optional city coordinates for visualization
            
        Returns:
            Best solution found
        """
        
        self.cost_matrix = cost_matrix
        self.coordinates = coordinates
        self.total_iterations = n_iterations
        
        # Initialize
        self._initialize()
        
        print(f"Starting {self.name} with {n_iterations:,} iterations...")
        print(f"Components: {self.initialization_strategy.__class__.__name__}, "
              f"{self.temperature_schedule.__class__.__name__}, "
              f"{self.proposal_generator.__class__.__name__}")
        print(f"Initial cost: {self.current_cost:.2f}")
        
        # Main optimization loop
        for self.iteration in range(1, n_iterations + 1):
            self._step()
            
            # Progress reporting
            if self.iteration % (n_iterations // 10) == 0:
                progress = self.iteration / n_iterations * 100
                improvement = ((self.cost_history[0] - self.best_cost) / self.cost_history[0] * 100)
                current_temp = self.temperature_schedule.get_temperature(self.iteration)
                print(f"Progress: {progress:.1f}% - Current: {self.current_cost:.2f}, "
                      f"Best: {self.best_cost:.2f} ({improvement:.1f}% improvement), "
                      f"T: {current_temp:.4f}")
        
        print(f"Optimization complete! Final best cost: {self.best_cost:.2f}")
        return self.best_solution.copy()
    
    def _initialize(self):
        """Initialize the algorithm state"""
        self.iteration = 0
        
        # Generate initial solution using the initialization strategy
        self.current_solution = self.initialization_strategy.initialize(self.cost_matrix)
        self.current_cost = self._evaluate_cost(self.current_solution)
        
        # Initialize best solution
        self.best_solution = self.current_solution.copy()
        self.best_cost = self.current_cost
        
        # Initialize history tracking
        if self.track_history:
            self.cost_history = [self.current_cost]
            self.best_cost_history = [self.best_cost]
            self.temperature_history = [self.temperature_schedule.get_temperature(0)]
            self.acceptance_history = []
            self.best_solution_snapshots = {0: self.best_solution.copy()}
            self.iteration_snapshots = [0]
    
    def _step(self):
        """Perform one iteration of the algorithm"""
        
        # Get current temperature
        temperature = self.temperature_schedule.get_temperature(self.iteration)
        
        # Generate neighbor solution
        neighbor = self.proposal_generator.generate_neighbor(self.current_solution)
        neighbor_cost = self._evaluate_cost(neighbor)
        
        # Calculate cost difference
        delta_cost = neighbor_cost - self.current_cost
        
        # Acceptance decision
        accepted = False
        if delta_cost < 0:
            # Accept improvement
            accepted = True
        else:
            # Accept with probability based on temperature
            if temperature > 0:
                acceptance_prob = np.exp(-delta_cost / temperature)
                accepted = np.random.rand() < acceptance_prob
        
        # Update current solution
        if accepted:
            self.current_solution = neighbor
            self.current_cost = neighbor_cost
            
            # Update best solution if necessary
            if neighbor_cost < self.best_cost:
                self.best_solution = neighbor.copy()
                self.best_cost = neighbor_cost
        
        # Update adaptive components
        if hasattr(self.temperature_schedule, 'update_temperature'):
            self.temperature_schedule.update_temperature(accepted)
        
        if hasattr(self.proposal_generator, 'update_success'):
            self.proposal_generator.update_success(accepted)
        
        # Track history
        if self.track_history:
            self.cost_history.append(self.current_cost)
            self.best_cost_history.append(self.best_cost)
            self.temperature_history.append(temperature)
            self.acceptance_history.append(accepted)
            
            # Save solution snapshots
            if self.iteration % self.save_frequency == 0:
                self.best_solution_snapshots[self.iteration] = self.best_solution.copy()
                self.iteration_snapshots.append(self.iteration)
    
    def _evaluate_cost(self, solution: np.ndarray) -> float:
        """Evaluate the cost of a solution"""
        cost = 0.0
        for i in range(len(solution)):
            from_city = solution[i]
            to_city = solution[(i + 1) % len(solution)]
            cost += self.cost_matrix[from_city][to_city]
        return cost
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the algorithm run"""
        if not self.track_history or not self.cost_history:
            return {}
        
        initial_cost = self.cost_history[0]
        final_best_cost = self.best_cost
        improvement = (initial_cost - final_best_cost) / initial_cost * 100
        
        # Find convergence iteration (when best cost stopped improving significantly)
        convergence_iter = len(self.best_cost_history) - 1
        threshold = final_best_cost * 1.001  # 0.1% threshold
        
        for i in reversed(range(len(self.best_cost_history))):
            if self.best_cost_history[i] > threshold:
                convergence_iter = i
                break
        
        # Calculate acceptance rate
        if self.acceptance_history:
            overall_acceptance_rate = sum(self.acceptance_history) / len(self.acceptance_history) * 100
        else:
            overall_acceptance_rate = 0
        
        return {
            'name': self.name,
            'initial_cost': initial_cost,
            'final_best_cost': final_best_cost,
            'improvement_percentage': improvement,
            'total_iterations': self.total_iterations,
            'convergence_iteration': convergence_iter,
            'overall_acceptance_rate': overall_acceptance_rate,
            'final_temperature': self.temperature_history[-1] if self.temperature_history else 0,
            'components': {
                'initialization': self.initialization_strategy.__class__.__name__,
                'temperature': self.temperature_schedule.__class__.__name__,
                'proposal': self.proposal_generator.__class__.__name__
            }
        }
    
    def get_component_info(self) -> Dict[str, str]:
        """Get information about the components used"""
        return {
            'initialization_strategy': self.initialization_strategy.__class__.__name__,
            'temperature_schedule': self.temperature_schedule.__class__.__name__,
            'proposal_generator': self.proposal_generator.__class__.__name__
        }
    
    def clone_with_different_components(self, **kwargs) -> 'ModularSimulatedAnnealing':
        """Create a clone with different components"""
        return ModularSimulatedAnnealing(
            initialization_strategy=kwargs.get('initialization_strategy', self.initialization_strategy),
            temperature_schedule=kwargs.get('temperature_schedule', self.temperature_schedule),
            proposal_generator=kwargs.get('proposal_generator', self.proposal_generator),
            track_history=kwargs.get('track_history', self.track_history),
            save_frequency=kwargs.get('save_frequency', self.save_frequency),
            name=kwargs.get('name', self.name)
        )


class SAExperimentRunner:
    """
    Class to manage and run multiple SA experiments for comparison
    """
    
    def __init__(self):
        self.experiments = []
        self.results = {}
    
    def add_experiment(self, sa_instance: ModularSimulatedAnnealing, 
                      experiment_name: Optional[str] = None) -> None:
        """Add a SA experiment to the runner"""
        name = experiment_name or f"Experiment_{len(self.experiments) + 1}"
        self.experiments.append((name, sa_instance))
    
    def run_all_experiments(self, cost_matrix: np.ndarray, n_iterations: int,
                           coordinates: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run all experiments and collect results"""
        
        print(f"\n{'='*80}")
        print(f"RUNNING {len(self.experiments)} SIMULATED ANNEALING EXPERIMENTS")
        print(f"{'='*80}")
        
        for name, sa_instance in self.experiments:
            print(f"\n🔧 Running experiment: {name}")
            
            # Run the experiment
            best_solution = sa_instance.run(cost_matrix, n_iterations, coordinates)
            
            # Store results
            self.results[name] = {
                'sa_instance': sa_instance,
                'best_solution': best_solution,
                'stats': sa_instance.get_statistics()
            }
            
            print(f"   ✅ Completed: {name} - Final Cost: {sa_instance.best_cost:.2f}")
        
        # Print summary
        self._print_results_summary()
        
        return self.results
    
    def _print_results_summary(self):
        """Print a summary table of all results"""
        print(f"\n{'='*100}")
        print(f"EXPERIMENT RESULTS SUMMARY")
        print(f"{'='*100}")
        
        headers = ['Experiment', 'Init Strategy', 'Temp Schedule', 'Proposal Gen', 
                  'Final Cost', 'Improvement %', 'Accept Rate %']
        
        # Print headers
        print(f"{headers[0]:<20} {headers[1]:<15} {headers[2]:<15} {headers[3]:<12} "
              f"{headers[4]:<12} {headers[5]:<12} {headers[6]:<12}")
        print("-" * 100)
        
        # Print results
        for name, result in self.results.items():
            stats = result['stats']
            components = stats['components']
            
            print(f"{name:<20} {components['initialization']:<15} "
                  f"{components['temperature']:<15} {components['proposal']:<12} "
                  f"{stats['final_best_cost']:<12.2f} {stats['improvement_percentage']:<12.1f} "
                  f"{stats['overall_acceptance_rate']:<12.1f}")
        
        # Find best result
        best_name = min(self.results.keys(), 
                       key=lambda x: self.results[x]['stats']['final_best_cost'])
        best_cost = self.results[best_name]['stats']['final_best_cost']
        
        print(f"\n🏆 Best Result: {best_name} with cost {best_cost:.2f}")
    
    def get_comparison_data(self) -> Dict[str, Any]:
        """Get data formatted for comparison plotting"""
        return self.results 
# Modular Simulated Annealing Framework

## Overview

This project implements a **modular, object-oriented Simulated Annealing framework** that moves away from exercise-driven scripts to a flexible, component-based architecture. The framework allows you to easily compose different algorithmic strategies and conduct systematic comparisons.

## Key Features

🧩 **Modular Architecture**: Separate classes for initialization strategies, temperature schedules, and proposal generators  
🔄 **Easy Component Swapping**: Change algorithmic components without modifying core logic  
📊 **Dynamic Plotting System**: Widget-based visualization that can be composed like a collage  
🧪 **Systematic Experiments**: Built-in experiment runner for comparative studies  
📈 **Comprehensive Tracking**: Detailed history and statistics for analysis  
🎯 **Object-Oriented Design**: Clean separation of concerns and extensible structure  

## Architecture

### Core Components

#### 1. Initialization Strategies (`classes/InitializationStrategies.py`)
Different methods to generate initial solutions:

- **RandomInitialization**: Random permutation
- **NearestNeighborInitialization**: Greedy nearest neighbor heuristic
- **GreedyInitialization**: Edge-based greedy construction
- **FarthestInsertionInitialization**: Farthest insertion heuristic

```python
from classes.InitializationStrategies import NearestNeighborInitialization

# Create an initialization strategy
init_strategy = NearestNeighborInitialization(start_city=0)
```

#### 2. Temperature Schedules (`classes/TemperatureSchedules.py`)
Various cooling strategies to control acceptance probability:

- **ExponentialCooling**: T = T₀ × α^k
- **LinearCooling**: Linear decrease
- **LogarithmicCooling**: T = T₀ / log(k + 2)
- **SqrtCooling**: T = T₀ / √(k + 1)
- **AdaptiveCooling**: Adjusts based on acceptance rate
- **ReheatSchedule**: Periodic reheating
- **HybridCooling**: Combines multiple strategies
- **GeometricCooling**: Step-wise geometric cooling

```python
from classes.TemperatureSchedules import ExponentialCooling

# Create a temperature schedule
temp_schedule = ExponentialCooling(initial_temp=100, cooling_rate=0.995)
```

#### 3. Proposal Generators (`classes/ProposalGenerators.py`)
Different neighborhood structures for solution exploration:

- **TwoSwapProposal**: Swap two random cities
- **TwoOptProposal**: Reverse a segment (2-opt move)
- **ThreeOptProposal**: More complex 3-opt rearrangements
- **OrOptProposal**: Relocate sequences of cities
- **LinKernighanProposal**: Simplified Lin-Kernighan moves
- **InsertionProposal**: Remove and reinsert cities
- **MixedProposal**: Randomly selects from multiple strategies
- **AdaptiveProposal**: Adapts selection based on success rates

```python
from classes.ProposalGenerators import TwoOptProposal, MixedProposal

# Create proposal generators
two_opt = TwoOptProposal()
mixed = MixedProposal([TwoOptProposal(), ThreeOptProposal()], weights=[0.7, 0.3])
```

#### 4. Modular Simulated Annealing (`classes/ModularSimulatedAnnealing.py`)
The main algorithm class that composes different strategies:

```python
from classes.ModularSimulatedAnnealing import ModularSimulatedAnnealing

# Create a configured SA instance
sa = ModularSimulatedAnnealing(
    initialization_strategy=NearestNeighborInitialization(),
    temperature_schedule=ExponentialCooling(initial_temp=100, cooling_rate=0.995),
    proposal_generator=TwoOptProposal(),
    name='My_SA_Config'
)

# Run the algorithm
best_solution = sa.run(cost_matrix, n_iterations=10000, coordinates=coordinates)
```

#### 5. Plot Widgets (`classes/PlotWidgets.py`)
Composable visualization components:

- **CostEvolutionWidget**: Cost evolution over iterations
- **TemperatureWidget**: Temperature schedule visualization
- **AcceptanceRateWidget**: Acceptance rate over time
- **SolutionPathWidget**: TSP path visualization
- **StatisticsWidget**: Algorithm statistics display
- **ComparisonWidget**: Multi-algorithm comparison
- **HeatmapWidget**: Matrix visualizations

```python
from classes.PlotWidgets import ComposablePlotter

# Create a dashboard
plotter = ComposablePlotter(figsize=(16, 12))
dashboard = plotter.create_dashboard(sa_instance, coordinates)

# Or compose custom visualizations
plotter.add_widget(CostEvolutionWidget())
plotter.add_widget(TemperatureWidget())
plotter.create_plot(data)
```

## Usage Examples

### 1. Single Experiment

```python
from classes.TravellingSalesman import TravellingSalesman
from classes.ModularSimulatedAnnealing import ModularSimulatedAnnealing
from classes.InitializationStrategies import NearestNeighborInitialization
from classes.TemperatureSchedules import ExponentialCooling
from classes.ProposalGenerators import TwoOptProposal
from classes.PlotWidgets import ComposablePlotter

# Create TSP problem
tsp = TravellingSalesman(n_stations=20)

# Configure SA
sa = ModularSimulatedAnnealing(
    initialization_strategy=NearestNeighborInitialization(),
    temperature_schedule=ExponentialCooling(initial_temp=100, cooling_rate=0.995),
    proposal_generator=TwoOptProposal(),
    name='My_Experiment'
)

# Run optimization
best_solution = sa.run(tsp.cost_matrix, n_iterations=10000, coordinates=tsp.coordinates)

# Create visualization dashboard
plotter = ComposablePlotter()
plotter.create_dashboard(sa, tsp.coordinates, save_path='my_experiment.png')
```

### 2. Comparative Study

```python
from classes.ModularSimulatedAnnealing import SAExperimentRunner

# Create experiment runner
runner = SAExperimentRunner()

# Add different configurations
runner.add_experiment(
    ModularSimulatedAnnealing(
        initialization_strategy=RandomInitialization(),
        temperature_schedule=ExponentialCooling(),
        proposal_generator=TwoOptProposal(),
        name='Config_A'
    ), 'Config_A'
)

runner.add_experiment(
    ModularSimulatedAnnealing(
        initialization_strategy=NearestNeighborInitialization(),
        temperature_schedule=SqrtCooling(),
        proposal_generator=ThreeOptProposal(),
        name='Config_B'
    ), 'Config_B'
)

# Run all experiments
results = runner.run_all_experiments(cost_matrix, n_iterations=15000, coordinates)

# Create comparison plot
plotter = ComposablePlotter()
plotter.create_comparison_plot(results, save_path='comparison.png')
```

### 3. Component Analysis

```python
# Study effect of different initialization strategies
base_temp = ExponentialCooling(initial_temp=100, cooling_rate=0.995)
base_proposal = TwoOptProposal()

init_strategies = [
    ('Random', RandomInitialization()),
    ('NN', NearestNeighborInitialization()),
    ('Greedy', GreedyInitialization())
]

runner = SAExperimentRunner()
for name, init_strategy in init_strategies:
    sa = ModularSimulatedAnnealing(
        initialization_strategy=init_strategy,
        temperature_schedule=base_temp,
        proposal_generator=base_proposal,
        name=f'Init_{name}'
    )
    runner.add_experiment(sa, f'Init_{name}')

results = runner.run_all_experiments(cost_matrix, n_iterations=10000)
```

## File Structure

```
├── classes/
│   ├── InitializationStrategies.py    # Initialization methods
│   ├── TemperatureSchedules.py        # Temperature cooling strategies
│   ├── ProposalGenerators.py          # Neighborhood generators
│   ├── ModularSimulatedAnnealing.py   # Main SA algorithm
│   ├── PlotWidgets.py                 # Modular plotting system
│   └── TravellingSalesman.py          # TSP problem class
├── examples/
│   └── modular_main.py                # Demonstration script
└── MODULAR_SA_README.md               # This file
```

## Running the Examples

```bash
# Run the comprehensive demonstration
cd examples
python modular_main.py
```

This will generate:
- Single experiment dashboard
- Comparative study results  
- Component effect analysis
- Multiple comparison plots

## Benefits of the Modular Architecture

### 🔧 **Easy Component Swapping**
```python
# Change just the temperature schedule
sa_v1 = sa.clone_with_different_components(
    temperature_schedule=SqrtCooling(initial_temp=100)
)

# Change multiple components
sa_v2 = sa.clone_with_different_components(
    initialization_strategy=GreedyInitialization(),
    proposal_generator=ThreeOptProposal()
)
```

### 📊 **Flexible Visualization**
```python
# Create custom plot layouts
plotter = ComposablePlotter()
plotter.add_widget(CostEvolutionWidget(), position=(0, 0))
plotter.add_widget(TemperatureWidget(), position=(0, 1))
plotter.add_widget(SolutionPathWidget(), position=(1, 0))
plotter.add_widget(StatisticsWidget(), position=(1, 1))
plotter.set_layout(2, 2)
plotter.create_plot(data)
```

### 🧪 **Systematic Experimentation**
```python
# Automatically run multiple configurations
configurations = create_sa_configurations()  # Generate many configs
runner = SAExperimentRunner()

for config in configurations:
    runner.add_experiment(config['sa'], config['name'])

results = runner.run_all_experiments(cost_matrix, n_iterations)
# Automatic summary table and statistics
```

### 📈 **Extensibility**
Adding new components is straightforward:

```python
class MyCustomCooling(TemperatureSchedule):
    def get_temperature(self, iteration: int) -> float:
        # Implement your cooling strategy
        return my_temperature_function(iteration)

# Use it immediately
sa = ModularSimulatedAnnealing(
    temperature_schedule=MyCustomCooling(),
    # ... other components
)
```

## Advanced Features

### Adaptive Components
Some components learn and adapt during optimization:

```python
# Adaptive temperature based on acceptance rate
adaptive_temp = AdaptiveCooling(target_acceptance_rate=0.4)

# Adaptive proposal selection based on success
adaptive_proposal = AdaptiveProposal([
    TwoOptProposal(),
    ThreeOptProposal(),
    OrOptProposal()
], learning_rate=0.1)
```

### Mixed Strategies
Combine multiple approaches:

```python
# Mixed proposal with custom weights
mixed_proposal = MixedProposal([
    TwoOptProposal(),
    ThreeOptProposal(),
    OrOptProposal()
], weights=[0.5, 0.3, 0.2])

# Hybrid cooling that changes strategy over time
hybrid_cooling = HybridCooling(initial_temp=100, max_iterations=10000)
```

### Statistical Analysis
Comprehensive statistics for each run:

```python
stats = sa.get_statistics()
print(f"Improvement: {stats['improvement_percentage']:.1f}%")
print(f"Convergence at iteration: {stats['convergence_iteration']}")
print(f"Components used: {stats['components']}")
```

## Migration from Old System

The old exercise-driven approach:
```python
# OLD WAY - Exercise specific functions
def run_single_experiment(n_stations=20, n_iterations=50000, temperature_schedule='sqrt_cooling'):
    # Hardcoded exercise logic
    # Fixed visualization
    # Limited flexibility
```

The new modular approach:
```python
# NEW WAY - Composable components
sa = ModularSimulatedAnnealing(
    initialization_strategy=choose_init_strategy(),
    temperature_schedule=choose_temp_schedule(), 
    proposal_generator=choose_proposal_generator()
)

plotter = ComposablePlotter()
plotter.add_widget(choose_widgets())
plotter.create_plot(data)
```

## Contributing

To add new components:

1. **New Initialization Strategy**: Inherit from `InitializationStrategy`
2. **New Temperature Schedule**: Inherit from `TemperatureSchedule`  
3. **New Proposal Generator**: Inherit from `ProposalGenerator`
4. **New Plot Widget**: Inherit from `PlotWidget`

Each component is self-contained and follows the strategy pattern, making the system highly extensible and maintainable.

---

This modular architecture provides a robust foundation for simulated annealing research and experimentation, offering both flexibility for researchers and ease of use for practical applications. 
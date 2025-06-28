import sys
import os
import pandas as pd
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
from classes.TSP.TravellingSalesman import TravellingSalesman
from classes.TSP.ModularSimulatedAnnealing import ModularSimulatedAnnealing, SAExperimentRunner
from classes.TSP.InitializationStrategies import (
    RandomInitialization, NearestNeighborInitialization, 
    GreedyInitialization, FarthestInsertionInitialization
)
from classes.TSP.TemperatureSchedules import (
    ExponentialCooling, LinearCooling, LogarithmicCooling, 
    SqrtCooling, FastCooling, AdaptiveCooling, ReheatSchedule, 
    HybridCooling, GeometricCooling, ConstantThermodynamicSpeedV2
)
from classes.TSP.ProposalGenerators import (
    TwoSwapProposal, TwoOptProposal, ThreeOptProposal, 
    OrOptProposal, LinKernighanProposal, InsertionProposal, 
    MixedProposal, AdaptiveProposal
)
from classes.TSP.PlotWidgets import ComposablePlotter
from classes.TSP.TemperatureSchedules import OptimalAnnealing
from classes.Plotter import Plotter

def exercise_1():
    n_cities = 20
    tsp = TravellingSalesman(n_stations=n_cities)

    configuration = ModularSimulatedAnnealing(
        initialization_strategy=RandomInitialization(seed=42),
        temperature_schedule=SqrtCooling(initial_temp=1),
        proposal_generator=TwoSwapProposal(),
        name='Exercise 1'
    )

    configuration.run(tsp.cost_matrix, n_iterations=10000, coordinates=tsp.coordinates)

    plotter = ComposablePlotter(figsize=(16, 12))
    plotter.create_dashboard(
        sa_instance=configuration,
        coordinates=tsp.coordinates,
        save_path='plots/exercise_1_dashboard.png'
    )
    
def exercise_2():
    cost_matrix = pd.read_csv('exercises/L6. SimAnn & Bootstrap/Simulated Annealing/cost.csv', header=None).to_numpy()

    tsp = TravellingSalesman(n_stations=20, cost_matrix=cost_matrix)

    configuration = ModularSimulatedAnnealing(
        initialization_strategy=RandomInitialization(),
        temperature_schedule=SqrtCooling(initial_temp=100),
        proposal_generator=TwoSwapProposal(),
        name='Exercise 2'
    )

    costs = []
    for _ in range(100):
        _, cost = configuration.run(tsp.cost_matrix, n_iterations=10000, coordinates=tsp.coordinates, verbose = False)
        costs.append(cost)

    plotter = Plotter()
    plotter.plot_histogram(costs, bins = 25, title = 'Histogram of costs', x_label = 'Cost', y_label = 'Frequency')

    # plotter = ComposablePlotter(figsize=(16, 12))
    # plotter.create_dashboard(
    #     sa_instance=configuration,
    #     coordinates=tsp.coordinates,
    #     save_path='plots/exercise_2_dashboard.png'
    # )

def exercise_3():
    n_cities = 100
    n_iterations = 50000
    tsp = TravellingSalesman(n_stations=n_cities)

    configuration = ModularSimulatedAnnealing(
        initialization_strategy=GreedyInitialization(),
        temperature_schedule=ConstantThermodynamicSpeedV2(
            initial_temp=100.0,
            final_temp=0.0001,
            max_iterations=n_iterations,
            system_size=n_cities
        ),
        proposal_generator=MixedProposal([TwoOptProposal(), TwoSwapProposal(), ThreeOptProposal()], [0.3, 0.3, 0.4]),
        name='Exercise 3'
    )

    configuration.run(tsp.cost_matrix, n_iterations=n_iterations, coordinates=tsp.coordinates)

    plotter = ComposablePlotter(figsize=(16, 12))
    plotter.create_dashboard(
        sa_instance=configuration,
        coordinates=tsp.coordinates,
        save_path='plots/exercise_3_dashboard.png'
    )

def exercise_4():
    n_cities = 20
    n_iterations = 10000
    coordinates = np.array([[50 * math.cos(i * 2 * math.pi / n_cities), 50 * math.sin(i * 2 * math.pi / n_cities)] for i in range(n_cities)])
    tsp = TravellingSalesman(coordinates=coordinates)

    configuration = ModularSimulatedAnnealing(
        initialization_strategy=RandomInitialization(),
        temperature_schedule=SqrtCooling(initial_temp=10),
        proposal_generator=TwoSwapProposal(),
        name='Exercise 4'
    )

    configuration.run(tsp.cost_matrix, n_iterations=n_iterations, coordinates=tsp.coordinates)

    plotter = ComposablePlotter(figsize=(16, 12))
    plotter.create_dashboard(
        sa_instance=configuration,
        coordinates=tsp.coordinates,
        save_path='plots/exercise_4_dashboard.png'
    )

def exercise_2_check():
    """
    Solves the TSP for the asymmetric cost matrix from exercise 2
    using Google's OR-Tools to find the optimal solution.
    """
    try:
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
    except ImportError:
        print("Please install Google OR-Tools by running: pip install ortools")
        return

    cost_matrix = pd.read_csv('exercises/L6. SimAnn & Bootstrap/Simulated Annealing/cost.csv', header=None).to_numpy()
    
    def create_data_model():
        """Stores the data for the problem."""
        data = {}
        data['cost_matrix'] = cost_matrix
        data['num_vehicles'] = 1
        data['depot'] = 0
        return data

    data = create_data_model()
    
    manager = pywrapcp.RoutingIndexManager(len(data['cost_matrix']),
                                           data['num_vehicles'], data['depot'])
    
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['cost_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        print(f"Optimal cost found with OR-Tools: {solution.ObjectiveValue()}")
    else:
        print("No solution found for the TSP problem with OR-Tools.")

def main():
    os.makedirs('plots', exist_ok=True)
    
    # exercise_1()
    exercise_2()
    # exercise_2_check()
    # exercise_3()
    # exercise_4()
    

if __name__ == "__main__":
    main() 
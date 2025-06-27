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
    cost_matrix = pd.read_csv('L6. SimAnn & Bootstrap/Simulated Annealing/cost.csv', header=None).to_numpy()

    tsp = TravellingSalesman(n_stations=20, cost_matrix=cost_matrix)

    configuration = ModularSimulatedAnnealing(
        initialization_strategy=RandomInitialization(seed=42),
        temperature_schedule=SqrtCooling(initial_temp=1),
        proposal_generator=TwoSwapProposal(),
        name='Exercise 2'
    )

    configuration.run(tsp.cost_matrix, n_iterations=10000, coordinates=tsp.coordinates)

    plotter = ComposablePlotter(figsize=(16, 12))
    plotter.create_dashboard(
        sa_instance=configuration,
        coordinates=tsp.coordinates,
        save_path='plots/exercise_2_dashboard.png'
    )

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
        initialization_strategy=RandomInitialization(seed=42),
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


def main():
    os.makedirs('plots', exist_ok=True)
    
    # exercise_1()
    # exercise_2()
    # exercise_3()
    exercise_4()
    

if __name__ == "__main__":
    main() 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from classes.DESP.System import System
from classes.CRVG import HyperExponential, Exponential, Erlang, Constant, Pareto
import random
import numpy as np

from classes.Plotter import Plotter 

def main():
    num_service_units = 10
    mean_service_time = 8
    mean_interarrival_time = 1

    ## DISTRIBUTION FOR SERVICE TIME

    service_time_distribution = Exponential(lambda_ = 1 / mean_service_time)
    # service_time_distribution = Pareto(k = 1.05, mean = mean_service_time)
    # service_time_distribution = Pareto(k = 2.05, mean = mean_service_time)
    # service_time_distribution = Constant(value = mean_service_time)
    # service_time_distribution = HyperExponential(lambdas = [0.2, 0.05], weights = [0.8, 0.2])


    ## DISTRIBUTION FOR INTERARRIVAL TIME

    interarrival_time_distribution = Exponential(lambda_ = 1 / mean_interarrival_time)
    # interarrival_time_distribution = Erlang(k = 2, lambda_ = 1)
    # interarrival_time_distribution = HyperExponential(lambdas = [0.8333, 5], weights = [0.8, 0.2])
    # interarrival_time_distribution = Constant(value = mean_interarrival_time)

    system = System(
        name="System",
        parameters={
            'n_service_units': num_service_units,
            'service_time_distribution': service_time_distribution,
            'interarrival_time_distribution': interarrival_time_distribution,
            'n_customers': 10000,
            'n_runs': 10,
        }
    )
    system.simulate(verbose = True)

    # variance_of_difference(
    #     same_seed = True, 
    #     num_service_units = num_service_units, 
    #     service_time_distribution = service_time_distribution, 
    #     interarrival_time_distribution1 = Exponential(lambda_ = 1 / mean_interarrival_time),
    #     interarrival_time_distribution2 = HyperExponential(lambdas = [0.8333, 5], weights = [0.8, 0.2]),
    #     n_customers = 10000,
    #     n_runs = 10
    # )

    # variance_of_difference(
    #     same_seed = False, 
    #     num_service_units = num_service_units, 
    #     service_time_distribution = service_time_distribution, 
    #     interarrival_time_distribution1 = Exponential(lambda_ = 1 / mean_interarrival_time),
    #     interarrival_time_distribution2 = HyperExponential(lambdas = [0.8333, 5], weights = [0.8, 0.2]),
    #     n_customers = 10000,
    #     n_runs = 10
    # )


def variance_of_difference(same_seed : bool = False, num_service_units = 10, service_time_distribution = None, interarrival_time_distribution1 = None, interarrival_time_distribution2 = None, n_customers = 10000, n_runs = 10):
    system1 = System(
        name="System",
        parameters={
            'n_service_units': num_service_units,
            'service_time_distribution': service_time_distribution,
            'interarrival_time_distribution': interarrival_time_distribution1,
            'n_customers': n_customers,
            'n_runs': n_runs,
        }
    )

    system2 = System(
        name="System2",
        parameters={
            'n_service_units': num_service_units,
            'service_time_distribution': service_time_distribution,
            'interarrival_time_distribution': interarrival_time_distribution2,
            'n_customers': n_customers,
            'n_runs': n_runs,
        }
    )

    for _ in range(n_runs):
        np.random.seed(random.randint(0, 1000000))
        U11, U21, U31 = np.random.uniform(0, 1, 3 * n_customers).reshape(3, n_customers)

        system1.initialize(U11, U21, U31)
        system1.run()

        if same_seed is False:
            np.random.seed(random.randint(0, 1000000))
            U12, U22, U32 = np.random.uniform(0, 1, 3 * n_customers).reshape(3, n_customers)
        else:
            U12 = U11
            U22 = U21
            U32 = U31
        
        system2.initialize(U12, U22, U32)
        system2.run()

    print(np.var([(system1.blocked_customers[i] / system1.parameters['n_customers'] - system2.blocked_customers[i] / system2.parameters['n_customers']) for i in range(10)]))


if __name__ == "__main__":
    main()
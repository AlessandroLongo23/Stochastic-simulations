import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from classes.DESP.System import System
from classes.CRVG import Erlang, Poisson, HyperExponential, Constant, Exponential, Pareto

def main():
    mean_service_time = 8
    mean_interarrival_time = 1

    ## DISTRIBUTION FOR SERVICE TIME

    # service_time_distribution = Exponential(lambda_ = 1 / mean_service_time)
    # service_time_distribution = Constant(value = mean_service_time)
    # service_time_distribution = Pareto(k = 1.05, beta = 1)
    service_time_distribution = Pareto(k = 2.05, beta = 1)


    ## DISTRIBUTION FOR INTERARRIVAL TIME

    # interarrival_time_distribution = Constant(value = mean_interarrival_time)
    interarrival_time_distribution = Poisson(lambda_ = mean_interarrival_time)
    # interarrival_time_distribution = Erlang(n = 1, lambda_ = 1)
    # interarrival_time_distribution = HyperExponential(lambdas = [0.8333, 5], weights = [0.8, 0.2])


    ## SYSTEM CREATION

    system = System(
        name="System",
        description="System", 
        parameters={
            'service_units': 10,
            'mean_service_time': mean_service_time,
            'service_time_distribution': service_time_distribution,
            'mean_interarrival_time': mean_interarrival_time,
            'interarrival_time_distribution': interarrival_time_distribution,
            'customers': 10000,
            'runs': 10,
        }
    )

    system.simulate()
    statistics = system.get_statistics()
    print(statistics)

if __name__ == "__main__":
    main()
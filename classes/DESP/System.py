import math
from classes.DESP.Customer import Customer
from classes.DESP.ServiceUnit import ServiceUnit
import numpy as np
from classes.Plotter import Plotter
from classes.Estimator import Control

class System:
    def __init__(self, name: str, parameters: dict):
        self.name = name
        self.parameters = parameters
        self.offered_loads = []
        self.blocked_customers = []

    def initialize(self, U1 = None, U2 = None, U3 = None):
        self.service_units = [
            ServiceUnit(
                name=f'ServiceUnit_{i}', 
                description=f'ServiceUnit_{i}', 
                parameters={
                    'service_time': self.parameters['service_time_distribution'],
                }
            ) for i in range(self.parameters['n_service_units'])
        ]
        
        self.interarrival_times = []
        self.interarrival_times = self.parameters['interarrival_time_distribution'].simulate(n = self.parameters['n_customers'])['observed']
        # self.interarrival_times = self.parameters['interarrival_time_distribution'].simulate(n = self.parameters['n_customers'], U1 = U1, U2 = U2)['observed']
        self.arrival_times = [sum(self.interarrival_times[:i + 1]) for i in range(len(self.interarrival_times))]
        if U3 is None:
            self.service_times = [None for _ in range(self.parameters['n_customers'])]
        else:
            self.service_times = self.parameters['service_time_distribution'].simulate(n = self.parameters['n_customers'])['observed']
            # self.service_times = self.parameters['service_time_distribution'].simulate(n = self.parameters['n_customers'], U2 = U3)['observed']

        self.customers = [
            Customer(
                name=f'Customer_{i}',
                description=f'Customer_{i}',
                parameters={
                    'arrival_time': self.arrival_times[i],
                }
            ) for i in range(self.parameters['n_customers'])
        ]

        self.time = 0
        self.num_blocked = 0
        self.statistics = {
            'service_times': [],
        }

    def simulate(self, verbose = False):
        for _ in range(self.parameters['n_runs']):
            self.initialize()
            self.run()

        self.compute_statistics()
        if verbose:
            self.print_statistics()
            self.ordinary_estimator()
            self.control_estimator()

    def run(self):
        for customer in self.customers:
            self.time = customer.arrival_time
            blocked = self.assign_customer_to_free_service_unit(customer, service_time = self.service_times.pop(0))
            if blocked:
                self.num_blocked += 1
            else:
                self.statistics['service_times'].append(customer.service_time)

        self.blocked_customers.append(self.num_blocked)

        total_arrival_time = sum(self.interarrival_times)
        total_service_time = sum(self.statistics['service_times'])
        offered_load = total_service_time / total_arrival_time
        self.offered_loads.append(offered_load)

    def assign_customer_to_free_service_unit(self, customer: Customer, service_time: float = None):
        for service_unit in self.service_units:
            service_unit.check_if_free(self.time)

            if service_unit.free:
                service_unit.serve_customer(customer, service_time)
                return False
        return True
    
    def compute_statistics(self):
        self.statistics['service_time_mean'] = float(np.mean(self.statistics['service_times']))
        self.statistics['service_time_variance'] = float(np.std(self.statistics['service_times']) ** 2)
        self.statistics['interarrival_time_mean'] = float(np.mean(self.interarrival_times))
        self.statistics['interarrival_time_variance'] = float(np.std(self.interarrival_times) ** 2)
    
        n_runs = self.parameters['n_runs']

        fraction_blocked = [self.blocked_customers[i] / self.parameters['n_customers'] for i in range(self.parameters['n_runs'])]
        self.statistics['blocking_probability_mean'] = sum(fraction_blocked) / n_runs
        self.statistics['blocking_probability_variance'] = np.var(fraction_blocked)
        self.statistics['blocking_probability_std'] = np.std(fraction_blocked)

        t = 2.262 # 95% confidence level, n_runs = 10
        self.statistics['blocking_probability_confidence_interval'] = (
            float(self.statistics['blocking_probability_mean'] - t * self.statistics['blocking_probability_std'] / math.sqrt(n_runs)),
            float(self.statistics['blocking_probability_mean'] + t * self.statistics['blocking_probability_std'] / math.sqrt(n_runs))
        )

        A = self.parameters['interarrival_time_distribution'].mean() * self.parameters['service_time_distribution'].mean()
        m = self.parameters['n_service_units']
        num = A ** m / math.factorial(m)
        den = sum(A ** i / math.factorial(i) for i in range(m + 1))
        self.statistics['theoretical_blocking_probability'] = num / den

    def print_statistics(self):
        print(f"""
---------------------------------- Statistics --------------------------------------------

Service time:
- Empirical mean: {self.statistics['service_time_mean']}
- Empirical var: {self.statistics['service_time_variance']}

Interarrival time:
- Empirical mean: {self.statistics['interarrival_time_mean']}
- Empirical var: {self.statistics['interarrival_time_variance']}
""")
    
    def ordinary_estimator(self):
        print(f"""
---------------------------------- Ordinary estimator --------------------------------------------

Blocking probability:
- Empirical mean: {self.statistics['blocking_probability_mean']}
- Empirical variance: {self.statistics['blocking_probability_variance']}
- Confidence interval: {self.statistics['blocking_probability_confidence_interval']}
- Theoretical value: {self.statistics['theoretical_blocking_probability']}
        """)


    def control_estimator(self):
        fraction_blocked = [self.blocked_customers[i] / self.parameters['n_customers'] for i in range(self.parameters['n_runs'])]
        self.statistics['mean_fraction_blocked'] = sum(fraction_blocked) / self.parameters['n_runs']
        self.statistics['mean_offered_load'] = sum(self.offered_loads) / self.parameters['n_runs']
        temp = [(fraction_blocked[i] - self.statistics['mean_fraction_blocked']) * (self.offered_loads[i] - self.statistics['mean_offered_load']) for i in range(self.parameters['n_runs'])]
        sample_cov = 1 / (self.parameters['n_runs'] - 1) * sum(temp)
        
        temp = [(self.offered_loads[i] - self.statistics['mean_offered_load']) ** 2 for i in range(self.parameters['n_runs'])]
        self.statistics['sample_var_offered_load'] = 1 / (self.parameters['n_runs'] - 1) * sum(temp)

        self.statistics['c'] = sample_cov / self.statistics['sample_var_offered_load']

        better_estimates = [fraction_blocked[i] - self.statistics['c'] * (self.offered_loads[i] - self.statistics['mean_offered_load']) for i in range(self.parameters['n_runs'])]
        self.statistics['mean_better_estimates'] = np.mean(better_estimates)
        self.statistics['variance_better_estimates'] = np.var(better_estimates)

        self.statistics['better_estimates_confidence_interval'] = (
            float(self.statistics['mean_better_estimates'] - 1.96 * np.sqrt(self.statistics['variance_better_estimates'] / self.parameters['n_runs'])),
            float(self.statistics['mean_better_estimates'] + 1.96 * np.sqrt(self.statistics['variance_better_estimates'] / self.parameters['n_runs']))
        )

        print(f"""
---------------------------------- Control estimator --------------------------------------------
              
Blocking probability:
- Empirical mean: {self.statistics['mean_better_estimates']}
- Empirical variance: {self.statistics['variance_better_estimates']}
- Confidence interval: {self.statistics['better_estimates_confidence_interval']}
- c: {self.statistics['c']}
""")
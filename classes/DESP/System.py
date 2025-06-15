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

    def initialize(self):
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
        self.interarrival_times, _, _ = self.parameters['interarrival_time_distribution'].simulate(self.parameters['n_customers'])
        self.arrival_times = [sum(self.interarrival_times[:i + 1]) for i in range(len(self.interarrival_times))]

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

            for customer in self.customers:
                self.time = customer.arrival_time
                blocked = self.assign_customer_to_free_service_unit(customer)
                if blocked:
                    self.num_blocked += 1
                else:
                    self.statistics['service_times'].append(customer.service_time)

            self.blocked_customers.append(self.num_blocked)

            total_arrival_time = sum(self.interarrival_times)
            total_service_time = sum(self.statistics['service_times'])
            offered_load = total_service_time / total_arrival_time
            self.offered_loads.append(offered_load)

        if verbose:
            self.print_statistics()
            self.ordinary_estimator()
            self.control_estimator()

    def assign_customer_to_free_service_unit(self, customer: Customer):
        for service_unit in self.service_units:
            service_unit.check_if_free(self.time)

            if service_unit.free:
                service_unit.serve_customer(customer)
                return False
        return True

    def print_statistics(self):
        print(f"""
---------------------------------- Statistics --------------------------------------------
              
Service time:
- Empirical mean: {float(np.mean(self.statistics['service_times']))}
- Empirical var: {float(np.std(self.statistics['service_times']) ** 2)}

Interarrival time:
- Empirical mean: {float(np.mean(self.interarrival_times))}
- Empirical var: {float(np.std(self.interarrival_times) ** 2)}
""")
    
    def ordinary_estimator(self):
        n_runs = self.parameters['n_runs']
        # emp_mean = sum(self.blocked_customers) / n_runs
        fraction_blocked = [self.blocked_customers[i] / self.parameters['n_customers'] for i in range(self.parameters['n_runs'])]
        emp_mean = sum(fraction_blocked) / n_runs
        # emp_var = (sum(x ** 2 for x in self.blocked_customers) - n_runs * emp_mean ** 2) / (n_runs - 1)
        emp_var = np.var(fraction_blocked)
        # emp_std = np.std(self.blocked_customers)
        emp_std = np.std(fraction_blocked)

        t = 2.262 # 95% confidence level, n_runs = 10
        blocking_probability_confidence_interval = (
            float(emp_mean - t * emp_std / math.sqrt(n_runs)),
            float(emp_mean + t * emp_std / math.sqrt(n_runs))
        )

        A = self.parameters['interarrival_time_distribution'].mean() * self.parameters['service_time_distribution'].mean()
        m = self.parameters['n_service_units']
        num = A ** m / math.factorial(m)
        den = sum(A ** i / math.factorial(i) for i in range(m + 1))
        theoretical_blocking_probability = num / den

        print(f"""
---------------------------------- Ordinary estimator --------------------------------------------

Blocking probability:
- Empirical mean: {emp_mean}
- Empirical variance: {emp_var}
- Confidence interval: {blocking_probability_confidence_interval}
- Theoretical value: {theoretical_blocking_probability}
        """)


    def control_estimator(self):
        fraction_blocked = [self.blocked_customers[i] / self.parameters['n_customers'] for i in range(self.parameters['n_runs'])]
        mean_fraction_blocked = sum(fraction_blocked) / self.parameters['n_runs']
        mean_offered_load = sum(self.offered_loads) / self.parameters['n_runs']
        temp = [(fraction_blocked[i] - mean_fraction_blocked) * (self.offered_loads[i] - mean_offered_load) for i in range(self.parameters['n_runs'])]
        sample_cov = 1 / (self.parameters['n_runs'] - 1) * sum(temp)
        
        temp = [(self.offered_loads[i] - mean_offered_load) ** 2 for i in range(self.parameters['n_runs'])]
        sample_var_offered_load = 1 / (self.parameters['n_runs'] - 1) * sum(temp)

        c = sample_cov / sample_var_offered_load

        better_estimates = [fraction_blocked[i] - c * (self.offered_loads[i] - mean_offered_load) for i in range(self.parameters['n_runs'])]
        mean_better_estimates = np.mean(better_estimates)
        variance_better_estimates = np.var(better_estimates)
        better_estimates_confidence_interval = (
            float(mean_better_estimates - 1.96 * np.sqrt(variance_better_estimates / self.parameters['n_runs'])),
            float(mean_better_estimates + 1.96 * np.sqrt(variance_better_estimates / self.parameters['n_runs']))
        )

        print(f"""
---------------------------------- Control estimator --------------------------------------------
              
Blocking probability:
- Empirical mean: {mean_better_estimates}
- Empirical variance: {variance_better_estimates}
- Confidence interval: {better_estimates_confidence_interval}
- c: {c}
""")
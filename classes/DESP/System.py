import math
from classes.CRVG import Poisson, Exponential, Erlang, HyperExponential
from classes.DESP.Customer import Customer
from classes.DESP.ServiceUnit import ServiceUnit

class System:
    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.parameters = parameters


    def initialize(self):
        self.service_units = [
            ServiceUnit(
                name=f'ServiceUnit_{i}', 
                description=f'ServiceUnit_{i}', 
                parameters={
                    'service_time': self.parameters['service_time_distribution'],
                }
            ) for i in range(self.parameters['service_units'])
        ]
        
        self.interarrival_times, _, _ = self.parameters['interarrival_time_distribution'].simulate(self.parameters['customers'])
        self.arrival_times = [sum(self.interarrival_times[:i + 1]) for i in range(len(self.interarrival_times))]

        self.customers = [
            Customer(
                name=f'Customer_{i}',
                description=f'Customer_{i}',
                parameters={
                    'arrival_time': self.arrival_times[i],
                }
            ) for i in range(self.parameters['customers'])
        ]

        self.time = 0
        self.num_blocked = 0


    def simulate(self):
        self.statistics = {
            'blocked_customers': [],
        }

        for _ in range(self.parameters['runs']):
            self.initialize()

            for customer in self.customers:
                self.time = customer.arrival_time
                blocked = self.assign_customer_to_free_service_unit(customer)
                if blocked:
                    self.num_blocked += 1

            self.statistics['blocked_customers'].append(self.num_blocked)


    def assign_customer_to_free_service_unit(self, customer: Customer):
        for service_unit in self.service_units:
            service_unit.check_if_free(self.time)

            if service_unit.free:
                service_unit.serve_customer(customer)
                return False
        return True
    
    def get_statistics(self):
        n_runs = self.parameters['runs']
        emp_mean = sum(self.statistics['blocked_customers']) / n_runs
        emp_std = math.sqrt((sum(x ** 2 for x in self.statistics['blocked_customers']) - n_runs * emp_mean ** 2) / (n_runs - 1))

        t = 2.262 # 95% confidence level, n_runs = 10
        confidence_interval = (
            emp_mean - t * emp_std / math.sqrt(n_runs),
            emp_mean + t * emp_std / math.sqrt(n_runs)
        )

        A = self.parameters['interarrival_time_distribution'].mean() * self.parameters['service_time_distribution'].mean()
        m = self.parameters['service_units']
        num = A ** m / math.factorial(m)
        den = sum(A ** i / math.factorial(i) for i in range(m + 1))
        theoretical_blocking_probability = num / den

        return {
            'empirical_mean': emp_mean,
            'empirical_standard_deviation': emp_std,
            'confidence_interval': confidence_interval,
            'theoretical_blocking_probability': theoretical_blocking_probability,
        }

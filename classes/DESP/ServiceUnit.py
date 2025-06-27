from classes.DESP.Customer import Customer

class ServiceUnit:
    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters

        self.served_customer = None
        self.service_time = None
        self.free = True

    def __str__(self):
        return f"ServiceUnit(name={self.name}, description={self.description}, parameters={self.parameters})"
    
    def check_if_free(self, time: int):
        if self.free:
            return
        
        if time >= self.served_customer.arrival_time + self.service_time:
            self.free = True
            self.served_customer = None
            self.service_time = None

    def serve_customer(self, customer: Customer, service_time: float = None):
        self.free = False
        self.served_customer = customer
        self.service_time = service_time if service_time is not None else self.parameters['service_time'].sample()

        customer.service_time = self.service_time





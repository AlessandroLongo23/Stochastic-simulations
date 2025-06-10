class Customer:
    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.arrival_time = parameters['arrival_time']

    def __str__(self):
        return f"Customer(name={self.name}, description={self.description}, parameters={self.parameters})"
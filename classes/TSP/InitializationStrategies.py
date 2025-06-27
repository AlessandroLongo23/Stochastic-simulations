import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class InitializationStrategy(ABC):
    @abstractmethod
    def initialize(self, cost_matrix: np.ndarray) -> np.ndarray:
        pass


class RandomInitialization(InitializationStrategy):
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
    
    def initialize(self, cost_matrix: np.ndarray) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.permutation(cost_matrix.shape[0])


class NearestNeighborInitialization(InitializationStrategy):
    def __init__(self, start_city: int = 0):
        self.start_city = start_city
    
    def initialize(self, cost_matrix: np.ndarray) -> np.ndarray:
        n_cities = cost_matrix.shape[0]
        unvisited = set(range(n_cities))
        current = self.start_city
        tour = [current]
        unvisited.remove(current)
        
        while unvisited:
            nearest = min(unvisited, key=lambda city: cost_matrix[current][city])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        return np.array(tour)


class GreedyInitialization(InitializationStrategy):
    def initialize(self, cost_matrix: np.ndarray) -> np.ndarray:
        n_cities = cost_matrix.shape[0]
        
        edges = []
        for i in range(n_cities):
            for j in range(i + 1, n_cities):
                edges.append((cost_matrix[i][j], i, j))
        
        edges.sort()
        
        degree = [0] * n_cities
        adjacency = [[] for _ in range(n_cities)]
        
        for cost, i, j in edges:
            if degree[i] < 2 and degree[j] < 2:
                adjacency[i].append(j)
                adjacency[j].append(i)
                degree[i] += 1
                degree[j] += 1
                
                if sum(degree) == 2 * n_cities:
                    break
        
        start = next((i for i in range(n_cities) if degree[i] > 0), 0)
        tour = [start]
        current = start
        visited = {start}
        
        while len(tour) < n_cities:
            next_city = None
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    next_city = neighbor
                    break
            
            if next_city is None:
                remaining = [i for i in range(n_cities) if i not in visited]
                tour.extend(remaining)
                break
            else:
                tour.append(next_city)
                visited.add(next_city)
                current = next_city
        
        return np.array(tour)


class FarthestInsertionInitialization(InitializationStrategy):
    def initialize(self, cost_matrix: np.ndarray) -> np.ndarray:
        n_cities = cost_matrix.shape[0]
        
        if n_cities <= 2:
            return np.arange(n_cities)
        
        max_distance = 0
        start_pair = (0, 1)
        
        for i in range(n_cities):
            for j in range(i + 1, n_cities):
                if cost_matrix[i][j] > max_distance:
                    max_distance = cost_matrix[i][j]
                    start_pair = (i, j)
        
        tour = list(start_pair)
        remaining = set(range(n_cities)) - set(tour)
        
        while remaining:
            farthest_city = None
            max_min_distance = -1
            
            for city in remaining:
                min_distance = min(cost_matrix[city][tour_city] for tour_city in tour)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    farthest_city = city
            
            best_position = 0
            best_increase = float('inf')
            
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                increase = (cost_matrix[tour[i]][farthest_city] + 
                           cost_matrix[farthest_city][tour[j]] - 
                           cost_matrix[tour[i]][tour[j]])
                
                if increase < best_increase:
                    best_increase = increase
                    best_position = i + 1
            
            tour.insert(best_position, farthest_city)
            remaining.remove(farthest_city)
        
        return np.array(tour) 
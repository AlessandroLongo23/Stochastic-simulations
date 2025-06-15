import numpy as np
from sklearn.manifold import MDS
import warnings

class TravellingSalesman:
    def __init__(self, n_stations: int = None, cost_matrix: np.ndarray = None):
        if cost_matrix is None:
            self.n_stations = n_stations
            self.coordinates = self.generate_coordinates()
            self.cost_matrix = self.generate_cost_matrix()
        else:
            self.n_stations = cost_matrix.shape[0]
            self.cost_matrix = cost_matrix
            self.coordinates = self.calculate_coordinates()

    def generate_coordinates(self):
        return np.random.rand(self.n_stations, 2)
    
    def generate_cost_matrix(self):
        cost_matrix = np.zeros((self.n_stations, self.n_stations))
        for i in range(self.n_stations):
            for j in range(self.n_stations):
                cost_matrix[i, j] = self.calculate_distance(i, j)
        return cost_matrix
    
    def calculate_distance(self, i: int, j: int):
        return np.linalg.norm(self.coordinates[i] - self.coordinates[j])
    
    def calculate_coordinates(self):
        """
        Calculate 2D coordinates from the cost matrix using Multi-Dimensional Scaling (MDS).
        This finds coordinates such that the Euclidean distances approximate the cost matrix values.
        """
        try:
            # Suppress sklearn convergence warnings for cleaner output
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Use MDS to embed the distance matrix into 2D space
                mds = MDS(
                    n_components=2,           # 2D coordinates
                    dissimilarity='precomputed',  # We provide distance matrix directly
                    random_state=42,          # For reproducible results
                    max_iter=1000,           # Maximum iterations
                    eps=1e-6                 # Convergence tolerance
                )
                
                coordinates = mds.fit_transform(self.cost_matrix)
                
                return coordinates
                
        except Exception as e:
            print(f"MDS failed: {e}")
            print("Falling back to classical MDS using eigenvalue decomposition...")
            
            return self.classical_mds(self.cost_matrix)
    
    def classical_mds(self, distance_matrix):
        """
        Classical Multi-Dimensional Scaling using eigenvalue decomposition.
        This is a fallback method if sklearn MDS fails.
        """
        n = distance_matrix.shape[0]
        
        H = np.eye(n) - np.ones((n, n)) / n
        
        D_squared = distance_matrix ** 2
        B = -0.5 * H @ D_squared @ H
        
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        positive_eigenvalues = np.maximum(eigenvalues[:2], 0)
        
        coordinates = eigenvectors[:, :2] @ np.diag(np.sqrt(positive_eigenvalues))
        
        return coordinates

        
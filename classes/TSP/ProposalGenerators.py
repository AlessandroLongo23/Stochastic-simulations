import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List


class ProposalGenerator(ABC):
    @abstractmethod
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        pass


class TwoSwapProposal(ProposalGenerator):
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        neighbor = solution.copy()
        n = len(solution)
        i, j = np.random.choice(n, 2, replace=False)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor


class TwoOptProposal(ProposalGenerator):
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        neighbor = solution.copy()
        n = len(solution)
        
        i, j = sorted(np.random.choice(n, 2, replace=False))
        
        neighbor[i:j+1] = neighbor[i:j+1][::-1]
        
        return neighbor


class ThreeOptProposal(ProposalGenerator):
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        neighbor = solution.copy()
        n = len(solution)
        
        if n < 6:
            return TwoOptProposal().generate_neighbor(solution)
        
        positions = sorted(np.random.choice(n, 3, replace=False))
        i, j, k = positions
        
        move_type = np.random.randint(0, 3)
        
        if move_type == 0:
            neighbor[i:j] = neighbor[i:j][::-1]
        elif move_type == 1:
            neighbor[j:k] = neighbor[j:k][::-1]
        else:
            segment1 = neighbor[i:j].copy()
            segment2 = neighbor[j:k].copy()
            neighbor[i:i+len(segment2)] = segment2
            neighbor[i+len(segment2):i+len(segment2)+len(segment1)] = segment1
        
        return neighbor


class OrOptProposal(ProposalGenerator):
    def __init__(self, max_sequence_length: int = 3):
        self.max_sequence_length = max_sequence_length
    
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        neighbor = solution.copy()
        n = len(solution)
        
        if n < 4:
            return TwoSwapProposal().generate_neighbor(solution)
        
        seq_len = np.random.randint(1, min(self.max_sequence_length + 1, n // 2))
        
        start_pos = np.random.randint(0, n - seq_len + 1)
        
        possible_positions = list(range(n - seq_len + 1))
        possible_positions = [pos for pos in possible_positions 
                            if pos <= start_pos - seq_len or pos >= start_pos + seq_len]
        
        if not possible_positions:
            return TwoSwapProposal().generate_neighbor(solution)
        
        insert_pos = np.random.choice(possible_positions)
        
        sequence = neighbor[start_pos:start_pos + seq_len].copy()
        
        neighbor = np.concatenate([neighbor[:start_pos], neighbor[start_pos + seq_len:]])
        
        if insert_pos > start_pos:
            insert_pos -= seq_len
        
        neighbor = np.concatenate([neighbor[:insert_pos], sequence, neighbor[insert_pos:]])
        
        return neighbor


class LinKernighanProposal(ProposalGenerator):
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        n = len(solution)
        
        if np.random.random() < 0.7:  # 70% chance for 2-opt
            return TwoOptProposal().generate_neighbor(solution)
        else:  # 30% chance for 3-opt
            return ThreeOptProposal().generate_neighbor(solution)


class InsertionProposal(ProposalGenerator):
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        neighbor = solution.copy()
        n = len(solution)
        
        if n < 3:
            return TwoSwapProposal().generate_neighbor(solution)
        
        remove_pos = np.random.randint(0, n)
        city = neighbor[remove_pos]
        
        neighbor = np.concatenate([neighbor[:remove_pos], neighbor[remove_pos+1:]])
        
        insert_pos = np.random.randint(0, n-1)
        
        neighbor = np.concatenate([neighbor[:insert_pos], [city], neighbor[insert_pos:]])
        
        return neighbor


class MixedProposal(ProposalGenerator):
    def __init__(self, proposals: List[ProposalGenerator], weights: Optional[List[float]] = None):
        self.proposals = proposals
        if weights is None:
            weights = [1.0] * len(proposals)
        self.weights = np.array(weights) / sum(weights)
    
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        selected_proposal = np.random.choice(self.proposals, p=self.weights)
        return selected_proposal.generate_neighbor(solution)


class AdaptiveProposal(ProposalGenerator):
    def __init__(self, proposals: List[ProposalGenerator], learning_rate: float = 0.1):
        self.proposals = proposals
        self.learning_rate = learning_rate
        self.weights = np.ones(len(proposals)) / len(proposals)
        self.success_counts = np.zeros(len(proposals))
        self.total_counts = np.zeros(len(proposals))
        self.last_used = None
    
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        self.last_used = np.random.choice(len(self.proposals), p=self.weights)
        return self.proposals[self.last_used].generate_neighbor(solution)
    
    def update_success(self, accepted: bool):
        if self.last_used is not None:
            self.total_counts[self.last_used] += 1
            if accepted:
                self.success_counts[self.last_used] += 1
            
            if self.total_counts[self.last_used] > 10:
                success_rate = self.success_counts[self.last_used] / self.total_counts[self.last_used]
                self.weights[self.last_used] = (1 - self.learning_rate) * self.weights[self.last_used] + \
                                             self.learning_rate * success_rate
                
                self.weights = self.weights / np.sum(self.weights) 
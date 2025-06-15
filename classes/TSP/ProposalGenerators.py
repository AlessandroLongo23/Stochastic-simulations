"""
Proposal generators for Simulated Annealing optimization.
Provides various neighborhood structures for solution exploration.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List


class ProposalGenerator(ABC):
    """Abstract base class for proposal generators"""
    
    @abstractmethod
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate a neighbor solution"""
        pass


class TwoSwapProposal(ProposalGenerator):
    """Two-swap neighborhood: swap two random cities"""
    
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        neighbor = solution.copy()
        n = len(solution)
        i, j = np.random.choice(n, 2, replace=False)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor


class TwoOptProposal(ProposalGenerator):
    """2-opt neighborhood: reverse a segment of the tour"""
    
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        neighbor = solution.copy()
        n = len(solution)
        
        # Select two different positions
        i, j = sorted(np.random.choice(n, 2, replace=False))
        
        # Reverse the segment between i and j
        neighbor[i:j+1] = neighbor[i:j+1][::-1]
        
        return neighbor


class ThreeOptProposal(ProposalGenerator):
    """3-opt neighborhood: more complex rearrangement"""
    
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        neighbor = solution.copy()
        n = len(solution)
        
        if n < 6:  # 3-opt needs at least 6 cities
            return TwoOptProposal().generate_neighbor(solution)
        
        # Select three different positions and sort them
        positions = sorted(np.random.choice(n, 3, replace=False))
        i, j, k = positions
        
        # There are several 3-opt moves, randomly choose one
        move_type = np.random.randint(0, 3)
        
        if move_type == 0:
            # Move 1: reverse first segment
            neighbor[i:j] = neighbor[i:j][::-1]
        elif move_type == 1:
            # Move 2: reverse second segment
            neighbor[j:k] = neighbor[j:k][::-1]
        else:
            # Move 3: swap segments
            segment1 = neighbor[i:j].copy()
            segment2 = neighbor[j:k].copy()
            neighbor[i:i+len(segment2)] = segment2
            neighbor[i+len(segment2):i+len(segment2)+len(segment1)] = segment1
        
        return neighbor


class OrOptProposal(ProposalGenerator):
    """Or-opt neighborhood: relocate a sequence of cities"""
    
    def __init__(self, max_sequence_length: int = 3):
        self.max_sequence_length = max_sequence_length
    
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        neighbor = solution.copy()
        n = len(solution)
        
        if n < 4:
            return TwoSwapProposal().generate_neighbor(solution)
        
        # Choose sequence length (1 to max_sequence_length)
        seq_len = np.random.randint(1, min(self.max_sequence_length + 1, n // 2))
        
        # Choose starting position of sequence to move
        start_pos = np.random.randint(0, n - seq_len + 1)
        
        # Choose insertion position (can't be within the sequence)
        possible_positions = list(range(n - seq_len + 1))
        # Remove positions that would overlap with current sequence
        possible_positions = [pos for pos in possible_positions 
                            if pos <= start_pos - seq_len or pos >= start_pos + seq_len]
        
        if not possible_positions:
            return TwoSwapProposal().generate_neighbor(solution)
        
        insert_pos = np.random.choice(possible_positions)
        
        # Extract sequence
        sequence = neighbor[start_pos:start_pos + seq_len].copy()
        
        # Remove sequence from current position
        neighbor = np.concatenate([neighbor[:start_pos], neighbor[start_pos + seq_len:]])
        
        # Insert sequence at new position
        if insert_pos > start_pos:
            insert_pos -= seq_len  # Adjust for removed sequence
        
        neighbor = np.concatenate([neighbor[:insert_pos], sequence, neighbor[insert_pos:]])
        
        return neighbor


class LinKernighanProposal(ProposalGenerator):
    """Simplified Lin-Kernighan style move (k-opt with k=2,3)"""
    
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        n = len(solution)
        
        if np.random.random() < 0.7:  # 70% chance for 2-opt
            return TwoOptProposal().generate_neighbor(solution)
        else:  # 30% chance for 3-opt
            return ThreeOptProposal().generate_neighbor(solution)


class InsertionProposal(ProposalGenerator):
    """Insertion neighborhood: remove and reinsert a city"""
    
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        neighbor = solution.copy()
        n = len(solution)
        
        if n < 3:
            return TwoSwapProposal().generate_neighbor(solution)
        
        # Select city to remove
        remove_pos = np.random.randint(0, n)
        city = neighbor[remove_pos]
        
        # Remove city
        neighbor = np.concatenate([neighbor[:remove_pos], neighbor[remove_pos+1:]])
        
        # Select new insertion position
        insert_pos = np.random.randint(0, n-1)
        
        # Insert city at new position
        neighbor = np.concatenate([neighbor[:insert_pos], [city], neighbor[insert_pos:]])
        
        return neighbor


class MixedProposal(ProposalGenerator):
    """Mixed proposal generator that randomly selects from multiple strategies"""
    
    def __init__(self, proposals: List[ProposalGenerator], weights: Optional[List[float]] = None):
        self.proposals = proposals
        if weights is None:
            weights = [1.0] * len(proposals)
        self.weights = np.array(weights) / sum(weights)  # Normalize
    
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        # Randomly select a proposal generator based on weights
        selected_proposal = np.random.choice(self.proposals, p=self.weights)
        return selected_proposal.generate_neighbor(solution)


class AdaptiveProposal(ProposalGenerator):
    """Adaptive proposal that adjusts selection based on success rates"""
    
    def __init__(self, proposals: List[ProposalGenerator], learning_rate: float = 0.1):
        self.proposals = proposals
        self.learning_rate = learning_rate
        self.weights = np.ones(len(proposals)) / len(proposals)
        self.success_counts = np.zeros(len(proposals))
        self.total_counts = np.zeros(len(proposals))
        self.last_used = None
    
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        # Select proposal based on current weights
        self.last_used = np.random.choice(len(self.proposals), p=self.weights)
        return self.proposals[self.last_used].generate_neighbor(solution)
    
    def update_success(self, accepted: bool):
        """Update success rates based on whether the proposal was accepted"""
        if self.last_used is not None:
            self.total_counts[self.last_used] += 1
            if accepted:
                self.success_counts[self.last_used] += 1
            
            # Update weights based on success rates
            if self.total_counts[self.last_used] > 10:  # Only update after some trials
                success_rate = self.success_counts[self.last_used] / self.total_counts[self.last_used]
                # Adjust weight based on success rate
                self.weights[self.last_used] = (1 - self.learning_rate) * self.weights[self.last_used] + \
                                             self.learning_rate * success_rate
                
                # Normalize weights
                self.weights = self.weights / np.sum(self.weights) 
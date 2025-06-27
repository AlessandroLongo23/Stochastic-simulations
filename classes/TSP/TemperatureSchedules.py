import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Optional, List, Callable


class TemperatureSchedule(ABC):
    @abstractmethod
    def get_temperature(self, iteration: int) -> float:
        pass


class ExponentialCooling(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0, cooling_rate: float = 0.995):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
    
    def get_temperature(self, iteration: int) -> float:
        return self.initial_temp * (self.cooling_rate ** iteration)


class LinearCooling(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0, max_iterations: int = 10000):
        self.initial_temp = initial_temp
        self.max_iterations = max_iterations
    
    def get_temperature(self, iteration: int) -> float:
        return max(0.01, self.initial_temp * (1 - iteration / self.max_iterations))


class LogarithmicCooling(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0):
        self.initial_temp = initial_temp
    
    def get_temperature(self, iteration: int) -> float:
        return self.initial_temp / math.log(iteration + 2)


class SqrtCooling(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0):
        self.initial_temp = initial_temp
    
    def get_temperature(self, iteration: int) -> float:
        return self.initial_temp / math.sqrt(iteration + 1)


class FastCooling(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0):
        self.initial_temp = initial_temp
    
    def get_temperature(self, iteration: int) -> float:
        return self.initial_temp / (iteration + 1)


class AdaptiveCooling(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0, target_acceptance_rate: float = 0.5, 
                 window_size: int = 100):
        self.initial_temp = initial_temp
        self.target_acceptance_rate = target_acceptance_rate
        self.current_temp = initial_temp
        self.recent_acceptances = []
        self.window_size = window_size
    
    def get_temperature(self, iteration: int) -> float:
        return self.current_temp
    
    def update_temperature(self, accepted: bool):
        self.recent_acceptances.append(accepted)
        
        if len(self.recent_acceptances) > self.window_size:
            self.recent_acceptances.pop(0)
        
        if len(self.recent_acceptances) >= self.window_size:
            acceptance_rate = sum(self.recent_acceptances) / len(self.recent_acceptances)
            
            if acceptance_rate > self.target_acceptance_rate:
                self.current_temp *= 0.99   
            else:
                self.current_temp *= 1.01  


class ReheatSchedule(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0, cooling_rate: float = 0.995,
                 reheat_interval: int = 1000, reheat_factor: float = 1.5):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.reheat_interval = reheat_interval
        self.reheat_factor = reheat_factor
        self.temp = initial_temp
    
    def get_temperature(self, iteration: int) -> float:
        self.temp *= self.cooling_rate
        
        if iteration > 0 and iteration % self.reheat_interval == 0:
            self.temp *= self.reheat_factor
        
        return self.temp


class HybridCooling(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0, max_iterations: int = 10000):
        self.initial_temp = initial_temp
        self.max_iterations = max_iterations
    
    def get_temperature(self, iteration: int) -> float:
        if iteration < self.max_iterations * 0.3:
            return self.initial_temp / math.sqrt(iteration + 1)
        else:
            adjusted_iter = iteration - self.max_iterations * 0.3
            return (self.initial_temp * 0.3) * (0.999 ** adjusted_iter)


class GeometricCooling(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0, cooling_factor: float = 0.9, 
                 steps_per_temp: int = 100):
        self.initial_temp = initial_temp
        self.cooling_factor = cooling_factor
        self.steps_per_temp = steps_per_temp
    
    def get_temperature(self, iteration: int) -> float:
        cooling_step = iteration // self.steps_per_temp
        return self.initial_temp * (self.cooling_factor ** cooling_step)


class ConstantThermodynamicSpeed(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0, final_temp: float = 0.1, 
                 max_iterations: int = 10000, system_size: int = 100):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.max_iterations = max_iterations
        self.system_size = system_size
        
        self.speed = self._calculate_thermodynamic_speed()
    
    def _calculate_thermodynamic_speed(self) -> float:
        temp_ratio = self.initial_temp / self.final_temp
        return math.log(temp_ratio) / self.max_iterations
    
    def get_temperature(self, iteration: int) -> float:
        if iteration >= self.max_iterations:
            return self.final_temp
        
        progress = iteration / self.max_iterations
        
        temp = self.initial_temp * math.exp(-self.speed * iteration)
        
        return max(temp, self.final_temp)


class ConstantThermodynamicSpeedV2(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0, final_temp: float = 0.1,
                 max_iterations: int = 10000, system_size: int = 100,
                 relaxation_factor: float = 1.0):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.max_iterations = max_iterations
        self.system_size = system_size
        self.relaxation_factor = relaxation_factor
        
        self.natural_timescale = self._calculate_natural_timescale()
        self.speed_parameter = self._calculate_speed_parameter()
    
    def _calculate_natural_timescale(self) -> float:
        return math.sqrt(self.system_size) * self.relaxation_factor
    
    def _calculate_speed_parameter(self) -> float:
        temp_ratio = self.initial_temp / self.final_temp
        return math.log(temp_ratio) / (self.max_iterations * self.natural_timescale)
    
    def get_temperature(self, iteration: int) -> float:
        if iteration >= self.max_iterations:
            return self.final_temp
        
        progress = iteration / self.max_iterations
        
        temp_ratio = math.log(self.initial_temp / self.final_temp)
        thermodynamic_progress = progress * temp_ratio
        
        temp = self.initial_temp * math.exp(-thermodynamic_progress)
        
        if iteration > 0:
            correction_amplitude = 0.05 * (1.0 - progress)
            non_equilibrium_correction = 1.0 + correction_amplitude * math.sin(iteration / 1000.0)
            temp *= non_equilibrium_correction
        
        return max(temp, self.final_temp)


class EntropyMinimizingCooling(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0, final_temp: float = 0.1,
                 max_iterations: int = 10000, entropy_weight: float = 1.0):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.max_iterations = max_iterations
        self.entropy_weight = entropy_weight
        
        self.total_entropy = 0.0
        self.entropy_history = []
    
    def get_temperature(self, iteration: int) -> float:
        if iteration >= self.max_iterations:
            return self.final_temp
        
        progress = iteration / self.max_iterations
        
        entropy_factor = 1.0 - (progress ** (1.0 + self.entropy_weight))
        
        temp = self.final_temp + (self.initial_temp - self.final_temp) * entropy_factor
        
        if iteration > 0:
            prev_temp = self.entropy_history[-1] if self.entropy_history else self.initial_temp
            entropy_change = abs(math.log(temp / prev_temp))
            self.total_entropy += entropy_change
        
        self.entropy_history.append(temp)
        return temp
    
    def get_total_entropy_production(self) -> float:
        return self.total_entropy


class RelaxationTimeAdaptive(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0, final_temp: float = 0.1,
                 max_iterations: int = 10000, adaptation_rate: float = 0.1):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.max_iterations = max_iterations
        self.adaptation_rate = adaptation_rate
        
        self.current_temp = initial_temp
        self.estimated_relaxation_time = 1.0
        self.energy_variance_history = []
    
    def get_temperature(self, iteration: int) -> float:
        if iteration >= self.max_iterations:
            return self.final_temp
        
        self._update_relaxation_time(iteration)
        
        base_progress = iteration / self.max_iterations
        relaxation_adjusted_progress = base_progress * (1.0 + self.estimated_relaxation_time)
        
        relaxation_adjusted_progress = min(relaxation_adjusted_progress, 1.0)
        
        temp_ratio = (self.initial_temp / self.final_temp) ** relaxation_adjusted_progress
        temp = self.initial_temp / temp_ratio
        
        self.current_temp = max(temp, self.final_temp)
        return self.current_temp
    
    def _update_relaxation_time(self, iteration: int):
        if iteration > 10:
            mock_energy_variance = 1.0 / (iteration + 1)
            self.energy_variance_history.append(mock_energy_variance)
            
            if len(self.energy_variance_history) > 100:
                self.energy_variance_history.pop(0)
            
            avg_variance = sum(self.energy_variance_history) / len(self.energy_variance_history)
            self.estimated_relaxation_time = (self.current_temp ** 2) * avg_variance * self.adaptation_rate


class OptimalAnnealing(TemperatureSchedule):
    def __init__(self, initial_temp: float = 100.0, final_temp: float = 0.1,
                 max_iterations: int = 10000, system_size: int = 100):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.max_iterations = max_iterations
        self.system_size = system_size
        
        self.tds2 = ConstantThermodynamicSpeedV2(
            initial_temp, final_temp, max_iterations, system_size, 
            relaxation_factor=0.1
        )
        self.entropy_minimizer = EntropyMinimizingCooling(
            initial_temp, final_temp, max_iterations, entropy_weight=0.5
        )
        
        self.exploration_phase_end = int(0.7 * max_iterations)
        
        self.best_energy_seen = float('inf')
        self.stagnation_counter = 0
        self.stagnation_threshold = max_iterations // 100
    
    def get_temperature(self, iteration: int) -> float:
        if iteration >= self.max_iterations:
            return self.final_temp
        
        progress = iteration / self.max_iterations
        
        if iteration < self.exploration_phase_end:
            temp = self.initial_temp * (0.998 ** iteration)
        
        else:
            remaining_progress = (iteration - self.exploration_phase_end) / (self.max_iterations - self.exploration_phase_end)
            exploration_temp = self.initial_temp * (0.998 ** self.exploration_phase_end)
            
            temp = exploration_temp * ((self.final_temp / exploration_temp) ** remaining_progress)
        
        return max(temp, self.final_temp)
    
    def update_performance(self, current_energy: float):
        if current_energy < self.best_energy_seen:
            self.best_energy_seen = current_energy
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
    
    def is_stagnating(self) -> bool:
        return self.stagnation_counter > self.stagnation_threshold 
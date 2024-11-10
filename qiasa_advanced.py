import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import requests
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt

@dataclass
class QIASAParams:
    population_size: int = 100
    max_iterations: int = 1000
    amplitude_decay: float = 0.9
    amplitude_gain: float = 1.1
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

class Problem:
    def __init__(self, name: str, dimension: int):
        self.name = name
        self.dimension = dimension
        
    def fitness(self, solution: np.ndarray) -> float:
        raise NotImplementedError
        
    def validate(self, solution: np.ndarray) -> bool:
        raise NotImplementedError

class TSP(Problem):
    def __init__(self, cities: Dict[str, Tuple[float, float]]):
        
        
        
        super().__init__("TSP", len(cities))
        self.cities = cities
        self.distance_matrix = self._compute_distance_matrix()
        
    def _compute_distance_matrix(self) -> np.ndarray:
        coords = np.array(list(self.cities.values()))
        n = len(coords)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
        return dist_matrix
        
    def fitness(self, tour: np.ndarray) -> float:
        total_distance = 0.0
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]
            total_distance += self.distance_matrix[from_city][to_city]
        return total_distance
        
    def validate(self, tour: np.ndarray) -> bool:
        return len(np.unique(tour)) == len(tour)

class KnapsackProblem(Problem):
    def __init__(self, items: List[Dict[str, float]], capacity: float):
        super().__init__("Knapsack", len(items))
        self.items = items
        self.capacity = capacity
        
    def fitness(self, solution: np.ndarray) -> float:
        total_weight = sum(item['weight'] * bit 
                         for item, bit in zip(self.items, solution))
        if total_weight > self.capacity:
            return float('-inf')
        return sum(item['value'] * bit 
                  for item, bit in zip(self.items, solution))
        
    def validate(self, solution: np.ndarray) -> bool:
        return all(bit in (0, 1) for bit in solution)

class JobScheduling(Problem):
    def __init__(self, jobs: List[Dict[str, Any]], machines: int):
        super().__init__("JobScheduling", len(jobs))
        self.jobs = jobs
        self.machines = machines
        
    def fitness(self, schedule: np.ndarray) -> float:
        machine_times = np.zeros(self.machines)
        for job_id in schedule:
            min_machine = np.argmin(machine_times)
            machine_times[min_machine] += self.jobs[job_id]['duration']
        return -np.max(machine_times)  # Negative because we want to minimize
        
    def validate(self, schedule: np.ndarray) -> bool:
        return len(np.unique(schedule)) == len(schedule)

class QIASA:
    def __init__(self, problem: Problem, params: QIASAParams):
        self.problem = problem
        self.params = params
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.history = []
        
    def initialize_population(self) -> List[np.ndarray]:
        if isinstance(self.problem, TSP):
            return [np.random.permutation(self.problem.dimension) 
                    for _ in range(self.params.population_size)]
        elif isinstance(self.problem, KnapsackProblem):
            return [np.random.randint(2, size=self.problem.dimension)
                    for _ in range(self.params.population_size)]
        else:
            return [np.random.permutation(self.problem.dimension)
                    for _ in range(self.params.population_size)]
    
    def adaptive_interference(self, 
                            fitness_values: np.ndarray, 
                            prob_amplitudes: np.ndarray) -> np.ndarray:
        avg_fitness = np.mean(fitness_values)
        new_amplitudes = np.copy(prob_amplitudes)
        
        for i in range(len(fitness_values)):
            if fitness_values[i] > avg_fitness:
                new_amplitudes[i] *= self.params.amplitude_gain
            else:
                new_amplitudes[i] *= self.params.amplitude_decay
                
        return new_amplitudes / np.sum(new_amplitudes)
    
    def mutate(self, solution: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.params.mutation_rate:
            return solution
            
        new_solution = solution.copy()
        if isinstance(self.problem, KnapsackProblem):
            idx = np.random.randint(len(solution))
            new_solution[idx] = 1 - new_solution[idx]
        else:
            i, j = np.random.choice(len(solution), 2, replace=False)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            
        return new_solution
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.params.crossover_rate:
            return parent1
            
        if isinstance(self.problem, KnapsackProblem):
            # Single-point crossover for binary representation
            crossover_point = np.random.randint(1, len(parent1))
            child = np.concatenate([
                parent1[:crossover_point],
                parent2[crossover_point:]
            ])
        else:
            # Order crossover (OX) for permutation problems
            size = len(parent1)
            start, end = sorted(np.random.choice(size, 2, replace=False))
            
            # Initialize child with invalid values
            child = np.full(size, -1)
            
            # Copy the selected segment from parent1
            child[start:end] = parent1[start:end]
            
            # Fill remaining positions with cities from parent2
            remaining = [x for x in parent2 if x not in parent1[start:end]]
            j = 0
            for i in range(size):
                if child[i] == -1:
                    child[i] = remaining[j]
                    j += 1
                    
        return child
    
    def optimize(self, callback=None) -> Tuple[np.ndarray, float, List[float]]:
        """
        Main optimization loop with optional callback for visualization
        
        Args:
            callback: Optional function called each iteration with current state
            
        Returns:
            Tuple of (best solution, best fitness, fitness history)
        """
        # Initialize population and probability amplitudes
        population = self.initialize_population()
        prob_amplitudes = np.ones(self.params.population_size) / self.params.population_size
        
        for iteration in range(self.params.max_iterations):
            # Evaluate fitness for current population
            fitness_values = np.array([
                self.problem.fitness(solution) 
                for solution in population
            ])
            
            # Update best solution found
            max_idx = np.argmax(fitness_values)
            if fitness_values[max_idx] > self.best_fitness:
                self.best_fitness = fitness_values[max_idx]
                self.best_solution = population[max_idx].copy()
            
            self.history.append(self.best_fitness)
            
            # Call callback if provided
            if callback:
                callback({
                    'iteration': iteration,
                    'best_fitness': self.best_fitness,
                    'best_solution': self.best_solution,
                    'population': population,
                    'fitness_values': fitness_values,
                    'amplitudes': prob_amplitudes
                })
            
            # Apply adaptive interference
            prob_amplitudes = self.adaptive_interference(fitness_values, prob_amplitudes)
            
            # Generate new population
            indices = np.arange(self.params.population_size)
            selected_indices = np.random.choice(
                indices, 
                size=self.params.population_size, 
                p=prob_amplitudes
            )
            
            new_population = []
            for i in range(0, self.params.population_size, 2):
                # Select parents
                parent1 = population[selected_indices[i]]
                parent2 = population[selected_indices[min(i + 1, len(selected_indices) - 1)]]
                
                # Create children through crossover and mutation
                child1 = self.mutate(self.crossover(parent1, parent2))
                child2 = self.mutate(self.crossover(parent2, parent1))
                
                new_population.extend([child1, child2])
            
            # Ensure population size remains constant
            population = new_population[:self.params.population_size]
            
        return self.best_solution, self.best_fitness, self.history

# Example usage and helper functions for real-world data
def load_city_data(country: str = 'US', n_cities: int = 20) -> Dict[str, Tuple[float, float]]:
    """
    Load real city data from an API or local source
    Returns dict of city names to (lat, lon) coordinates
    """
    # You could use a real API here, this is a simplified example
    cities = {
        'New York': (40.7128, -74.0060),
        'Los Angeles': (34.0522, -118.2437),
        'Chicago': (41.8781, -87.6298),
        'Houston': (29.7604, -95.3698),
        'Phoenix': (33.4484, -112.0740),
        'Philadelphia': (39.9526, -75.1652),
        'San Antonio': (29.4241, -98.4936),
        'San Diego': (32.7157, -117.1611),
        'Dallas': (32.7767, -96.7970),
        'San Jose': (37.3382, -121.8863)
    }
    return dict(list(cities.items())[:n_cities])

def load_knapsack_data(n_items: int = 50) -> Tuple[List[Dict[str, float]], float]:
    """
    Generate or load knapsack problem data
    Returns (items, capacity) where items is list of dict with 'weight' and 'value'
    """
    items = []
    for i in range(n_items):
        items.append({
            'weight': np.random.uniform(1, 10),
            'value': np.random.uniform(1, 100)
        })
    capacity = sum(item['weight'] for item in items) * 0.5  # 50% of total weight
    return items, capacity

def load_job_data(n_jobs: int = 30, n_machines: int = 5) -> Tuple[List[Dict[str, Any]], int]:
    """
    Generate or load job scheduling data
    Returns (jobs, n_machines) where jobs is list of dict with 'duration' and other properties
    """
    jobs = []
    for i in range(n_jobs):
        jobs.append({
            'id': i,
            'duration': np.random.uniform(1, 20),
            'priority': np.random.uniform(1, 5)
        })
    return jobs, n_machines

def visualize_tsp_solution(cities: Dict[str, Tuple[float, float]], tour: np.ndarray):
    """
    Visualize TSP solution using matplotlib
    """
    coords = np.array(list(cities.values()))
    plt.figure(figsize=(12, 8))
    
    # Plot cities
    plt.scatter(coords[:, 1], coords[:, 0], c='red', s=50)
    
    # Plot tour
    for i in range(len(tour)):
        start = coords[tour[i]]
        end = coords[tour[(i + 1) % len(tour)]]
        plt.plot([start[1], end[1]], [start[0], end[0]], 'b-', alpha=0.5)
    
    plt.title('TSP Solution')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Solve TSP
    cities = load_city_data(n_cities=15)
    tsp = TSP(cities)
    params = QIASAParams(
        population_size=100,
        max_iterations=1000,
        amplitude_decay=0.9,
        amplitude_gain=1.1,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    optimizer = QIASA(tsp, params)
    best_solution, best_fitness, history = optimizer.optimize()
    
    print(f"Best TSP distance: {best_fitness}")
    visualize_tsp_solution(cities, best_solution)
    
    # Solve Knapsack
    items, capacity = load_knapsack_data(n_items=30)
    knapsack = KnapsackProblem(items, capacity)
    optimizer = QIASA(knapsack, params)
    best_solution, best_fitness, history = optimizer.optimize()
    
    print(f"Best knapsack value: {best_fitness}")
    
    # Solve Job Scheduling
    jobs, n_machines = load_job_data(n_jobs=20, n_machines=3)
    scheduling = JobScheduling(jobs, n_machines)
    optimizer = QIASA(scheduling, params)
    best_solution, best_fitness, history = optimizer.optimize()
    
    print(f"Best scheduling makespan: {-best_fitness}")
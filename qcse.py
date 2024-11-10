import numpy as np
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from dataclasses import dataclass
import logging
from functools import partial
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QCSEConfig:
    """Configuration for QCSE algorithm"""
    n_particles: int = 50
    n_dimensions: int = 30
    cognitive_depth: int = 4
    quantum_layers: int = 3
    learning_rate: float = 0.01
    interaction_strength: float = 0.1
    entanglement_decay: float = 0.95
    memory_size: int = 100

class QuantumState:
    """Represents a quantum state in the Hilbert space"""
    
    def __init__(self, n_dimensions: int, n_quantum_layers: int):
        self.n_dimensions = n_dimensions
        self.n_quantum_layers = n_quantum_layers
        self.amplitude = self._initialize_amplitude()
        self.phase = self._initialize_phase()
        
    def _initialize_amplitude(self) -> np.ndarray:
        """Initialize quantum amplitude"""
        amplitude = np.random.randn(self.n_dimensions, self.n_quantum_layers) + \
                   1j * np.random.randn(self.n_dimensions, self.n_quantum_layers)
        return amplitude / np.linalg.norm(amplitude)
    
    def _initialize_phase(self) -> np.ndarray:
        """Initialize quantum phase"""
        return np.random.uniform(0, 2*np.pi, (self.n_dimensions, self.n_quantum_layers))
    
    def apply_quantum_operator(self, operator: np.ndarray) -> None:
        """Apply quantum operator to state"""
        self.amplitude = np.dot(operator, self.amplitude)
        self.amplitude = self.amplitude / np.linalg.norm(self.amplitude)
        
    def get_probability_distribution(self) -> np.ndarray:
        """Get probability distribution from quantum state"""
        return np.abs(self.amplitude) ** 2
    
    def collapse_state(self) -> np.ndarray:
        """Collapse quantum state to classical state"""
        prob_dist = self.get_probability_distribution()
        collapsed = np.zeros_like(prob_dist[:, 0])
        
        for i in range(self.n_dimensions):
            collapsed[i] = np.random.choice(self.n_quantum_layers, p=prob_dist[i]/np.sum(prob_dist[i]))
            
        return collapsed

class CognitiveState:
    """Represents cognitive state and memory"""
    
    def __init__(self, n_dimensions: int, memory_size: int):
        self.n_dimensions = n_dimensions
        self.memory_size = memory_size
        self.memory = []
        self.current_state = self._initialize_state()
        
    def _initialize_state(self) -> np.ndarray:
        """Initialize cognitive state"""
        return np.random.randn(self.n_dimensions)
    
    def update_memory(self, state: np.ndarray, fitness: float):
        """Update cognitive memory"""
        self.memory.append((state.copy(), fitness))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
            
    def get_best_memory(self) -> Tuple[np.ndarray, float]:
        """Get best state from memory"""
        if not self.memory:
            return self.current_state, float('-inf')
        return max(self.memory, key=lambda x: x[1])

class SwarmTopology:
    """Manages swarm interaction topology"""
    
    def __init__(self, n_particles: int, interaction_strength: float):
        self.n_particles = n_particles
        self.interaction_strength = interaction_strength
        self.topology_matrix = self._initialize_topology()
        
    def _initialize_topology(self) -> np.ndarray:
        """Initialize topology matrix"""
        topology = np.random.rand(self.n_particles, self.n_particles)
        topology = (topology + topology.T) / 2  # Make symmetric
        np.fill_diagonal(topology, 1)
        return topology
    
    def update_topology(self, fitness_values: np.ndarray):
        """Update topology based on fitness"""
        fitness_diff = np.abs(fitness_values[:, np.newaxis] - fitness_values)
        self.topology_matrix *= np.exp(-fitness_diff * self.interaction_strength)
        self.topology_matrix = (self.topology_matrix + self.topology_matrix.T) / 2
        np.fill_diagonal(self.topology_matrix, 1)

class QCSEParticle:
    """Individual particle in the swarm"""
    
    def __init__(self, config: QCSEConfig):
        self.quantum_state = QuantumState(config.n_dimensions, config.quantum_layers)
        self.cognitive_state = CognitiveState(config.n_dimensions, config.memory_size)
        self.current_position = np.random.randn(config.n_dimensions)
        self.current_velocity = np.zeros(config.n_dimensions)
        self.best_position = self.current_position.copy()
        self.best_fitness = float('-inf')
        
    def update(self, global_best: np.ndarray, topology_weights: np.ndarray, 
              particle_positions: np.ndarray, learning_rate: float):
        """Update particle state"""
        # Quantum evolution
        quantum_influence = self.quantum_state.collapse_state()
        
        # Cognitive influence
        cognitive_best, _ = self.cognitive_state.get_best_memory()
        
        # Social influence
        social_influence = np.average(particle_positions, weights=topology_weights, axis=0)
        
        # Update velocity
        self.current_velocity = self.current_velocity * 0.7 + \
                              learning_rate * (quantum_influence + \
                              0.5 * (cognitive_best - self.current_position) + \
                              0.5 * (social_influence - self.current_position))
        
        # Update position
        self.current_position += self.current_velocity
        
    def evaluate_fitness(self, fitness_function) -> float:
        """Evaluate particle fitness"""
        fitness = fitness_function(self.current_position)
        
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.current_position.copy()
            
        self.cognitive_state.update_memory(self.current_position, fitness)
        return fitness

class QCSE:
    """Quantum-Inspired Cognitive Swarm Evolution"""
    
    def __init__(self, config: QCSEConfig, fitness_function):
        self.config = config
        self.fitness_function = fitness_function
        self.particles = [QCSEParticle(config) for _ in range(config.n_particles)]
        self.topology = SwarmTopology(config.n_particles, config.interaction_strength)
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        self.history = []
        
    def optimize(self, n_iterations: int) -> Tuple[np.ndarray, float, List[float]]:
        """Run optimization"""
        logger.info("Starting QCSE optimization...")
        
        for iteration in range(n_iterations):
            # Evaluate all particles
            positions = np.array([p.current_position for p in self.particles])
            fitness_values = np.array([p.evaluate_fitness(self.fitness_function) 
                                     for p in self.particles])
            
            # Update global best
            best_idx = np.argmax(fitness_values)
            if fitness_values[best_idx] > self.global_best_fitness:
                self.global_best_fitness = fitness_values[best_idx]
                self.global_best_position = positions[best_idx].copy()
            
            # Update topology
            self.topology.update_topology(fitness_values)
            
            # Update particles
            for i, particle in enumerate(self.particles):
                particle.update(
                    self.global_best_position,
                    self.topology.topology_matrix[i],
                    positions,
                    self.config.learning_rate
                )
            
            # Record history
            self.history.append(self.global_best_fitness)
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Best Fitness = {self.global_best_fitness:.4f}")
        
        return self.global_best_position, self.global_best_fitness, self.history
    
    def plot_convergence(self):
        """Plot convergence history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.title('QCSE Convergence History')
        plt.grid(True)
        plt.show()

def example_fitness_function(x: np.ndarray) -> float:
    """Example fitness function (maximize negative squared distance from origin)"""
    return -np.sum(x**2)

def main():
    """Main function demonstrating QCSE usage"""
    
    # Configuration
    config = QCSEConfig(
        n_particles=1643,
        n_dimensions=10,
        cognitive_depth=4,
        quantum_layers=6,
        learning_rate=0.01,
        interaction_strength=0.1,
        entanglement_decay=0.95,
        memory_size=1000
    )
    
    # Initialize optimizer
    qcse = QCSE(config, example_fitness_function)
    
    # Run optimization
    best_position, best_fitness, history = qcse.optimize(n_iterations=100)
    
    # Print results
    logger.info("\nOptimization Results:")
    logger.info(f"Best Fitness: {best_fitness:.4f}")
    logger.info(f"Best Position Norm: {np.linalg.norm(best_position):.4f}")
    
    # Plot convergence
    qcse.plot_convergence()

if __name__ == "__main__":
    main()
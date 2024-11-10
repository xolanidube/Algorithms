import numpy as np
from typing import List, Dict, Tuple, Optional
import heapq
from dataclasses import dataclass
import random

@dataclass
class ResourceNode:
    """Represents a node in the distributed system with its resources."""
    id: int
    cpu_capacity: float
    memory_capacity: float
    network_bandwidth: float
    current_load: float
    location: Tuple[float, float]  # Geographical coordinates
    
class QuantumState:
    """Represents a quantum-inspired state for optimization."""
    def __init__(self, num_nodes: int, num_dimensions: int):
        # Initialize with complex numbers
        self.amplitudes = np.random.uniform(0, 1, (num_nodes, num_dimensions)).astype(np.complex128)
        self.normalize()
    
    def normalize(self):
        """Normalize the quantum state."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes /= norm
    
    def collapse(self) -> np.ndarray:
        """Collapse the quantum state to get a classical solution."""
        # Get real probabilities from complex amplitudes
        probabilities = np.square(np.abs(self.amplitudes))
        # Normalize across nodes
        return probabilities / np.sum(probabilities, axis=0)

class QIANOOptimizer:
    """Quantum-Inspired Adaptive Network Optimizer"""
    
    def __init__(self, 
                 nodes: List[ResourceNode],
                 learning_rate: float = 0.01,
                 quantum_iterations: int = 100):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.learning_rate = learning_rate
        self.quantum_iterations = quantum_iterations
        self.quantum_state = QuantumState(self.num_nodes, 3)  # 3 dimensions: CPU, Memory, Network
        self.history: List[Dict] = []
        
    def adaptive_phase_estimation(self, workload: Dict[str, float]) -> np.ndarray:
        """
        Perform adaptive phase estimation based on current workload requirements.
        """
        total_cpu = sum(node.cpu_capacity for node in self.nodes)
        total_memory = sum(node.memory_capacity for node in self.nodes)
        total_bandwidth = sum(node.network_bandwidth for node in self.nodes)
        
        normalized_workload = {
            'cpu': workload['cpu'] / total_cpu,
            'memory': workload['memory'] / total_memory,
            'network': workload['network'] / total_bandwidth
        }
        
        phases = np.zeros(self.num_nodes, dtype=np.complex128)
        for i, node in enumerate(self.nodes):
            node_capacity = np.array([
                node.cpu_capacity / total_cpu,
                node.memory_capacity / total_memory,
                node.network_bandwidth / total_bandwidth
            ])
            
            # Calculate phase based on capacity matching
            phase = np.dot(node_capacity, list(normalized_workload.values()))
            phases[i] = np.exp(1j * phase * np.pi)  # Convert to complex rotation
            
        return phases
    
    def quantum_rotation(self, phases: np.ndarray):
        """Apply quantum rotation gates to the quantum state."""
        # Reshape phases for broadcasting
        phases_reshaped = phases[:, np.newaxis]
        # Apply rotation
        self.quantum_state.amplitudes = self.quantum_state.amplitudes * phases_reshaped
        self.quantum_state.normalize()
    
    def calculate_fitness(self, allocation: np.ndarray, workload: Dict[str, float]) -> float:
        """Calculate the fitness of a given resource allocation."""
        fitness = 0.0
        
        for i, node in enumerate(self.nodes):
            if np.any(allocation[i] > 0):
                # Calculate resource utilization efficiency
                cpu_util = min(workload['cpu'] * allocation[i, 0] / node.cpu_capacity, 1.0)
                mem_util = min(workload['memory'] * allocation[i, 1] / node.memory_capacity, 1.0)
                net_util = min(workload['network'] * allocation[i, 2] / node.network_bandwidth, 1.0)
                
                # Calculate balanced resource utilization score
                utilization_score = (cpu_util + mem_util + net_util) / 3
                
                # Consider node's current load
                load_penalty = node.current_load / 100
                
                # Final node fitness considering both utilization and load
                node_fitness = utilization_score * (1 - load_penalty)
                fitness += node_fitness
        
        return fitness / np.sum(allocation > 0) if np.sum(allocation > 0) > 0 else 0
    
    def optimize(self, workload: Dict[str, float]) -> Dict[int, float]:
        """
        Optimize resource allocation for given workload.
        """
        best_fitness = 0.0
        best_allocation = None
        
        for _ in range(self.quantum_iterations):
            # Perform adaptive phase estimation
            phases = self.adaptive_phase_estimation(workload)
            
            # Apply quantum rotation
            self.quantum_rotation(phases)
            
            # Collapse quantum state to get classical solution
            allocation = self.quantum_state.collapse()
            
            # Calculate fitness
            fitness = self.calculate_fitness(allocation, workload)
            
            # Update best solution
            if fitness > best_fitness:
                best_fitness = fitness
                best_allocation = allocation.copy()
            
            # Record history
            self.history.append({
                'fitness': fitness,
                'allocation': allocation.copy()
            })
        
        # Convert best allocation to node mapping
        result = {}
        if best_allocation is not None:
            # Take the average allocation across dimensions for each node
            node_allocations = np.mean(best_allocation, axis=1)
            for i, alloc in enumerate(node_allocations):
                if alloc > 0.01:  # Threshold to avoid tiny allocations
                    result[self.nodes[i].id] = float(alloc)
        
        return result

    def get_optimization_metrics(self) -> Dict:
        """Return metrics about the optimization process."""
        fitness_history = [h['fitness'] for h in self.history]
        return {
            'convergence_rate': (max(fitness_history) - min(fitness_history)) / len(fitness_history),
            'final_fitness': fitness_history[-1] if fitness_history else 0,
            'iterations': len(fitness_history),
            'average_fitness': sum(fitness_history) / len(fitness_history) if fitness_history else 0
        }

def create_test_scenario():
    """Create a test scenario with sample nodes and workload."""
    nodes = [
        ResourceNode(id=1, cpu_capacity=100, memory_capacity=32, network_bandwidth=1000,
                    current_load=30, location=(0, 0)),
        ResourceNode(id=2, cpu_capacity=200, memory_capacity=64, network_bandwidth=2000,
                    current_load=50, location=(1, 1)),
        ResourceNode(id=3, cpu_capacity=150, memory_capacity=48, network_bandwidth=1500,
                    current_load=20, location=(2, 2))
    ]
    
    workload = {
        'cpu': 80,  # CPU units required
        'memory': 16,  # GB of memory required
        'network': 800  # Mbps network bandwidth required
    }
    
    return nodes, workload

def run_test():
    """Run a test of the QIANO algorithm."""
    print("Starting QIANO algorithm test...")
    
    # Create test scenario
    nodes, workload = create_test_scenario()
    
    print("\nTest scenario created with:")
    print(f"Number of nodes: {len(nodes)}")
    print(f"Workload requirements: {workload}")
    
    # Initialize optimizer
    optimizer = QIANOOptimizer(nodes, learning_rate=0.01, quantum_iterations=100)
    
    print("\nRunning optimization...")
    # Run optimization
    allocation = optimizer.optimize(workload)
    
    # Get metrics
    metrics = optimizer.get_optimization_metrics()
    
    # Print results
    print("\nResource Allocation Results:")
    total_allocation = 0
    for node_id, alloc in allocation.items():
        percentage = alloc * 100
        total_allocation += percentage
        print(f"Node {node_id}: {percentage:.2f}% of workload")
    print(f"Total allocation: {total_allocation:.2f}%")
    
    print("\nOptimization Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    run_test()
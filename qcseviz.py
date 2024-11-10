import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

class QCSEVisualizer:
    """Visualization tools for QCSE algorithm"""
    
    def __init__(self, qcse_instance):
        self.qcse = qcse_instance
        plt.style.use('seaborn')
        
    def plot_quantum_states(self):
        """Visualize quantum states of particles"""
        fig = plt.figure(figsize=(12, 6))
        
        # Plot amplitude distribution
        ax1 = fig.add_subplot(121)
        amplitudes = np.array([p.quantum_state.get_probability_distribution() 
                             for p in self.qcse.particles])
        sns.heatmap(amplitudes.mean(axis=0), ax=ax1, cmap='viridis')
        ax1.set_title('Average Quantum Amplitude Distribution')
        ax1.set_xlabel('Quantum Layer')
        ax1.set_ylabel('Dimension')
        
        # Plot phase distribution
        ax2 = fig.add_subplot(122)
        phases = np.array([p.quantum_state.phase for p in self.qcse.particles])
        sns.heatmap(phases.mean(axis=0), ax=ax2, cmap='twilight')
        ax2.set_title('Average Quantum Phase Distribution')
        ax2.set_xlabel('Quantum Layer')
        ax2.set_ylabel('Dimension')
        
        plt.tight_layout()
        plt.show()
        
    def plot_swarm_topology(self):
        """Visualize swarm interaction topology"""
        plt.figure(figsize=(10, 10))
        G = nx.from_numpy_array(self.qcse.topology.topology_matrix)
        
        # Calculate node sizes based on particle fitness
        fitness_values = np.array([p.best_fitness for p in self.qcse.particles])
        node_sizes = 100 + 1000 * (fitness_values - fitness_values.min()) / \
                    (fitness_values.max() - fitness_values.min())
        
        # Calculate edge weights
        edge_weights = self.qcse.topology.topology_matrix[np.triu_indices(len(G), k=1)]
        edge_weights = edge_weights[edge_weights > 0]
        
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                             node_color=fitness_values, cmap='viridis')
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=edge_weights*2)
        
        plt.title('Swarm Interaction Topology')
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'))
        plt.axis('off')
        plt.show()
        
    def plot_particle_trajectories(self, dimensions: List[int] = [0, 1, 2]):
        """Visualize particle trajectories in 3D"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get position history for selected dimensions
        positions = np.array([p.current_position[dimensions] for p in self.qcse.particles])
        
        # Plot current positions
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='blue', marker='o', label='Current Positions')
        
        # Plot global best
        best_pos = self.qcse.global_best_position[dimensions]
        ax.scatter([best_pos[0]], [best_pos[1]], [best_pos[2]], 
                  c='red', marker='*', s=200, label='Global Best')
        
        ax.set_xlabel(f'Dimension {dimensions[0]}')
        ax.set_ylabel(f'Dimension {dimensions[1]}')
        ax.set_zlabel(f'Dimension {dimensions[2]}')
        ax.set_title('Particle Positions in Solution Space')
        ax.legend()
        
        plt.show()
        
    def plot_cognitive_memory(self):
        """Visualize cognitive memory distribution"""
        plt.figure(figsize=(12, 6))
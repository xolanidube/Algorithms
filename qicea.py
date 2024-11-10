# Import necessary libraries
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumCognitiveState:
    """
    Represents a quantum cognitive state in the combined Hilbert space.
    """
    def __init__(self, n_qubits: int, n_cognitive_dims: int):
        self.n_qubits = n_qubits
        self.n_cognitive_dims = n_cognitive_dims
        self.quantum_state = self._initialize_quantum_state()
        self.cognitive_state = self._initialize_cognitive_state()
        self.phase = 0.0
        
    def _initialize_quantum_state(self) -> np.ndarray:
        """Initialize quantum state vector"""
        state = np.random.rand(2**self.n_qubits).astype(np.complex128)
        # Calculate the norm of the complex state array
        norm = np.linalg.norm(state)
        # Normalize by dividing each element by the norm
        state = (state / norm).reshape(-1)
        return state
    
    def _initialize_cognitive_state(self) -> np.ndarray:
        """Initialize cognitive state vector"""
        state = np.random.rand(self.n_cognitive_dims).astype(np.complex128)
        # Calculate the norm of the complex state array
        norm = np.linalg.norm(state)
        # Normalize by dividing each element by the norm
        state = (state / norm).reshape(-1)
        return state
    
    def _initialize_cognitive_state(self) -> np.ndarray:
        """Initialize cognitive state vector"""
        state = np.random.rand(self.n_cognitive_dims).astype(np.complex128)
        # Calculate the norm of the complex state array
        norm = np.linalg.norm(state)
        # Normalize by dividing each element by the norm
        state = (state / norm).reshape(-1)
        return state
    
    def apply_quantum_gate(self, gate: np.ndarray) -> None:
        """Apply quantum gate operation"""
        self.quantum_state = np.dot(gate, self.quantum_state)
        # Calculate the norm of the complex state array
        norm = np.linalg.norm(self.quantum_state)
        # Normalize by dividing each element by the norm
        self.quantum_state = (self.quantum_state / norm).reshape(-1)
    
    def apply_cognitive_operator(self, operator: np.ndarray) -> None:
        """Apply cognitive evolution operator"""
        self.cognitive_state = np.dot(operator, self.cognitive_state)
        # Calculate the norm of the complex state array
        norm = np.linalg.norm(self.cognitive_state)
        # Normalize by dividing each element by the norm
        self.cognitive_state = (self.cognitive_state / norm).reshape(-1)
        
    def update_phase(self, delta_phase: float) -> None:
        """Update quantum phase"""
        self.phase += delta_phase
        self.phase %= 2 * np.pi

class NeuralEvolutionOperator(nn.Module):
    """
    Neural network-based evolution operator for cognitive state transformation.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class QICEA:
    """
    Quantum-Inspired Cognitive Evolution Algorithm implementation.
    """
    def __init__(self, 
                 n_qubits: int, 
                 n_cognitive_dims: int, 
                 population_size: int,
                 learning_rate: float = 0.01):
        self.n_qubits = n_qubits
        self.n_cognitive_dims = n_cognitive_dims
        self.population_size = population_size
        self.learning_rate = learning_rate
        
        # Initialize population
        self.population = [
            QuantumCognitiveState(n_qubits, n_cognitive_dims) 
            for _ in range(population_size)
        ]
        
        # Initialize neural evolution operator
        self.neo = NeuralEvolutionOperator(n_cognitive_dims, n_cognitive_dims * 2)
        self.optimizer = torch.optim.Adam(self.neo.parameters(), lr=learning_rate)
        
        # Initialize quantum gates
        self.hadamard = self._create_hadamard_gate()
        self.phase_gate = self._create_phase_gate()
        
    def _create_hadamard_gate(self) -> np.ndarray:
        """Create Hadamard gate matrix"""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        return np.kron(H, np.eye(2**(self.n_qubits-1)))
    
    def _create_phase_gate(self) -> np.ndarray:
        """Create phase gate matrix"""
        P = np.array([[1, 0], [0, np.exp(1j * np.pi/4)]])
        return np.kron(P, np.eye(2**(self.n_qubits-1)))
    
    def quantum_cognitive_superposition(self, state: QuantumCognitiveState) -> QuantumCognitiveState:
        """Apply quantum cognitive superposition"""
        # Apply Hadamard gate to create superposition
        state.apply_quantum_gate(self.hadamard)
        # Apply phase shift
        state.apply_quantum_gate(self.phase_gate)
        return state
    
    def neural_evolution_step(self, state: QuantumCognitiveState) -> QuantumCognitiveState:
        """Apply neural evolution operator"""
        # Convert cognitive state to torch tensor
        cognitive_tensor = torch.FloatTensor(state.cognitive_state)
        
        # Apply neural evolution
        with torch.no_grad():
            evolved_state = self.neo(cognitive_tensor).numpy()
        
        # Update cognitive state
        state.cognitive_state = normalize(evolved_state.reshape(-1, 1), norm='l2').reshape(-1)
        return state
    
    def adaptive_dimension_transform(self, state: QuantumCognitiveState) -> QuantumCognitiveState:
        """Apply adaptive dimension transformation"""
        # Create transformation matrix
        transform = np.random.rand(self.n_cognitive_dims, self.n_cognitive_dims)
        transform = normalize(transform, norm='l2', axis=0)
        
        # Apply transformation
        state.apply_cognitive_operator(transform)
        return state
    
    def evaluate_fitness(self, state: QuantumCognitiveState, protein_sequence: str) -> float:
        """
        Evaluate the fitness of a quantum cognitive state for protein folding.
        """
        # Convert quantum and cognitive states to protein conformation
        conformation = self._state_to_conformation(state, len(protein_sequence))
        
        # Calculate energy score (lower is better)
        energy_score = self._calculate_energy(conformation, protein_sequence)
        
        return -energy_score  # Negative because we want to minimize energy
    
    def _state_to_conformation(self, state: QuantumCognitiveState, seq_length: int) -> List[Tuple[float, float, float]]:
        """Convert quantum cognitive state to 3D protein conformation"""
        # Use quantum state for angles and cognitive state for spatial organization
        angles = 2 * np.pi * np.abs(state.quantum_state[:seq_length])
        spatial = state.cognitive_state[:seq_length * 3].reshape(-1, 3)
        
        # Combine to create 3D coordinates
        coordinates = []
        current_pos = np.array([0., 0., 0.])
        
        for i in range(seq_length):
            theta = angles[i]
            direction = spatial[i]
            
            # Normalize direction vector directly
            norm = np.linalg.norm(direction)
            if norm != 0:
                direction = direction / norm

            # Calculate new position
            new_pos = current_pos + direction
            coordinates.append(tuple(new_pos))
            current_pos = new_pos
            
        return coordinates

    
    def _calculate_energy(self, conformation: List[Tuple[float, float, float]], sequence: str) -> float:
        """Calculate energy score for protein conformation"""
        energy = 0.0
        coords = np.array(conformation)
        
        # Calculate pairwise distances
        distances = np.sqrt(np.sum((coords[:, np.newaxis] - coords[np.newaxis, :]) ** 2, axis=2))
        
        # Simple energy model based on hydrophobic interactions
        for i in range(len(sequence)):
            for j in range(i + 2, len(sequence)):
                if sequence[i] in 'FILVWY' and sequence[j] in 'FILVWY':  # Hydrophobic residues
                    # Lennard-Jones potential
                    r = distances[i, j]
                    if r > 0:
                        energy += 4 * ((1/r)**12 - (1/r)**6)
                        
        return energy
    
    def train(self, 
              protein_sequence: str, 
              n_generations: int, 
              n_elite: int = 2) -> Tuple[QuantumCognitiveState, float]:
        """
        Train the QICEA algorithm on protein folding problem.
        """
        best_fitness = float('-inf')
        best_state = None
        
        for generation in range(n_generations):
            # Evaluate population
            fitness_scores = [
                self.evaluate_fitness(state, protein_sequence) 
                for state in self.population
            ]
            
            # Sort population by fitness
            sorted_population = [x for _, x in sorted(
                zip(fitness_scores, self.population), 
                key=lambda pair: pair[0],
                reverse=True
            )]
            
            # Update best solution
            if max(fitness_scores) > best_fitness:
                best_fitness = max(fitness_scores)
                best_state = sorted_population[0]
                
            # Keep elite solutions
            new_population = sorted_population[:n_elite]
            
            # Generate new solutions
            while len(new_population) < self.population_size:
                # Select parent
                parent = np.random.choice(sorted_population[:n_elite])
                
                # Create child through evolution
                child = QuantumCognitiveState(self.n_qubits, self.n_cognitive_dims)
                child = self.quantum_cognitive_superposition(child)
                child = self.neural_evolution_step(child)
                child = self.adaptive_dimension_transform(child)
                
                new_population.append(child)
                
            self.population = new_population
            
            # Log progress
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best Fitness = {best_fitness}")
        
        return best_state, best_fitness

# Example usage
def main():
    # Example protein sequence (small peptide)
    protein_sequence = "MVKVYAPA"
    
    # Initialize QICEA
    qicea = QICEA(
        n_qubits=len(protein_sequence),  # Number of qubits (based on sequence length)
        n_cognitive_dims=24,  # 3D coordinates for each residue
        population_size=100
    )
    
    # Train algorithm
    best_state, best_fitness = qicea.train(
        protein_sequence=protein_sequence,
        n_generations=100,
        n_elite=2
    )
    
    # Get final conformation
    final_conformation = qicea._state_to_conformation(best_state, len(protein_sequence))
    
    logger.info(f"Training completed!")
    logger.info(f"Best fitness achieved: {best_fitness}")
    logger.info(f"Final conformation coordinates:")
    for i, coord in enumerate(final_conformation):
        logger.info(f"Residue {i+1} ({protein_sequence[i]}): {coord}")

if __name__ == "__main__":
    main()
    
    
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List, Tuple, Dict
import seaborn as sns

class QICEAVisualizer:
    def __init__(self):
        """Initialize the visualizer with default settings"""
        plt.style.use('seaborn')
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def plot_conformation(self, conformation: List[Tuple[float, float, float]], sequence: str):
        """
        Plot the 3D protein conformation
        
        Args:
            conformation: List of 3D coordinates for each residue
            sequence: Protein sequence string
        """
        # Convert conformation to numpy array for easier manipulation
        coords = np.array(conformation)
        
        # Plot the backbone
        self.ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'b-', label='Backbone', linewidth=2)
        
        # Plot individual residues
        for i, (coord, residue) in enumerate(zip(coords, sequence)):
            # Color based on residue type
            color = self._get_residue_color(residue)
            self.ax.scatter(coord[0], coord[1], coord[2], 
                          c=color, s=100, label=f'{residue}{i+1}',
                          edgecolor='black', linewidth=1)
            
            # Add residue labels
            self.ax.text(coord[0], coord[1], coord[2], f'{residue}{i+1}',
                        fontsize=8, fontweight='bold')
        
        # Set labels and title
        self.ax.set_xlabel('X (Å)', fontsize=12, labelpad=10)
        self.ax.set_ylabel('Y (Å)', fontsize=12, labelpad=10)
        self.ax.set_zlabel('Z (Å)', fontsize=12, labelpad=10)
        self.ax.set_title('Protein Conformation', fontsize=14, pad=20)
        
        # Add legend
        handles, labels = self.ax.get_legend_handles_labels()
        unique_labels = {l:h for l,h in zip(labels, handles)}
        self.ax.legend(unique_labels.values(), unique_labels.keys(),
                      bbox_to_anchor=(1.15, 1), loc='upper left')
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([1,1,1])
        
    def _get_residue_color(self, residue: str) -> str:
        """
        Get color based on residue type and properties
        
        Args:
            residue: Single letter amino acid code
            
        Returns:
            Color code for the residue
        """
        # Color scheme based on residue properties
        color_map = {
            # Hydrophobic
            'A': '#FF4D4D', 'V': '#FF6B6B', 'L': '#FF8787', 
            'I': '#FFA3A3', 'M': '#FFBFBF', 'F': '#FFD4D4', 
            'W': '#FFEBEB', 'P': '#FF9999',
            # Polar
            'S': '#4CAF50', 'T': '#66BB6A', 'N': '#81C784', 
            'Q': '#A5D6A7', 'Y': '#C8E6C9',
            # Basic
            'K': '#2196F3', 'R': '#42A5F5', 'H': '#64B5F6',
            # Acidic
            'D': '#FFC107', 'E': '#FFCA28',
            # Others
            'G': '#9C27B0', 'C': '#FF9800'
        }
        return color_map.get(residue, '#808080')
    
    def plot_energy_landscape(self, energies: List[float], generations: int):
        """
        Plot the energy landscape during optimization
        
        Args:
            energies: List of energy values during optimization
            generations: Number of generations
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(generations), energies, 'b-', linewidth=2)
        plt.fill_between(range(generations), energies, alpha=0.2)
        
        # Add rolling average
        window = min(10, len(energies)//5)
        rolling_avg = np.convolve(energies, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, generations), rolling_avg, 'r--', 
                linewidth=2, label=f'{window}-gen Rolling Average')
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Energy (kcal/mol)', fontsize=12)
        plt.title('Energy Landscape During Optimization', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
    def plot_contact_map(self, conformation: List[Tuple[float, float, float]], 
                        sequence: str, contact_threshold: float = 8.0):
        """
        Plot the contact map for the protein conformation
        
        Args:
            conformation: List of 3D coordinates
            sequence: Protein sequence
            contact_threshold: Distance threshold for contacts (in Angstroms)
        """
        coords = np.array(conformation)
        n_residues = len(sequence)
        
        # Calculate distance matrix
        distances = np.zeros((n_residues, n_residues))
        for i in range(n_residues):
            for j in range(n_residues):
                distances[i,j] = np.linalg.norm(coords[i] - coords[j])
        
        # Create contact map
        contact_map = distances <= contact_threshold
        
        # Plot
        plt.figure(figsize=(8, 8))
        sns.heatmap(contact_map, cmap='coolwarm', square=True,
                   xticklabels=list(sequence),
                   yticklabels=list(sequence))
        plt.title('Protein Contact Map', fontsize=14)
        plt.xlabel('Residue', fontsize=12)
        plt.ylabel('Residue', fontsize=12)
        
    def plot_ramachandran(self, conformation: List[Tuple[float, float, float]]):
        """
        Plot Ramachandran plot for the protein conformation
        
        Args:
            conformation: List of 3D coordinates
        """
        coords = np.array(conformation)
        n_residues = len(coords)
        
        # Calculate phi and psi angles
        phi = []
        psi = []
        
        for i in range(1, n_residues-1):
            # Calculate phi angle (C-N-CA-C)
            v1 = coords[i-1] - coords[i]
            v2 = coords[i+1] - coords[i]
            phi.append(np.arctan2(np.linalg.norm(np.cross(v1, v2)),
                                np.dot(v1, v2)) * 180/np.pi)
            
            # Calculate psi angle (N-CA-C-N)
            if i < n_residues-2:
                v3 = coords[i+1] - coords[i]
                v4 = coords[i+2] - coords[i+1]
                psi.append(np.arctan2(np.linalg.norm(np.cross(v3, v4)),
                                    np.dot(v3, v4)) * 180/np.pi)
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(phi, psi, c='blue', alpha=0.6)
        plt.xlabel('Phi (degrees)', fontsize=12)
        plt.ylabel('Psi (degrees)', fontsize=12)
        plt.title('Ramachandran Plot', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add typical regions
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
    def save_plot(self, filename: str):
        """
        Save the current plot to a file
        
        Args:
            filename: Output filename
        """
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

class QICEAAnalyzer:
    """Analyzer class for QICEA results"""
    
    def __init__(self, conformation: List[Tuple[float, float, float]], 
                 sequence: str, energies: List[float]):
        self.conformation = np.array(conformation)
        self.sequence = sequence
        self.energies = energies
        
    def analyze_convergence(self) -> Dict:
        """Analyze convergence of the optimization"""
        results = {
            'final_energy': self.energies[-1],
            'initial_energy': self.energies[0],
            'energy_improvement': self.energies[0] - self.energies[-1],
            'convergence_gen': self._find_convergence_generation(),
            'stability_score': self._calculate_stability()
        }
        return results
    
    def _find_convergence_generation(self) -> int:
        """Find the generation where convergence occurred"""
        window = min(10, len(self.energies)//5)
        rolling_std = np.std(
            [self.energies[i:i+window] 
             for i in range(len(self.energies)-window)],
            axis=1
        )
        # Find where standard deviation becomes small
        threshold = np.mean(rolling_std) * 0.1
        conv_gen = np.where(rolling_std < threshold)[0]
        return conv_gen[0] if len(conv_gen) > 0 else len(self.energies)
    
    def _calculate_stability(self) -> float:
        """Calculate stability score based on contacts"""
        coords = np.array(self.conformation)
        distances = np.linalg.norm(
            coords[:, np.newaxis] - coords[np.newaxis, :],
            axis=2
        )
        # Count stable contacts
        contacts = (distances < 8.0).sum() - len(self.sequence)  # Exclude self-contacts
        return contacts / (len(self.sequence) * (len(self.sequence)-1)/2)

def main():
    # Example usage
    protein_sequence = "MVKVGVNG"
    qicea = QICEA(n_qubits=8, n_cognitive_dims=24, population_size=50)
    
    # Train
    best_state, best_fitness = qicea.train(
        protein_sequence=protein_sequence,
        n_generations=100,
        n_elite=2
    )
    
    # Get final conformation
    final_conformation = qicea._state_to_conformation(best_state, len(protein_sequence))
    
    # Create visualizer
    visualizer = QICEAVisualizer()
    
    # Plot protein conformation
    visualizer.plot_conformation(final_conformation, protein_sequence)
    visualizer.save_plot('protein_conformation.png')
    
    # Plot energy landscape
    #visualizer.plot_energy_landscape(qicea.energy_history, 100)
    #visualizer.save_plot('energy_landscape.png')
    
    # Plot contact map
    visualizer.plot_contact_map(final_conformation, protein_sequence)
    visualizer.save_plot('contact_map.png')
    
    # Plot Ramachandran plot
    visualizer.plot_ramachandran(final_conformation)
    visualizer.save_plot('ramachandran_plot.png')
    
    # Analyze results
    analyzer = QICEAAnalyzer(final_conformation, protein_sequence, qicea.energy_history)
    results = analyzer.analyze_convergence()
    
    print("\nAnalysis Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
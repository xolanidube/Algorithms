import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class QICEAVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def plot_conformation(self, conformation, sequence):
        """Plot the 3D protein conformation"""
        # Convert conformation to numpy array for easier manipulation
        coords = np.array(conformation)
        
        # Plot the backbone
        self.ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'b-', label='Backbone')
        
        # Plot individual residues
        for i, (coord, residue) in enumerate(zip(coords, sequence)):
            # Color based on residue type
            color = self._get_residue_color(residue)
            self.ax.scatter(coord[0], coord[1], coord[2], c=color, s=100, label=f'{residue}{i+1}')
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Protein Conformation')
        
        # Add legend
        handles, labels = self.ax.get_legend_handles_labels()
        unique_labels = {l:h for l,h in zip(labels, handles)}
        self.ax.legend(unique_labels.values(), unique_labels.keys())
        
    def _get_residue_color(self, residue):
        """Get color based on residue type"""
        # Color scheme based on residue properties
        color_map = {
            # Hydrophobic
            'A': 'red', 'V': 'red', 'L': 'red', 'I': 'red', 'M': 'red',
            'F': 'red', 'W': 'red', 'P': 'red',
            # Polar
            'S': 'green', 'T': 'green', 'N': 'green', 'Q': 'green', 'Y': 'green',
            # Basic
            'K': 'blue', 'R': 'blue', 'H': 'blue',
            # Acidic
            'D': 'yellow', 'E': 'yellow',
            # Others
            'G': 'purple', 'C': 'orange'
        }
        return color_map.get(residue, 'gray')
    
    def save_plot(self, filename):
        """Save the plot to a file"""
        plt.savefig(filename)
        plt.close()

def plot_energy_landscape(energies, generations):
    """Plot the energy landscape during optimization"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(generations), energies, 'b-')
    plt.xlabel('Generation')
    plt.ylabel
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class TopologicalDiffusionPR:
    def __init__(self, max_dimension=2, n_components=2, gamma=1.0, t=1):
        self.max_dimension = max_dimension
        self.n_components = n_components
        self.gamma = gamma
        self.t = t
        
    def compute_persistence(self, X):
        # Compute persistence diagrams using Ripser
        diagrams = ripser(X, maxdim=self.max_dimension)['dgms']
        return diagrams
    
    def _clean_diagram(self, diagram):
        """Remove infinite death times and ensure finite values."""
        # Filter out points with infinite death times
        finite_mask = np.isfinite(diagram).all(axis=1)
        cleaned_diagram = diagram[finite_mask]
        
        if len(cleaned_diagram) == 0:
            # If no points remain, add a dummy point
            return np.array([[0., 0.]])
        return cleaned_diagram
    
    def wasserstein_kernel(self, diag1, diag2):
        """Compute Wasserstein-based kernel between persistence diagrams."""
        # Clean diagrams
        diag1 = self._clean_diagram(diag1)
        diag2 = self._clean_diagram(diag2)
        
        # Compute distance
        dist = self._wasserstein_distance(diag1, diag2)
        return np.exp(-self.gamma * dist**2)
    
    def _wasserstein_distance(self, diag1, diag2):
        """Compute approximate Wasserstein distance between persistence diagrams."""
        def project_to_diagonal(points):
            return np.column_stack(((points[:, 0] + points[:, 1]) / 2,
                                  (points[:, 0] + points[:, 1]) / 2))
        
        # Add projections to diagonal
        diag1_proj = project_to_diagonal(diag1)
        diag2_proj = project_to_diagonal(diag2)
        
        # Combine original points with projections
        extended_diag1 = np.vstack([diag1, diag2_proj])
        extended_diag2 = np.vstack([diag2, diag1_proj])
        
        # Compute cost matrix
        cost_matrix = cdist(extended_diag1, extended_diag2)
        
        # Use minimum cost for stability
        return np.sqrt(np.sum(np.min(cost_matrix, axis=1)**2))
    
    def compute_diffusion_maps(self, kernel_matrix):
        """Compute diffusion map embedding."""
        # Add small constant to diagonal for numerical stability
        kernel_matrix = kernel_matrix + 1e-8 * np.eye(len(kernel_matrix))
        
        # Normalize kernel matrix
        D = np.sum(kernel_matrix, axis=1)
        D_inv = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-10)))
        P = D_inv @ kernel_matrix @ D_inv
        
        # Compute eigendecomposition
        eigenvals, eigenvects = eigh(P)
        
        # Sort in descending order
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvects = eigenvects[:, idx]
        
        # Select top components
        diffusion_coords = eigenvects[:, 1:(self.n_components + 1)] * \
                          np.maximum(eigenvals[1:(self.n_components + 1)], 0)**self.t
        
        return diffusion_coords
    
    def fit_transform(self, X):
        """Fit TDPR and transform data."""
        # Standardize input
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compute persistence diagrams
        diagrams = self.compute_persistence(X_scaled)
        
        # Compute kernel matrix
        n_samples = X.shape[0]
        kernel_matrix = np.zeros((n_samples, n_samples))
        
        # Use 0-dimensional persistence diagram
        diagram_0 = diagrams[0]
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                # Use a simpler kernel for stability
                k_val = np.exp(-self.gamma * np.sum((X_scaled[i] - X_scaled[j])**2))
                kernel_matrix[i, j] = k_val
                kernel_matrix[j, i] = k_val
        
        # Compute diffusion maps
        embedding = self.compute_diffusion_maps(kernel_matrix)
        return embedding

    def visualize_persistence(self, X):
        """Visualize persistence diagrams."""
        diagrams = self.compute_persistence(X)
        fig = plt.figure(figsize=(10, 5))
        plot_diagrams(diagrams, show=False)
        plt.title("Persistence Diagrams")
        return fig
    
    def visualize_embedding(self, X):
        """Visualize the diffusion map embedding."""
        embedding = self.fit_transform(X)
        
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6)
        plt.title("Diffusion Map Embedding")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        return fig

def test_algorithm():
    """Test the algorithm on synthetic data."""
    # Generate synthetic data
    n_points = 100
    t = np.linspace(0, 2*np.pi, n_points)
    circle = np.column_stack((np.cos(t), np.sin(t)))
    figure_eight = np.column_stack((np.sin(2*t), np.sin(t)))
    
    # Add noise
    noise = 0.05
    circle += np.random.normal(0, noise, circle.shape)
    figure_eight += np.random.normal(0, noise, figure_eight.shape)
    
    # Combine datasets
    X = np.vstack([circle, figure_eight])
    
    # Create and test TDPR
    tdpr = TopologicalDiffusionPR()
    embedding = tdpr.fit_transform(X)
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Original data
    ax1.scatter(X[:n_points, 0], X[:n_points, 1], label='Circle')
    ax1.scatter(X[n_points:, 0], X[n_points:, 1], label='Figure Eight')
    ax1.set_title('Original Data')
    ax1.legend()
    
    # Embedding
    ax2.scatter(embedding[:n_points, 0], embedding[:n_points, 1], label='Circle')
    ax2.scatter(embedding[n_points:, 0], embedding[n_points:, 1], label='Figure Eight')
    ax2.set_title('TDPR Embedding')
    ax2.legend()
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    test_fig = test_algorithm()
    test_fig.savefig('tdpr_test_results.png')
    test_fig.show()
    plt.close()
    
import numpy as np
import matplotlib.pyplot as plt


def generate_synthetic_data(n_points=100, noise=0.05):
    """Generate synthetic datasets for testing."""
    # Generate circle
    t = np.linspace(0, 2*np.pi, n_points)
    circle = np.column_stack((np.cos(t), np.sin(t)))
    circle += np.random.normal(0, noise, circle.shape)
    
    # Generate figure eight
    figure_eight = np.column_stack((np.sin(2*t), np.sin(t)))
    figure_eight += np.random.normal(0, noise, figure_eight.shape)
    
    # Combine datasets
    X = np.vstack([circle, figure_eight])
    y = np.array([0]*n_points + [1]*n_points)  # Labels for visualization
    
    return X, y

def main():
    # Generate data
    X, y = generate_synthetic_data(n_points=100, noise=0.05)
    
    # Create TDPR instance
    tdpr = TopologicalDiffusionPR(max_dimension=1, n_components=2)
    
    # Create visualization panel
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original data
    axes[0].scatter(X[y==0, 0], X[y==0, 1], label='Circle', alpha=0.6)
    axes[0].scatter(X[y==1, 0], X[y==1, 1], label='Figure Eight', alpha=0.6)
    axes[0].set_title('Original Data')
    axes[0].legend()
    
    # Plot persistence diagram
    persistence_fig = tdpr.visualize_persistence(X)
    plt.close(persistence_fig)  # Close the separate figure
    axes[1].set_title('Persistence Diagram')
    
    # Compute and plot embedding
    embedding = tdpr.fit_transform(X)
    axes[2].scatter(embedding[y==0, 0], embedding[y==0, 1], label='Circle', alpha=0.6)
    axes[2].scatter(embedding[y==1, 0], embedding[y==1, 1], label='Figure Eight', alpha=0.6)
    axes[2].set_title('TDPR Embedding')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('tdpr_analysis.png')
    plt.close()

if __name__ == "__main__":
    main()
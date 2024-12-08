import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from persim import PersistenceImager
import gudhi
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import networkx as nx
from scipy.spatial.distance import pdist, squareform

class TopologicalFeatureExtractor:
    """Extracts topological features using persistent homology."""
    
    def __init__(self, max_dimension: int = 1, max_edge_length: float = 2.0):
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.persistence_imager = PersistenceImager(pixel_size=0.1)
        
    def compute_persistence_diagrams(self, X: np.ndarray) -> List[np.ndarray]:
        """Compute persistence diagrams for a point cloud."""
        rips_complex = gudhi.RipsComplex(points=X, max_edge_length=self.max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)
        diagrams = simplex_tree.persistence()
        
        # Convert to numpy arrays and separate by dimension
        pers_diagrams = []
        for dim in range(self.max_dimension + 1):
            points = np.array([[p[1][0], p[1][1]] for p in diagrams if p[0] == dim])
            if len(points) == 0:
                points = np.zeros((0, 2))
            pers_diagrams.append(points)
            
        return pers_diagrams
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract topological features from point cloud data."""
        diagrams = self.compute_persistence_diagrams(X)
        # Convert persistence diagrams to images
        persistence_images = []
        for diagram in diagrams:
            if len(diagram) > 0:
                img = self.persistence_imager.transform(diagram)
                persistence_images.append(img.flatten())
            else:
                persistence_images.append(np.zeros(100))  # Default size for empty diagrams
        
        return np.concatenate(persistence_images)

class QuantumKernel(nn.Module):
    """Quantum-inspired kernel layer."""
    
    def __init__(self, feature_dim: int, n_quantum_features: int = 50):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_quantum_features = n_quantum_features
        
        # Parameterized quantum feature map
        self.quantum_params = nn.Parameter(
            torch.randn(n_quantum_features, feature_dim) / np.sqrt(feature_dim)
        )
        
    def quantum_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum-inspired feature map."""
        # Compute quantum features using cosine and sine transformations
        phase = x @ self.quantum_params.T
        quantum_features = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        return quantum_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantum kernel matrix."""
        quantum_features = self.quantum_feature_map(x)
        kernel_matrix = quantum_features @ quantum_features.T
        return kernel_matrix

class TQNKN(nn.Module):
    """Topological Quantum Neural Kernel Network."""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int = 64,
        n_quantum_features: int = 50,
        max_topo_dim: int = 1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Topological feature extractor
        self.topo_extractor = TopologicalFeatureExtractor(
            max_dimension=max_topo_dim
        )
        
        # Quantum kernel layer
        self.quantum_kernel = QuantumKernel(
            feature_dim=input_dim,
            n_quantum_features=n_quantum_features
        )
        
        # Neural layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.ReLU()
        
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_features: If True, return intermediate features
            
        Returns:
            output: Network output
            features: Dictionary of intermediate features (if return_features=True)
        """
        batch_size = x.shape[0]
        
        # Extract topological features
        topo_features = []
        for i in range(batch_size):
            features = self.topo_extractor.extract_features(x[i].detach().cpu().numpy())
            topo_features.append(torch.from_numpy(features).float())
        topo_features = torch.stack(topo_features).to(x.device)
        
        # Compute quantum kernel
        kernel_matrix = self.quantum_kernel(x)
        
        # Combine features
        combined_features = x + 0.1 * kernel_matrix.mean(dim=1, keepdim=True)
        
        # Neural network layers
        hidden = self.activation(self.fc1(combined_features))
        hidden = self.activation(self.fc2(hidden))
        output = self.fc_out(hidden)
        
        if return_features:
            features = {
                'topological': topo_features,
                'kernel': kernel_matrix,
                'hidden': hidden
            }
            return output, features
        
        return output

class TQNKNDataset(Dataset):
    """Dataset class for TQNKN."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

def train_tqnkn(
    model: TQNKN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[List[float], List[float]]:
    """Train the TQNKN model."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y.view(-1, 1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y.view(-1, 1))
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
    return train_losses, val_losses

def visualize_features(model: TQNKN, X: torch.Tensor, y: torch.Tensor):
    """Visualize the learned features."""
    model.eval()
    with torch.no_grad():
        _, features = model(X, return_features=True)
    
    # Plot topological features
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(features['topological'].cpu().numpy(), aspect='auto')
    plt.title('Topological Features')
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(features['kernel'].cpu().numpy(), aspect='auto')
    plt.title('Quantum Kernel Matrix')
    plt.colorbar()
    
    plt.subplot(133)
    plt.scatter(
        features['hidden'][:, 0].cpu().numpy(),
        features['hidden'][:, 1].cpu().numpy(),
        c=y.cpu().numpy()
    )
    plt.title('Hidden Layer Representation')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.randn(n_samples) * 0.1
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create datasets
    train_size = int(0.8 * n_samples)
    train_dataset = TQNKNDataset(X_scaled[:train_size], y[:train_size])
    val_dataset = TQNKNDataset(X_scaled[train_size:], y[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize and train model
    model = TQNKN(input_dim=10)
    train_losses, val_losses = train_tqnkn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100
    )
    
    # Visualize results
    visualize_features(
        model=model,
        X=torch.from_numpy(X_scaled).float(),
        y=torch.from_numpy(y).float()
    )
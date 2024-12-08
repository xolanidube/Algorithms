import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class PolytopeParams:
    A: torch.Tensor  # Polytope boundary matrices
    b: torch.Tensor  # Polytope boundary vectors
    W: torch.Tensor  # Linear transformation matrices
    c: torch.Tensor  # Linear transformation biases

class PGNN(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 n_polytopes: int,
                 learning_rate: float = 0.01):
        """
        Initialize a Polytope-Galois Neural Network.
        
        Args:
            input_dim: Dimension of input space
            output_dim: Dimension of output space
            n_polytopes: Number of polytopes to partition the space
            learning_rate: Learning rate for parameter updates
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_polytopes = n_polytopes
        self.lr = learning_rate
        
        # Initialize polytope parameters
        self.params = self._init_parameters()

    def _init_parameters(self) -> PolytopeParams:
        """Initialize the network parameters defining polytopes and linear maps."""
        # Initialize boundary matrices/vectors for each polytope
        A = torch.randn(self.n_polytopes, self.input_dim, self.input_dim, requires_grad=True)
        b = torch.randn(self.n_polytopes, self.input_dim, requires_grad=True)
        
        # Initialize linear transformation parameters for each polytope
        W = torch.randn(self.n_polytopes, self.output_dim, self.input_dim, requires_grad=True)
        c = torch.randn(self.n_polytopes, self.output_dim, requires_grad=True)
        
        return PolytopeParams(A=A, b=b, W=W, c=c)

    def _compute_polytope_membership(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute soft membership values for each polytope.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, n_polytopes) containing membership values
        """
        batch_size = x.shape[0]
        memberships = torch.zeros(batch_size, self.n_polytopes)
        
        for i in range(self.n_polytopes):
            # Compute Ax <= b for each polytope
            inequalities = torch.matmul(self.params.A[i], x.T).T - self.params.b[i]
            # Soft membership using sigmoid
            membership_i = torch.sigmoid(-inequalities.sum(dim=1))
            memberships[:, i] = membership_i
            
        # Normalize memberships
        memberships = memberships / (memberships.sum(dim=1, keepdim=True) + 1e-6)
        return memberships

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PGNN.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        
        # Compute polytope memberships
        memberships = self._compute_polytope_membership(x)
        
        # Apply linear transformations for each polytope
        outputs = torch.zeros(batch_size, self.output_dim)
        for i in range(self.n_polytopes):
            linear_output = torch.matmul(x, self.params.W[i].T) + self.params.c[i]
            outputs += memberships[:, i].unsqueeze(1) * linear_output
            
        return outputs

    def galois_update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update parameters using the Galois connection-based update rule.
        
        Args:
            x: Input tensor
            y: Target tensor
        """
        # Forward pass
        output = self.forward(x)
        
        # Compute error
        error = output - y
        
        # Compute memberships for gradient updates
        memberships = self._compute_polytope_membership(x)
        
        # Update parameters using a simplified Galois-inspired rule
        with torch.no_grad():
            for i in range(self.n_polytopes):
                # Weight contributions by polytope membership
                weighted_error = error * memberships[:, i].unsqueeze(1)
                weighted_x = x * memberships[:, i].unsqueeze(1)
                
                # Update linear transformation parameters
                self.params.W[i] -= self.lr * torch.matmul(weighted_error.T, x)
                self.params.c[i] -= self.lr * weighted_error.mean(dim=0)
                
                # Update polytope boundaries based on error magnitude
                error_magnitude = torch.norm(weighted_error, dim=1)
                weighted_x = x * error_magnitude.unsqueeze(1)  # shape (32,2)
                self.params.A[i] -= self.lr * torch.matmul(weighted_x.T, x)  # shape (2,2), matches A[i]
                self.params.b[i] -= self.lr * error_magnitude.mean()

    def fit(self, 
            X: torch.Tensor, 
            y: torch.Tensor, 
            epochs: int = 100, 
            batch_size: int = 32,
            verbose: bool = True) -> List[float]:
        """
        Train the PGNN using the Galois update rule.
        
        Args:
            X: Input data
            y: Target data
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print training progress
            
        Returns:
            List of losses during training
        """
        losses = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                
                # Forward pass
                output = self.forward(batch_X)
                loss = torch.mean((output - batch_y) ** 2)
                
                # Update parameters
                self.galois_update(batch_X, batch_y)
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return losses

    def visualize_polytopes_2d(self, 
                              x_range: Tuple[float, float] = (-5, 5),
                              y_range: Tuple[float, float] = (-5, 5),
                              n_points: int = 100) -> None:
        """
        Visualize the polytope partitioning for 2D input space.
        
        Args:
            x_range: Range for x-axis
            y_range: Range for y-axis
            n_points: Number of points per dimension
        """
        if self.input_dim != 2:
            raise ValueError("Visualization only supported for 2D input space")
            
        # Create grid of points
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        
        # Convert to torch tensor
        points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
        
        # Compute polytope memberships
        memberships = self._compute_polytope_membership(points)
        
        # Plot
        plt.figure(figsize=(10, 10))
        
        # Plot membership values for each polytope
        membership_map = memberships.argmax(dim=1).numpy().reshape(n_points, n_points)
        plt.imshow(membership_map, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                  origin='lower', cmap='viridis')
        
        plt.colorbar(label='Polytope Index')
        plt.title('PGNN Polytope Partitioning')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.grid(True)
        plt.show()
        
        
# Utility functions for testing
def generate_synthetic_data(n_samples: int = 1000, 
                          input_dim: int = 2,
                          output_dim: int = 1,
                          noise_level: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data for testing."""
    X = torch.randn(n_samples, input_dim)
    # Generate piecewise linear function
    y = torch.zeros(n_samples, output_dim)
    
    # Create different linear regions
    mask1 = X[:, 0] > 0
    mask2 = X[:, 0] <= 0
    
    # Different linear combinations in different regions
    # Use unsqueeze to match dimensions
    y[mask1, 0] = 2 * X[mask1, 0] + X[mask1, 1]
    y[mask2, 0] = -X[mask2, 0] - 2 * X[mask2, 1]
    
    # Add noise
    y += noise_level * torch.randn_like(y)
    
    return X, y

def evaluate_model(model: PGNN, 
                  X_test: torch.Tensor, 
                  y_test: torch.Tensor) -> float:
    """Evaluate model performance."""
    with torch.no_grad():
        y_pred = model(X_test)
        mse = torch.mean((y_pred - y_test) ** 2).item()
    return mse
# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data
    X_train, y_train = generate_synthetic_data(n_samples=1000)
    X_test, y_test = generate_synthetic_data(n_samples=200)
    
    # Initialize and train model
    model = PGNN(input_dim=2, output_dim=1, n_polytopes=4)
    losses = model.fit(X_train, y_train, epochs=100, batch_size=32)
    
    # Evaluate model
    test_mse = evaluate_model(model, X_test, y_test)
    print(f"Test MSE: {test_mse:.6f}")
    
    # Visualize polytopes
    model.visualize_polytopes_2d()
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
import numpy as np
import torch
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
 # Import PGNN from previous implementation

def load_and_preprocess_data():
    """Load and preprocess the Boston Housing dataset."""
    # Load dataset
    boston = load_boston()
    X, y = boston.data, boston.target
    
    # Scale features and targets
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    return X_train, X_test, y_train, y_test, y_scaler

def train_and_evaluate():
    """Train and evaluate PGNN on the Boston Housing dataset."""
    # Load and preprocess data
    X_train, X_test, y_train, y_test, y_scaler = load_and_preprocess_data()
    
    # Create model
    model = PGNN(
        input_dim=X_train.shape[1],
        output_dim=1,
        n_polytopes=16,
        learning_rate=0.01
    )
    
    # Train model
    losses = model.fit(X_train, y_train, epochs=200, batch_size=32)
    
    # Evaluate model
    with torch.no_grad():
        # Training predictions
        y_train_pred = model(X_train)
        train_mse = torch.mean((y_train_pred - y_train) ** 2).item()
        
        # Test predictions
        y_test_pred = model(X_test)
        test_mse = torch.mean((y_test_pred - y_test) ** 2).item()
        
        # Convert back to original scale for interpretability
        y_test_orig = y_scaler.inverse_transform(y_test.numpy())
        y_test_pred_orig = y_scaler.inverse_transform(y_test_pred.numpy())
        
        # Calculate R² score
        y_test_mean = np.mean(y_test_orig)
        ss_tot = np.sum((y_test_orig - y_test_mean) ** 2)
        ss_res = np.sum((y_test_orig - y_test_pred_orig) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    
    # Plot 2: Predictions vs Actual
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_orig, y_test_pred_orig, alpha=0.5)
    plt.plot([y_test_orig.min(), y_test_orig.max()], 
             [y_test_orig.min(), y_test_orig.max()], 
             'r--', label='Perfect Prediction')
    plt.title('Predictions vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"R² Score: {r2_score:.6f}")
    
    return model, losses, (y_test_orig, y_test_pred_orig)

if __name__ == "__main__":
    # Run training and evaluation
    model, losses, (y_true, y_pred) = train_and_evaluate()
    
    # Additional analysis: Feature importance
    print("\nAnalyzing polytope structure...")
    
    # Look at average polytope activation for each feature
    X_test_standardized = torch.FloatTensor(StandardScaler().fit_transform(load_boston().data))
    with torch.no_grad():
        memberships = model._compute_polytope_membership(X_test_standardized)
    
    print(f"\nNumber of active polytopes: {(memberships > 0.1).sum(dim=1).mean():.2f}")
    print(f"Max membership value: {memberships.max().item():.4f}")
    print(f"Min membership value: {memberships.min().item():.4f}")
    
    # Analyze polytope boundaries
    boundary_strengths = torch.norm(model.params.A, dim=2).mean(dim=0)
    print("\nFeature influence on polytope boundaries:")
    for i, feature_name in enumerate(load_boston().feature_names):
        print(f"{feature_name}: {boundary_strengths[i]:.4f}")
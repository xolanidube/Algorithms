import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math
import numpy as np
from typing import Tuple, List, Optional

class HyperbolicManifold:
    """
    Implements operations in the Poincaré ball model of hyperbolic space.
    """
    def __init__(self, dim: int, c: float = 1.0):
        self.dim = dim
        self.c = c

    def mobius_addition(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition in the Poincaré ball.
        """
        assert x.shape == y.shape, f"Shape mismatch: x:{x.shape}, y:{y.shape}"
        
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
        return num / denom.clamp(min=1e-15)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from tangent space to manifold.
        """
        assert x.shape[-1] == v.shape[-1], f"Dimension mismatch: x:{x.shape}, v:{v.shape}"
        
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        v_norm = v_norm.clamp(min=1e-15)
        sqrt_c = math.sqrt(self.c)
        
        second_term = torch.tanh(sqrt_c * v_norm / 2) * v / (sqrt_c * v_norm)
        return self.mobius_addition(x, second_term)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map from manifold to tangent space.
        """
        sqrt_c = math.sqrt(self.c)
        diff = self.mobius_addition(-x, y)
        diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
        diff_norm = diff_norm.clamp(min=1e-15)
        return 2 / sqrt_c * torch.atanh(sqrt_c * diff_norm) * diff / diff_norm

class QuasiconformalLayer(nn.Module):
    """
    Neural network layer implementing a quasiconformal map on hyperbolic space.
    """
    def __init__(self, manifold: HyperbolicManifold, in_dim: int, out_dim: int):
        super().__init__()
        self.manifold = manifold
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Project to and from hyperbolic space while maintaining dimensions
        self.projection = nn.Sequential(
            nn.Linear(in_dim, manifold.dim),
            nn.ReLU(),
            nn.Linear(manifold.dim, out_dim)
        )
        
        # Initialize weights
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def compute_distortion(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute quasiconformal distortion using cached intermediate values if available.
        """
        if cache is None:
            cache = self.forward(x, return_intermediate=True)
            
        # Compute Jacobian of the transformation
        with torch.enable_grad():
            x_mapped = cache.detach().requires_grad_()
            output = self.projection[-1](x_mapped)
            jac = torch.autograd.grad(output.sum(), x_mapped, create_graph=True)[0]
        
        # Compute singular values
        s = torch.linalg.svdvals(jac.reshape(jac.shape[0], -1))
        return s.max() / s.min().clamp(min=1e-6)

    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> torch.Tensor:
        """
        Apply quasiconformal map to input points.
        """
        batch_size = x.shape[0]
        
        # First linear projection
        h = self.projection[0](x)
        h = self.projection[1](h)  # ReLU
        
        # Map to hyperbolic space
        zero = torch.zeros(batch_size, self.manifold.dim, device=x.device)
        h_hyp = self.manifold.exp_map(zero, h)
        
        if return_intermediate:
            return h_hyp
            
        # Final projection
        return self.projection[-1](h_hyp)

class HQNME(nn.Module):
    """
    Hyperbolic Quasiconformal Neural Metric Embedding (HQNME) network.
    """
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        manifold_dim: int = 32,
        c: float = 1.0
    ):
        super().__init__()
        
        self.manifold = HyperbolicManifold(manifold_dim, c)
        
        # Build sequential layers
        dimensions = [input_dim] + hidden_dims + [output_dim]
        layers = []
        
        for i in range(len(dimensions) - 1):
            layers.append(QuasiconformalLayer(
                self.manifold, 
                dimensions[i], 
                dimensions[i + 1]
            ))
        
        self.layers = nn.ModuleList(layers)

    def compute_pde_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE-based loss incorporating reconstruction and distortion.
        """
        intermediates = []
        h = x
        
        # Forward pass with intermediate values
        for layer in self.layers:
            h_inter = layer(h, return_intermediate=True)
            intermediates.append(h_inter)
            h = layer.projection[-1](h_inter)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(h, x)
        
        # Compute distortion loss using cached intermediates
        distortion_loss = torch.tensor(0., device=x.device)
        for layer, intermediate in zip(self.layers, intermediates):
            distortion_loss = distortion_loss + layer.compute_distortion(x, intermediate)
        
        # Combined loss
        lambda_qc = 0.1
        return recon_loss + lambda_qc * distortion_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

def train_hqnme(
    model: HQNME,
    train_data: torch.Tensor,
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> HQNME:
    """
    Train the HQNME model on given data.
    """
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    n_batches = (train_data.shape[0] + batch_size - 1) // batch_size
    
    print(f"\nStarting training with {n_batches} batches per epoch")
    print(f"Input dimension: {train_data.shape[1]}")
    print(f"Device: {device}\n")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        # Shuffle data
        perm = torch.randperm(train_data.shape[0])
        train_data = train_data[perm]
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, train_data.shape[0])
            batch = train_data[start_idx:end_idx].to(device)
            
            optimizer.zero_grad()
            
            # Compute loss
            loss = model.compute_pde_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    return model

def example_usage():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Parameters
    input_dim = 10
    hidden_dims = [20, 15]
    output_dim = 10  # Same as input for autoencoder-style training
    num_samples = 100
    manifold_dim = 32
    
    print("\nInitializing HQNME model...")
    print(f"Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}")
    print(f"Manifold dimension: {manifold_dim}")
    
    # Initialize model
    model = HQNME(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        manifold_dim=manifold_dim
    )
    
    # Create synthetic data
    print("\nGenerating synthetic data...")
    data = torch.randn(num_samples, input_dim)
    
    # Train model
    print("\nTraining model...")
    trained_model = train_hqnme(model, data)
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    with torch.no_grad():
        embeddings = trained_model(data)
    
    print(f"\nResults:")
    print(f"Input shape: {data.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    
    return embeddings

if __name__ == "__main__":
    embeddings = example_usage()
    print(embeddings)
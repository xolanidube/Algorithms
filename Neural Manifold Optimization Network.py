import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

class ManifoldLayer(nn.Module):
    def __init__(self, in_features, out_features, manifold_dim=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold_dim = manifold_dim
        
        # Learnable manifold parameters
        self.manifold_params = nn.Parameter(torch.randn(manifold_dim, in_features))
        self.curvature_params = nn.Parameter(torch.randn(manifold_dim, manifold_dim))
        
        # Position-dependent weight field
        self.weight_field = nn.Parameter(torch.randn(out_features, in_features, manifold_dim))
        
    def compute_manifold_metric(self):
        # Compute Riemannian metric tensor
        G = torch.matmul(self.manifold_params, self.manifold_params.t())
        return G + torch.eye(self.manifold_dim)
    
    def manifold_gradient(self, x):
        # Project input onto manifold
        proj = torch.matmul(x, self.manifold_params.t())
        metric = self.compute_manifold_metric()
        
        # Compute covariant derivative
        grad = torch.matmul(proj, torch.inverse(metric))
        return grad
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Compute manifold coordinates
        manifold_coords = self.manifold_gradient(x)
        
        # Evaluate weight field at manifold positions
        weights = torch.sum(self.weight_field * manifold_coords.unsqueeze(1), dim=2)
        
        # Apply curvature correction
        curvature = torch.matmul(manifold_coords, self.curvature_params)
        curved_weights = weights * (1 + torch.tanh(curvature.sum(dim=1, keepdim=True)))
        
        return F.linear(x, curved_weights)

class NMON(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, manifold_dim=3):
        super().__init__()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            ManifoldLayer(dims[i], dims[i+1], manifold_dim)
            for i in range(len(dims)-1)
        ])
        
        self.manifold_dim = manifold_dim
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
    
    def compute_manifold_complexity(self):
        total_complexity = 0
        for layer in self.layers:
            metric = layer.compute_manifold_metric()
            # Compute Ricci scalar curvature (simplified)
            complexity = torch.diagonal(metric).sum() / self.manifold_dim
            total_complexity += complexity
        return total_complexity

# Example usage and validation
def train_and_validate(model, train_loader, test_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch_x)
            
            # Compute loss with manifold regularization
            main_loss = F.mse_loss(output, batch_y)
            manifold_complexity = model.compute_manifold_complexity()
            loss = main_loss + 0.01 * manifold_complexity
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for val_x, val_y in test_loader:
                    val_output = model(val_x)
                    val_loss += F.mse_loss(val_output, val_y).item()
            print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, Manifold Complexity = {manifold_complexity:.4f}")
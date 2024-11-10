"""
Adaptive Mesh Learning Algorithm (AMLA) - Version 2
A novel approach to distributed learning with dynamic topology adaptation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn

@dataclass
class NodeState:
    """Represents the state of a node in the learning mesh"""
    id: int
    features: np.ndarray
    confidence: float
    connections: List[int]
    local_model: Optional[nn.Module]
    performance_history: List[float]

class AdaptiveMeshLearning:
    """
    Implements the Adaptive Mesh Learning Algorithm (AMLA)
    
    Key Features:
    1. Dynamic topology adaptation
    2. Local-global knowledge synthesis
    3. Autonomous node specialization
    4. Resilient distributed learning
    """
    
    def __init__(self, 
                 dim_features: int,
                 num_initial_nodes: int,
                 adaptation_rate: float = 0.1,
                 confidence_threshold: float = 0.75,
                 k_neighbors: int = 3):
        
        self.dim_features = dim_features
        self.adaptation_rate = adaptation_rate
        self.confidence_threshold = confidence_threshold
        self.k_neighbors = min(k_neighbors, num_initial_nodes - 1)  # Ensure k is valid
        
        # Initialize mesh structure
        self.nodes: Dict[int, NodeState] = {}
        self.global_topology = nx.Graph()
        self.performance_history = []
        
        # Setup initial mesh
        self._initialize_mesh(num_initial_nodes)
        
    def _initialize_mesh(self, num_nodes: int):
        """Initialize the learning mesh with starting nodes"""
        # Create nodes with random initial states
        for i in range(num_nodes):
            features = np.random.randn(self.dim_features)
            node = NodeState(
                id=i,
                features=features,
                confidence=0.5,
                connections=[],
                local_model=self._create_local_model(),
                performance_history=[]
            )
            self.nodes[i] = node
            self.global_topology.add_node(i)
        
        # Create initial connections using k-nearest neighbors
        self._update_topology()
    
    def _create_local_model(self) -> nn.Module:
        """Create a local learning model for a node"""
        model = nn.Sequential(
            nn.Linear(self.dim_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.dim_features)
        )
        return model
    
    def _update_topology(self):
        """Update mesh topology based on node performance and relationships"""
        if len(self.nodes) < 2:
            return  # Can't create connections with less than 2 nodes
            
        # Get node positions in feature space
        positions = np.vstack([node.features for node in self.nodes.values()])
        node_ids = list(self.nodes.keys())
        
        # Use k-nearest neighbors to establish connections
        k = min(self.k_neighbors, len(self.nodes) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(positions)
        distances, indices = nbrs.kneighbors(positions)
        
        # Update topology
        self.global_topology.clear_edges()
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip first neighbor (self)
                self.global_topology.add_edge(node_ids[i], node_ids[j])
        
        # Update node connections
        for node_id in self.nodes:
            self.nodes[node_id].connections = list(self.global_topology.neighbors(node_id))
    
    def _local_learning_phase(self, data: np.ndarray):
        """Execute local learning for each node"""
        for node in self.nodes.values():
            # Select relevant data for this node based on feature similarity
            distances = np.linalg.norm(data - node.features, axis=1)
            relevant_indices = np.argsort(distances)[:100]  # Take 100 closest samples
            local_data = data[relevant_indices]
            
            # Update local model (simplified training loop)
            if isinstance(node.local_model, nn.Module):
                optimizer = torch.optim.Adam(node.local_model.parameters())
                local_data_tensor = torch.FloatTensor(local_data)
                
                for _ in range(5):  # Mini training loop
                    optimizer.zero_grad()
                    output = node.local_model(local_data_tensor)
                    loss = nn.MSELoss()(output, local_data_tensor)
                    loss.backward()
                    optimizer.step()
                
                # Update node confidence based on reconstruction error
                with torch.no_grad():
                    reconstruction_error = nn.MSELoss()(
                        node.local_model(local_data_tensor),
                        local_data_tensor
                    ).item()
                    node.confidence = 1.0 / (1.0 + reconstruction_error)
    
    def learn(self, data: np.ndarray, iterations: int = 100) -> List[float]:
        """
        Execute the distributed learning process
        
        Args:
            data: Training data of shape (n_samples, dim_features)
            iterations: Number of learning iterations
        """
        performance_history = []
        
        for iteration in range(iterations):
            # Local learning phase
            self._local_learning_phase(data)
            
            # Knowledge synthesis between connected nodes
            self._knowledge_synthesis()
            
            # Topology adaptation every 10 iterations
            if iteration % 10 == 0:
                self._update_topology()
            
            # Record performance
            avg_confidence = np.mean([node.confidence for node in self.nodes.values()])
            performance_history.append(avg_confidence)
            
        return performance_history
    
    def _knowledge_synthesis(self):
        """Synthesize knowledge across connected nodes"""
        for node in self.nodes.values():
            if node.confidence < self.confidence_threshold:
                # Get models from higher confidence neighbors
                neighbor_models = []
                neighbor_weights = []
                
                for neighbor_id in node.connections:
                    neighbor = self.nodes[neighbor_id]
                    if neighbor.confidence > node.confidence:
                        neighbor_models.append(neighbor.local_model)
                        neighbor_weights.append(neighbor.confidence)
                
                if neighbor_models:
                    # Average the model parameters weighted by confidence
                    with torch.no_grad():
                        for param in node.local_model.parameters():
                            param.data.zero_()
                            total_weight = sum(neighbor_weights)
                            
                            for model, weight in zip(neighbor_models, neighbor_weights):
                                for p1, p2 in zip(node.local_model.parameters(),
                                                model.parameters()):
                                    p1.data.add_(p2.data * (weight / total_weight))
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Generate predictions using the mesh"""
        predictions = []
        data_tensor = torch.FloatTensor(data)
        
        for sample in data_tensor:
            # Find relevant nodes
            distances = [np.linalg.norm(sample.numpy() - node.features) 
                        for node in self.nodes.values()]
            node_weights = 1.0 / (np.array(distances) + 1e-6)
            node_weights = node_weights / node_weights.sum()
            
            # Weighted prediction from all nodes
            sample_predictions = []
            for node, weight in zip(self.nodes.values(), node_weights):
                with torch.no_grad():
                    pred = node.local_model(sample.unsqueeze(0))
                    sample_predictions.append(pred.numpy() * weight)
            
            predictions.append(np.sum(sample_predictions, axis=0))
        
        return np.array(predictions)

# Example usage
if __name__ == "__main__":
    # Create synthetic dataset
    num_samples = 1000
    feature_dim = 10
    X = np.random.randn(num_samples, feature_dim)
    
    # Initialize AMLA
    amla = AdaptiveMeshLearning(
        dim_features=feature_dim,
        num_initial_nodes=5,
        k_neighbors=2  # Reduced number of neighbors for small networks
    )
    
    # Train the mesh
    performance_history = amla.learn(X, iterations=50)
    
    # Generate predictions
    test_data = np.random.randn(10, feature_dim)
    predictions = amla.predict(test_data)
    
    print("Training complete!")
    print(f"Final average confidence: {performance_history[-1]:.4f}")
    print(f"Number of active nodes: {len(amla.nodes)}")
    print(f"Average node connections: {np.mean([len(node.connections) for node in amla.nodes.values()]):.2f}")
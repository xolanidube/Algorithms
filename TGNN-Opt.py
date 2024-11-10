import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GlobalAttention
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import gym
from gym import spaces
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

@dataclass
class TGNNOptConfig:
    input_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 32
    num_layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 0.001
    max_stm_size: int = 10
    ltm_update_frequency: int = 100
    reward_alpha: float = 0.5  # Weight for efficiency vs. information preservation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class TimeLayeredGraphMemory:
    def __init__(self, config: TGNNOptConfig):
        self.config = config
        self.stm = deque(maxlen=config.max_stm_size)  # Short-term memory
        self.ltm = []  # Long-term memory
        self.update_counter = 0
    
    def add_graph(self, graph_data: Data):
        """Add a new graph to short-term memory"""
        self.stm.append(graph_data)
        self.update_counter += 1
        
        if self.update_counter >= self.config.ltm_update_frequency:
            self._update_ltm()
            self.update_counter = 0
    
    def _update_ltm(self):
        """Update long-term memory with aggregated patterns"""
        if len(self.stm) == 0:
            return
            
        # Aggregate node features and edge patterns
        agg_features = []
        agg_edges = []
        
        for graph in self.stm:
            agg_features.append(graph.x)
            agg_edges.append(graph.edge_index)
        
        # Compute average node features
        avg_features = torch.stack(agg_features).mean(dim=0)
        
        # Store aggregated pattern
        self.ltm.append({
            'features': avg_features,
            'timestamp': len(self.ltm)
        })
    
    def get_temporal_context(self) -> Tuple[List[Data], Optional[torch.Tensor]]:
        """Get current temporal context from STM and LTM"""
        stm_graphs = list(self.stm)
        ltm_context = None
        
        if len(self.ltm) > 0:
            ltm_context = torch.stack([m['features'] for m in self.ltm[-5:]])
        
        return stm_graphs, ltm_context

class TemporalGNN(nn.Module):
    def __init__(self, config: TGNNOptConfig):
        super().__init__()
        self.config = config
        
        # GNN layers
        self.convs = nn.ModuleList([
            GCNConv(config.input_dim if i == 0 else config.hidden_dim,
                   config.hidden_dim)
            for i in range(config.num_layers)
        ])
        
        # Attention mechanism
        self.attention = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1)
            )
        )
        
        # Output layers
        self.fc_out = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        
        # Apply GNN layers with residual connections
        for conv in self.convs:
            x_new = conv(x, edge_index)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.config.dropout, training=self.training)
            x = x + x_new if x.shape == x_new.shape else x_new
        
        # Apply attention
        x = self.attention(x)
        
        # Final output
        return self.fc_out(x)

class EdgeRewiringEnv(gym.Env):
    def __init__(self, initial_graph: Data, config: TGNNOptConfig):
        super().__init__()
        self.config = config
        self.initial_graph = initial_graph
        self.current_graph = initial_graph
        
        # Define action and observation spaces
        num_nodes = initial_graph.num_nodes
        self.action_space = spaces.Discrete(num_nodes * (num_nodes - 1) // 2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_nodes, config.input_dim)
        )
    
    def step(self, action: int) -> Tuple[Data, float, bool, dict]:
        # Convert action to edge modification
        edge_pair = self._action_to_edge(action)
        
        # Apply rewiring
        new_graph = self._rewire_edge(edge_pair)
        
        # Calculate reward
        reward = self._calculate_reward(new_graph)
        
        self.current_graph = new_graph
        
        return new_graph, reward, False, {}
    
    def reset(self) -> Data:
        self.current_graph = self.initial_graph
        return self.current_graph
    
    def _action_to_edge(self, action: int) -> Tuple[int, int]:
        """Convert action index to edge pair"""
        num_nodes = self.initial_graph.num_nodes
        row = int(np.floor((-1 + np.sqrt(1 + 8 * action)) / 2))
        col = action - row * (row + 1) // 2
        return (row, col + row + 1)
    
    def _rewire_edge(self, edge_pair: Tuple[int, int]) -> Data:
        """Modify graph by adding or removing edge"""
        edge_index = self.current_graph.edge_index.cpu().numpy()
        edge_list = set(map(tuple, edge_index.T))
        
        if edge_pair in edge_list:
            edge_list.remove(edge_pair)
        else:
            edge_list.add(edge_pair)
            edge_list.add(edge_pair[::-1])  # Add reverse edge for undirected graph
        
        new_edges = np.array(list(edge_list)).T
        new_graph = Data(
            x=self.current_graph.x,
            edge_index=torch.tensor(new_edges, device=self.config.device)
        )
        
        return new_graph
    
    def _calculate_reward(self, graph: Data) -> float:
        """Calculate reward based on efficiency and information preservation"""
        # Efficiency metric: inverse of average shortest path length
        nx_graph = self._to_networkx(graph)
        try:
            efficiency = 1.0 / nx.average_shortest_path_length(nx_graph)
        except nx.NetworkXError:
            efficiency = -1.0  # Penalize disconnected graphs
        
        # Information preservation: similarity to original graph
        orig_adj = self._to_adjacency(self.initial_graph)
        new_adj = self._to_adjacency(graph)
        info_preserved = F.cosine_similarity(orig_adj.flatten(), new_adj.flatten(), dim=0)
        
        # Combined reward
        reward = (self.config.reward_alpha * efficiency + 
                 (1 - self.config.reward_alpha) * info_preserved)
        
        return float(reward)
    
    @staticmethod
    def _to_networkx(graph: Data) -> nx.Graph:
        """Convert PyG graph to NetworkX"""
        G = nx.Graph()
        G.add_nodes_from(range(graph.num_nodes))
        G.add_edges_from(graph.edge_index.cpu().numpy().T)
        return G
    
    @staticmethod
    def _to_adjacency(graph: Data) -> torch.Tensor:
        """Convert graph to adjacency matrix"""
        num_nodes = graph.num_nodes
        adj = torch.zeros((num_nodes, num_nodes))
        adj[graph.edge_index[0], graph.edge_index[1]] = 1
        return adj

class TGNNOpt:
    def __init__(self, config: TGNNOptConfig):
        self.config = config
        self.memory = TimeLayeredGraphMemory(config)
        self.gnn = TemporalGNN(config).to(config.device)
        self.optimizer = torch.optim.Adam(
            self.gnn.parameters(), 
            lr=config.learning_rate
        )
    
    def process_graph(self, graph: Data) -> Data:
        """Process a new graph and optimize its structure"""
        # Add to memory
        self.memory.add_graph(graph)
        
        # Get temporal context
        stm_graphs, ltm_context = self.memory.get_temporal_context()
        
        # Create environment for edge rewiring
        env = EdgeRewiringEnv(graph, self.config)
        
        # Optimize graph structure
        optimized_graph = self._optimize_structure(env, stm_graphs, ltm_context)
        
        return optimized_graph
    
    def _optimize_structure(
        self, 
        env: EdgeRewiringEnv, 
        stm_graphs: List[Data],
        ltm_context: Optional[torch.Tensor]
    ) -> Data:
        """Optimize graph structure using GNN predictions"""
        self.gnn.eval()
        
        with torch.no_grad():
            # Get current graph embedding
            current_graph = env.current_graph
            embedding = self.gnn(current_graph)
            
            # Use embedding to predict optimal rewiring
            action_scores = F.softmax(embedding, dim=-1)
            action = action_scores.argmax().item()
            
            # Apply rewiring
            new_graph, reward, _, _ = env.step(action)
        
        return new_graph
    
    def train_step(self, batch_graphs: List[Data]):
        """Train the GNN on a batch of graphs"""
        self.gnn.train()
        self.optimizer.zero_grad()
        
        # Create batch
        batch = Batch.from_data_list(batch_graphs)
        
        # Forward pass
        output = self.gnn(batch)
        
        # Calculate loss (example: reconstruction loss)
        target = batch.x
        loss = F.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def visualize_graph(graph: Data, title: str = "Graph Visualization"):
    """Visualize graph structure"""
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(graph.edge_index.cpu().numpy().T)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold')
    plt.title(title)
    plt.show()
    
    
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import requests
import io
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import networkx as nx
from tqdm import tqdm

class TrafficDataProcessor:
    def __init__(self, config: TGNNOptConfig):
        self.config = config
        self.scaler = StandardScaler()
        
    def fetch_pems_data(self) -> pd.DataFrame:
        """
        Fetch PeMS traffic data from UCI ML Repository
        Returns processed DataFrame with traffic measurements
        """
        # URL for PEMS-SF data (you might need to update this URL)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip"
        
        try:
            # Download and extract data
            response = requests.get(url)
            if response.status_code == 200:
                with io.BytesIO(response.content) as buf:
                    df = pd.read_csv(buf, compression='zip')
                return df
            else:
                raise Exception("Failed to download data")
        except Exception as e:
            print(f"Error downloading data: {e}")
            # Generate synthetic data for testing if download fails
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic traffic data for testing"""
        num_sensors = 100
        num_timepoints = 1000
        
        # Generate timestamps
        base_time = datetime.now()
        timestamps = [base_time + timedelta(minutes=5*i) for i in range(num_timepoints)]
        
        # Generate synthetic measurements
        data = {
            'timestamp': timestamps,
            'sensor_id': [],
            'speed': [],
            'flow': [],
            'occupancy': []
        }
        
        for sensor in range(num_sensors):
            for _ in range(num_timepoints):
                data['sensor_id'].append(sensor)
                data['speed'].append(np.random.normal(65, 10))  # Speed in mph
                data['flow'].append(np.random.normal(1000, 200))  # Vehicles per hour
                data['occupancy'].append(np.random.uniform(0, 100))  # Percentage
        
        return pd.DataFrame(data)
    
    def process_traffic_data(self, df: pd.DataFrame) -> List[Data]:
        """
        Process traffic data into temporal graph sequence
        Returns list of PyG Data objects
        """
        # Group by timestamp
        grouped = df.groupby('timestamp')
        
        graph_sequence = []
        prev_features = None
        
        for timestamp, group in tqdm(grouped):
            # Create node features
            features = group[['speed', 'flow', 'occupancy']].values
            features = self.scaler.fit_transform(features)
            
            # Create edges based on spatial proximity and flow correlation
            edge_index = self._create_edge_index(features, prev_features)
            
            # Convert to PyG Data object
            graph_data = Data(
                x=torch.FloatTensor(features),
                edge_index=torch.LongTensor(edge_index),
                timestamp=timestamp
            )
            
            graph_sequence.append(graph_data)
            prev_features = features
        
        return graph_sequence
    
    def _create_edge_index(self, 
                          current_features: np.ndarray, 
                          prev_features: Optional[np.ndarray]) -> np.ndarray:
        """Create edge index based on feature similarity and temporal correlation"""
        num_nodes = len(current_features)
        edges = []
        
        # Spatial edges based on feature similarity
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                similarity = np.corrcoef(current_features[i], current_features[j])[0, 1]
                if similarity > 0.7:  # Threshold for edge creation
                    edges.extend([[i, j], [j, i]])
        
        # Temporal edges if previous features exist
        if prev_features is not None:
            for i in range(num_nodes):
                if i < len(prev_features):
                    correlation = np.corrcoef(current_features[i], prev_features[i])[0, 1]
                    if correlation > 0.5:
                        edges.append([i, i])
        
        return np.array(edges).T if edges else np.zeros((2, 0), dtype=int)

class TGNNOptEvaluator:
    def __init__(self, model: TGNNOpt, config: TGNNOptConfig):
        self.model = model
        self.config = config
        self.metrics = {
            'efficiency': [],
            'preservation': [],
            'processing_time': []
        }
    
    def evaluate_sequence(self, graph_sequence: List[Data]):
        """Evaluate TGNN-Opt on a sequence of temporal graphs"""
        for graph in tqdm(graph_sequence):
            start_time = datetime.now()
            
            # Process graph
            optimized_graph = self.model.process_graph(graph)
            
            # Calculate metrics
            efficiency = self._calculate_efficiency(optimized_graph)
            preservation = self._calculate_preservation(graph, optimized_graph)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store metrics
            self.metrics['efficiency'].append(efficiency)
            self.metrics['preservation'].append(preservation)
            self.metrics['processing_time'].append(processing_time)
    
    def _calculate_efficiency(self, graph: Data) -> float:
        """Calculate graph efficiency metric"""
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(graph.edge_index.cpu().numpy().T)
        
        try:
            return 1.0 / nx.average_shortest_path_length(nx_graph)
        except:
            return 0.0
    
    def _calculate_preservation(self, original: Data, optimized: Data) -> float:
        """Calculate information preservation metric"""
        orig_adj = torch.zeros((original.num_nodes, original.num_nodes))
        opt_adj = torch.zeros((optimized.num_nodes, optimized.num_nodes))
        
        orig_adj[original.edge_index[0], original.edge_index[1]] = 1
        opt_adj[optimized.edge_index[0], optimized.edge_index[1]] = 1
        
        return float(torch.nn.functional.cosine_similarity(
            orig_adj.flatten(), 
            opt_adj.flatten(), 
            dim=0
        ))
    
    def plot_metrics(self):
        """Plot evaluation metrics"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot efficiency
        ax1.plot(self.metrics['efficiency'])
        ax1.set_title('Graph Efficiency Over Time')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Efficiency')
        
        # Plot preservation
        ax2.plot(self.metrics['preservation'])
        ax2.set_title('Information Preservation Over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Preservation Score')
        
        # Plot processing time
        ax3.plot(self.metrics['processing_time'])
        ax3.set_title('Processing Time Over Time')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Time (seconds)')
        
        plt.tight_layout()
        plt.show()

def main():
    # Initialize configuration
    config = TGNNOptConfig(
        input_dim=3,  # speed, flow, occupancy
        hidden_dim=64,
        output_dim=32,
        num_layers=3,
        dropout=0.2,
        learning_rate=0.001,
        max_stm_size=10,
        ltm_update_frequency=100
    )
    
    # Initialize processor and model
    processor = TrafficDataProcessor(config)
    model = TGNNOpt(config)
    evaluator = TGNNOptEvaluator(model, config)
    
    # Fetch and process data
    print("Fetching traffic data...")
    traffic_data = processor.fetch_pems_data()
    
    print("Processing into graph sequence...")
    graph_sequence = processor.process_traffic_data(traffic_data)
    
    # Train model
    print("Training model...")
    num_epochs = 5
    batch_size = 32
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = len(graph_sequence) // batch_size
        
        for i in range(num_batches):
            batch_graphs = graph_sequence[i*batch_size:(i+1)*batch_size]
            loss = model.train_step(batch_graphs)
            total_loss += loss
            
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    # Evaluate model
    print("Evaluating model...")
    evaluator.evaluate_sequence(graph_sequence[:100])  # Evaluate on first 100 graphs
    
    # Plot results
    print("Plotting results...")
    evaluator.plot_metrics()

if __name__ == "__main__":
    main()
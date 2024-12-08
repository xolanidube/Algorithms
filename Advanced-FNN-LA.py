import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import deque
import random

class ChannelAttention(nn.Module):
    """
    Channel-wise attention mechanism for fractal units
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for both pooling branches
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.mlp(self.max_pool(x).view(x.size(0), -1))
        
        attention = torch.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        return x * attention

class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for fractal units
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention_map = torch.sigmoid(self.conv(attention_map))
        return x * attention_map

class MemoryModule(nn.Module):
    """
    Memory-augmented module for storing and retrieving activation patterns
    """
    def __init__(self, feature_dim: int, memory_size: int = 64):
        super(MemoryModule, self).__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        
        # Memory bank
        self.memory = nn.Parameter(torch.randn(memory_size, feature_dim))
        self.memory_keys = nn.Parameter(torch.randn(memory_size, feature_dim))
        
        # Memory controller
        self.controller = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Reshape input to (batch_size, feature_dim)
        x_flat = x.view(batch_size, -1)
        
        # Generate query from input
        query = self.controller(x_flat)
        
        # Compute attention weights
        attention = torch.matmul(query, self.memory_keys.t())
        attention = F.softmax(attention / np.sqrt(self.feature_dim), dim=1)
        
        # Retrieve memory
        retrieved = torch.matmul(attention, self.memory)
        
        # Compute gate values
        gate_input = torch.cat([x_flat, retrieved], dim=1)
        gate_values = self.gate(gate_input)
        
        # Combine input with memory
        output = gate_values * retrieved + (1 - gate_values) * x_flat
        
        # Update memory (using moving average)
        with torch.no_grad():
            update_mask = (attention > 0.1).float()
            self.memory.data = (
                0.99 * self.memory.data + 
                0.01 * torch.matmul(update_mask.t(), x_flat)
            )
        
        return output.view_as(x), attention

class SpikingNeuron(nn.Module):
    """
    Spiking neuron implementation for neuromorphic features
    """
    def __init__(self, threshold: float = 1.0, leak_rate: float = 0.1):
        super(SpikingNeuron, self).__init__()
        self.threshold = threshold
        self.leak_rate = leak_rate
        self.membrane_potential = None
        self.spike_history = []
    
    def reset_state(self):
        self.membrane_potential = None
        self.spike_history = []
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros_like(x)
        
        # Update membrane potential
        self.membrane_potential = (
            (1 - self.leak_rate) * self.membrane_potential + 
            self.leak_rate * x
        )
        
        # Generate spikes
        spikes = (self.membrane_potential >= self.threshold).float()
        
        # Reset membrane potential where spikes occurred
        self.membrane_potential = self.membrane_potential * (1 - spikes)
        
        # Store spike history
        self.spike_history.append(spikes)
        if len(self.spike_history) > 100:
            self.spike_history.pop(0)
        
        # Compute firing rate
        firing_rate = torch.stack(self.spike_history[-10:]).mean(0)
        
        return spikes, firing_rate

class SelfSupervisedPredictor(nn.Module):
    """
    Self-supervised predictor for dynamic depth selection
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(SelfSupervisedPredictor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.predictor(x)
        confidence = self.confidence_estimator(prediction)
        return prediction, confidence

class RLAgent:
    """
    Reinforcement Learning agent for controlling fractal expansion
    """
    def __init__(self, state_size: int, action_size: int = 2):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Q-Network
        self.model = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def get_action(self, state: torch.Tensor) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            action_values = self.model(state)
            return torch.argmax(action_values).item()
    
    def train(self, batch_size: int = 32):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch])
        next_states = torch.stack([b[3] for b in batch])
        dones = torch.tensor([b[4] for b in batch])
        
        # Get Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class AdvancedFractalUnit(nn.Module):
    """
    Advanced Fractal Unit with self-supervised learning, memory augmentation,
    and neuromorphic features
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 depth: int = 0,
                 max_depth: int = 3):
        super(AdvancedFractalUnit, self).__init__()
        
        self.depth = depth
        self.max_depth = max_depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Base convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Memory module
        self.memory = MemoryModule(out_channels)
        
        # Spiking neurons
        self.spiking_neurons = nn.ModuleList([
            SpikingNeuron() for _ in range(out_channels)
        ])
        
        # Self-supervised predictor
        self.predictor = SelfSupervisedPredictor(out_channels, out_channels)
        
        # Attention mechanisms (from previous version)
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()
        
        # Recursive units
        self.recursive_units: Optional[List[AdvancedFractalUnit]] = None
        
        # Feature integration
        self.integrate = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def reset_states(self):
        """Reset all stateful components"""
        for neuron in self.spiking_neurons:
            neuron.reset_state()
        if self.recursive_units is not None:
            for unit in self.recursive_units:
                unit.reset_states()
    
    def process_spikes(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through spiking neurons"""
        # Split input along channel dimension
        channel_inputs = x.split(1, dim=1)
        
        # Process each channel through its corresponding spiking neuron
        spikes_list = []
        firing_rates = []
        for i, (input_channel, neuron) in enumerate(zip(channel_inputs, self.spiking_neurons)):
            spikes, rate = neuron(input_channel)
            spikes_list.append(spikes)
            firing_rates.append(rate)
        
        # Combine results
        spikes = torch.cat(spikes_list, dim=1)
        firing_rates = torch.cat(firing_rates, dim=1)
        
        # Use firing rates to modulate the output
        return spikes * firing_rates
    
    def should_expand(self, features: torch.Tensor, prediction: torch.Tensor, 
                     confidence: torch.Tensor) -> bool:
        """
        Determine if the unit should expand based on prediction confidence
        and feature complexity
        """
        if self.depth >= self.max_depth:
            return False
        
        # Check prediction confidence
        if confidence.mean() > 0.9:  # High confidence threshold
            return False
        
        # Check feature complexity (from previous version)
        complexity = self.estimate_complexity(features)
        if complexity < 0.3:  # Low complexity threshold
            return False
        
        return True
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        identity = self.shortcut(x)
        metrics = {}
        
        # Base transformation
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply attention (from previous version)
        out = self.channel_attention(out)
        out = self.spatial_attention(out)
        
        # Process through memory module
        memory_out, memory_attention = self.memory(out)
        metrics['memory_attention'] = memory_attention.mean().item()
        
        # Process through spiking neurons
        spike_out = self.process_spikes(memory_out)
        metrics['spike_rate'] = torch.cat([n.spike_history[-1] for n in self.spiking_neurons]).mean().item()
        
        # Self-supervised prediction
        prediction, confidence = self.predictor(spike_out)
        metrics['prediction_confidence'] = confidence.mean().item()
        
        # Decide whether to expand based on combined signals
        if self.should_expand(spike_out, prediction, confidence):
            if self.recursive_units is None:
                self.recursive_units = nn.ModuleList([
                    AdvancedFractalUnit(
                        self.out_channels,
                        self.out_channels,
                        depth=self.depth + 1,
                        max_depth=self.max_depth
                    ) for _ in range(2)
                ])
            
            # Process through recursive units
            recursive_features = []
            recursive_metrics = []
            for unit in self.recursive_units:
                rec_out, rec_metrics = unit(spike_out)
                recursive_features.append(rec_out)
                recursive_metrics.append(rec_metrics)
            
            # Combine recursive features
            if recursive_features:
                combined = torch.cat(recursive_features, dim=1)
                spike_out = spike_out + self.integrate(combined)
                
                # Aggregate recursive metrics
                for i, rec_metrics in enumerate(recursive_metrics):
                    for k, v in rec_metrics.items():
                        metrics[f'recursive_{i}_{k}'] = v
        
        out = F.relu(spike_out + identity)
        return out, metrics

class AdvancedFNNLA(nn.Module):
    """
    Advanced Fractal Neural Network Learning Architecture
    """
    def __init__(self, 
                 in_channels: int, 
                 num_classes: int,
                 base_channels: int = 64,
                 max_depth: int = 3):
        super(AdvancedFNNLA, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Create fractal stages with advanced units
        self.stage1 = AdvancedFractalUnit(base_channels, base_channels, max_depth=max_depth)
        self.stage2 = AdvancedFractalUnit(base_channels, base_channels*2, max_depth=max_depth)
        self.stage3 = AdvancedFractalUnit(base_channels*2, base_channels*4, max_depth=max_depth)
        
        # Global pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels*4, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def reset_states(self):
        """Reset all stateful components in the network"""
        self.stage1.reset_states()
        self.stage2.reset_states()
        self.stage3.reset_states()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        metrics = {}
        
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Process through fractal stages
        x, metrics_1 = self.stage1(x)
        x, metrics_2 = self.stage2(x)
        x, metrics_3 = self.stage3(x)
        
        # Aggregate metrics
        metrics.update({f'stage1_{k}': v for k, v in metrics_1.items()})
        metrics.update({f'stage2_{k}': v for k, v in metrics_2.items()})
        metrics.update({f'stage3_{k}': v for k, v in metrics_3.items()})
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x, metrics
    
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import wandb  # For experiment tracking

class AdvancedFNNLATrainer:
    """
    Advanced trainer for FNN-LA with comprehensive monitoring and management
    of memory, spikes, and self-supervised learning
    """
    def __init__(
        self,
        model: AdvancedFNNLA,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        save_dir: str = './advanced_fnnla_checkpoints',
        enable_wandb: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.enable_wandb = enable_wandb
        
        # Initialize optimizers
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=50,  # Total epochs
            steps_per_epoch=len(train_loader),
            pct_start=0.1,  # Warmup percentage
            anneal_strategy='cos'
        )
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.self_supervised_criterion = nn.MSELoss()
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'memory_stats': defaultdict(list),
            'spike_stats': defaultdict(list),
            'expansion_stats': defaultdict(list),
            'prediction_stats': defaultdict(list)
        }
        
        self.best_accuracy = 0.0
        
        # Initialize wandb if enabled
        if self.enable_wandb:
            wandb.init(
                project="advanced-fnnla",
                config={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "architecture": "Advanced-FNNLA",
                    "dataset": "CIFAR10"
                }
            )
    
    def collect_unit_stats(self, unit: AdvancedFractalUnit, prefix: str = '') -> Dict:
        """
        Collect detailed statistics from a fractal unit
        """
        stats = {
            f'{prefix}memory_attention': unit.memory.memory.abs().mean().item(),
            f'{prefix}spike_rate': np.mean([len(n.spike_history) for n in unit.spiking_neurons]),
            f'{prefix}prediction_confidence': unit.predictor.confidence_estimator(
                unit.predictor.predictor(torch.zeros_like(unit.memory.memory)).detach()
            ).mean().item()
        }
        
        if unit.recursive_units is not None:
            for i, recursive_unit in enumerate(unit.recursive_units):
                recursive_stats = self.collect_unit_stats(
                    recursive_unit,
                    prefix=f'{prefix}recursive_{i}_'
                )
                stats.update(recursive_stats)
        
        return stats
    
    def compute_self_supervised_loss(self, 
                                   features: torch.Tensor, 
                                   predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute self-supervised loss for feature prediction
        """
        # Use the next layer's features as targets
        with torch.no_grad():
            targets = F.adaptive_avg_pool2d(features, predictions.shape[-2:])
        return self.self_supervised_criterion(predictions, targets)
    
    def train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """
        Train for one epoch with comprehensive monitoring
        """
        self.model.train()
        total_loss = 0.0
        total_self_supervised_loss = 0.0
        batch_metrics = defaultdict(list)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass with metrics
            output, metrics = self.model(data)
            
            # Compute classification loss
            class_loss = self.classification_criterion(output, target)
            
            # Compute self-supervised losses for each stage
            self_supervised_loss = torch.tensor(0.0, device=self.device)
            for stage_name in ['stage1', 'stage2', 'stage3']:
                stage = getattr(self.model, stage_name)
                for unit in [stage] + (stage.recursive_units or []):
                    pred, conf = unit.predictor(output)
                    self_supervised_loss += self.compute_self_supervised_loss(
                        output, pred
                    ) * conf.mean()
            
            # Combined loss
            loss = class_loss + 0.1 * self_supervised_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_self_supervised_loss += self_supervised_loss.item()
            
            # Collect detailed metrics every 100 batches
            if batch_idx % 100 == 0:
                stats = self.collect_unit_stats(self.model.stage1, 'stage1_')
                stats.update(self.collect_unit_stats(self.model.stage2, 'stage2_'))
                stats.update(self.collect_unit_stats(self.model.stage3, 'stage3_'))
                
                for k, v in stats.items():
                    batch_metrics[k].append(v)
                
                if self.enable_wandb:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'batch_self_supervised_loss': self_supervised_loss.item(),
                        **stats
                    })
                
                print(f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}]'
                      f' Loss: {loss.item():.4f}'
                      f' Self-Supervised Loss: {self_supervised_loss.item():.4f}')
        
        # Compute average metrics
        avg_metrics = {
            'loss': total_loss / len(self.train_loader),
            'self_supervised_loss': total_self_supervised_loss / len(self.train_loader)
        }
        
        for k, v in batch_metrics.items():
            avg_metrics[k] = np.mean(v)
        
        return avg_metrics
    
    def evaluate(self) -> Tuple[float, float, Dict]:
        """
        Evaluate the model with detailed metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        eval_metrics = defaultdict(list)
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, metrics = self.model(data)
                
                # Compute loss
                loss = self.classification_criterion(output, target)
                total_loss += loss.item()
                
                # Compute accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                # Collect stage metrics
                for k, v in metrics.items():
                    eval_metrics[k].append(v)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / len(self.val_loader.dataset)
        
        # Average evaluation metrics
        avg_eval_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
        
        return avg_loss, accuracy, avg_eval_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """
        Save comprehensive checkpoint including memory states and metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_accuracy': self.best_accuracy
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
        
        # Save detailed metrics
        with open(self.save_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    
    # Continuing from the previous code...

    def plot_metrics(self):
        """
        Generate comprehensive visualization of network behavior
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Loss curves
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(self.metrics['train_loss'], label='Train Loss')
        ax1.plot(self.metrics['val_loss'], label='Val Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot 2: Accuracy
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(self.metrics['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        
        # Plot 3: Memory Statistics
        ax3 = plt.subplot(3, 2, 3)
        for key, values in self.metrics['memory_stats'].items():
            ax3.plot(values, label=key)
        ax3.set_title('Memory Module Statistics')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Memory Usage')
        ax3.legend()
        
        # Plot 4: Spike Statistics
        ax4 = plt.subplot(3, 2, 4)
        for key, values in self.metrics['spike_stats'].items():
            ax4.plot(values, label=key)
        ax4.set_title('Spiking Neuron Statistics')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Spike Rate')
        ax4.legend()
        
        # Plot 5: Expansion Statistics
        ax5 = plt.subplot(3, 2, 5)
        expansion_data = np.array([
            [stats[k] for k in self.metrics['expansion_stats'].keys()]
            for stats in self.metrics['expansion_stats'].values()
        ])
        ax5.plot(expansion_data)
        ax5.set_title('Network Expansion Statistics')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Number of Units')
        
        # Plot 6: Prediction Confidence
        ax6 = plt.subplot(3, 2, 6)
        for key, values in self.metrics['prediction_stats'].items():
            ax6.plot(values, label=key)
        ax6.set_title('Self-Supervised Prediction Confidence')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Confidence Score')
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_metrics.png')
        plt.close()
    
    def train(self, num_epochs: int):
        """
        Full training loop with comprehensive monitoring and adaptation
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            start_time = time.time()
            
            # Reset stateful components at epoch start
            self.model.reset_states()
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            self.metrics['train_loss'].append(train_metrics['loss'])
            
            # Evaluation phase
            val_loss, accuracy, eval_metrics = self.evaluate()
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_accuracy'].append(accuracy)
            
            # Update detailed metrics
            for category in ['memory_stats', 'spike_stats', 'expansion_stats', 'prediction_stats']:
                for k, v in train_metrics.items():
                    if k.startswith(category.split('_')[0]):
                        self.metrics[category][k].append(v)
            
            # Log to wandb if enabled
            if self.enable_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_loss,
                    'accuracy': accuracy,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    **train_metrics,
                    **eval_metrics
                })
            
            # Save checkpoint if best model
            is_best = accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = accuracy
            
            self.save_checkpoint(epoch, self.metrics, is_best)
            
            # Plot current metrics
            self.plot_metrics()
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1} completed in {epoch_time:.2f}s')
            print(f'Train Loss: {train_metrics["loss"]:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Best Accuracy: {self.best_accuracy:.4f}')
            
            # Print interesting statistics
            print('\nNetwork Statistics:')
            print(f'Average Memory Usage: {np.mean(list(train_metrics.values())):.4f}')
            print(f'Average Spike Rate: {train_metrics.get("spike_rate", 0):.4f}')
            print(f'Prediction Confidence: {train_metrics.get("prediction_confidence", 0):.4f}')

def train_advanced_fnnla(
    dataset: str = 'cifar10',
    epochs: int = 50,
    batch_size: int = 128,
    save_dir: str = './advanced_fnnla_results',
    enable_wandb: bool = True
):
    """
    Main training function for Advanced FNN-LA
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading and preprocessing
    if dataset.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10('./data', train=True, 
                                       download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('./data', train=False,
                                      transform=transform_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        model = AdvancedFNNLA(
            in_channels=3, 
            num_classes=10,
            base_channels=64,
            max_depth=3
        ).to(device)
        
        # Initialize trainer
        trainer = AdvancedFNNLATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            device=device,
            save_dir=save_dir,
            enable_wandb=enable_wandb
        )
        
        # Train model
        trainer.train(num_epochs=epochs)
        
        return model, trainer.metrics

if __name__ == "__main__":
    model, metrics = train_advanced_fnnla(
        epochs=50,
        enable_wandb=False  # Set to False if not using wandb
    )
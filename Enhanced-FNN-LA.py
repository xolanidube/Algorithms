import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from collections import deque
import random
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Mixup:
    """Mixup augmentation with proper training mode handling"""
    def __init__(
        self,
        alpha: float = 0.2,
        num_classes: int = 10,
        p: float = 0.5
    ):
        self.alpha = alpha
        self.num_classes = num_classes
        self.p = p
        self.distribution = torch.distributions.Beta(alpha, alpha)
        self._training = True  # Use private attribute for training state
    
    def train(self, mode: bool = True) -> 'Mixup':
        """Set training mode"""
        self._training = mode
        return self
    
    def eval(self) -> 'Mixup':
        """Set evaluation mode"""
        self._training = False
        return self
    
    def __call__(
        self,
        x: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup to the input batch"""
        if not self._training or torch.rand(1) > self.p:
            return x, self._to_one_hot(target)
        
        batch_size = x.size(0)
        
        # Sample mixing parameter
        lam = self.distribution.sample()
        lam = torch.clamp(lam, min=0.1, max=0.9)
        
        # Generate permutation
        index = torch.randperm(batch_size, device=x.device)
        
        # Mix the samples
        mixed_x = lam * x + (1 - lam) * x[index]
        
        # Mix the targets
        target_one_hot = self._to_one_hot(target)
        target_shuffled = target_one_hot[index]
        mixed_target = lam * target_one_hot + (1 - lam) * target_shuffled
        
        return mixed_x, mixed_target
    
    def _to_one_hot(self, target: torch.Tensor) -> torch.Tensor:
        """Convert class indices to one-hot encoded targets"""
        one_hot = torch.zeros(
            target.size(0),
            self.num_classes,
            device=target.device
        )
        return one_hot.scatter_(1, target.unsqueeze(1), 1)
    
    def training(self, mode: bool = True):
        """Set training mode"""
        self.training = mode
        return self

class MixupLoss(nn.Module):
    """
    Loss function for mixup training that handles soft targets
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input: Model predictions [B, C]
            target: Soft targets from mixup [B, C]
        """
        loss = -(target * F.log_softmax(input, dim=1)).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class HybridAttention(nn.Module):
    """
    Enhanced attention mechanism combining channel and spatial attention with
    adaptive weighting and resource-aware computation.
    """
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
        temperature: float = 1.0
    ):
        super().__init__()
        
        # Channel attention components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for channel attention
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel_size,
                                    padding=spatial_kernel_size//2)
        
        # Adaptive fusion mechanism
        self.fusion_weights = nn.Parameter(torch.FloatTensor([0.5, 0.5]))
        self.temperature = temperature
        
        # Resource usage tracking
        self.computation_history = deque(maxlen=100)
        self.attention_scores = None
    
    def compute_channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute channel attention with efficiency improvements
        """
        batch_size, channels, _, _ = x.size()
        
        # Compute pooling efficiently
        avg_pool = self.avg_pool(x).view(batch_size, channels)
        max_pool = self.max_pool(x).view(batch_size, channels)
        
        # Shared MLP processing
        avg_out = self.channel_mlp(avg_pool)
        max_out = self.channel_mlp(max_pool)
        
        # Combine attention scores
        channel_attention = torch.sigmoid((avg_out + max_out) / self.temperature)
        return channel_attention.view(batch_size, channels, 1, 1)
    
    def compute_spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention with efficiency improvements
        """
        # Efficient pooling operations
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        
        # Concatenate and process
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = torch.sigmoid(self.spatial_conv(spatial_features))
        
        return spatial_attention
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive fusion and resource tracking
        """
        # Record input size for resource tracking
        input_size = x.size()
        
        # Compute attention mechanisms
        channel_attention = self.compute_channel_attention(x)
        spatial_attention = self.compute_spatial_attention(x)
        
        # Adaptive fusion with learned weights
        fusion_weights = F.softmax(self.fusion_weights / self.temperature, dim=0)
        
        # Apply attention
        channel_out = x * channel_attention
        spatial_out = x * spatial_attention
        
        # Combine outputs
        output = fusion_weights[0] * channel_out + fusion_weights[1] * spatial_out
        
        # Track computation cost
        computation_cost = np.prod(input_size) * (
            np.sum(fusion_weights.detach().cpu().numpy())
        )
        self.computation_history.append(computation_cost)
        
        # Store attention scores for analysis
        self.attention_scores = {
            'channel': channel_attention.detach(),
            'spatial': spatial_attention.detach(),
            'fusion_weights': fusion_weights.detach()
        }
        
        return output
    
    def get_computation_stats(self) -> Dict[str, float]:
        """
        Get statistics about computation costs
        """
        history = np.array(self.computation_history)
        return {
            'mean_cost': float(np.mean(history)),
            'max_cost': float(np.max(history)),
            'min_cost': float(np.min(history)),
            'std_cost': float(np.std(history))
        }

class AdaptivePruningMechanism:
    """
    Adaptive pruning mechanism that adjusts thresholds based on unit performance
    and resource utilization.
    """
    def __init__(
        self,
        initial_threshold: float = 0.3,
        min_threshold: float = 0.1,
        max_threshold: float = 0.5,
        history_size: int = 100,
        adaptation_rate: float = 0.01
    ):
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.history = deque(maxlen=history_size)
        self.adaptation_rate = adaptation_rate
        self.complexity_history = deque(maxlen=history_size)
        self.memory_usage_history = deque(maxlen=history_size)
        
    def update_threshold(
        self, 
        current_complexity: float,
        memory_usage: float,
        performance_metric: float
    ) -> float:
        """
        Update pruning threshold based on recent history and current metrics
        """
        self.complexity_history.append(current_complexity)
        self.memory_usage_history.append(memory_usage)
        
        # Calculate trends
        complexity_trend = self._calculate_trend(self.complexity_history)
        memory_trend = self._calculate_trend(self.memory_usage_history)
        
        # Adjust threshold based on trends and performance
        adjustment = 0.0
        
        # If complexity is increasing but performance isn't improving
        if complexity_trend > 0 and performance_metric < 0:
            adjustment -= self.adaptation_rate
        
        # If memory usage is high
        if memory_trend > 0:
            adjustment -= self.adaptation_rate * 0.5
        
        # If performance is improving
        if performance_metric > 0:
            adjustment += self.adaptation_rate * 0.25
        
        # Update threshold with bounds
        self.threshold = np.clip(
            self.threshold + adjustment,
            self.min_threshold,
            self.max_threshold
        )
        
        return self.threshold
    
    def _calculate_trend(self, history: deque) -> float:
        """
        Calculate the trend in recent history
        """
        if len(history) < 2:
            return 0.0
        
        recent = list(history)[-10:]
        if len(recent) < 2:
            return 0.0
            
        return np.polyfit(range(len(recent)), recent, 1)[0]

    def get_pruning_decision(
        self,
        unit_complexity: float,
        memory_usage: float,
        performance_delta: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Make pruning decision based on current metrics
        """
        current_threshold = self.update_threshold(
            unit_complexity,
            memory_usage,
            performance_delta
        )
        
        metrics = {
            'threshold': current_threshold,
            'complexity': unit_complexity,
            'memory_usage': memory_usage,
            'performance_delta': performance_delta
        }
        
        should_prune = unit_complexity < current_threshold
        return should_prune, metrics

class MemoryAwarePruning:
    """
    Memory-aware pruning mechanism that considers both performance and resource constraints
    """
    def __init__(
        self,
        memory_threshold: float = 0.8,  # 80% memory utilization threshold
        min_unit_size: int = 1000,      # Minimum unit size to consider for pruning
        history_size: int = 100
    ):
        self.memory_threshold = memory_threshold
        self.min_unit_size = min_unit_size
        self.pruning_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        
    def estimate_memory_usage(self, unit: nn.Module) -> float:
        """
        Estimate memory usage of a unit
        """
        total_params = 0
        total_buffers = 0
        
        # Count parameters
        for param in unit.parameters():
            total_params += param.nelement() * param.element_size()
            
        # Count buffers
        for buffer in unit.buffers():
            total_buffers += buffer.nelement() * buffer.element_size()
            
        return (total_params + total_buffers) / 1024 / 1024  # Convert to MB
    
    def should_prune(
        self,
        unit: nn.Module,
        performance_metric: float,
        current_memory_usage: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Decide whether to prune based on memory and performance considerations
        """
        unit_memory = self.estimate_memory_usage(unit)
        self.memory_history.append(current_memory_usage)
        
        # Calculate memory trend
        memory_trend = (
            np.mean(list(self.memory_history)[-10:])
            if len(self.memory_history) >= 10
            else current_memory_usage
        )
        
        # Decision metrics
        metrics = {
            'unit_memory': unit_memory,
            'total_memory': current_memory_usage,
            'memory_trend': memory_trend,
            'performance': performance_metric
        }
        
        # Pruning decision based on multiple factors
        should_prune = (
            (current_memory_usage > self.memory_threshold and 
             unit_memory > self.min_unit_size and
             performance_metric < 0.1) or
            (memory_trend > self.memory_threshold * 0.9 and
             performance_metric < 0.05)
        )
        
        self.pruning_history.append(1 if should_prune else 0)
        
        return should_prune, metrics
    
    def get_pruning_stats(self) -> Dict[str, float]:
        """
        Get statistics about pruning decisions
        """
        history = np.array(self.pruning_history)
        return {
            'prune_rate': float(np.mean(history)),
            'total_pruned': int(np.sum(history)),
            'recent_prune_rate': float(np.mean(history[-10:]))
            if len(history) >= 10 else 0.0
        }

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
        self.learning_rate = 0.002
        
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

class EnhancedFractalUnit(nn.Module):
    """
    Enhanced Fractal Unit with adaptive pruning, hybrid attention, and memory awareness
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 0,
        max_depth: int = 3,
        reduction_ratio: int = 16
    ):
        super().__init__()
        
        self.depth = depth
        self.max_depth = max_depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize enhanced components
        self.adaptive_pruning = AdaptivePruningMechanism()
        self.memory_pruning = MemoryAwarePruning()
        self.hybrid_attention = HybridAttention(
            out_channels,
            reduction_ratio=reduction_ratio
        )
        
        # Base convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Recursive units
        self.recursive_units = None
        
        # Feature integration
        self.integrate = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.complexity_history = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        
        # Pruning state
        self.is_pruned = False
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Current metrics storage
        self.current_metrics = {}
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics"""
        return self.current_metrics
    
    def get_state(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get state representation for RL agent
        """
        with torch.no_grad():
            variance = torch.var(features, dim=[2, 3]).mean().item()
            sparsity = (features.abs() < 1e-3).float().mean().item()
            mean_activation = features.abs().mean().item()
            current_depth = self.depth / self.max_depth
            
            return torch.tensor([variance, sparsity, mean_activation, current_depth])
    
    def expand(self, features: torch.Tensor) -> bool:
        """
        Use RL agent to decide whether to expand
        """
        if self.depth >= self.max_depth:
            return False
            
        state = self.get_state(features)
        action = self.rl_agent.get_action(state)
        
        if action == 1:  # Expand
            if self.recursive_units is None:
                self.recursive_units = nn.ModuleList([
                    EnhancedFractalUnit(
                        self.out_channels,
                        self.out_channels,
                        depth=self.depth + 1,
                        max_depth=self.max_depth
                    ) for _ in range(2)
                ])
            return True
        return False
    
    def estimate_complexity(self, features: torch.Tensor) -> float:
        """
        Estimate feature complexity using activation statistics
        """
        with torch.no_grad():
            # Calculate feature activation variance
            var = torch.var(features, dim=[2, 3]).mean().item()
            
            # Calculate feature sparsity
            sparsity = (features.abs() < 1e-3).float().mean().item()
            
            # Combine metrics
            self.feature_complexity = var * (1 - sparsity)
            return self.feature_complexity
    
    def calculate_unit_performance(self, features: torch.Tensor) -> float:
        """
        Calculate unit performance based on feature quality and resource usage
        """
        with torch.no_grad():
            # Feature effectiveness
            feature_variance = torch.var(features, dim=[2, 3]).mean().item()
            feature_sparsity = (features.abs() < 1e-3).float().mean().item()
            
            # Resource metrics
            memory_usage = self.memory_pruning.estimate_memory_usage(self)
            computation_stats = self.hybrid_attention.get_computation_stats()
            
            # Combine metrics
            performance = (
                feature_variance * (1 - feature_sparsity) / 
                (1 + memory_usage * computation_stats['mean_cost'])
            )
            
            self.performance_history.append(performance)
            return performance
    
    def update_pruning_state(
        self,
        features: torch.Tensor,
        current_memory_usage: float
    ) -> Dict[str, float]:
        """
        Update pruning state based on performance and resource usage
        """
        # Calculate current performance
        current_performance = self.calculate_unit_performance(features)
        
        # Calculate performance trend
        performance_delta = (
            current_performance - self.performance_history[-2]
            if len(self.performance_history) > 1
            else 0.0
        )
        
        # Get complexity metrics
        unit_complexity = self.adaptive_pruning.get_pruning_decision(
            current_performance,
            current_memory_usage,
            performance_delta
        )[1]['complexity']
        
        # Update memory usage history
        self.memory_usage_history.append(current_memory_usage)
        
        # Make pruning decisions
        should_prune_adaptive, adaptive_metrics = self.adaptive_pruning.get_pruning_decision(
            unit_complexity,
            current_memory_usage,
            performance_delta
        )
        
        should_prune_memory, memory_metrics = self.memory_pruning.should_prune(
            self,
            performance_delta,
            current_memory_usage
        )
        
        # Combined decision
        self.is_pruned = should_prune_adaptive or should_prune_memory
        
        # Return combined metrics
        return {
            **adaptive_metrics,
            **memory_metrics,
            'is_pruned': self.is_pruned,
            'performance': current_performance,
            'performance_delta': performance_delta
        }
    
    def expand_recursive_units(self) -> bool:
        """
        Decide whether to expand recursive units based on current state
        """
        if self.depth >= self.max_depth:
            return False
            
        if self.recursive_units is None and not self.is_pruned:
            # Create new recursive units
            self.recursive_units = nn.ModuleList([
                EnhancedFractalUnit(
                    self.out_channels,
                    self.out_channels,
                    depth=self.depth + 1,
                    max_depth=self.max_depth
                ) for _ in range(2)
            ])
            return True
            
        return False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with metrics collection"""
        # Reset metrics
        self.current_metrics = {}
        
        # Check pruning state
        if self.is_pruned:
            self.current_metrics['pruned'] = True
            return self.shortcut(x)
        
        identity = self.shortcut(x)
        
        # Base transformation
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply hybrid attention
        out = self.hybrid_attention(out)
        
        # Update metrics (stored separately)
        current_memory = self.memory_pruning.estimate_memory_usage(self)
        with torch.no_grad():
            # Feature effectiveness
            feature_variance = torch.var(out, dim=[2, 3]).mean().item()
            feature_sparsity = (out.abs() < 1e-3).float().mean().item()
            
            self.current_metrics.update({
                'feature_variance': feature_variance,
                'feature_sparsity': feature_sparsity,
                'memory_usage': current_memory,
                'is_pruned': self.is_pruned
            })
        
        # Process through recursive units if not pruned
        if not self.is_pruned and self.recursive_units is not None:
            recursive_features = []
            
            for unit in self.recursive_units:
                rec_out = unit(out)
                recursive_features.append(rec_out)
                # Collect recursive metrics
                self.current_metrics[f'recursive_unit_{id(unit)}'] = unit.get_metrics()
            
            if recursive_features:
                combined = torch.cat(recursive_features, dim=1)
                out = out + self.integrate(combined)
        
        out = F.relu(out + identity)
        return out

    def expand_recursive_units(self) -> bool:
        """
        Decide whether to expand recursive units based on current state
        """
        if self.depth >= self.max_depth:
            return False
            
        if self.recursive_units is None and not self.is_pruned:
            # Create new recursive units
            self.recursive_units = nn.ModuleList([
                EnhancedFractalUnit(
                    self.out_channels,
                    self.out_channels,
                    depth=self.depth + 1,
                    max_depth=self.max_depth
                ) for _ in range(2)
            ])
            return True
            
        return False

    def get_unit_stats(self) -> Dict[str, float]:
        """
        Get comprehensive unit statistics
        """
        return {
            'depth': self.depth,
            'is_pruned': self.is_pruned,
            'memory_usage': float(np.mean(list(self.memory_usage_history))),
            'performance': float(np.mean(list(self.performance_history))),
            'pruning_stats': self.memory_pruning.get_pruning_stats(),
            'attention_stats': self.hybrid_attention.get_computation_stats()
        }

class EnhancedFNNLA(nn.Module):
    """
    Enhanced Fractal Neural Network with advanced components
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 64,
        max_depth: int = 3,
        dropout_rate=0.6,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.5
    ):
        super().__init__()
        
        # Initialize mixup
        self.mixup = Mixup(
            alpha=mixup_alpha,
            num_classes=num_classes,
            p=mixup_prob
        )
        self.mixup_loss = MixupLoss()
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.stochastic_depth_prob = 0.1
        
        # Enhanced fractal stages
        self.stage1 = EnhancedFractalUnit(
            base_channels, base_channels,
            max_depth=max_depth
        )
        self.stage2 = EnhancedFractalUnit(
            base_channels, base_channels*2,
            max_depth=max_depth
        )
        self.stage3 = EnhancedFractalUnit(
            base_channels*2, base_channels*4,
            max_depth=max_depth
        )
        
        # Global pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels*4, num_classes)
        
        # Performance tracking
        self.memory_tracker = MemoryAwarePruning()
        self.metrics_history = []
        
        self._initialize_weights()
        
        # Current metrics storage
        self.current_metrics = {}
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_metrics(self) -> Dict[str, any]:
        """Get current metrics from all components"""
        metrics = {
            'stage1': self.stage1.get_metrics(),
            'stage2': self.stage2.get_metrics(),
            'stage3': self.stage3.get_metrics(),
            'total_memory': self.memory_tracker.estimate_memory_usage(self)
        }
        metrics.update(self.current_metrics)
        return metrics
    
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        apply_mixup: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional mixup augmentation
        
        Args:
            x: Input tensor [B, C, H, W]
            target: Optional target tensor [B]
            apply_mixup: Whether to apply mixup (True during training)
            
        Returns:
            If target is None:
                output: Model predictions
            If target is provided:
                tuple: (output, mixed_target)
        """
        # Apply mixup if in training mode and targets provided
        if self.training and target is not None and apply_mixup:
            x, mixed_target = self.mixup(x, target)
        else:
            mixed_target = None
        
        # Reset metrics
        self.current_metrics = {}
        
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.dropout(x)  # Apply dropout after initial convolution
        x = self.stage1(x)
        
        x = self.dropout(x)  # Dropout after each stage to prevent overfitting
        x = self.stage2(x)
        
        x = self.dropout(x)
        x = self.stage3(x)
        
        x = self.dropout(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mixup_target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate loss with mixup if applicable
        
        Args:
            output: Model predictions [B, C]
            target: Original targets [B]
            mixup_target: Optional mixed targets [B, C]
        """
        if mixup_target is not None:
            return self.mixup_loss(output, mixup_target)
        return F.cross_entropy(output, target)
    
    def train(self, mode: bool = True):
        """Override train mode to handle mixup state"""
        super().train(mode)
        if hasattr(self, 'mixup'):
            self.mixup.train(mode)  # Use train() method instead of setting attribute
        return self

    def _save_epoch_metrics(self, epoch: int, metrics: Dict):
        """Save detailed metrics for current epoch"""
        # Create metrics directory if it doesn't exist
        metrics_dir = self.save_dir / 'epoch_metrics'
        metrics_dir.mkdir(exist_ok=True)
        
        # Prepare metrics for saving (convert tensors/arrays to python types)
        saveable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.ndarray, np.number)):
                saveable_metrics[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                saveable_metrics[key] = value.detach().cpu().tolist()
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # Handle lists of tensors/arrays
                if isinstance(value[0], (np.ndarray, np.number, torch.Tensor)):
                    saveable_metrics[key] = [
                        v.tolist() if isinstance(v, (np.ndarray, torch.Tensor))
                        else float(v) for v in value
                    ]
                else:
                    saveable_metrics[key] = value
            else:
                saveable_metrics[key] = value
        
        # Add timestamp and epoch info
        saveable_metrics['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        saveable_metrics['epoch'] = epoch
        
        # Save to file
        metrics_file = metrics_dir / f'epoch_{epoch:03d}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(saveable_metrics, f, indent=4, default=str)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, List, Tuple
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping handler to prevent overfitting"""
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        mode: str = 'min',
        baseline: Optional[float] = None,
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.counter = 0
        self.best_score = None
        self.stop_training = False
        
        # Set comparison function based on mode
        if mode == 'min':
            self._is_better = lambda score, best: score < (best - min_delta)
        else:
            self._is_better = lambda score, best: score > (best + min_delta)
    
    def __call__(self, model: nn.Module, current_score: float) -> bool:
        """
        Check if training should stop and update best weights
        
        Args:
            model: Current model
            current_score: Current validation metric
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            # First call
            self.best_score = current_score
            self.best_weights = {
                k: v.cpu().clone().detach() 
                for k, v in model.state_dict().items()
            }
            return False
            
        if self._is_better(current_score, self.best_score):
            # Score improved
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone().detach() 
                    for k, v in model.state_dict().items()
                }
        else:
            # Score did not improve
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        return False
    
    def reset(self):
        """Reset the early stopping handler"""
        self.counter = 0
        self.best_score = None
        self.stop_training = False
        self.best_weights = None

class CheckpointHandler:
    """
    Handles saving and loading of model checkpoints
    """
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.save_dir / 'checkpoint.pth'
        self.best_model_path = self.save_dir / 'best_model.pth'
        self.metrics_path = self.save_dir / 'training_metrics.json'
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        metrics: Dict,
        is_best: bool = False
    ):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_path)
        
        # Save best model if needed
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            
        # Save metrics separately for easy access
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        load_best: bool = False
    ) -> Tuple[int, Dict]:
        """Load training checkpoint"""
        checkpoint_path = self.best_model_path if load_best else self.checkpoint_path
        
        if not checkpoint_path.exists():
            logger.info(f"No checkpoint found at {checkpoint_path}")
            return 0, {}
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['epoch'], checkpoint['metrics']

class EnhancedFNNLATrainer:
    """
    Enhanced trainer with adaptive pruning and resource management
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        checkpoint_handler: Optional[CheckpointHandler] = None,
        metrics: Optional[Dict] = None,
        save_dir: str = './enhanced_fnnla_results',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 7,
        min_delta: float = 0.001
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or use provided optimizer
        self.optimizer = optimizer if optimizer is not None else optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize or use provided scheduler
        self.scheduler = scheduler if scheduler is not None else optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,       # Reduced from 10
            T_mult=2,
            eta_min=1e-6 # Added minimum LR
        )
        
        # Initialize or use provided checkpoint handler
        self.checkpoint_handler = checkpoint_handler if checkpoint_handler is not None else CheckpointHandler(save_dir)
        
        # Initialize metrics with default values if not provided
        self.metrics = self._initialize_metrics(metrics)
        
        # Initialize best accuracy
        self.best_accuracy = max(self.metrics['accuracy']) if self.metrics['accuracy'] else 0.0
        
        # Initialize epoch metrics history
        self.epoch_metrics_history = []
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            mode='min',  # We're monitoring validation loss
            restore_best_weights=True
        )
        
        # Training history for analysis
        self.training_history = []
    
    def _initialize_metrics(self, provided_metrics: Optional[Dict] = None) -> Dict:
        """Initialize metrics dictionary with required fields"""
        default_metrics = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'learning_rates': [],
            'memory_usage': [],
            'pruning_stats': [],
            'attention_stats': [],
            'stage_metrics': [],
            'epoch_times': [],
            'cumulative_time': [],
            'batch_sizes': [],
            'stage1_metrics': [],
            'stage2_metrics': [],
            'stage3_metrics': []
        }
        
        if provided_metrics is not None:
            # Ensure all required keys exist in provided metrics
            for key in default_metrics:
                if key not in provided_metrics:
                    provided_metrics[key] = []
            return provided_metrics
        
        return default_metrics
    
    def _update_metrics(self, epoch_metrics: Dict[str, any]):
        """Update metrics history with new epoch data"""
        for key in self.metrics:
            if key in epoch_metrics:
                self.metrics[key].append(epoch_metrics[key])
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Training loop for one epoch with enhanced monitoring"""
        self.model.train()
        total_loss = 0.0
        total_metrics = defaultdict(float)
        batch_metrics = []
        epoch_grad_norms = []  # Track gradient norms across epoch
        
        epoch_start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            
            # Get metrics after forward pass
            batch_model_metrics = self.model.get_metrics()
            
            # Backward pass
            loss.backward()
            
            # Gradient monitoring
            batch_grad_norms = []
            for param in self.model.parameters():
                if param.grad is not None:
                    batch_grad_norms.append(param.grad.norm().item())
            
            current_grad_norm = np.mean(batch_grad_norms)
            epoch_grad_norms.append(current_grad_norm)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            batch_metrics.append({
                'batch_idx': batch_idx,
                'loss': loss.item(),
                'grad_norm': current_grad_norm,
                **batch_model_metrics
            })
            
            # Update running metrics
            for key, value in batch_model_metrics.items():
                if isinstance(value, (int, float)):
                    total_metrics[key] += value
            
            # Log progress
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}]'
                      f' Loss: {loss.item():.4f}')
                print(f'Gradient Norm: {current_grad_norm:.4f}')
                
                # Print resource usage
                print(f'Memory Usage: {batch_model_metrics.get("total_memory", 0):.2f} MB')
                
                # Print stage metrics
                for stage in ['stage1', 'stage2', 'stage3']:
                    if stage in batch_model_metrics:
                        print(f'{stage} metrics:', batch_model_metrics[stage])
        
        # Compute epoch metrics
        epoch_time = time.time() - epoch_start_time
        num_batches = len(self.train_loader)
        
        avg_metrics = {
            'train_loss': total_loss / num_batches,
            'epoch_time': epoch_time,
            'batches_processed': num_batches,
            'mean_grad_norm': np.mean(epoch_grad_norms),
            'max_grad_norm': np.max(epoch_grad_norms),
            'min_grad_norm': np.min(epoch_grad_norms),
            **{k: v / num_batches for k, v in total_metrics.items()},
            'batch_metrics': batch_metrics,
            'grad_norm_history': epoch_grad_norms
        }
        
        # Log gradient statistics
        print(f'\nGradient Statistics:')
        print(f'Mean Gradient Norm: {avg_metrics["mean_grad_norm"]:.4f}')
        print(f'Max Gradient Norm: {avg_metrics["max_grad_norm"]:.4f}')
        print(f'Min Gradient Norm: {avg_metrics["min_grad_norm"]:.4f}')
        
        return avg_metrics
    def validate(self) -> Dict[str, float]:
        """Validation with enhanced metrics collection"""
        self.model.eval()
        val_loss = 0
        correct = 0
        val_metrics = defaultdict(float)
        batch_metrics = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Get metrics
                batch_model_metrics = self.model.get_metrics()
                
                # Calculate validation metrics
                batch_loss = F.cross_entropy(output, target).item()
                val_loss += batch_loss
                pred = output.argmax(dim=1, keepdim=True)
                batch_correct = pred.eq(target.view_as(pred)).sum().item()
                correct += batch_correct
                
                # Store batch metrics
                batch_metrics.append({
                    'batch_idx': batch_idx,
                    'loss': batch_loss,
                    'correct': batch_correct,
                    **batch_model_metrics
                })
                
                # Update running metrics
                for key, value in batch_model_metrics.items():
                    if isinstance(value, (int, float)):
                        val_metrics[key] += value
        
        # Average metrics
        num_batches = len(self.val_loader)
        dataset_size = len(self.val_loader.dataset)
        
        return {
            'val_loss': val_loss / num_batches,
            'accuracy': correct / dataset_size,
            **{k: v / num_batches for k, v in val_metrics.items()},
            'batch_metrics': batch_metrics
        }
    
    def train(self, num_epochs: int, start_epoch: int = 0) -> Dict:
        """Full training loop with enhanced monitoring and early stopping"""
        training_start_time = time.time()
        best_model_path = self.save_dir / 'best_model.pth'
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            epoch_start_time = time.time()
            print(f'\nEpoch {epoch+1}/{start_epoch + num_epochs}')
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['val_loss'])  # Using validation loss for scheduling
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Collect epoch metrics
            epoch_metrics = {
                **train_metrics,
                **val_metrics,
                'epoch': epoch,
                'learning_rate': current_lr,
                'epoch_time': time.time() - epoch_start_time,
                'total_time': time.time() - training_start_time
            }
            
            # Update metrics history
            self._update_metrics(epoch_metrics)
            self.training_history.append(epoch_metrics)
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = val_metrics['accuracy']
            
            self.checkpoint_handler.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=self.metrics,
                is_best=is_best
            )
            
            # Print epoch summary
            self._print_epoch_summary(epoch_metrics)
            
            # Early stopping check
            if self.early_stopping(self.model, val_metrics['val_loss']):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {self.early_stopping.best_score:.4f}")
                
                # Save final state
                self.checkpoint_handler.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=self.metrics,
                    is_best=True
                )
                break
            
            # Save detailed metrics to file
            self._save_epoch_metrics(epoch, epoch_metrics)
            
            # Plot current progress
            self._plot_metrics()
        
        # Final training summary
        self._print_training_summary()
        return self.metrics
    
    def _print_epoch_summary(self, metrics: Dict):
        """Print comprehensive epoch summary"""
        print(f'\nEpoch Summary:')
        print(f'Train Loss: {metrics["train_loss"]:.4f}')
        print(f'Val Loss: {metrics["val_loss"]:.4f}')
        print(f'Accuracy: {metrics["accuracy"]:.4f}')
        print(f'Best Accuracy: {self.best_accuracy:.4f}')
        print(f'Learning Rate: {metrics["learning_rate"]:.6f}')
        print(f'Epoch Time: {metrics["epoch_time"]:.2f}s')
        
        # Print resource usage
        print('\nResource Usage:')
        print(f'Memory Usage: {metrics.get("total_memory", 0):.2f} MB')
        
        # Print early stopping status
        if hasattr(self, 'early_stopping'):
            print(f'Early Stopping Counter: {self.early_stopping.counter}/{self.early_stopping.patience}')
    
    def _print_training_summary(self):
        """Print comprehensive training summary"""
        print("\nTraining Summary:")
        print(f"Best Validation Accuracy: {self.best_accuracy:.4f}")
        print(f"Best Validation Loss: {min(self.metrics['val_loss']):.4f}")
        print(f"Total Training Time: {sum(self.metrics['epoch_times']):.2f}s")
        
        # Print convergence information
        print("\nConvergence Information:")
        print(f"Final Learning Rate: {self.metrics['learning_rates'][-1]:.6f}")
        print(f"Early Stopping Triggered: {self.early_stopping.stop_training}")
        
        # Print resource usage summary
        print("\nResource Usage Summary:")
        avg_memory = np.mean(self.metrics['memory_usage'])
        peak_memory = max(self.metrics['memory_usage'])
        print(f"Average Memory Usage: {avg_memory:.2f} MB")
        print(f"Peak Memory Usage: {peak_memory:.2f} MB")
    
    def _plot_metrics(self):
        """Plot training metrics if matplotlib is available"""
        try:
            import matplotlib.pyplot as plt
            
            # Create directory for plots
            plots_dir = self.save_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Plot loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['train_loss'], label='Train Loss')
            plt.plot(self.metrics['val_loss'], label='Val Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(plots_dir / 'loss_curves.png')
            plt.close()
            
            # Plot accuracy
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['accuracy'], label='Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(plots_dir / 'accuracy.png')
            plt.close()
            
            # Plot memory usage
            if self.metrics['memory_usage']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics['memory_usage'], label='Memory Usage')
                plt.title('Memory Usage Over Time')
                plt.xlabel('Epoch')
                plt.ylabel('Memory (MB)')
                plt.legend()
                plt.savefig(plots_dir / 'memory_usage.png')
                plt.close()
        
        except ImportError:
            print("Warning: matplotlib not available for plotting")

    def _save_epoch_metrics(self, epoch: int, metrics: Dict):
        """Save detailed metrics for current epoch"""
        # Create metrics directory if it doesn't exist
        metrics_dir = self.save_dir / 'epoch_metrics'
        metrics_dir.mkdir(exist_ok=True)
        
        # Prepare metrics for saving (convert tensors/arrays to python types)
        saveable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.ndarray, np.number)):
                saveable_metrics[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                saveable_metrics[key] = value.detach().cpu().tolist()
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # Handle lists of tensors/arrays
                if isinstance(value[0], (np.ndarray, np.number, torch.Tensor)):
                    saveable_metrics[key] = [
                        v.tolist() if isinstance(v, (np.ndarray, torch.Tensor))
                        else float(v) for v in value
                    ]
                else:
                    saveable_metrics[key] = value
            else:
                saveable_metrics[key] = value
        
        # Add timestamp and epoch info
        saveable_metrics['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        saveable_metrics['epoch'] = epoch
        
        # Save to file
        metrics_file = metrics_dir / f'epoch_{epoch:03d}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(saveable_metrics, f, indent=4, default=str)



class FNNLATrainer:
    """
    Trainer class for Enhanced FNNLA with monitoring capabilities
    """
    def __init__(
        self,
        model: EnhancedFNNLA,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        save_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'pruning_stats': defaultdict(list),
            'complexity_stats': defaultdict(list),
            'attention_stats': defaultdict(list),
            'expansion_stats': defaultdict(list)
        }
        
        self.best_accuracy = 0.0
        
    def collect_unit_stats(self, unit: EnhancedFractalUnit, prefix: str = '') -> Dict:
        """
        Collect statistics from a fractal unit
        """
        stats = {
            f'{prefix}complexity': unit.feature_complexity,
            f'{prefix}is_pruned': int(unit.is_pruned),
            f'{prefix}prune_counter': unit.prune_counter
        }
        
        if unit.recursive_units is not None:
            for i, recursive_unit in enumerate(unit.recursive_units):
                recursive_stats = self.collect_unit_stats(
                    recursive_unit,
                    prefix=f'{prefix}recursive_{i}_'
                )
                stats.update(recursive_stats)
        
        return stats
    
    def collect_network_stats(self) -> Dict:
        """
        Collect statistics from the entire network
        """
        stats = {}
        
        # Collect stats from each stage
        for i, stage in enumerate([self.model.stage1, self.model.stage2, self.model.stage3]):
            stage_stats = self.collect_unit_stats(stage, prefix=f'stage_{i}_')
            stats.update(stage_stats)
        
        return stats
    
    def update_metrics(self, epoch_stats: Dict):
        """
        Update training metrics with new statistics
        """
        for key, value in epoch_stats.items():
            category = key.split('_')[0]
            if category in self.metrics:
                self.metrics[f'{category}_stats'][key].append(value)
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """
        Save model checkpoint and metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            
        # Save metrics
        with open(self.save_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """
        Train for one epoch and collect statistics
        """
        self.model.train()
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        epoch_stats = defaultdict(float)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect statistics every 100 batches
            if batch_idx % 100 == 0:
                stats = self.collect_network_stats()
                for key, value in stats.items():
                    epoch_stats[key] += value
                
                print(f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}]'
                      f' Loss: {loss.item():.4f}')
                
                # Print some interesting statistics
                pruned_units = sum(1 for k, v in stats.items() 
                                 if 'is_pruned' in k and v > 0)
                print(f'Currently pruned units: {pruned_units}')
        
        # Average statistics over batches
        num_batches = len(self.train_loader)
        epoch_stats = {k: v / num_batches for k, v in epoch_stats.items()}
        
        return total_loss / num_batches, epoch_stats
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / len(self.val_loader.dataset)
        
        return avg_loss, accuracy
    
    def plot_metrics(self):
        """
        Plot training metrics and statistics with error handling and data validation
        """
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Loss curves
        if self.metrics['train_loss'] and self.metrics['val_loss']:
            ax1.plot(self.metrics['train_loss'], label='Train Loss', color='blue')
            ax1.plot(self.metrics['val_loss'], label='Val Loss', color='red')
            ax1.set_title('Loss Curves')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
        
        # Plot 2: Accuracy
        if self.metrics['val_accuracy']:
            ax2.plot(self.metrics['val_accuracy'], label='Validation Accuracy', color='green')
            ax2.set_title('Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
        
        # Plot 3: Pruning Statistics
        if self.metrics['pruning_stats']:
            # Safely convert dictionary data to plottable format
            epochs = len(next(iter(self.metrics['pruning_stats'].values())))
            stats_keys = list(self.metrics['pruning_stats'].keys())
            
            pruning_data = np.zeros((epochs, len(stats_keys)))
            for i, key in enumerate(stats_keys):
                pruning_data[:, i] = self.metrics['pruning_stats'][key]
            
            # Plot each statistic
            for i, key in enumerate(stats_keys):
                ax3.plot(pruning_data[:, i], label=key)
            
            ax3.set_title('Pruning Statistics')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Number of Pruned Units')
            if len(stats_keys) > 1:
                ax3.legend()
        
        # Plot 4: Complexity Statistics
        if self.metrics['complexity_stats']:
            # Safely convert dictionary data to plottable format
            epochs = len(next(iter(self.metrics['complexity_stats'].values())))
            stats_keys = list(self.metrics['complexity_stats'].keys())
            
            complexity_data = np.zeros((epochs, len(stats_keys)))
            for i, key in enumerate(stats_keys):
                complexity_data[:, i] = self.metrics['complexity_stats'][key]
            
            # Plot mean complexity with confidence interval
            mean_complexity = np.mean(complexity_data, axis=1)
            min_complexity = np.min(complexity_data, axis=1)
            max_complexity = np.max(complexity_data, axis=1)
            
            ax4.plot(mean_complexity, label='Mean Complexity', color='purple')
            ax4.fill_between(
                range(len(mean_complexity)),
                min_complexity,
                max_complexity,
                alpha=0.3,
                color='purple'
            )
            
            ax4.set_title('Feature Complexity')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Complexity Score')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_metrics.png')
        plt.close()
        
    def plot_detailed_metrics(self):
        """
        Generate additional detailed plots for network analysis
        """
        # Create a larger figure for detailed metrics
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Memory Usage Distribution
        ax1 = plt.subplot(3, 2, 1)
        if self.metrics['memory_stats']:
            memory_data = np.array(list(self.metrics['memory_stats'].values()))
            ax1.hist(memory_data.flatten(), bins=50, alpha=0.7)
            ax1.set_title('Memory Usage Distribution')
            ax1.set_xlabel('Memory Usage')
            ax1.set_ylabel('Frequency')
        
        # Plot 2: Spike Rate Over Time
        ax2 = plt.subplot(3, 2, 2)
        if any('spike_rate' in key for key in self.metrics.keys()):
            spike_rates = [m.get('spike_rate', 0) for m in self.metrics['spike_stats'].values()]
            ax2.plot(spike_rates, label='Spike Rate', color='orange')
            ax2.set_title('Spike Rate Evolution')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Spike Rate')
        
        # Plot 3: Prediction Confidence
        ax3 = plt.subplot(3, 2, 3)
        if self.metrics['prediction_stats']:
            confidence_data = np.array(list(self.metrics['prediction_stats'].values()))
            ax3.plot(np.mean(confidence_data, axis=1), label='Mean Confidence')
            ax3.fill_between(
                range(len(confidence_data)),
                np.min(confidence_data, axis=1),
                np.max(confidence_data, axis=1),
                alpha=0.3
            )
            ax3.set_title('Prediction Confidence')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Confidence Score')
        
        # Plot 4: Network Growth
        ax4 = plt.subplot(3, 2, 4)
        if self.metrics['expansion_stats']:
            expansion_data = np.array(list(self.metrics['expansion_stats'].values()))
            cumsum_expansion = np.cumsum(expansion_data, axis=0)
            ax4.plot(cumsum_expansion, label='Cumulative Growth')
            ax4.set_title('Network Growth Over Time')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Number of Units')
        
        # Plot 5: Learning Rate
        ax5 = plt.subplot(3, 2, 5)
        if hasattr(self, 'scheduler'):
            lrs = [self.scheduler.get_last_lr()[0] for _ in range(len(self.metrics['train_loss']))]
            ax5.plot(lrs, label='Learning Rate', color='red')
            ax5.set_title('Learning Rate Schedule')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Learning Rate')
        
        # Plot 6: Combined Metrics
        ax6 = plt.subplot(3, 2, 6)
        if self.metrics['train_loss'] and self.metrics['val_accuracy']:
            # Normalize data for comparison
            normalized_loss = np.array(self.metrics['train_loss']) / max(self.metrics['train_loss'])
            normalized_acc = np.array(self.metrics['val_accuracy'])
            
            ax6.plot(normalized_loss, label='Normalized Loss', color='red')
            ax6.plot(normalized_acc, label='Accuracy', color='blue')
            ax6.set_title('Training Progress')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Score')
            ax6.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'detailed_metrics.png')
        plt.close()
    
    def train(self, num_epochs: int):
        """
        Full training loop with metric collection
        """
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            start_time = time.time()
            
            # Training phase
            train_loss, epoch_stats = self.train_epoch(epoch)
            self.metrics['train_loss'].append(train_loss)
            self.update_metrics(epoch_stats)
            
            # Evaluation phase
            val_loss, accuracy = self.evaluate()
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_accuracy'].append(accuracy)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save checkpoint if best model
            is_best = accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = accuracy
            
            self.save_checkpoint(epoch, self.metrics, is_best)
            
            # Plot current metrics
            self.plot_metrics()
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1} completed in {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Best Accuracy: {self.best_accuracy:.4f}')
            
            # Print interesting statistics
            stats = self.collect_network_stats()
            pruned_units = sum(1 for k, v in stats.items() if 'is_pruned' in k and v > 0)
            avg_complexity = np.mean([v for k, v in stats.items() 
                                   if 'complexity' in k])
            
            print(f'Network Statistics:')
            print(f'- Pruned Units: {pruned_units}')
            print(f'- Average Complexity: {avg_complexity:.4f}')

class EnhancedFNNLATrainerOld:
    """
    Enhanced trainer with adaptive pruning and resource management
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        checkpoint_handler: Optional[CheckpointHandler] = None,
        metrics: Optional[Dict] = None,
        save_dir: str = './enhanced_fnnla_results',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or use provided optimizer
        self.optimizer = optimizer if optimizer is not None else optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize or use provided scheduler
        self.scheduler = scheduler if scheduler is not None else optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,       # Reduced from 10
            T_mult=2,
            eta_min=1e-6 # Added minimum LR
        )
        
        # Initialize or use provided checkpoint handler
        self.checkpoint_handler = checkpoint_handler if checkpoint_handler is not None else CheckpointHandler(save_dir)
        
        # Initialize or use provided metrics
        self.metrics = metrics if metrics is not None else {
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'pruning_stats': [],
            'memory_usage': [],
            'attention_stats': [],
            'unit_stats': []
        }
        
        self.best_accuracy = max(self.metrics.get('accuracy', [0.0]))
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Training loop for one epoch with enhanced monitoring
        """
        self.model.train()
        total_loss = 0.0
        total_metrics = defaultdict(float)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass with metrics
            output, metrics = self.model(data)
            loss = F.cross_entropy(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    total_metrics[key] += value
                
            # Log progress
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}]'
                      f' Loss: {loss.item():.4f}')
                
                # Print pruning and memory stats
                print(f'Pruned Units: {metrics.get("pruned_units", 0)}')
                print(f'Memory Usage: {metrics.get("memory_usage", 0):.2f} MB')
        
        # Average metrics
        num_batches = len(self.train_loader)
        avg_metrics = {
            'train_loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in total_metrics.items()}
        }
        
        return avg_metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validation with enhanced metrics collection
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        metrics_sum = defaultdict(float)
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, metrics = self.model(data)
                
                # Calculate validation metrics
                val_loss += F.cross_entropy(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                # Accumulate other metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_sum[key] += value
        
        # Average metrics
        num_batches = len(self.val_loader)
        return {
            'val_loss': val_loss / num_batches,
            'accuracy': correct / len(self.val_loader.dataset),
            **{k: v / num_batches for k, v in metrics_sum.items()}
        }
    
    def train(self, num_epochs: int, start_epoch: int = 0) -> Dict:
        """
        Full training loop with enhanced monitoring and checkpoint handling
        """
        for epoch in range(start_epoch, start_epoch + num_epochs):
            print(f'\nEpoch {epoch+1}/{start_epoch + num_epochs}')
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            self.metrics['train_loss'].append(train_metrics['train_loss'])
            
            # Validation phase
            val_metrics = self.validate()
            self.metrics['val_loss'].append(val_metrics['val_loss'])
            self.metrics['accuracy'].append(val_metrics['accuracy'])
            
            # Update learning rate
            self.scheduler.step()
            
            # Collect and store comprehensive metrics
            epoch_metrics = {
                **train_metrics,
                **val_metrics,
                'epoch': epoch,
                'learning_rate': self.scheduler.get_last_lr()[0]
            }
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = val_metrics['accuracy']
            
            self.checkpoint_handler.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=self.metrics,
                is_best=is_best
            )
            
            # Print epoch summary
            print(f'\nEpoch Summary:')
            print(f'Train Loss: {train_metrics["train_loss"]:.4f}')
            print(f'Val Loss: {val_metrics["val_loss"]:.4f}')
            print(f'Accuracy: {val_metrics["accuracy"]:.4f}')
            print(f'Best Accuracy: {self.best_accuracy:.4f}')
            
            # Print resource usage
            print('\nResource Usage:')
            print(f'Memory Usage: {epoch_metrics.get("memory_usage", 0):.2f} MB')
            print(f'Pruned Units: {epoch_metrics.get("pruned_units", 0)}')
            print(f'Active Units: {epoch_metrics.get("active_units", 0)}')
        
        return self.metrics

def train_enhanced_fnnla(
    dataset: str = 'cifar10',
    epochs: int = 50,
    batch_size: int = 128,
    save_dir: str = './enhanced_fnnla_results',
    resume_training: bool = False,
    load_best: bool = False,
    patience = 7,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4
):
    """
    Enhanced training function with checkpoint handling
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data loading and preprocessing
    if dataset.lower() == 'cifar10':
        transform_train = transforms.Compose([
            # PIL Image transforms (before ToTensor)
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            
            # Convert to tensor
            transforms.ToTensor(),
            
            # Tensor operations (after ToTensor)
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            ),
            
            # Additional augmentations
            transforms.RandomErasing(
                p=0.2,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=0,
                inplace=False
            )
        ])
        
        # Test transforms - keep simple for consistent evaluation
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
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
            pin_memory=True,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model with Mixup
        model = EnhancedFNNLA(
            in_channels=3,
            num_classes=10,
            base_channels=64,
            max_depth=3,
            dropout_rate=0.5,
            mixup_alpha=0.2,   # Controls interpolation strength
            mixup_prob=0.5     # Probability of applying mixup
        ).to(device)
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization

        
        # Initialize scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,       # Reduced from 10
            T_mult=2,
            eta_min=1e-6 # Added minimum LR
        )
        
        
        
        # Initialize checkpoint handler
        checkpoint_handler = CheckpointHandler(save_dir)
        
        # Resume training if requested
        start_epoch = 0
        metrics = {}
        
        if resume_training:
            start_epoch, metrics = checkpoint_handler.load_checkpoint(
                model,
                optimizer,
                scheduler,
                load_best=load_best
            )
            logger.info(f"Resuming training from epoch {start_epoch}")
        
        # Initialize trainer with resumed state
        trainer = EnhancedFNNLATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            checkpoint_handler=checkpoint_handler,
            save_dir=save_dir
        )
        
        # Train model
        final_metrics = trainer.train(
            num_epochs=epochs,
            start_epoch=start_epoch
        )
        
        return model, final_metrics
    else:
        raise ValueError(f"Dataset {dataset} not supported")

if __name__ == "__main__":
    # Resume training from best checkpoint
    model, metrics = train_enhanced_fnnla(
    epochs=50,
    save_dir='./enhanced_fnnla_results',
    resume_training=True,
    load_best=True,
)
    

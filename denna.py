import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import time
from typing import List, Dict, Optional
from matplotlib.animation import FuncAnimation
import matplotlib
import seaborn as sns
import matplotlib.animation as animation

# Replace the style settings with:
sns.set_theme(style="whitegrid")
matplotlib.rcParams.update({
    'figure.figsize': (15, 10),
    'figure.dpi': 100,
    'font.size': 10,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'animation.html': 'jshtml'
})

class DynamicBlock(nn.Module):
    """Enhanced Dynamic Block with more features"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super(DynamicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, groups=groups)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                              padding=padding, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None
        self.active = True
        self.importance_score = 0.0
        
    def forward(self, x):
        if not self.active:
            return x
            
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.activation(out)
        return out

class DynamicNet(nn.Module):
    """Enhanced Dynamic Network with larger initial architecture"""
    def __init__(self, num_classes: int = 10, initial_channels: int = 64):
        super(DynamicNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Dynamic blocks organized in stages
        self.stages = nn.ModuleList()
        channels = [initial_channels, initial_channels * 2, 
                   initial_channels * 4, initial_channels * 8]
        
        for i in range(4):  # 4 stages with increasing channels
            stage = nn.ModuleList()
            in_channels = channels[i-1] if i > 0 else initial_channels
            out_channels = channels[i]
            
            # Each stage starts with 4 blocks
            for j in range(4):
                stride = 2 if j == 0 and i > 0 else 1
                block = DynamicBlock(in_channels if j == 0 else out_channels,
                                   out_channels, stride=stride)
                stage.append(block)
            self.stages.append(stage)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)
        self.to(self.device)
        
        # Training history
        self.training_history = {
            'accuracy': [], 'loss': [], 'block_count': [],
            'layer_activations': deque(maxlen=100),
            'gradient_flow': deque(maxlen=100)
        }

    def forward(self, x):
        activation_maps = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)
        activation_maps.append(x.detach())
        
        for stage in self.stages:
            for block in stage:
                if block.active:
                    x = block(x)
                    activation_maps.append(x.detach())
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x, activation_maps

    def add_block(self, stage_idx: int):
        """Add a block to a specific stage"""
        if stage_idx < len(self.stages):
            stage = self.stages[stage_idx]
            if len(stage) > 0:
                in_channels = stage[-1].conv2.out_channels
                new_block = DynamicBlock(in_channels, in_channels).to(self.device)
                stage.append(new_block)

    def prune_block(self, stage_idx: int, block_idx: int):
        """Prune specific block by setting it inactive"""
        if stage_idx < len(self.stages):
            stage = self.stages[stage_idx]
            if block_idx < len(stage):
                stage[block_idx].active = False

    def get_architecture_info(self) -> Dict:
        """Get current architecture information"""
        info = {'total_blocks': 0, 'active_blocks': 0, 'stage_channels': []}
        for stage in self.stages:
            stage_info = {
                'blocks': len(stage),
                'active_blocks': sum(1 for block in stage if block.active),
                'channels': stage[0].conv2.out_channels if len(stage) > 0 else 0
            }
            info['stage_channels'].append(stage_info)
            info['total_blocks'] += stage_info['blocks']
            info['active_blocks'] += stage_info['active_blocks']
        return info

class NetworkVisualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = self.fig.add_gridspec(3, 3)
        self.setup_plots()
        self.prev_positions = {}  # Store previous node positions
        self.anim = None  # Store animation object
        plt.tight_layout()
        
    def _animate_network(self, frame, G, pos_new, node_colors):
        """Animate network transitions"""
        self.ax_arch.clear()
        
        # Interpolate positions
        alpha = frame / 10  # 10 frames for transition
        pos_current = {}
        
        for node in G.nodes():
            if node in self.prev_positions:
                old_pos = self.prev_positions[node]
                new_pos = pos_new[node]
                pos_current[node] = (
                    old_pos[0] * (1 - alpha) + new_pos[0] * alpha,
                    old_pos[1] * (1 - alpha) + new_pos[1] * alpha
                )
            else:
                pos_current[node] = pos_new[node]
        
        # Draw network with current positions
        nx.draw(G, pos_current, ax=self.ax_arch, 
               node_color=node_colors,
               node_size=1000,
               arrows=True, 
               arrowsize=20,
               with_labels=True)
        
        # Add channel information
        channels = nx.get_node_attributes(G, 'channels')
        labels = {n: f'{channels.get(n, "")}' for n in G.nodes()}
        nx.draw_networkx_labels(G, pos_current, labels, ax=self.ax_arch)
        
        self.ax_arch.set_title('Network Architecture\n(Green: Active, Red: Pruned)')
        
        
    def setup_plots(self):
        # Network architecture plot
        self.ax_arch = self.fig.add_subplot(self.gs[0, :])
        self.ax_arch.set_title('Network Architecture')
        
        # Training metrics
        self.ax_metrics = self.fig.add_subplot(self.gs[1, 0])
        self.ax_metrics.set_title('Training Metrics')
        
        # Loss plot
        self.ax_loss = self.fig.add_subplot(self.gs[1, 1])
        self.ax_loss.set_title('Loss History')
        
        # Block distribution
        self.ax_blocks = self.fig.add_subplot(self.gs[1, 2])
        self.ax_blocks.set_title('Block Distribution')
        
        # Layer activations heatmap
        self.ax_activations = self.fig.add_subplot(self.gs[2, :])
        self.ax_activations.set_title('Layer Activations')
        
        # Store line objects for updating
        self.lines = {}
        self.images = {}
        
    def update(self, net, history):
        try:
            self._update_architecture(net)
            self._update_metrics(history)
            self._update_loss(history)
            self._update_blocks(net)
            if history['layer_activations']:
                self._update_activations(history['layer_activations'][-1])
            
            plt.pause(0.01)  # Small pause to update display
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Visualization error: {e}")
            
    def _update_architecture(self, net):
        """Update network architecture with animation"""
        G = nx.DiGraph()
        
        # Add input node
        G.add_node('input', pos=(0, 0))
        
        # Add nodes for each stage and block
        active_blocks = []
        max_blocks = max(len(stage) for stage in net.stages)
        
        for stage_idx, stage in enumerate(net.stages):
            stage_active_blocks = 0
            for block_idx, block in enumerate(stage):
                if block.active:
                    node_id = f's{stage_idx}b{block_idx}'
                    # Position blocks in a grid layout with active blocks grouped together
                    x = (stage_active_blocks + 1) / (max_blocks + 1)
                    y = 1 - (stage_idx + 1) / (len(net.stages) + 1)
                    G.add_node(node_id, 
                              pos=(x, y),
                              active=True,
                              channels=block.conv1.out_channels)
                    
                    # Add edges
                    if stage_active_blocks == 0:
                        prev_node = f's{stage_idx-1}b0' if stage_idx > 0 else 'input'
                    else:
                        prev_node = f's{stage_idx}b{stage_active_blocks-1}'
                    G.add_edge(prev_node, node_id)
                    
                    stage_active_blocks += 1
            active_blocks.append(stage_active_blocks)
        
        # Get positions and colors
        pos_new = nx.get_node_attributes(G, 'pos')
        node_colors = ['lightblue' if n == 'input' else 'green' for n in G.nodes()]
        
        # Create animation if positions have changed
        if self.prev_positions and pos_new != self.prev_positions:
            anim = animation.FuncAnimation(
                self.fig, 
                self._animate_network,
                frames=10,
                fargs=(G, pos_new, node_colors),
                interval=50,
                blit=False
            )
            plt.pause(0.5)  # Allow animation to complete
        else:
            # First draw or no change
            nx.draw(G, pos_new, ax=self.ax_arch,
                   node_color=node_colors,
                   node_size=1000,
                   arrows=True,
                   arrowsize=20,
                   with_labels=True)
            
            # Add channel information
            channels = nx.get_node_attributes(G, 'channels')
            labels = {n: f'{channels.get(n, "")}' for n in G.nodes()}
            nx.draw_networkx_labels(G, pos_new, labels, ax=self.ax_arch)
        
        # Store current positions for next update
        self.prev_positions = pos_new
        self.ax_arch.set_title('Network Architecture\n(Green: Active, Red: Pruned)')

    def _update_metrics(self, history):
        self.ax_metrics.clear()
        if history['accuracy']:
            epochs = range(1, len(history['accuracy']) + 1)
            self.ax_metrics.plot(epochs, history['accuracy'], 'b-', label='Accuracy')
            self.ax_metrics.set_xlabel('Epoch')
            self.ax_metrics.set_ylabel('Accuracy (%)')
            self.ax_metrics.grid(True)
            self.ax_metrics.legend()
            
    def _update_loss(self, history):
        self.ax_loss.clear()
        if history['loss']:
            self.ax_loss.plot(history['loss'], 'r-', label='Loss')
            self.ax_loss.set_xlabel('Iteration')
            self.ax_loss.set_ylabel('Loss')
            self.ax_loss.grid(True)
            self.ax_loss.legend()
            
    def _update_blocks(self, net):
        self.ax_blocks.clear()
        info = net.get_architecture_info()
        stages = range(len(info['stage_channels']))
        active_blocks = [s['active_blocks'] for s in info['stage_channels']]
        total_blocks = [s['blocks'] for s in info['stage_channels']]
        
        width = 0.35
        self.ax_blocks.bar(stages, total_blocks, width, label='Total Blocks', color='lightgray')
        self.ax_blocks.bar(stages, active_blocks, width, label='Active Blocks', color='green')
        
        self.ax_blocks.set_xlabel('Stage')
        self.ax_blocks.set_ylabel('Number of Blocks')
        self.ax_blocks.legend()
        
    def _update_activations(self, activations):
        """Update layer activations visualization with proper tensor handling"""
        self.ax_activations.clear()
        
        try:
            # Process activations
            processed = []
            for act in activations:
                # Average across batch dimension
                mean_act = act.mean(dim=0).cpu()
                
                # Resize to common size (8x8)
                if mean_act.shape[-1] != 8 or mean_act.shape[-2] != 8:
                    resized = F.adaptive_avg_pool2d(mean_act.unsqueeze(0), (8, 8)).squeeze(0)
                else:
                    resized = mean_act
                    
                # Average across channels to get a single 8x8 heatmap
                heatmap = resized.mean(dim=0).numpy()
                processed.append(heatmap)
                
            if processed:
                # Stack processed activations into a single array and reshape
                combined = np.stack(processed)
                
                # Reshape to 2D matrix where each row is a flattened 8x8 heatmap
                num_layers = combined.shape[0]
                combined_2d = combined.reshape(num_layers, -1)
                
                # Create heatmap
                im = self.ax_activations.imshow(
                    combined_2d,
                    aspect='auto',
                    cmap='viridis',
                    interpolation='nearest'
                )
                
                # Add colorbar and labels
                plt.colorbar(im, ax=self.ax_activations, label='Average Activation')
                self.ax_activations.set_xlabel('Spatial Position (flattened)')
                self.ax_activations.set_ylabel('Layer')
                
                # Add layer ticks
                self.ax_activations.set_yticks(range(num_layers))
                self.ax_activations.set_yticklabels([f'Layer {i+1}' for i in range(num_layers)])
                
                # Add spatial position ticks
                num_positions = combined_2d.shape[1]
                step = max(1, num_positions // 8)  # Show at most 8 ticks
                self.ax_activations.set_xticks(range(0, num_positions, step))
                
                # Title with shape information
                self.ax_activations.set_title(
                    f'Layer Activations\n'
                    f'{num_layers} layers, {int(np.sqrt(num_positions))}x{int(np.sqrt(num_positions))} spatial resolution'
                )
                
            else:
                self.ax_activations.text(0.5, 0.5, 'No activation data available',
                                    ha='center', va='center')
            
        except Exception as e:
            print(f"Error in activation visualization: {str(e)}")
            self.ax_activations.text(0.5, 0.5, f'Activation visualization error: {str(e)}',
                                    ha='center', va='center')


def train_dynamic_net(net, trainloader, testloader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                            epochs=epochs,
                                            steps_per_epoch=len(trainloader))
    
    # Create visualizer
    visualizer = NetworkVisualizer()
    
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(net.device), targets.to(net.device)
            optimizer.zero_grad()
            
            outputs, activations = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            # Update history
            running_loss += loss.item()
            if batch_idx % 50 == 0:  # Update more frequently
                net.training_history['loss'].append(running_loss / (batch_idx + 1))
                net.training_history['layer_activations'].append(activations)
                visualizer.update(net, net.training_history)
        
        # Evaluate and adjust architecture
        test_accuracy = evaluate(net, testloader)
        net.training_history['accuracy'].append(test_accuracy)
        net.training_history['block_count'].append(
            net.get_architecture_info()['active_blocks']
        )
        
        print(f'Epoch {epoch+1}/{epochs}, Test Accuracy: {test_accuracy:.2f}%')
        
        # Dynamic architecture adjustment
        adjust_architecture(net, test_accuracy)
        
        # Update visualization
        visualizer.update(net, net.training_history)
        
    plt.ioff()
    plt.show()
    return net

def evaluate(net, testloader):
    """Evaluate network performance"""
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(net.device), targets.to(net.device)
            outputs, _ = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def adjust_architecture(net, accuracy):
    """Adjust network architecture based on performance"""
    info = net.get_architecture_info()
    
    # Add blocks if performance is good
    if accuracy > 85 and info['active_blocks'] < 32:
        stage_idx = np.random.randint(len(net.stages))
        net.add_block(stage_idx)
        
    # Prune blocks if performance is poor
    elif accuracy < 75 and info['active_blocks'] > 8:
        # Find least important blocks
        importance_scores = []
        for stage_idx, stage in enumerate(net.stages):
            for block_idx, block in enumerate(stage):
                if block.active:
                    grad_magnitude = torch.norm(block.conv1.weight.grad).item() if block.conv1.weight.grad is not None else 0
                    importance_scores.append((grad_magnitude, stage_idx, block_idx))
        
        if importance_scores:
            # Sort by importance and prune least important block
            importance_scores.sort()
            _, stage_idx, block_idx = importance_scores[0]
            net.prune_block(stage_idx, block_idx)

def main():
    # Setup data loading
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                           shuffle=False, num_workers=2)

    # Create and train network
    net = DynamicNet(num_classes=10, initial_channels=64)
    net = train_dynamic_net(net, trainloader, testloader, epochs=20)
        
if __name__ == "__main__":
    main()
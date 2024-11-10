import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import math
import time

class SpikingNeuron(nn.Module):
    """
    Implementation of a spiking neuron with threshold-based activation
    """
    def __init__(self, threshold: float = 1.0, decay: float = 0.5):
        super(SpikingNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay
        self.membrane_potential = 0.0
        self.spike_history = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Update membrane potential
        self.membrane_potential = self.membrane_potential * self.decay + x
        
        # Generate spike if threshold is reached
        spike = torch.zeros_like(x)
        mask = self.membrane_potential >= self.threshold
        spike[mask] = 1.0
        
        # Reset membrane potential where spikes occurred
        self.membrane_potential[mask] = 0.0
        
        self.spike_history.append(spike)
        return spike

class SparseLinear(nn.Module):
    """
    Sparse linear layer with dynamic connection pruning
    """
    def __init__(self, in_features: int, out_features: int, sparsity: float = 0.9):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        
        # Initialize sparse weight matrix
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Mask for sparse connections
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.reset_parameters()
        self.apply_sparsity()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def apply_sparsity(self):
        with torch.no_grad():
            # Keep only top (1-sparsity)% of connections
            k = int(self.weight.numel() * (1 - self.sparsity))
            threshold = torch.topk(torch.abs(self.weight.view(-1)), k)[0][-1]
            self.mask = (torch.abs(self.weight) >= threshold).float()
            self.weight.data *= self.mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

class NSAMLayer(nn.Module):
    """
    Neuromorphic Sparse Adaptive Meta-Learning Layer with fixed gradient handling
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 sparsity: float = 0.9,
                 threshold: float = 1.0):
        super(NSAMLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Base sparse linear layer
        self.sparse_linear = SparseLinear(in_features, out_features, sparsity)
        
        # Meta-learning parameters (ensure requires_grad=True)
        self.meta_weights = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.meta_bias = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
        
        # Spiking neurons (wrap parameters in nn.Parameter)
        self.spiking_neurons = nn.ModuleList([
            SpikingNeuron(threshold) for _ in range(out_features)
        ])
        
        self.reset_meta_parameters()
    
    def forward(self, x: torch.Tensor, meta_learn: bool = False) -> torch.Tensor:
        # Ensure input is properly shaped
        if len(x.shape) == 3:  # If input is [N, H, W]
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        elif len(x.shape) > 2:  # For any other higher dimensional input
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            
        # Forward pass
        if meta_learn:
            out = F.linear(x, self.meta_weights, self.meta_bias)
        else:
            out = self.sparse_linear(x)
            
        # Process through spiking neurons
        spikes = []
        for i, neuron in enumerate(self.spiking_neurons):
            spike = neuron(out[:, i:i+1])
            spikes.append(spike)
            
        return torch.cat(spikes, dim=1)
        
    
    def reset_meta_parameters(self):
        nn.init.kaiming_uniform_(self.meta_weights, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.meta_bias, -bound, bound)   
    
    
    
class NSAMLearn(nn.Module):
    """
    Complete Neuromorphic Sparse Adaptive Meta-Learning Network
    """
    def __init__(self,
                input_size: int,
                hidden_sizes: List[int],
                output_size: int,
                sparsity: float = 0.9,
                threshold: float = 1.0):
        super(NSAMLearn, self).__init__()
        
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList([
            NSAMLayer(
                layer_sizes[i],
                layer_sizes[i+1],
                sparsity=sparsity,
                threshold=threshold
            )
            for i in range(len(layer_sizes)-1)
        ])
        
        # Task adaptation parameters
        self.task_memories = {}
        self.adaptation_lr = 0.01
    
    def forward(self, x: torch.Tensor, meta_learn: bool = False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, meta_learn)
        return x
    
    def adapt_to_task(self, 
                      task_id: str,
                      support_x: torch.Tensor,
                      support_y: torch.Tensor,
                      num_adaptation_steps: int = 5):
        """
        Adapt the model to a new task using few-shot learning
        """
        # Store task-specific parameters
        task_params = {
            name: param.clone()
            for name, param in self.named_parameters()
        }
        
        # Perform adaptation steps
        optimizer = optim.Adam(self.parameters(), lr=self.adaptation_lr)
        
        for _ in range(num_adaptation_steps):
            optimizer.zero_grad()
            output = self(support_x)
            loss = F.cross_entropy(output, support_y)
            loss.backward()
            optimizer.step()
            
            # Apply sparsity after each adaptation step
            for layer in self.layers:
                layer.sparse_linear.apply_sparsity()
        
        # Store adapted parameters
        self.task_memories[task_id] = {
            name: param.clone()
            for name, param in self.named_parameters()
        }
        
        # Restore original parameters
        for name, param in self.named_parameters():
            param.data.copy_(task_params[name])
    
    def load_task(self, task_id: str):
        """
        Load parameters for a specific task
        """
        if task_id in self.task_memories:
            for name, param in self.named_parameters():
                param.data.copy_(self.task_memories[task_id][name])

def train_meta_batch(model: NSAMLearn,
                    tasks: List[Tuple[torch.Tensor, torch.Tensor]],
                    meta_optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    num_inner_steps: int = 5):
    """
    Train on a batch of tasks with proper gradient handling
    """
    meta_loss = torch.tensor(0.0, requires_grad=True, device=device)
    
    # Ensure model is in training mode
    model.train()
    
    for task_x, task_y in tasks:
        # Ensure proper input shape
        if len(task_x.shape) > 2:
            task_x = task_x.view(task_x.size(0), -1)
            
        # Move data to device and enable gradients
        task_x = task_x.to(device).requires_grad_(True)
        task_y = task_y.to(device)
            
        # Split into support and query sets
        split_idx = task_x.size(0) // 2
        support_x = task_x[:split_idx]
        query_x = task_x[split_idx:]
        support_y = task_y[:split_idx]
        query_y = task_y[split_idx:]
        
        # Store original parameters
        orig_params = {
            name: param.clone()
            for name, param in model.named_parameters()
        }
        
        # Inner loop adaptation
        model.train()  # Ensure model is in training mode
        inner_optimizer = torch.optim.SGD(model.parameters(), lr=model.adaptation_lr)
        
        for _ in range(num_inner_steps):
            inner_optimizer.zero_grad()
            output = model(support_x, meta_learn=True)
            loss = F.cross_entropy(output, support_y)
            loss.backward(retain_graph=True)
            inner_optimizer.step()
        
        # Compute meta-loss on query set
        query_output = model(query_x, meta_learn=True)
        task_meta_loss = F.cross_entropy(query_output, query_y)
        meta_loss = meta_loss + task_meta_loss
        
        # Restore original parameters
        for name, param in model.named_parameters():
            param.data.copy_(orig_params[name].data)
    
    # Average meta-loss
    meta_loss = meta_loss / len(tasks)
    
    # Update meta-parameters
    meta_optimizer.zero_grad()
    meta_loss.backward()
    meta_optimizer.step()
    
    return meta_loss.item()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple
import random

def create_task_batch(dataset, num_tasks: int, num_samples: int, num_classes: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create a batch of few-shot learning tasks with proper tensor shapes
    """
    tasks = []
    
    for _ in range(num_tasks):
        # Randomly select classes for this task
        classes = random.sample(range(10), num_classes)
        
        # Collect samples for these classes
        task_x = []
        task_y = []
        
        for class_idx, class_label in enumerate(classes):
            # Get indices for this class
            class_indices = (dataset.targets == class_label).nonzero().squeeze()
            selected_indices = class_indices[torch.randperm(len(class_indices))[:num_samples]]
            
            # Get samples and process them
            selected_samples = dataset.data[selected_indices]
            
            # Ensure proper shape and normalization
            if len(selected_samples.shape) == 3:  # [N, H, W]
                selected_samples = selected_samples.view(selected_samples.size(0), -1)
            
            task_x.append(selected_samples)
            task_y.extend([class_idx] * num_samples)
        
        # Convert to tensors and normalize
        task_x = torch.cat(task_x, dim=0).float() / 255.0
        task_y = torch.tensor(task_y, dtype=torch.long)
        
        tasks.append((task_x, task_y))
    
    return tasks

def train_nsam(
    num_epochs: int = 100,
    meta_batch_size: int = 4,
    num_samples_per_class: int = 5,
    num_classes_per_task: int = 5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    device = torch.device(device)
    
    # Load dataset
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=None)
    mnist_test = datasets.MNIST('./data', train=False, transform=None)
    
    # Initialize model
    model = NSAMLearn(
        input_size=784,
        hidden_sizes=[256, 128],
        output_size=num_classes_per_task,
        sparsity=0.9,
        threshold=1.0
    ).to(device)
    
    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
    # Initialize optimizer with correct parameters
    meta_optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001
    )
    
    # Training metrics
    metrics = {
        'meta_losses': [],
        'test_accuracies': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        # Create and process tasks
        tasks = create_task_batch(
            mnist_train,
            num_tasks=meta_batch_size,
            num_samples=num_samples_per_class * 2,
            num_classes=num_classes_per_task
        )
        
        # Move tasks to device and ensure proper shape
        tasks = [(x.view(x.size(0), -1).to(device), y.to(device)) 
                for x, y in tasks]
        
        # Train on meta-batch
        meta_loss = train_meta_batch(
            model,
            tasks,
            meta_optimizer,
            device,
            num_inner_steps=5
        )
        
        metrics['meta_losses'].append(meta_loss)
        
        # Evaluation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_tasks = create_task_batch(
                    mnist_test,
                    num_tasks=10,
                    num_samples=num_samples_per_class * 2,
                    num_classes=num_classes_per_task
                )
                
                test_tasks = [(x.view(x.size(0), -1).to(device), y.to(device)) 
                             for x, y in test_tasks]
                accuracies = []
                
                for task_id, (task_x, task_y) in enumerate(test_tasks):
                    split_idx = task_x.size(0) // 2
                    support_x = task_x[:split_idx]
                    support_y = task_y[:split_idx]
                    query_x = task_x[split_idx:]
                    query_y = task_y[split_idx:]
                    
                    model.adapt_to_task(f"test_task_{task_id}", support_x, support_y)
                    output = model(query_x)
                    pred = output.argmax(dim=1)
                    accuracy = (pred == query_y).float().mean().item()
                    accuracies.append(accuracy)
                
                avg_accuracy = np.mean(accuracies)
                metrics['test_accuracies'].append(avg_accuracy)
                
                print(f"Epoch {epoch + 1}")
                print(f"Meta Loss: {meta_loss:.4f}")
                print(f"Few-Shot Test Accuracy: {avg_accuracy:.4f}")
                print("-" * 50)
            
            model.train()
    
    return model, metrics


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Training configuration
    config = {
        'num_epochs': 100,
        'meta_batch_size': 4,
        'num_samples_per_class': 5,  # k-shot learning with k=5
        'num_classes_per_task': 5,   # n-way classification with n=5
        'hidden_sizes': [256, 128],
        'sparsity': 0.9,
        'threshold': 1.0,
        'learning_rate': 0.001,
        'adaptation_steps': 5
    }
    
    # Initialize training metrics
    metrics = {
        'meta_losses': [],
        'test_accuracies': [],
        'sparsity_levels': [],
        'adaptation_times': []
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Train model
        print("Starting NSAM-Learn training...")
        print(f"Configuration: {config}")
        print("-" * 50)
        
        model = train_nsam(
            num_epochs=config['num_epochs'],
            meta_batch_size=config['meta_batch_size'],
            num_samples_per_class=config['num_samples_per_class'],
            num_classes_per_task=config['num_classes_per_task'],
            device=device
        )
        
        # Save trained model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': metrics
        }, 'nsam_model.pth')
        
        print("\nTraining completed successfully!")
        print("Model saved as 'nsam_model.pth'")
        
        # Evaluate final model
        print("\nFinal Evaluation:")
        model.eval()
        with torch.no_grad():
            # Create test tasks
            test_tasks = create_task_batch(
                datasets.MNIST('./data', train=False, download=True),
                num_tasks=20,  # More tasks for final evaluation
                num_samples=config['num_samples_per_class'],
                num_classes=config['num_classes_per_task']
            )
            
            test_tasks = [(x.to(device), y.to(device)) for x, y in test_tasks]
            accuracies = []
            adaptation_times = []
            
            for task_id, (support_x, support_y) in enumerate(test_tasks):
                # Time the adaptation process
                start_time = time.time()
                
                # Adapt to task
                model.adapt_to_task(
                    f"final_test_task_{task_id}",
                    support_x[:config['num_samples_per_class'] * config['num_classes_per_task'] // 2],
                    support_y[:config['num_samples_per_class'] * config['num_classes_per_task'] // 2],
                    num_adaptation_steps=config['adaptation_steps']
                )
                
                adaptation_time = time.time() - start_time
                adaptation_times.append(adaptation_time)
                
                # Evaluate on query set
                query_x = support_x[config['num_samples_per_class'] * config['num_classes_per_task'] // 2:]
                query_y = support_y[config['num_samples_per_class'] * config['num_classes_per_task'] // 2:]
                
                output = model(query_x)
                pred = output.argmax(dim=1)
                accuracy = (pred == query_y).float().mean().item()
                accuracies.append(accuracy)
            
            # Calculate final metrics
            final_accuracy = np.mean(accuracies)
            final_std = np.std(accuracies)
            avg_adaptation_time = np.mean(adaptation_times)
            
            print(f"\nFinal Test Results:")
            print(f"Average Accuracy: {final_accuracy:.4f} ± {final_std:.4f}")
            print(f"Average Adaptation Time: {avg_adaptation_time:.4f} seconds")
            
            # Calculate sparsity
            total_params = 0
            nonzero_params = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    total_params += param.numel()
                    nonzero_params += (param != 0).sum().item()
            
            final_sparsity = 1 - (nonzero_params / total_params)
            print(f"Final Model Sparsity: {final_sparsity:.4f}")
        
        # Plot training progress
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        # Plot meta-loss
        plt.subplot(1, 2, 1)
        plt.plot(metrics['meta_losses'])
        plt.title('Meta-Learning Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Plot test accuracy
        plt.subplot(1, 2, 2)
        plt.plot(metrics['test_accuracies'])
        plt.title('Few-Shot Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        print("\nTraining progress plot saved as 'training_progress.png'")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise e

def evaluate_on_new_task(model, 
                        support_x: torch.Tensor, 
                        support_y: torch.Tensor,
                        query_x: torch.Tensor,
                        query_y: torch.Tensor,
                        device: torch.device):
    """
    Evaluate model on a new task
    """
    model.eval()
    with torch.no_grad():
        # Adapt to new task
        model.adapt_to_task(
            "evaluation_task",
            support_x.to(device),
            support_y.to(device)
        )
        
        # Evaluate on query set
        output = model(query_x.to(device))
        pred = output.argmax(dim=1)
        accuracy = (pred == query_y.to(device)).float().mean().item()
        
        return accuracy

# Example usage for a new task
def test_on_new_task():
    # Load trained model
    checkpoint = torch.load('nsam_model.pth')
    config = checkpoint['config']
    
    model = NSAMLearn(
        input_size=784,
        hidden_sizes=config['hidden_sizes'],
        output_size=config['num_classes_per_task'],
        sparsity=config['sparsity'],
        threshold=config['threshold']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create a new task
    new_task = create_task_batch(
        datasets.MNIST('./data', train=False, download=True),
        num_tasks=1,
        num_samples=config['num_samples_per_class'],
        num_classes=config['num_classes_per_task']
    )[0]
    
    # Split into support and query sets
    support_x = new_task[0][:config['num_samples_per_class'] * config['num_classes_per_task'] // 2]
    support_y = new_task[1][:config['num_samples_per_class'] * config['num_classes_per_task'] // 2]
    query_x = new_task[0][config['num_samples_per_class'] * config['num_classes_per_task'] // 2:]
    query_y = new_task[1][config['num_samples_per_class'] * config['num_classes_per_task'] // 2:]
    
    # Evaluate
    accuracy = evaluate_on_new_task(model, support_x, support_y, query_x, query_y, device)
    print(f"\nAccuracy on new task: {accuracy:.4f}")

if __name__ == "__main__":
    # After training
    test_on_new_task()
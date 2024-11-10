import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple, Optional
import math

class FractalUnit(nn.Module):
    """
    Base Fractal Unit that can recursively expand
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 depth: int = 0,
                 max_depth: int = 3):
        super(FractalUnit, self).__init__()
        
        self.depth = depth
        self.max_depth = max_depth
        self.complexity_threshold = 0.7
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Base convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Recursive units (initialized as None)
        self.recursive_units: Optional[List[FractalUnit]] = None
        
        # Feature integration layer
        self.integrate = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        
        # Complexity estimation metrics
        self.feature_complexity = 0.0
        self.activation_variance = 0.0
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def expand(self) -> bool:
        """
        Expand the fractal unit if complexity threshold is met
        """
        if self.depth >= self.max_depth:
            return False
            
        if self.feature_complexity > self.complexity_threshold:
            if self.recursive_units is None:
                self.recursive_units = nn.ModuleList([
                    FractalUnit(
                        self.out_channels,
                        self.out_channels,
                        depth=self.depth + 1,
                        max_depth=self.max_depth
                    ) for _ in range(2)  # Binary fractal expansion
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        # Base transformation
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Estimate complexity and potentially expand
        self.estimate_complexity(out)
        
        if self.recursive_units is not None:
            # Process through recursive units
            recursive_features = []
            for unit in self.recursive_units:
                if unit.expand():  # Dynamically expand if needed
                    recursive_features.append(unit(out))
            
            if recursive_features:
                # Combine recursive features
                combined = torch.cat(recursive_features, dim=1)
                out = out + self.integrate(combined)
        
        out = F.relu(out + identity)
        return out

class FNNLA(nn.Module):
    """
    Fractal Neural Network Learning Architecture
    """
    def __init__(self, 
                 in_channels: int, 
                 num_classes: int,
                 base_channels: int = 64,
                 max_depth: int = 3):
        super(FNNLA, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Create fractal stages
        self.stage1 = FractalUnit(base_channels, base_channels, max_depth=max_depth)
        self.stage2 = FractalUnit(base_channels, base_channels*2, max_depth=max_depth)
        self.stage3 = FractalUnit(base_channels*2, base_channels*4, max_depth=max_depth)
        
        # Global pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels*4, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Process through fractal stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def train_fnnla(dataset='cifar10', epochs=50, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading and preprocessing
    if dataset.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
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
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=2, pin_memory=True)
        
        model = FNNLA(in_channels=3, num_classes=10).to(device)
        
        # Initialize optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_accuracy = 0.0
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = train_epoch(model, train_loader, optimizer, device)
            
            # Evaluation phase
            val_loss, accuracy = evaluate(model, test_loader, device)
            scheduler.step()
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(accuracy)
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                }, 'best_fnnla_model.pth')
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Best Accuracy: {best_accuracy:.4f}')
            print('-' * 30)
        
        return model, history

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    """Evaluate the model"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = correct / len(val_loader.dataset)
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy

if __name__ == "__main__":
    model, history = train_fnnla(dataset='cifar10', epochs=50)
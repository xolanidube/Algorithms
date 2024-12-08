import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
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


class HierarchicalDataset(Dataset):
    """
    Dataset class for hierarchical data with both features and labels.
    """
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, hierarchy: Dict[int, List[int]]):
        self.features = features
        self.labels = labels
        self.hierarchy = hierarchy
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_cifar100_hierarchy() -> Dict[int, List[int]]:
    """
    Create hierarchy for CIFAR100 based on its coarse and fine labels.
    Returns a dictionary mapping superclasses to their subclasses.
    """
    # CIFAR100 has 20 superclasses, each containing 5 fine-grained classes
    hierarchy = {
        0: [0, 1, 2, 3, 4],           # aquatic mammals: beaver, dolphin, otter, seal, whale
        1: [5, 6, 7, 8, 9],           # fish: aquarium fish, flatfish, ray, shark, trout
        2: [10, 11, 12, 13, 14],      # flowers: orchids, poppies, roses, sunflowers, tulips
        3: [15, 16, 17, 18, 19],      # food containers: bottles, bowls, cans, cups, plates
        4: [20, 21, 22, 23, 24],      # fruit and vegetables: apples, mushrooms, oranges, pears, sweet peppers
        5: [25, 26, 27, 28, 29],      # household electrical devices: clock, keyboard, lamp, telephone, television
        6: [30, 31, 32, 33, 34],      # household furniture: bed, chair, couch, table, wardrobe
        7: [35, 36, 37, 38, 39],      # insects: bee, beetle, butterfly, caterpillar, cockroach
        8: [40, 41, 42, 43, 44],      # large carnivores: bear, leopard, lion, tiger, wolf
        9: [45, 46, 47, 48, 49],      # large man-made outdoor things: bridge, castle, house, road, skyscraper
        10: [50, 51, 52, 53, 54],     # large natural outdoor scenes: cloud, forest, mountain, plain, sea
        11: [55, 56, 57, 58, 59],     # large omnivores and herbivores: camel, cattle, chimpanzee, elephant, kangaroo
        12: [60, 61, 62, 63, 64],     # medium-sized mammals: fox, porcupine, possum, raccoon, skunk
        13: [65, 66, 67, 68, 69],     # non-insect invertebrates: crab, lobster, snail, spider, worm
        14: [70, 71, 72, 73, 74],     # people: baby, boy, girl, man, woman
        15: [75, 76, 77, 78, 79],     # reptiles: crocodile, dinosaur, lizard, snake, turtle
        16: [80, 81, 82, 83, 84],     # small mammals: hamster, mouse, rabbit, shrew, squirrel
        17: [85, 86, 87, 88, 89],     # trees: maple, oak, palm, pine, willow
        18: [90, 91, 92, 93, 94],     # vehicles 1: bicycle, bus, motorcycle, pickup truck, train
        19: [95, 96, 97, 98, 99],     # vehicles 2: lawn-mower, rocket, streetcar, tank, tractor
    }
    return hierarchy


class HQNMEClassifier(nn.Module):
    """
    HQNME-based classifier for hierarchical data.
    """
    def __init__(
        self,
        base_model: HQNME,
        num_classes: int,
        hierarchy: Dict[int, List[int]]
    ):
        super().__init__()
        self.base_model = base_model
        self.hierarchy = hierarchy
        self.num_classes = num_classes
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(base_model.layers[-1].out_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize hierarchy-aware loss weights
        self.register_buffer('class_weights', self._compute_class_weights())
        
    def _compute_class_weights(self) -> torch.Tensor:
        """
        Compute weights for each class based on hierarchy level.
        """
        weights = torch.ones(self.num_classes)
        level_weights = {0: 1.0, 1: 1.2}  # Superclass and subclass weights
        
        # Assign weights based on hierarchy level
        for superclass, subclasses in self.hierarchy.items():
            # Superclass gets base weight
            for subclass in subclasses:
                weights[subclass] = level_weights[1]  # Subclasses get higher weight
                
        return weights / weights.mean()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both embeddings and class predictions.
        """
        # Get hyperbolic embeddings
        embeddings = self.base_model(x)
        
        # Get class predictions
        logits = self.classifier(embeddings)
        
        return embeddings, logits
    
    def compute_hierarchical_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss that considers hierarchical relationships.
        """
        # Basic cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        
        # Add hierarchical consistency loss
        hier_loss = torch.tensor(0., device=logits.device)
        probs = F.softmax(logits, dim=1)
        
        # Create reverse mapping from subclass to superclass
        subclass_to_super = {}
        for super_idx, subclasses in self.hierarchy.items():
            for sub_idx in subclasses:
                subclass_to_super[sub_idx] = super_idx
        
        # Compute superclass probabilities
        super_probs = torch.zeros((probs.shape[0], len(self.hierarchy)), device=logits.device)
        for super_idx, subclasses in self.hierarchy.items():
            super_probs[:, super_idx] = probs[:, subclasses].sum(dim=1)
        
        # For each target, ensure superclass probability is higher than other superclasses
        for i, target in enumerate(targets):
            super_class = subclass_to_super[target.item()]
            other_supers = [idx for idx in range(len(self.hierarchy)) if idx != super_class]
            
            # Superclass probability should be higher than other superclass probabilities
            violations = F.relu(super_probs[i, other_supers] - super_probs[i, super_class] + 0.1)
            hier_loss = hier_loss + violations.mean()
        
        # Combine losses
        total_loss = ce_loss + 0.2 * hier_loss
        return total_loss

def prepare_cifar100_data() -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[int]]]:
    """
    Prepare CIFAR100 dataset.
    """
    # Load CIFAR100
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_dataset = datasets.CIFAR100(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Extract features and labels
    features = torch.stack([x for x, _ in train_dataset])
    labels = torch.tensor([y for _, y in train_dataset])
    
    # Get hierarchy
    hierarchy = create_cifar100_hierarchy()
    
    return features, labels, hierarchy




def prepare_cifar100_hierarchy() -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[int]]]:
    """
    Prepare CIFAR100 dataset with its natural hierarchy.
    Returns features, labels, and hierarchy dictionary.
    """
    # Load CIFAR100
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR100(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Extract features and labels
    features = torch.stack([x for x, _ in train_dataset])
    labels = torch.tensor([y for _, y in train_dataset])
    
    # Define hierarchy (simplified version with 20 superclasses)
    hierarchy = {
        # Superclass 0: animals
        0: [4, 30, 55, 72, 95],  # e.g., [fish, mammals, reptiles, insects, aquatic]
        # Superclass 1: vehicles
        1: [1, 32, 48, 61, 88],  # e.g., [automobiles, ships, planes, trains, bikes]
        # ... more superclasses
    }
    
    return features, labels, hierarchy

def train_classifier(
    model: HQNMEClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> HQNMEClassifier:
    """
    Train the HQNME classifier.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam([
        {'params': model.base_model.parameters(), 'lr': lr * 0.1},
        {'params': model.classifier.parameters(), 'lr': lr}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Training loop
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.reshape(batch_features.shape[0], -1)  # Flatten images
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            embeddings, logits = model(batch_features)
            
            # Compute loss
            loss = model.compute_hierarchical_loss(logits, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.reshape(batch_features.shape[0], -1)
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                embeddings, logits = model(batch_features)
                loss = model.compute_hierarchical_loss(logits, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {total_loss/len(train_loader):.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {accuracy:.2f}%\n')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def evaluate_classifier(
    model: HQNMEClassifier,
    test_loader: DataLoader,
    hierarchy: Dict[int, List[int]],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[float, Dict]:
    """
    Evaluate the classifier and compute metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.reshape(features.shape[0], -1)
            features, labels = features.to(device), labels.to(device)
            _, logits = model(features)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Create reverse mapping from subclass to superclass
    subclass_to_super = {}
    for super_idx, subclasses in hierarchy.items():
        for sub_idx in subclasses:
            subclass_to_super[sub_idx] = super_idx
    
    # Compute hierarchical accuracy
    hier_correct = 0
    total = len(all_labels)
    
    for pred, label in zip(all_preds, all_labels):
        if pred == label:
            hier_correct += 1
        else:
            # Check if prediction is in the same superclass
            pred_super = subclass_to_super[pred]
            label_super = subclass_to_super[label]
            if pred_super == label_super:
                hier_correct += 0.5  # Partial credit for same-superclass prediction
    
    hier_accuracy = hier_correct / total
    
    # Generate detailed report
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    metrics = {
        'accuracy': accuracy,
        'hierarchical_accuracy': hier_accuracy,
        'detailed_report': report
    }
    
    return accuracy, metrics


def main():
    # Set random seed
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("\nLoading CIFAR100 dataset...")
    features, labels, hierarchy = prepare_cifar100_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Data shapes:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    # Create datasets
    train_dataset = HierarchicalDataset(X_train, y_train, hierarchy)
    val_dataset = HierarchicalDataset(X_val, y_val, hierarchy)
    test_dataset = HierarchicalDataset(X_test, y_test, hierarchy)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize base HQNME model
    print("\nInitializing HQNME model...")
    base_model = HQNME(
        input_dim=3072,  # 32x32x3 flattened
        hidden_dims=[2048, 1024, 512],
        output_dim=256,
        manifold_dim=128
    )
    
    # Create classifier
    print("Creating hierarchical classifier...")
    classifier = HQNMEClassifier(
        base_model=base_model,
        num_classes=100,
        hierarchy=hierarchy
    )
    
    # Train classifier
    print("\nTraining classifier...")
    trained_classifier = train_classifier(
        classifier,
        train_loader,
        val_loader,
        num_epochs=100,
        lr=1e-3,
        device=device
    )
    
    # Evaluate
    print("\nEvaluating classifier...")
    accuracy, metrics = evaluate_classifier(
        trained_classifier,
        test_loader,
        hierarchy,
        device=device
    )
    
    print("\nFinal Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Hierarchical Accuracy: {metrics['hierarchical_accuracy']:.4f}")
    
    # Print detailed performance metrics
    report_df = pd.DataFrame(metrics['detailed_report']).transpose()
    report_df = report_df.round(4)
    
    # Save results
    print("\nSaving results...")
    torch.save(trained_classifier.state_dict(), 'final_model.pth')
    report_df.to_csv('classification_report.csv')
    
    print("\nResults saved to 'final_model.pth' and 'classification_report.csv'")
    
    return trained_classifier, metrics

if __name__ == "__main__":
    trained_model, final_metrics = main()
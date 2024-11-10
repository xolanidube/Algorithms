import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import random
from abc import ABC, abstractmethod


@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation."""
    # Performance weights
    accuracy_weight: float = 0.4
    resource_weight: float = 0.2
    latency_weight: float = 0.15
    robustness_weight: float = 0.15
    complexity_weight: float = 0.1
    
    # Resource thresholds
    max_memory_usage: float = 8.0  # GB
    max_compute_usage: float = 1e12  # FLOPS
    target_latency: float = 100.0  # ms
    max_energy_usage: float = 1.0  # kWh
    
    # Robustness parameters
    noise_levels: List[float] = (0.1, 0.2, 0.3)
    num_adversarial_samples: int = 100
    epsilon_adversarial: float = 0.1
    
    # Evaluation parameters
    batch_size: int = 32
    num_validation_batches: int = 50
    
class FitnessEvaluator:
    """Comprehensive fitness evaluation for NETOR models."""
    
    def __init__(self, config: FitnessConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute_fitness(self, model: nn.Module, resources: Dict,
                       train_loader: DataLoader, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Compute comprehensive fitness score for a model."""
        model = model.to(self.device)
        model.eval()
        
        # Evaluate different aspects
        accuracy_score, accuracy_metrics = self._evaluate_accuracy(model, val_loader)
        resource_score, resource_metrics = self._evaluate_resources(model, resources)
        latency_score, latency_metrics = self._evaluate_latency(model, val_loader)
        robustness_score, robustness_metrics = self._evaluate_robustness(model, val_loader)
        complexity_score, complexity_metrics = self._evaluate_complexity(model)
        
        # Compute weighted final score
        final_score = (
            self.config.accuracy_weight * accuracy_score +
            self.config.resource_weight * resource_score +
            self.config.latency_weight * latency_score +
            self.config.robustness_weight * robustness_score +
            self.config.complexity_weight * complexity_score
        )
        
        # Compile metrics
        metrics = {
            'accuracy': accuracy_metrics,
            'resource': resource_metrics,
            'latency': latency_metrics,
            'robustness': robustness_metrics,
            'complexity': complexity_metrics,
            'final_score': final_score
        }
        
        return final_score, metrics
    
    def _evaluate_accuracy(self, model: nn.Module,
                          val_loader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate model accuracy and related metrics."""
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx >= self.config.num_validation_batches:
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                # Accuracy
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Store predictions and labels for additional metrics
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
                
                # Per-class accuracy
                for t, p in zip(target, pred):
                    t = t.item()
                    class_correct[t] = class_correct.get(t, 0) + (t == p.item())
                    class_total[t] = class_total.get(t, 0) + 1
        
        # Calculate metrics
        accuracy = correct / total
        per_class_accuracy = {
            cls: class_correct[cls] / class_total[cls]
            for cls in class_total.keys()
        }
        
        # Calculate confusion entropy
        pred_probs = np.bincount(predictions) / len(predictions)
        confusion_entropy = entropy(pred_probs)
        
        metrics = {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_accuracy,
            'confusion_entropy': confusion_entropy
        }
        
        # Normalize score between 0 and 1
        score = accuracy * (1 - confusion_entropy / np.log(len(class_total)))
        
        return score, metrics
    
    def _evaluate_resources(self, model: nn.Module,
                           resources: Dict) -> Tuple[float, Dict]:
        """Evaluate resource utilization efficiency."""
        # Memory efficiency
        memory_usage = resources.get('memory', 0.0)
        memory_efficiency = 1.0 - (memory_usage / self.config.max_memory_usage)
        memory_efficiency = max(0.0, min(1.0, memory_efficiency))
        
        # Compute efficiency
        compute_usage = resources.get('compute', 0.0)
        compute_efficiency = 1.0 - (compute_usage / self.config.max_compute_usage)
        compute_efficiency = max(0.0, min(1.0, compute_efficiency))
        
        # Energy efficiency
        energy_usage = resources.get('energy', 0.0)
        energy_efficiency = 1.0 - (energy_usage / self.config.max_energy_usage)
        energy_efficiency = max(0.0, min(1.0, energy_efficiency))
        
        metrics = {
            'memory_efficiency': memory_efficiency,
            'compute_efficiency': compute_efficiency,
            'energy_efficiency': energy_efficiency,
            'memory_usage': memory_usage,
            'compute_usage': compute_usage,
            'energy_usage': energy_usage
        }
        
        # Compute weighted resource score
        score = (memory_efficiency * 0.4 +
                compute_efficiency * 0.4 +
                energy_efficiency * 0.2)
        
        return score, metrics
    
    def _evaluate_latency(self, model: nn.Module,
                         val_loader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate model latency and throughput."""
        batch_times = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_loader):
                if batch_idx >= self.config.num_validation_batches:
                    break
                
                data = data.to(self.device)
                
                # Measure batch processing time
                start_time = time.time()
                _ = model(data)
                torch.cuda.synchronize()
                batch_time = (time.time() - start_time) * 1000  # Convert to ms
                batch_times.append(batch_time)
        
        # Calculate metrics
        avg_latency = np.mean(batch_times)
        p95_latency = np.percentile(batch_times, 95)
        p99_latency = np.percentile(batch_times, 99)
        throughput = self.config.batch_size / (avg_latency / 1000)  # samples/second
        
        metrics = {
            'avg_latency': avg_latency,
            'p95_latency': p95_latency,
            'p99_latency': p99_latency,
            'throughput': throughput
        }
        
        # Compute latency score
        latency_ratio = avg_latency / self.config.target_latency
        score = 1.0 / (1.0 + np.exp(latency_ratio - 1))  # Sigmoid scaling
        
        return score, metrics
    
    def _evaluate_robustness(self, model: nn.Module,
                            val_loader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate model robustness to noise and adversarial attacks."""
        base_accuracy = 0
        noise_accuracies = []
        adversarial_accuracy = 0
        
        # Get a batch of validation data
        data, target = next(iter(val_loader))
        data, target = data.to(self.device), target.to(self.device)
        
        # Base accuracy
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)
            base_accuracy = pred.eq(target).float().mean().item()
        
        # Noise robustness
        for noise_level in self.config.noise_levels:
            noisy_data = data + torch.randn_like(data) * noise_level
            with torch.no_grad():
                output = model(noisy_data)
                pred = output.argmax(dim=1)
                noise_accuracy = pred.eq(target).float().mean().item()
                noise_accuracies.append(noise_accuracy)
        
        # Adversarial robustness
        adversarial_data = self._generate_adversarial_samples(
            model, data, target, self.config.epsilon_adversarial)
        with torch.no_grad():
            output = model(adversarial_data)
            pred = output.argmax(dim=1)
            adversarial_accuracy = pred.eq(target).float().mean().item()
        
        metrics = {
            'base_accuracy': base_accuracy,
            'noise_accuracies': noise_accuracies,
            'adversarial_accuracy': adversarial_accuracy
        }
        
        # Compute robustness score
        noise_score = np.mean(noise_accuracies) / base_accuracy
        adversarial_score = adversarial_accuracy / base_accuracy
        score = 0.6 * noise_score + 0.4 * adversarial_score
        
        return score, metrics
    
    def _evaluate_complexity(self, model: nn.Module) -> Tuple[float, Dict]:
        """Evaluate model complexity and architecture efficiency."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Analyze layer distribution
        layer_counts = {}
        for module in model.modules():
            layer_type = type(module).__name__
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        # Calculate parameter efficiency
        param_efficiency = trainable_params / total_params if total_params > 0 else 0
        
        # Analyze model depth and width
        depth = len([m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))])
        max_width = max([
            m.out_features if isinstance(m, nn.Linear) else m.out_channels
            for m in model.modules()
            if isinstance(m, (nn.Linear, nn.Conv2d))
        ], default=0)
        
        metrics = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'layer_distribution': layer_counts,
            'depth': depth,
            'max_width': max_width,
            'param_efficiency': param_efficiency
        }
        
        # Compute complexity score
        size_score = 1.0 / (1.0 + np.log10(total_params))
        efficiency_score = param_efficiency
        score = 0.7 * size_score + 0.3 * efficiency_score
        
        return score, metrics
    
    def _generate_adversarial_samples(self, model: nn.Module,
                                    data: torch.Tensor,
                                    target: torch.Tensor,
                                    epsilon: float) -> torch.Tensor:
        """Generate adversarial samples using FGSM."""
        data.requires_grad = True
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        # Calculate gradients
        loss.backward()
        data_grad = data.grad.data
        
        # Create adversarial samples
        perturbed_data = data + epsilon * data_grad.sign()
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        return perturbed_data

@dataclass
class ArchitectureConfig:
    """Configuration for neural architecture."""
    input_size: int
    output_size: int
    max_layers: int
    layer_types: List[str]
    connection_types: List[str]

@dataclass
class ResourceConfig:
    """Configuration for resource management."""
    max_memory: float  # GB
    max_compute: float  # FLOPS
    energy_budget: float  # kWh
    target_latency: float  # ms

class ArchitectureGene:
    """Representation of neural network architecture."""
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.layers = []
        self.connections = {}
        self.transfer_points = []
        self.resource_profile = {}
        
    def initialize_random(self):
        """Initialize a random architecture."""
        num_layers = random.randint(1, self.config.max_layers)
        
        for i in range(num_layers):
            layer_type = random.choice(self.config.layer_types)
            layer_config = self._generate_layer_config(layer_type)
            self.layers.append((layer_type, layer_config))
            
        self._initialize_connections()
        self._initialize_transfer_points()
        
    def _generate_layer_config(self, layer_type: str) -> Dict:
        """Generate configuration for a specific layer type."""
        if layer_type == 'linear':
            return {
                'in_features': random.randint(32, 512),
                'out_features': random.randint(32, 512)
            }
        elif layer_type == 'conv2d':
            return {
                'in_channels': random.randint(3, 64),
                'out_channels': random.randint(32, 256),
                'kernel_size': random.choice([3, 5, 7])
            }
        return {}
    
    def _initialize_connections(self):
        """Initialize layer connections."""
        for i in range(len(self.layers)):
            self.connections[i] = []
            for j in range(i + 1, len(self.layers)):
                if random.random() < 0.3:  # 30% chance of connection
                    self.connections[i].append(j)
                    
    def _initialize_transfer_points(self):
        """Initialize transfer learning points."""
        for i in range(len(self.layers)):
            if random.random() < 0.2:  # 20% chance of transfer point
                self.transfer_points.append(i)
                
    def mutate(self, mutation_rate: float):
        """Mutate the architecture."""
        if random.random() < mutation_rate:
            self._mutate_layers()
        if random.random() < mutation_rate:
            self._mutate_connections()
        if random.random() < mutation_rate:
            self._mutate_transfer_points()
            
    def _mutate_layers(self):
        """Mutate layer structure."""
        mutation_type = random.choice(['add', 'remove', 'modify'])
        
        if mutation_type == 'add' and len(self.layers) < self.config.max_layers:
            layer_type = random.choice(self.config.layer_types)
            layer_config = self._generate_layer_config(layer_type)
            insert_pos = random.randint(0, len(self.layers))
            self.layers.insert(insert_pos, (layer_type, layer_config))
            
        elif mutation_type == 'remove' and len(self.layers) > 1:
            remove_pos = random.randint(0, len(self.layers) - 1)
            self.layers.pop(remove_pos)
            
        elif mutation_type == 'modify' and self.layers:
            modify_pos = random.randint(0, len(self.layers) - 1)
            layer_type = self.layers[modify_pos][0]
            new_config = self._generate_layer_config(layer_type)
            self.layers[modify_pos] = (layer_type, new_config)

class TransferOptimizer:
    """Optimizer for transfer learning strategies."""
    
    def __init__(self):
        self.knowledge_bank = {}
        self.transfer_history = {}
        self.performance_cache = {}
        
    def optimize_transfer(self, source_domain: str, target_domain: str,
                         source_model: nn.Module, target_model: nn.Module) -> Dict:
        """Optimize transfer learning strategy."""
        similarity = self._compute_domain_similarity(source_domain, target_domain)
        transfer_strategy = self._select_transfer_strategy(similarity)
        transfer_points = self._identify_transfer_points(source_model, target_model)
        
        return {
            'strategy': transfer_strategy,
            'transfer_points': transfer_points,
            'similarity_score': similarity
        }
        
    def _compute_domain_similarity(self, source_domain: str, target_domain: str) -> float:
        """Compute similarity between domains."""
        # Implement domain similarity computation
        return random.random()  # Placeholder
        
    def _select_transfer_strategy(self, similarity: float) -> str:
        """Select appropriate transfer strategy based on domain similarity."""
        if similarity > 0.8:
            return 'direct_transfer'
        elif similarity > 0.5:
            return 'partial_transfer'
        else:
            return 'adaptive_transfer'
            
    def _identify_transfer_points(self, source_model: nn.Module,
                                target_model: nn.Module) -> List[int]:
        """Identify optimal transfer points between models."""
        transfer_points = []
        source_layers = list(source_model.modules())
        target_layers = list(target_model.modules())
        
        for i, (s_layer, t_layer) in enumerate(zip(source_layers, target_layers)):
            if type(s_layer) == type(t_layer):
                if self._is_compatible(s_layer, t_layer):
                    transfer_points.append(i)
                    
        return transfer_points
        
    def _is_compatible(self, source_layer: nn.Module,
                      target_layer: nn.Module) -> bool:
        """Check if two layers are compatible for transfer."""
        if isinstance(source_layer, nn.Linear):
            return (source_layer.in_features == target_layer.in_features and
                   source_layer.out_features == target_layer.out_features)
        elif isinstance(source_layer, nn.Conv2d):
            return (source_layer.in_channels == target_layer.in_channels and
                   source_layer.out_channels == target_layer.out_channels and
                   source_layer.kernel_size == target_layer.kernel_size)
        return False

import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import random
from abc import ABC, abstractmethod

# ... (previous code remains the same until ResourceManager class) ...

class ResourceManager:
    """Manager for computational resources."""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.resource_usage = {}
        self.allocation_history = []
        self.monitors = {
            'memory': self._monitor_memory,
            'compute': self._monitor_compute,
            'energy': self._monitor_energy,
            'latency': self._monitor_latency
        }
        
    def optimize_resources(self, model: nn.Module,
                         performance_metrics: Dict) -> Dict:
        """Optimize resource allocation for model."""
        current_usage = self._get_current_usage()
        requirements = self._estimate_requirements(model)
        
        allocation = self._compute_optimal_allocation(
            current_usage, requirements, performance_metrics)
            
        return self._apply_allocation(allocation)
        
    def _get_current_usage(self) -> Dict:
        """Get current resource usage."""
        usage = {}
        for resource, monitor in self.monitors.items():
            usage[resource] = monitor()
        return usage
        
    def _estimate_requirements(self, model: nn.Module) -> Dict:
        """Estimate resource requirements for model."""
        total_params = sum(p.numel() for p in model.parameters())
        
        return {
            'memory': total_params * 4 / (1024 * 1024 * 1024),  # Convert to GB
            'compute': self._estimate_flops(model),
            'energy': self._estimate_energy_consumption(model),
            'latency': self._estimate_latency(model)
        }
        
    def _compute_optimal_allocation(self, current_usage: Dict,
                                  requirements: Dict,
                                  performance_metrics: Dict) -> Dict:
        """Compute optimal resource allocation."""
        allocation = {}
        
        # Memory allocation
        memory_headroom = self.config.max_memory - current_usage['memory']
        allocation['memory'] = min(requirements['memory'], memory_headroom)
        
        # Compute allocation
        compute_efficiency = performance_metrics.get('compute_efficiency', 0.8)
        allocation['compute'] = min(
            requirements['compute'] * compute_efficiency,
            self.config.max_compute
        )
        
        # Energy allocation
        energy_budget_remaining = self.config.energy_budget - current_usage['energy']
        allocation['energy'] = min(requirements['energy'], energy_budget_remaining)
        
        # Latency optimization
        target_latency_ratio = performance_metrics.get('latency', 1.0) / self.config.target_latency
        allocation['batch_size'] = self._optimize_batch_size(target_latency_ratio)
        
        return allocation
        
    def _apply_allocation(self, allocation: Dict) -> Dict:
        """Apply resource allocation to the system."""
        success = True
        applied_allocation = {}
        
        try:
            # Apply memory constraints
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(
                    allocation['memory'] / self.config.max_memory
                )
            
            # Apply compute constraints
            if torch.cuda.is_available():
                torch.cuda.set_device(0)  # Assuming single GPU
                
            # Track energy usage
            self.resource_usage['energy'] = allocation['energy']
            
            # Update batch size
            applied_allocation['batch_size'] = allocation['batch_size']
            
        except Exception as e:
            logging.error(f"Error applying allocation: {str(e)}")
            success = False
            
        applied_allocation['success'] = success
        self.allocation_history.append(applied_allocation)
        
        return applied_allocation
        
    def _monitor_memory(self) -> float:
        """Monitor current memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        return 0.0
        
    def _monitor_compute(self) -> float:
        """Monitor current compute usage."""
        if torch.cuda.is_available():
            return torch.cuda.utilization()
        return 0.0
        
    def _monitor_energy(self) -> float:
        """Monitor energy consumption."""
        return self.resource_usage.get('energy', 0.0)
        
    def _monitor_latency(self) -> float:
        """Monitor current latency."""
        return self.resource_usage.get('latency', 0.0)
        
    def _estimate_flops(self, model: nn.Module) -> float:
        """Estimate FLOPS for the model."""
        total_flops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                out_h = module.output_size[0] if hasattr(module, 'output_size') else 0
                out_w = module.output_size[1] if hasattr(module, 'output_size') else 0
                total_flops += (module.in_channels * module.out_channels *
                              module.kernel_size[0] * module.kernel_size[1] *
                              out_h * out_w)
            elif isinstance(module, nn.Linear):
                total_flops += module.in_features * module.out_features
                
        return total_flops
        
    def _estimate_energy_consumption(self, model: nn.Module) -> float:
        """Estimate energy consumption for the model."""
        # Simplified energy estimation based on FLOPS
        flops = self._estimate_flops(model)
        energy_per_flop = 1e-9  # Example: 1 nanojoule per FLOP
        return flops * energy_per_flop
        
    def _estimate_latency(self, model: nn.Module) -> float:
        """Estimate model latency."""
        # Simplified latency estimation
        flops = self._estimate_flops(model)
        return flops / self.config.max_compute * 1000  # Convert to ms
        
    def _optimize_batch_size(self, target_latency_ratio: float) -> int:
        """Optimize batch size based on target latency."""
        base_batch_size = 32
        if target_latency_ratio > 1.2:  # Too slow
            return max(1, int(base_batch_size / target_latency_ratio))
        elif target_latency_ratio < 0.8:  # Too fast
            return min(256, int(base_batch_size * (1 / target_latency_ratio)))
        return base_batch_size

class NETOR:
    """Main NETOR algorithm implementation."""
    
    def __init__(self, arch_config: ArchitectureConfig,
                 resource_config: ResourceConfig):
        self.arch_config = arch_config
        self.resource_config = resource_config
        self.resource_manager = ResourceManager(resource_config)
        self.transfer_optimizer = TransferOptimizer()
        self.population = []
        self.best_model = None
        self.generation = 0
        
    def initialize_population(self, population_size: int):
        """Initialize population of neural architectures."""
        self.population = []
        for _ in range(population_size):
            gene = ArchitectureGene(self.arch_config)
            gene.initialize_random()
            self.population.append(gene)
            
    def evolve(self, num_generations: int, fitness_fn: callable):
        """Execute the main evolutionary loop."""
        for gen in range(num_generations):
            self.generation = gen
            
            # Evaluate fitness
            fitness_scores = []
            for gene in self.population:
                model = self._construct_model(gene)
                resources = self.resource_manager.optimize_resources(
                    model, {'compute_efficiency': 0.8}
                )
                fitness = fitness_fn(model, resources)
                fitness_scores.append(fitness)
                
            # Select parents
            parents = self._select_parents(fitness_scores)
            
            # Create offspring
            offspring = self._create_offspring(parents)
            
            # Apply transfer learning
            self._apply_transfer_learning(offspring)
            
            # Update population
            self.population = offspring
            
            # Update best model
            best_idx = np.argmax(fitness_scores)
            if self.best_model is None or fitness_scores[best_idx] > self.best_fitness:
                self.best_model = self._construct_model(self.population[best_idx])
                self.best_fitness = fitness_scores[best_idx]
                
    def _construct_model(self, gene: ArchitectureGene) -> nn.Module:
        """Construct PyTorch model from architecture gene."""
        layers = []
        for layer_type, config in gene.layers:
            if layer_type == 'linear':
                layers.append(nn.Linear(**config))
            elif layer_type == 'conv2d':
                layers.append(nn.Conv2d(**config))
            layers.append(nn.ReLU())
            
        return nn.Sequential(*layers)
        
    def _select_parents(self, fitness_scores: List[float]) -> List[ArchitectureGene]:
        """Select parents using tournament selection."""
        tournament_size = 3
        num_parents = len(self.population) // 2
        parents = []
        
        for _ in range(num_parents):
            tournament_idx = np.random.choice(
                len(self.population), tournament_size, replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx])
            
        return parents
        
    def _create_offspring(self, parents: List[ArchitectureGene]) -> List[ArchitectureGene]:
        """Create offspring through crossover and mutation."""
        offspring = []
        
        while len(offspring) < len(self.population):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = self._crossover(parent1, parent2)
            child1.mutate(0.1)
            child2.mutate(0.1)
            offspring.extend([child1, child2])
            
        return offspring[:len(self.population)]
        
    def _crossover(self, parent1: ArchitectureGene,
                  parent2: ArchitectureGene) -> Tuple[ArchitectureGene, ArchitectureGene]:
        """Perform crossover between two parent architectures."""
        child1 = ArchitectureGene(self.arch_config)
        child2 = ArchitectureGene(self.arch_config)
        
        # Crossover layers
        crossover_point = random.randint(1, min(len(parent1.layers), len(parent2.layers)))
        child1.layers = parent1.layers[:crossover_point] + parent2.layers[crossover_point:]
        child2.layers = parent2.layers[:crossover_point] + parent1.layers[crossover_point:]
        
        # Initialize connections and transfer points
        child1._initialize_connections()
        child2._initialize_connections()
        child1._initialize_transfer_points()
        child2._initialize_transfer_points()
        
        return child1, child2
        
    def _apply_transfer_learning(self, population: List[ArchitectureGene]):
        """Apply transfer learning to the population."""
        if self.best_model is not None:
            for gene in population:
                model = self._construct_model(gene)
                transfer_strategy = self.transfer_optimizer.optimize_transfer(
                    'source', 'target', self.best_model, model
                )
                self._apply_transfer_strategy(model, transfer_strategy)
                
    def _apply_transfer_strategy(self, model: nn.Module, strategy: Dict):
        """Apply transfer strategy to model."""
        if strategy['strategy'] == 'direct_transfer':
            self._direct_transfer(model)
        elif strategy['strategy'] == 'partial_transfer':
            self._partial_transfer(model, strategy['transfer_points'])
        else:
            self._adaptive_transfer(model, strategy['similarity_score'])
            
    def _direct_transfer(self, model: nn.Module):
        """Apply direct transfer learning."""
        with torch.no_grad():
            for target_param, source_param in zip(
                model.parameters(), self.best_model.parameters()
            ):
                target_param.data.copy_(source_param.data)
                
    def _partial_transfer(self, model: nn.Module, transfer_points: List[int]):
        """Apply partial transfer learning."""
        with torch.no_grad():
            for i, (target_param, source_param) in enumerate(
                zip(model.parameters(), self.best_model.parameters())
            ):
                if i in transfer_points:
                    target_param.data.copy_(source_param.data)
                    
    def _adaptive_transfer(self, model: nn.Module, similarity_score: float):
        """Apply adaptive transfer learning."""
        with torch.no_grad():
            for target_param, source_param in zip(
                model.parameters(), self.best_model.parameters()
            ):
                target_param.data.copy_(
                    similarity_score * source_param.data +
                    (1 - similarity_score) * target_param.data
                )


class NETORExperiment:
    """NETOR experiment manager."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = f"logs/{self.experiment_name}_{self.timestamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories."""
        directories = ['logs', 'models', 'results', 'visualizations']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def save_config(self, config: Dict):
        """Save experiment configuration."""
        config_file = f"logs/{self.experiment_name}_{self.timestamp}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
            
    def save_results(self, results: Dict):
        """Save experiment results."""
        results_file = f"results/{self.experiment_name}_{self.timestamp}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
    def plot_training_progress(self, history: Dict):
        """Plot training progress metrics."""
        plt.figure(figsize=(15, 10))
        
        # Plot fitness progression
        plt.subplot(2, 2, 1)
        plt.plot(history['fitness_scores'])
        plt.title('Best Fitness Score per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        
        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history['accuracy'])
        plt.title('Model Accuracy per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        
        # Plot resource efficiency
        plt.subplot(2, 2, 3)
        plt.plot(history['resource_efficiency'])
        plt.title('Resource Efficiency per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Efficiency')
        
        # Plot model complexity
        plt.subplot(2, 2, 4)
        plt.plot(history['model_complexity'])
        plt.title('Model Complexity per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Complexity Score')
        
        plt.tight_layout()
        plt.savefig(f"visualizations/{self.experiment_name}_{self.timestamp}_progress.png")
        plt.close()

def prepare_data():
    """Prepare MNIST dataset for training and validation."""
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST('data', train=True, download=True,
                                 transform=transform)
    test_dataset = datasets.MNIST('data', train=False,
                                transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return train_loader, test_loader

def main():
    """Main execution function."""
    # Initialize experiment
    experiment = NETORExperiment("mnist_netor")
    
    # Configure the algorithm
    arch_config = ArchitectureConfig(
        input_size=784,  # 28x28 MNIST images
        output_size=10,  # 10 digits
        max_layers=10,
        layer_types=['linear', 'conv2d'],
        connection_types=['sequential', 'residual']
    )

    resource_config = ResourceConfig(
        max_memory=8.0,  # 8GB
        max_compute=1e12,  # 1 TFLOPS
        energy_budget=1.0,  # 1 kWh
        target_latency=100.0  # 100ms
    )
    
    # Save configurations
    experiment.save_config({
        'arch_config': arch_config.__dict__,
        'resource_config': resource_config.__dict__
    })
    
    # Prepare data
    train_loader, test_loader = prepare_data()
    
    # Initialize NETOR
    netor = NETOR(arch_config, resource_config)
    netor.initialize_population(population_size=50)
    
    # Create fitness function
    fitness_config = FitnessConfig()
    fitness_evaluator = FitnessEvaluator(fitness_config)
    
    def fitness_fn(model: nn.Module, resources: Dict) -> float:
        """Fitness function for model evaluation."""
        score, metrics = fitness_evaluator.compute_fitness(
            model, resources, train_loader, test_loader)
        return score, metrics
    
    # Training history
    history = {
        'fitness_scores': [],
        'accuracy': [],
        'resource_efficiency': [],
        'model_complexity': [],
        'best_models': []
    }
    
    # Run evolution
    try:
        for generation in range(100):  # 100 generations
            experiment.logger.info(f"Starting generation {generation + 1}")
            
            # Evolve one generation
            generation_metrics = netor.evolve(num_generations=1, fitness_fn=fitness_fn)
            
            # Update history
            history['fitness_scores'].append(generation_metrics['best_fitness'])
            history['accuracy'].append(generation_metrics['best_accuracy'])
            history['resource_efficiency'].append(generation_metrics['resource_efficiency'])
            history['model_complexity'].append(generation_metrics['model_complexity'])
            
            # Save best model from this generation
            if generation_metrics['best_fitness'] == max(history['fitness_scores']):
                model_path = f"models/{experiment.experiment_name}_gen{generation}.pt"
                torch.save(netor.best_model.state_dict(), model_path)
                history['best_models'].append(model_path)
            
            # Log progress
            experiment.logger.info(
                f"Generation {generation + 1} completed:\n"
                f"Best Fitness: {generation_metrics['best_fitness']:.4f}\n"
                f"Accuracy: {generation_metrics['best_accuracy']:.4f}\n"
                f"Resource Efficiency: {generation_metrics['resource_efficiency']:.4f}"
            )
            
            # Plot progress every 10 generations
            if (generation + 1) % 10 == 0:
                experiment.plot_training_progress(history)
    
    except KeyboardInterrupt:
        experiment.logger.info("Training interrupted by user")
    except Exception as e:
        experiment.logger.error(f"Error during training: {str(e)}")
    finally:
        # Save final results
        experiment.save_results(history)
        experiment.plot_training_progress(history)
        
        # Final evaluation of best model
        if netor.best_model is not None:
            final_score, final_metrics = fitness_fn(netor.best_model, {})
            experiment.logger.info(
                f"\nFinal Results:\n"
                f"Best Model Score: {final_score:.4f}\n"
                f"Final Metrics: {json.dumps(final_metrics, indent=2)}"
            )

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    
    # Execute main function
    main()
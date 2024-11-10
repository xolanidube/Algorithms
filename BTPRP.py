import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import heapq
from scipy.stats import entropy
import warnings

@dataclass
class TemporalPattern:
    """Represents a discovered temporal pattern with its characteristics."""
    sequence: np.ndarray
    frequency: int
    confidence: float
    temporal_range: Tuple[int, int]
    context_vector: np.ndarray
    evolution_rate: float

class NeuralMemoryCell:
    """Biomimetic neural memory cell for pattern storage and retrieval."""
    
    def __init__(self, input_dimension: int, context_dimension: int, plasticity: float = 0.1):
        self.input_dimension = input_dimension
        self.context_dimension = context_dimension
        self.weights = np.random.normal(0, 0.1, (input_dimension, input_dimension))
        self.context_weights = np.random.normal(0, 0.1, (input_dimension, context_dimension))
        self.plasticity = plasticity
        self.activation_history = []
        self.pattern_memory = {}
        self.context_vector = np.zeros(context_dimension)
        
    def update(self, input_pattern: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Update cell weights based on input pattern and context."""
        # Ensure input_pattern is correctly shaped
        input_pattern = input_pattern.reshape(self.input_dimension)
        context = context.reshape(self.context_dimension)
        
        # Update context vector with exponential moving average
        self.context_vector = 0.9 * self.context_vector + 0.1 * context
        
        # Compute activation with both pattern and context influence
        pattern_activation = np.tanh(np.dot(self.weights, input_pattern))
        context_influence = np.tanh(np.dot(self.context_weights, self.context_vector))
        
        # Combine activations
        activation = 0.7 * pattern_activation + 0.3 * context_influence
        
        # Hebbian learning with context modulation
        weight_update = self.plasticity * np.outer(activation, input_pattern)
        context_weight_update = self.plasticity * np.outer(activation, self.context_vector)
        
        # Update weights
        self.weights += weight_update
        self.context_weights += 0.1 * context_weight_update
        
        # Record activation
        self.activation_history.append(np.mean(activation))
        
        return activation

class BTPRP:
    """Biomimetic Temporal Pattern Recognition and Prediction Algorithm"""
    
    def __init__(self, 
                 input_dimension: int,
                 memory_cells: int = 10,
                 plasticity: float = 0.1,
                 pattern_threshold: float = 0.75):
        self.input_dimension = input_dimension
        # Context dimension is 4 * input_dimension due to temporal features
        self.context_dimension = 4 * input_dimension
        
        self.memory_cells = [
            NeuralMemoryCell(
                input_dimension=input_dimension,
                context_dimension=self.context_dimension,
                plasticity=plasticity
            ) for _ in range(memory_cells)
        ]
        
        self.pattern_threshold = pattern_threshold
        self.discovered_patterns: List[TemporalPattern] = []
        self.temporal_context = np.zeros(self.context_dimension)
        self.adaptation_rate = 0.1
        
    def _compute_temporal_context(self, sequence: np.ndarray) -> np.ndarray:
        """Compute temporal context vector from recent history."""
        if len(sequence) == 0:
            return np.zeros(self.context_dimension)
            
        # Multi-scale temporal features
        short_term = np.mean(sequence[-5:], axis=0) if len(sequence) >= 5 else np.mean(sequence, axis=0)
        long_term = np.mean(sequence, axis=0)
        
        # Compute temporal derivatives
        derivatives = np.diff(sequence, axis=0) if len(sequence) > 1 else np.zeros((1, self.input_dimension))
        trend = np.mean(derivatives, axis=0)
        
        # Compute volatility (standard deviation)
        volatility = np.std(sequence, axis=0)
        
        # Combine features into context vector
        context = np.concatenate([
            short_term,
            long_term,
            trend,
            volatility
        ])
        
        # Normalize context vector
        norm = np.linalg.norm(context)
        return context / norm if norm > 0 else context
    
    def _detect_anomalies(self, pattern: np.ndarray, history: np.ndarray) -> float:
        """Detect anomalies using multi-scale entropy analysis."""
        if len(history) < 2:
            return 0.0
            
        # Compute multi-scale entropy
        scales = [1, 2, 4]
        entropies = []
        
        for scale in scales:
            # Create coarse-grained time series
            coarse_grained = np.array([
                np.mean(history[i:i+scale], axis=0) 
                for i in range(0, len(history), scale)
            ])
            
            # Compute sample entropy
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ent = entropy(np.histogram(coarse_grained.flatten(), bins=20)[0])
            entropies.append(ent)
        
        # Compare pattern entropy with historical entropy
        pattern_entropy = entropy(np.histogram(pattern.flatten(), bins=20)[0])
        avg_historical_entropy = np.mean(entropies)
        
        return abs(pattern_entropy - avg_historical_entropy) / avg_historical_entropy \
            if avg_historical_entropy > 0 else 0.0
    
    def recognize_patterns(self, sequence: np.ndarray) -> List[TemporalPattern]:
        """Recognize temporal patterns in the input sequence."""
        patterns = []
        sequence_length = len(sequence)
        
        # Ensure sequence has correct shape
        if sequence.ndim == 1:
            sequence = sequence.reshape(-1, self.input_dimension)
        
        # Sliding window analysis
        for window_size in [5, 10, 20]:  # Multiple temporal scales
            if window_size > sequence_length:
                continue
                
            for i in range(sequence_length - window_size + 1):
                window = sequence[i:i+window_size]
                
                # Compute context from sequence history
                context = self._compute_temporal_context(sequence[:i])
                
                # Flatten window for neural processing while preserving dimensionality
                window_flat = window.reshape(-1, self.input_dimension)[-1]
                
                # Activate memory cells
                cell_activations = []
                for cell in self.memory_cells:
                    activation = cell.update(window_flat, context)
                    cell_activations.append(np.mean(activation))
                
                # Check for pattern recognition
                max_activation = max(cell_activations)
                if max_activation > self.pattern_threshold:
                    # Compute pattern confidence and characteristics
                    anomaly_score = self._detect_anomalies(window, sequence[:i])
                    confidence = max_activation * (1 - anomaly_score)
                    
                    if confidence > self.pattern_threshold:
                        pattern = TemporalPattern(
                            sequence=window.copy(),
                            frequency=1,
                            confidence=confidence,
                            temporal_range=(i, i+window_size),
                            context_vector=context.copy(),
                            evolution_rate=anomaly_score
                        )
                        patterns.append(pattern)
        
        # Merge similar patterns
        merged_patterns = self._merge_similar_patterns(patterns)
        self.discovered_patterns.extend(merged_patterns)
        
        return merged_patterns

    def _merge_similar_patterns(self, patterns: List[TemporalPattern]) -> List[TemporalPattern]:
        """Merge similar patterns using hierarchical clustering."""
        if not patterns:
            return []
            
        # Compute pattern similarity matrix
        n_patterns = len(patterns)
        similarity_matrix = np.zeros((n_patterns, n_patterns))
        
        for i in range(n_patterns):
            for j in range(i+1, n_patterns):
                similarity = self._compute_pattern_similarity(patterns[i], patterns[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Hierarchical clustering
        merged = []
        used = set()
        
        for i in range(n_patterns):
            if i in used:
                continue
                
            cluster = [patterns[i]]
            used.add(i)
            
            # Find similar patterns
            for j in range(i+1, n_patterns):
                if j in used:
                    continue
                    
                if similarity_matrix[i, j] > 0.8:  # Similarity threshold
                    cluster.append(patterns[j])
                    used.add(j)
            
            # Merge cluster into single pattern
            if cluster:
                merged_pattern = self._combine_patterns(cluster)
                merged.append(merged_pattern)
        
        return merged
    
    def _compute_pattern_similarity(self, p1: TemporalPattern, p2: TemporalPattern) -> float:
        """Compute similarity between two patterns."""
        # Dynamic Time Warping distance
        len1, len2 = len(p1.sequence), len(p2.sequence)
        dtw_matrix = np.full((len1 + 1, len2 + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = np.linalg.norm(p1.sequence[i-1] - p2.sequence[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                            dtw_matrix[i, j-1],    # deletion
                                            dtw_matrix[i-1, j-1])  # match
        
        dtw_distance = dtw_matrix[len1, len2]
        
        # Context similarity
        context_similarity = np.dot(p1.context_vector, p2.context_vector)
        
        # Combine similarities
        return 1 / (1 + dtw_distance) * 0.7 + context_similarity * 0.3

    def _combine_patterns(self, patterns: List[TemporalPattern]) -> TemporalPattern:
        """Combine a cluster of similar patterns into a single pattern."""
        # Average sequence and context vectors
        avg_sequence = np.mean([p.sequence for p in patterns], axis=0)
        avg_context = np.mean([p.context_vector for p in patterns], axis=0)
        
        # Combine metadata
        total_frequency = sum(p.frequency for p in patterns)
        avg_confidence = np.mean([p.confidence for p in patterns])
        avg_evolution = np.mean([p.evolution_rate for p in patterns])
        
        # Find temporal range covering all patterns
        min_start = min(p.temporal_range[0] for p in patterns)
        max_end = max(p.temporal_range[1] for p in patterns)
        
        return TemporalPattern(
            sequence=avg_sequence,
            frequency=total_frequency,
            confidence=avg_confidence,
            temporal_range=(min_start, max_end),
            context_vector=avg_context,
            evolution_rate=avg_evolution
        )

def test_algorithm():
    """Test the BTPRP algorithm on synthetic data."""
    # Generate synthetic time series with patterns
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    
    # Create pattern components
    trend = 0.1 * t
    seasonal = 2 * np.sin(2 * np.pi * 0.5 * t)
    noise = np.random.normal(0, 0.2, len(t))
    
    # Combine components
    series = trend + seasonal + noise
    
    # Add some anomalies
    series[300:310] += 3
    series[600:610] -= 2
    
    # Create 2D time series
    data = np.column_stack([series, np.roll(series, 5)])
    
    # Initialize algorithm
    btprp = BTPRP(input_dimension=2)
    
    # Process data in chunks
    chunk_size = 50
    n_chunks = len(data) // chunk_size
    
    print("Testing BTPRP Algorithm...")
    
    all_patterns = []
    for i in range(n_chunks):
        chunk = data[i*chunk_size:(i+1)*chunk_size]
        patterns = btprp.recognize_patterns(chunk)
        all_patterns.extend(patterns)
        
        if patterns:
            print(f"\nChunk {i+1}: Found {len(patterns)} patterns")
            for j, pattern in enumerate(patterns):
                print(f"Pattern {j+1}:")
                print(f"- Confidence: {pattern.confidence:.3f}")
                print(f"- Evolution Rate: {pattern.evolution_rate:.3f}")
                print(f"- Temporal Range: {pattern.temporal_range}")
    
    print(f"\nTotal patterns discovered: {len(all_patterns)}")
    print(f"Average confidence: {np.mean([p.confidence for p in all_patterns]):.3f}")
    print(f"Average evolution rate: {np.mean([p.evolution_rate for p in all_patterns]):.3f}")
    
    return btprp, data

if __name__ == "__main__":
    btprp, data = test_algorithm()
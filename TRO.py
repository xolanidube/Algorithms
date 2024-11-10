import numpy as np
import pandas as pd
from scipy import signal
import pywt
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

class TemporalResonanceOptimizer:
    def __init__(self, dimensions, time_horizon, learning_rate=0.01):
        self.dimensions = dimensions
        self.time_horizon = time_horizon
        self.learning_rate = learning_rate
        self.resonance_patterns = []
        self.interference_matrix = np.zeros((dimensions, dimensions))
        self.history = defaultdict(list)
        self.scaler = MinMaxScaler()

    def compute_temporal_wave(self, resource_pattern, frequency):
        """
        Enhanced temporal wave computation with fixed wavelet analysis
        """
        # Preprocess the pattern
        processed_pattern = self.preprocess_data(resource_pattern)
        
        # Multi-resolution decomposition
        wavelet = 'db4'
        max_level = pywt.dwt_max_level(len(processed_pattern), pywt.Wavelet(wavelet).dec_len)
        level = min(3, max_level)
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(processed_pattern, wavelet, level=level)
        
        # Reconstruct with modified coefficients
        modified_coeffs = list(coeffs)
        for i in range(1, len(modified_coeffs)):
            modified_coeffs[i] = modified_coeffs[i] * (1.0 / (i + 1))
        
        # Reconstruct signal
        enhanced_pattern = pywt.waverec(modified_coeffs, wavelet)
        
        # Trim to original length if necessary
        if len(enhanced_pattern) > len(processed_pattern):
            enhanced_pattern = enhanced_pattern[:len(processed_pattern)]
        
        # Generate base oscillation
        time_steps = np.arange(len(processed_pattern))
        base_wave = np.sin(2 * np.pi * frequency * time_steps)
        
        # Combine patterns
        combined_wave = enhanced_pattern * base_wave
        
        # Apply envelope detection
        analytic_signal = signal.hilbert(combined_wave)
        envelope = np.abs(analytic_signal)
        
        return (combined_wave * envelope) / np.max(np.abs(combined_wave * envelope))

    def preprocess_data(self, data, sampling_rate=None):
        """Preprocess input data with advanced signal processing"""
        data = np.array(data, dtype=float)
        
        # Remove outliers using IQR method
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        data = np.clip(data, Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        
        # Apply Savitzky-Golay filter for smoothing
        window_length = min(5, len(data) - 1)
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 3:
            data = signal.savgol_filter(data, window_length=window_length, polyorder=2)
        
        # Normalize data
        data = self.scaler.fit_transform(data.reshape(-1, 1)).ravel()
        
        if sampling_rate:
            original_points = np.arange(len(data))
            resampled_points = np.linspace(0, len(data)-1, int(len(data)*sampling_rate))
            data = np.interp(resampled_points, original_points, data)
            
        return data

    def calculate_base_frequency(self, pattern):
        """Calculate base frequency using FFT"""
        fft_result = np.fft.fft(pattern)
        frequencies = np.fft.fftfreq(len(pattern))
        
        # Find peak frequency excluding DC component
        peak_freq = frequencies[np.argmax(np.abs(fft_result[1:])) + 1]
        return np.abs(peak_freq)

    def calculate_interference(self, wave1, wave2):
        """Enhanced interference calculation"""
        # Normalize waves
        norm_wave1 = (wave1 - np.mean(wave1)) / (np.std(wave1) + 1e-8)
        norm_wave2 = (wave2 - np.mean(wave2)) / (np.std(wave2) + 1e-8)
        
        # Calculate cross-correlation
        correlation = signal.correlate(norm_wave1, norm_wave2, mode='full')
        max_corr = np.max(np.abs(correlation))
        
        return np.clip(max_corr / len(norm_wave1), -1, 1)

    def generate_action(self, state):
        """Generate an action based on current state"""
        action = np.dot(state, np.random.randn(state.shape[1], self.dimensions))
        return np.clip(action, 0, 1)

    def get_current_state(self):
        """Get current state of the system"""
        if not self.resonance_patterns:
            return np.zeros((self.time_horizon, self.dimensions))
        return np.array(self.resonance_patterns).T

    def calculate_reward(self, allocation):
        """
        Calculate reward for current allocation
        """
        # Calculate mean utilization across time
        utilization = np.mean(allocation)
        
        # Calculate balance across resources
        resource_means = np.mean(allocation, axis=1)
        balance = 1 - np.std(resource_means)
        
        return 0.7 * utilization + 0.3 * balance

    def apply_resonance(self, action, patterns=None):
        """
        Apply resonance patterns to modify actions
        
        Args:
            action (np.array): Original action array of shape (dimensions, time_horizon)
            patterns (list): Optional list of temporal patterns
            
        Returns:
            np.array: Modified action considering interference effects
        """
        patterns = patterns if patterns is not None else self.resonance_patterns
        patterns_array = np.array(patterns)
        
        # Ensure action has correct shape (dimensions, time_horizon)
        if len(action.shape) == 1:
            action = action.reshape(-1, 1) * np.ones((1, self.time_horizon))
        
        modified_action = action.copy()
        
        # Calculate interference effects for each dimension
        for i in range(self.dimensions):
            # Reshape interference matrix row for proper broadcasting
            interference_weights = self.interference_matrix[i].reshape(-1, 1)
            
            # Calculate weighted sum of patterns
            interference_effects = np.sum(
                interference_weights * patterns_array,
                axis=0
            )
            
            # Apply interference effects
            modified_action[i, :] *= (1 + interference_effects)
        
        return np.clip(modified_action, 0, 1)

    def optimize_resource_allocation(self, current_state, constraints, n_episodes=1000):
        """
        Optimizes resource allocation with real-time adaptation
        """
        self.training_episodes = n_episodes
        optimization_metrics = {'episode_rewards': [], 'constraint_violations': 0}
        
        # Generate temporal waves
        self.resonance_patterns = []
        for dim in range(self.dimensions):
            resource_pattern = current_state[f'resource_{dim}']
            processed_pattern = self.preprocess_data(resource_pattern)
            base_freq = self.calculate_base_frequency(processed_pattern)
            wave = self.compute_temporal_wave(processed_pattern, base_freq)
            self.resonance_patterns.append(wave)
        
        # Calculate interference matrix
        for i in range(self.dimensions):
            for j in range(i + 1, self.dimensions):
                interference = self.calculate_interference(
                    self.resonance_patterns[i],
                    self.resonance_patterns[j]
                )
                self.interference_matrix[i, j] = interference
                self.interference_matrix[j, i] = interference
        
        # Optimization loop
        best_reward = float('-inf')
        best_allocation = None
        
        for episode in range(self.training_episodes):
            # Generate base action
            action = np.random.rand(self.dimensions, self.time_horizon)
            
            # Apply resonance patterns
            modified_action = self.apply_resonance(action)
            
            if self.validate_constraints(modified_action, constraints):
                reward = self.calculate_reward(modified_action)
                optimization_metrics['episode_rewards'].append(reward)
                
                if reward > best_reward:
                    best_reward = reward
                    best_allocation = modified_action.copy()
            else:
                optimization_metrics['constraint_violations'] += 1
        
        # Ensure final allocation has correct shape (dimensions, time_horizon)
        final_allocation = best_allocation if best_allocation is not None else np.zeros((self.dimensions, self.time_horizon))
        uncertainty = self.estimate_uncertainty(final_allocation)
        
        return {
            'allocation': final_allocation,
            'uncertainty': uncertainty,
            'metrics': optimization_metrics
        }

    def validate_constraints(self, allocation, constraints):
        """
        Validate if allocation meets constraints
        """
        # Calculate mean allocation across time horizon for constraint checking
        mean_allocation = np.mean(allocation, axis=1)
        
        if np.any(mean_allocation > constraints['max_capacity']):
            return False
        if np.any(mean_allocation < constraints['min_requirements']):
            return False
        
        interference_sum = np.sum(
            self.interference_matrix * (mean_allocation.reshape(-1, 1) @ mean_allocation.reshape(1, -1))
        )
        if interference_sum > constraints['max_interference']:
            return False
            
        return True
    
    def generate_action(self, state):
        """
        Generate an action based on current state with proper shapes
        """
        # Ensure proper shapes for matrix multiplication
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        action = np.dot(state, np.random.randn(state.shape[1], self.dimensions))
        return np.clip(action, 0, 1)

    def get_current_state(self):
        """
        Get current state of the system with proper shape handling
        """
        if not self.resonance_patterns:
            return np.zeros(self.time_horizon)
        return np.mean(np.array(self.resonance_patterns), axis=1)

    def estimate_uncertainty(self, allocation):
        """Estimate uncertainty using bootstrap sampling"""
        n_samples = 100
        bootstrap_allocations = []
        
        for _ in range(n_samples):
            sample_indices = np.random.choice(
                len(self.resonance_patterns[0]),
                size=len(self.resonance_patterns[0]),
                replace=True
            )
            bootstrap_patterns = [pattern[sample_indices] for pattern in self.resonance_patterns]
            bootstrap_allocation = self.apply_resonance(allocation, patterns=bootstrap_patterns)
            bootstrap_allocations.append(bootstrap_allocation)
            
        return {
            'std': np.std(bootstrap_allocations, axis=0),
            'confidence_intervals': np.percentile(bootstrap_allocations, [2.5, 97.5], axis=0)
        }

# Example usage with synthetic data
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    time_steps = 100
    
    # Generate synthetic patterns with different characteristics
    t = np.linspace(0, 2*np.pi, time_steps)
    traffic_data = {
        'resource_0': 0.5 + 0.3*np.sin(2*t) + 0.1*np.random.randn(time_steps),
        'resource_1': 0.6 + 0.2*np.sin(3*t + np.pi/4) + 0.05*np.random.randn(time_steps),
        'resource_2': 0.4 + 0.25*np.sin(4*t + np.pi/3) + 0.08*np.random.randn(time_steps)
    }
    
    # Define constraints
    constraints = {
        'max_capacity': np.array([1.0, 0.8, 0.9]),
        'min_requirements': np.array([0.2, 0.3, 0.1]),
        'max_interference': 0.7
    }
    
    # Initialize and run optimizer
    optimizer = TemporalResonanceOptimizer(dimensions=3, time_horizon=time_steps)
    result = optimizer.optimize_resource_allocation(traffic_data, constraints, n_episodes=500)
    
    # Print results
    print("\nOptimization Results:")
    print(f"Final allocation shape: {result['allocation'].shape}")
    print(f"Average reward: {np.mean(result['metrics']['episode_rewards']):.4f}")
    print(f"Constraint violations: {result['metrics']['constraint_violations']}")
    print(f"Average uncertainty (std): {np.mean(result['uncertainty']['std']):.4f}")
    
    # Calculate resource utilization
    final_allocation = result['allocation']
    utilization = np.mean(final_allocation, axis=1)
    print("\nResource Utilization:")
    for i, util in enumerate(utilization):
        print(f"Resource {i}: {util:.2%}")
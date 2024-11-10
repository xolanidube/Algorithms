import numpy as np
import pandas as pd
from scipy.stats import entropy
from itertools import combinations

class ConvergentDimensionalHarmony:
    def __init__(self, correlation_threshold=0.8):
        self.correlation_threshold = correlation_threshold
        self.harmonies = []

    def calculate_entropy(self, data):
        """
        Calculate entropy for each column (dimension) of the dataset.
        """
        entropies = {}
        for col in data.columns:
            p_data = data[col].value_counts(normalize=True)  # Get the probability distribution
            entropies[col] = entropy(p_data)
        return entropies

    def calculate_correlations(self, data):
        """
        Calculate pairwise correlations between dimensions.
        """
        correlation_matrix = data.corr()
        return correlation_matrix

    def fuse_dimensions(self, data, correlations, entropies):
        """
        Fuse dimensions based on correlation threshold and create harmonies.
        """
        remaining_dims = set(data.columns)
        fused_dims = set()
        harmonies = []

        # Sort dimensions by entropy, descending, to prioritize high-information dimensions
        sorted_dims = sorted(entropies.keys(), key=lambda k: entropies[k], reverse=True)
        
        # Iterate over all pairs of dimensions to identify highly correlated pairs
        for dim1, dim2 in combinations(sorted_dims, 2):
            if dim1 in fused_dims or dim2 in fused_dims:
                continue

            correlation = correlations.loc[dim1, dim2]
            if abs(correlation) >= self.correlation_threshold:
                # Fuse the dimensions and create a new harmony
                harmony_name = f"{dim1}_{dim2}_harmony"
                harmonies.append((harmony_name, [dim1, dim2]))
                fused_dims.add(dim1)
                fused_dims.add(dim2)
                remaining_dims.discard(dim1)
                remaining_dims.discard(dim2)

        # Add remaining individual dimensions as harmonies
        for dim in remaining_dims:
            harmonies.append((dim, [dim]))

        self.harmonies = harmonies
        return harmonies

    def create_harmony_dataframe(self, data):
        """
        Create a new dataframe where the harmonies are represented as individual columns.
        """
        harmony_data = {}
        for harmony_name, dims in self.harmonies:
            if len(dims) == 1:
                harmony_data[harmony_name] = data[dims[0]]
            else:
                harmony_data[harmony_name] = data[dims].mean(axis=1)  # Taking the mean as a simple fusion strategy
        return pd.DataFrame(harmony_data)

    def fit_transform(self, data):
        """
        Full process of fitting the CDH model to the data and returning the transformed dataset.
        """
        entropies = self.calculate_entropy(data)
        correlations = self.calculate_correlations(data)
        self.fuse_dimensions(data, correlations, entropies)
        harmony_df = self.create_harmony_dataframe(data)
        return harmony_df

# Example Usage
data = pd.DataFrame({
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100) * 0.5,
    'Feature3': np.random.rand(100) * 2,
    'Feature4': np.random.rand(100) + 1,
    'Feature5': np.random.rand(100) * 1.5
})

cdh = ConvergentDimensionalHarmony(correlation_threshold=0.7)
harmony_data = cdh.fit_transform(data)

print("Original Data:\n", data.head())
print("\nHarmony Data:\n", harmony_data.head())

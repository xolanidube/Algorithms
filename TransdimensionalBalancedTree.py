import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

class TransdimensionalBalancedTree:
    """
    Transdimensional Balanced Tree (TBT) implementation.
    """

    def __init__(self, dimension=3, mod=997):
        """
        Initialize the TBT structure.
        :param dimension: Number of dimensions for the transdimensional space.
        :param mod: Prime number for modular arithmetic to ensure group operations.
        """
        self.dimension = dimension
        self.mod = mod
        self.data_store = {}  # Store transdimensional representations

    def _transdimensional_mapping(self, key):
        """
        Map a key to a transdimensional space using hash-based embeddings.
        :param key: The key to map.
        :return: A vector in the transdimensional space.
        """
        np.random.seed(hash(key) % self.mod)  # Ensure deterministic mapping
        return np.random.randint(1, self.mod, size=self.dimension)

    def _group_operation(self, vec1, vec2):
        """
        Group operation using modular addition.
        :param vec1: First vector.
        :param vec2: Second vector.
        :return: Resultant vector.
        """
        return (vec1 + vec2) % self.mod

    def _inverse_operation(self, vec):
        """
        Compute the inverse of a vector under modular arithmetic.
        :param vec: Input vector.
        :return: Inverse vector.
        """
        return (-vec) % self.mod

    def insert(self, key, data):
        """
        Insert a new element into the TBT.
        :param key: Unique key for the data.
        :param data: Data to store.
        """
        mapping = self._transdimensional_mapping(key)
        self.data_store[key] = (mapping, data)

    def delete(self, key):
        """
        Delete an element from the TBT.
        :param key: Unique key of the element to delete.
        """
        if key in self.data_store:
            del self.data_store[key]

    def search(self, key):
        """
        Search for an element in the TBT.
        :param key: Key of the element to search.
        :return: Data if found, else None.
        """
        return self.data_store.get(key, None)

    def visualize(self):
        """
        Visualize the TBT keys in 2D using PCA (if dimension > 2).
        """
        if len(self.data_store) == 0:
            print("No data to visualize.")
            return

        keys, data = zip(*self.data_store.values())
        data_matrix = np.array(keys)

        # Reduce dimensionality if needed
        if self.dimension > 2:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(data_matrix)
        else:
            reduced_data = data_matrix

        # Scatter plot
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', label='Data Points')
        plt.title("Transdimensional Balanced Tree Visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid()
        plt.show()

# Example Usage with Real-World Data
if __name__ == "__main__":
    # Initialize the TBT
    tbt = TransdimensionalBalancedTree(dimension=3)

    # Load real-world data: Iris dataset
    iris = load_iris()
    data = iris.data
    target = iris.target
    target_names = iris.target_names

    # Insert data into the TBT
    for i, (features, label) in enumerate(zip(data, target)):
        key = f"iris_{i}"
        tbt.insert(key, (features, target_names[label]))

    # Search for a specific element
    search_key = "iris_10"
    result = tbt.search(search_key)
    print(f"Search result for {search_key}: {result}")

    # Visualize the TBT
    tbt.visualize()

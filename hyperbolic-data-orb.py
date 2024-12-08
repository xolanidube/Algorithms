import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from math import acosh, sqrt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

class HyperbolicDataOrb:
    """
    Hyperbolic Data Orb implementation using the Poincaré Disk model.
    """

    def __init__(self, dimension=2):
        """
        Initialize the HDO structure.
        :param dimension: Number of dimensions for hyperbolic space. Default is 2.
        """
        self.dimension = dimension
        self.data_points = {}  # Stores data points with keys
        self.unit_disk_radius = 1.0  # Radius of the Poincaré Disk

    def _hyperbolic_embedding(self, point):
        """
        Normalize a point to fit within the Poincaré Disk.
        :param point: Input point in Euclidean space.
        :return: Embedded point in hyperbolic space.
        """
        norm = np.linalg.norm(point)
        if norm >= self.unit_disk_radius:
            point = (point / norm) * (self.unit_disk_radius - 1e-6)  # Adjust to fit within the unit disk
        return point

    def _poincare_distance(self, u, v):
        """
        Compute the hyperbolic distance between two points in the Poincaré Disk model.
        :param u: First point in hyperbolic space.
        :param v: Second point in hyperbolic space.
        :return: Hyperbolic distance.
        """
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        numerator = 2 * np.linalg.norm(u - v)**2
        denominator = (1 - u_norm**2) * (1 - v_norm**2)
        if denominator <= 0:
            return float('inf')  # Handle edge cases
        return acosh(1 + numerator / denominator)

    def insert(self, key, point):
        """
        Insert a new point into the HDO.
        :param key: Unique identifier for the point.
        :param point: The point in Euclidean space.
        """
        embedded_point = self._hyperbolic_embedding(np.array(point))
        self.data_points[key] = embedded_point

    def delete(self, key):
        """
        Delete a point from the HDO.
        :param key: Unique identifier of the point to delete.
        """
        if key in self.data_points:
            del self.data_points[key]

    def query(self, point, k=1):
        """
        Query the k-nearest neighbors to a given point.
        :param point: The query point in Euclidean space.
        :param k: Number of nearest neighbors to retrieve.
        :return: List of k-nearest neighbors with their distances.
        """
        query_point = self._hyperbolic_embedding(np.array(point))
        distances = []
        for key, data_point in self.data_points.items():
            dist = self._poincare_distance(query_point, data_point)
            distances.append((key, dist))
        distances.sort(key=lambda x: x[1])  # Sort by distance
        return distances[:k]

    def visualize(self):
        """
        Visualize the points in the 2D Poincaré Disk (only if dimension is 2).
        """
        if self.dimension != 2:
            print("Visualization is only supported for 2D data.")
            return

        fig, ax = plt.subplots()
        disk = plt.Circle((0, 0), self.unit_disk_radius, color='lightgray', fill=False, linewidth=2)
        ax.add_artist(disk)

        for key, point in self.data_points.items():
            ax.plot(point[0], point[1], 'bo')
            ax.text(point[0], point[1], key, fontsize=8)

        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_aspect('equal', 'box')
        plt.title("Hyperbolic Data Orb Visualization (Poincaré Disk)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.show()

# Testing with Iris Dataset
if __name__ == "__main__":
    # Load the Iris dataset
    iris = load_iris()
    data = iris.data
    target = iris.target
    target_names = iris.target_names

    # Reduce dimensions to 2D using PCA for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # Initialize the HDO
    hdo = HyperbolicDataOrb(dimension=2)

    # Insert reduced data into HDO
    for i, point in enumerate(reduced_data):
        hdo.insert(f"iris_{i}", point)

    # Query the 3 nearest neighbors for a specific point
    sample_point = reduced_data[0]  # Use the first point in the dataset as a sample query
    neighbors = hdo.query(sample_point, k=3)
    print("Query Result for Iris Data (3 Nearest Neighbors):", neighbors)

    # Visualize the points in the Poincaré Disk
    hdo.visualize()

    # Visualize the PCA reduced dataset for comparison
    plt.figure()
    for i, target_name in enumerate(target_names):
        plt.scatter(
            reduced_data[target == i, 0],
            reduced_data[target == i, 1],
            label=target_name
        )
    plt.title("PCA Reduced Iris Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

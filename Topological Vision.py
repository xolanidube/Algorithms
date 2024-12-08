import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from ripser import ripser
from persim import plot_diagrams
import cv2
from PIL import Image
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologicalFeatureExtractor:
    def __init__(self, max_dimension=2, max_scale=2.0):
        self.max_dimension = max_dimension
        self.max_scale = max_scale
        
    def extract_features(self, image):
        """Extract topological features from image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Extract points of interest using SIFT
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        if descriptors is None or len(descriptors) < 3:
            return None, None
            
        # Normalize descriptors
        scaler = StandardScaler()
        normalized_desc = scaler.fit_transform(descriptors)
        
        # Compute persistent homology
        diagrams = ripser(normalized_desc, maxdim=self.max_dimension)['dgms']
        
        return diagrams, normalized_desc
    
    def compute_wasserstein_distance(self, dgm1, dgm2):
        """Compute Wasserstein distance between persistence diagrams."""
        return wasserstein_distance(dgm1, dgm2)

class InformationGeometricLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x):
        # Fisher-Rao metric inspired transformation
        proj = F.linear(x, self.weight, self.bias)
        return F.relu(proj) / (torch.norm(proj, dim=1, keepdim=True) + 1e-8)

class TopologicalVisionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.info_geom = InformationGeometricLayer(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.info_geom(x)
        return self.fc(x)

class TopologicalVision:
    def __init__(self):
        self.feature_extractor = TopologicalFeatureExtractor()
        self.model = None
        
    def preprocess_image(self, image_path):
        """Load and preprocess image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.resize(img, (224, 224))
    
    def extract_topological_features(self, image):
        """Extract topological features and return visualization data."""
        diagrams, descriptors = self.feature_extractor.extract_features(image)
        if diagrams is None:
            return None, None, None
            
        # Convert persistence diagrams to feature vectors
        feature_vectors = []
        for dim in range(len(diagrams)):
            if len(diagrams[dim]) > 0:
                # Extract birth-death pairs and compute statistical features
                birth_times = diagrams[dim][:, 0]
                death_times = diagrams[dim][:, 1]
                persistence = death_times - birth_times
                
                features = [
                    np.mean(persistence),
                    np.std(persistence),
                    np.max(persistence),
                    len(persistence)
                ]
                feature_vectors.extend(features)
                
        return diagrams, descriptors, np.array(feature_vectors)
    
    def visualize_persistence(self, diagrams):
        """Create persistence diagram visualization."""
        plt.figure(figsize=(10, 10))
        plot_diagrams(diagrams, show=False)
        plt.title("Persistence Diagrams")
        return plt

    def visualize_point_cloud(self, descriptors):
        """Create point cloud visualization."""
        plt.figure(figsize=(10, 10))
        if descriptors.shape[1] > 2:
            from sklearn.manifold import TSNE
            descriptors_2d = TSNE(n_components=2).fit_transform(descriptors)
        else:
            descriptors_2d = descriptors
            
        plt.scatter(descriptors_2d[:, 0], descriptors_2d[:, 1], alpha=0.6)
        plt.title("Feature Space Point Cloud")
        return plt

    def analyze_image(self, image_path):
        """Perform complete topological analysis of an image."""
        # Load and preprocess image
        image = self.preprocess_image(image_path)
        
        # Extract features
        diagrams, descriptors, feature_vector = self.extract_topological_features(image)
        if diagrams is None:
            return None
            
        # Create visualizations
        persistence_viz = self.visualize_persistence(diagrams)
        point_cloud_viz = self.visualize_point_cloud(descriptors)
        
        return {
            'diagrams': diagrams,
            'descriptors': descriptors,
            'feature_vector': feature_vector,
            'persistence_viz': persistence_viz,
            'point_cloud_viz': point_cloud_viz
        }

def main():
    # Example usage
    topo_vision = TopologicalVision()
    
    # Analyze sample image
    sample_image_path = "C:/Users/Xolan/Downloads/GdbEMFyXsAAV0ZZ.jpg"  # Replace with actual image path
    results = topo_vision.analyze_image(sample_image_path)
    
    if results:
        # Display visualizations
        results['persistence_viz'].show()
        results['point_cloud_viz'].show()
        
        # Print feature vector
        print("Topological feature vector:", results['feature_vector'])
        
        # Save visualizations
        results['persistence_viz'].savefig('persistence_diagram.png')
        results['point_cloud_viz'].savefig('point_cloud.png')

if __name__ == "__main__":
    main()
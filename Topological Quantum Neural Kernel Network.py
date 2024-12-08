import numpy as np
from typing import Tuple, List
from sklearn.metrics import accuracy_score
import math

# Mock Persistent Homology Extraction
# In practice, use: from gudhi import RipsComplex, ...
def persistent_homology_features(X: np.ndarray) -> np.ndarray:
    # X: shape (N, d) point cloud
    # Mock: return some stable topological signature vector
    # Real TDA: compute persistence diagram and return summary features
    # For demonstration, we pretend topological complexity is the variance of distances.
    dist_matrix = np.sqrt(((X[:, None, :] - X[None, :, :])**2).sum(axis=2))
    # Topological feature: average pairwise distance, max distance, variance
    avg_dist = np.mean(dist_matrix)
    max_dist = np.max(dist_matrix)
    var_dist = np.var(dist_matrix)
    return np.array([avg_dist, max_dist, var_dist])

# Mock Quantum Kernel
# A parameterized feature map: phi(x) = exp(i * theta * x) (just a toy model)
def quantum_kernel_feature_map(x: np.ndarray, theta: float=0.5) -> np.ndarray:
    # Map input to complex exponential features
    # Real quantum kernels would use parameterized circuits
    return np.concatenate([np.sin(theta * x), np.cos(theta * x)], axis=-1)

def quantum_kernel(X: np.ndarray, Xp: np.ndarray, theta: float=0.5) -> np.ndarray:
    # Compute a kernel matrix
    phi_X = quantum_kernel_feature_map(X, theta)
    phi_Xp = quantum_kernel_feature_map(Xp, theta)
    # Inner product (simulate a kernel)
    K = np.dot(phi_X, phi_Xp.T)
    # Return the element-wise squared magnitude
    return K**2

# Neural Readout Layer (simple linear + softmax)
class NeuralLayer:
    def __init__(self, input_dim: int, output_dim: int):
        self.W = 0.01 * np.random.randn(input_dim, output_dim)
        self.b = np.zeros((output_dim,))
    
    def forward(self, X):
        return X @ self.W + self.b
    
    def backward(self, grad_out, X, lr=0.01):
        grad_W = X.T @ grad_out
        grad_b = grad_out.sum(axis=0)
        self.W -= lr * grad_W
        self.b -= lr * grad_b

# Simple training loop
def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / expZ.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, y):
    N = len(y)
    correct_logprobs = -np.log(probs[range(N), y])
    return correct_logprobs.mean()

def grad_wrt_logits(probs, y):
    N = len(y)
    probs[range(N), y] -= 1
    return probs / N

# Construct synthetic data: classify shapes
def generate_shape_data(num_samples=100, shape_type='sphere', noise=0.01):
    # Sphere: points on a sphere surface in 3D
    # Torus: points on a torus
    # Just mock different distributions
    if shape_type == 'sphere':
        phi = np.random.uniform(0, 2*math.pi, size=num_samples)
        costheta = np.random.uniform(-1, 1, size=num_samples)
        theta = np.arccos(costheta)
        r = 1.0
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        X = np.stack([x,y,z], axis=1) + noise*np.random.randn(num_samples,3)
        y_label = 0
    elif shape_type == 'torus':
        R = 1.0
        r = 0.3
        theta = np.random.uniform(0,2*math.pi,num_samples)
        phi = np.random.uniform(0,2*math.pi,num_samples)
        x = (R + r*np.cos(phi))*np.cos(theta)
        y = (R + r*np.cos(phi))*np.sin(theta)
        z = r*np.sin(phi)
        X = np.stack([x,y,z], axis=1) + noise*np.random.randn(num_samples,3)
        y_label = 1
    else:
        # Another shape: figure-eight (two spheres)
        half = num_samples//2
        X_sphere_1, _ = generate_shape_data(half, 'sphere', noise)
        X_sphere_2, _ = generate_shape_data(half, 'sphere', noise)
        X_sphere_2[:,0] += 2.0  # shift
        X = np.vstack([X_sphere_1, X_sphere_2])
        y_label = 2
    return X, np.full(num_samples, y_label)

# Combine topological + quantum kernel + neural layer
def extract_features(X):
    # Extract topological features
    topo_feat = np.array([persistent_homology_features(xi[None,:]) for xi in X]) 
    # topo_feat shape: (N, 3)
    # Combine original input + topo features for kernel mapping
    combined = np.hstack([X, topo_feat])
    # Quantum kernel feature map
    phi_X = quantum_kernel_feature_map(combined)
    return phi_X

# Training demonstration
def train_model(X_train, y_train, X_test, y_test, epochs=50, lr=0.01):
    input_dim = extract_features(X_train).shape[1]
    output_dim = len(np.unique(y_train))
    layer = NeuralLayer(input_dim, output_dim)

    for epoch in range(epochs):
        # Forward
        phi_train = extract_features(X_train)
        logits = layer.forward(phi_train)
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, y_train)

        # Backward
        grad = grad_wrt_logits(probs, y_train)
        layer.backward(grad, phi_train, lr)

        if epoch % 10 == 0:
            phi_test = extract_features(X_test)
            test_probs = softmax(layer.forward(phi_test))
            test_pred = np.argmax(test_probs, axis=1)
            acc = accuracy_score(y_test, test_pred)
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

    return layer

# Example usage:
X_sphere, y_sphere = generate_shape_data(200, 'sphere')
X_torus, y_torus = generate_shape_data(200, 'torus')
X_train = np.vstack([X_sphere[:100], X_torus[:100]])
y_train = np.concatenate([y_sphere[:100], y_torus[:100]])
X_test = np.vstack([X_sphere[100:], X_torus[100:]])
y_test = np.concatenate([y_sphere[100:], y_torus[100:]])

model = train_model(X_train, y_train, X_test, y_test)

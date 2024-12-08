
"""
Algebraic Global Convergence Neural Training (AGCNT) - Extended Version
-----------------------------------------------------------------------
This script demonstrates a hypothetical polynomial-time global training approach
for a tiny neural network with a polynomial activation function, now applied to a
synthetically generated regression dataset. This simulates a "real-world" scenario
in a controlled manner.

Key Steps:
1. Generate a small regression dataset (using sklearn) with one input feature.
2. Split into train and test sets.
3. Define a single-hidden-neuron polynomial network (phi(x) = x^2).
4. Formulate the training loss as a polynomial in parameters (W, B, V).
5. Symbolically compute gradients and solve the polynomial system for global minima.
6. Verify the solution by brute force over a parameter grid on the training set.
7. Evaluate the selected solution on the test set.
8. Optional visualization of a slice of the loss landscape.

Author: Algorithm Innovation Architect
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

############################################
# Data Generation
############################################

def generate_data(n_samples=5, noise=0.1, random_state=42):
    """
    Generate a small regression dataset with one input feature.
    We keep the dataset very small and simple.
    """
    X, Y = make_regression(n_samples=n_samples, n_features=1, noise=noise, random_state=random_state)
    # X shape: (n_samples,1)
    # Y shape: (n_samples,)
    X = X.flatten()
    # Normalize X and Y to avoid large values
    X = X / np.max(np.abs(X))  
    Y = Y / np.max(np.abs(Y))
    return X, Y

############################################
# Network and Loss Definition
############################################

def network_output(x, w, b, v):
    # Single hidden neuron with polynomial activation phi(x)=x^2
    # out = v * ((w*x + b)^2)
    return v * ((w*x + b)**2)

def loss_function(W, B, V, X, Y):
    # Mean squared error over the training set
    m = len(X)
    s = 0
    for i in range(m):
        pred = network_output(X[i], W, B, V)
        s += (pred - Y[i])**2
    return s / m

############################################
# Symbolic Setup
############################################

def symbolic_setup(X, Y):
    W, B, V = sp.symbols('W B V', real=True)
    m = len(X)
    L_expr = 0
    for i in range(m):
        L_expr += (V*((W*X[i]+B)**2) - Y[i])**2
    L_expr = L_expr/m
    
    # Compute gradients
    dLdW = sp.diff(L_expr, W)
    dLdB = sp.diff(L_expr, B)
    dLdV = sp.diff(L_expr, V)
    
    # Solve system of equations
    solutions = sp.solve([dLdW, dLdB, dLdV], [W, B, V], dict=True)
    return L_expr, solutions, (W,B,V)

############################################
# Brute Force Search
############################################

def brute_force_search(X, Y, param_range=(-2,2), steps=50):
    Wvals = np.linspace(param_range[0], param_range[1], steps)
    Bvals = np.linspace(param_range[0], param_range[1], steps)
    Vvals = np.linspace(param_range[0], param_range[1], steps)
    best_loss = float('inf')
    best_params = None
    for w in Wvals:
        for b in Bvals:
            for v in Vvals:
                L = loss_function(w,b,v,X,Y)
                if L < best_loss:
                    best_loss = L
                    best_params = (w,b,v)
    return best_loss, best_params

############################################
# Visualization (Optional)
############################################

def visualize_loss_landscape(X, Y, fixed_v=1.0, param_range=(-2,2), steps=50):
    Wvals = np.linspace(param_range[0], param_range[1], steps)
    Bvals = np.linspace(param_range[0], param_range[1], steps)
    L_grid = np.zeros((steps, steps))
    for i, w in enumerate(Wvals):
        for j, b in enumerate(Bvals):
            L_grid[i,j] = loss_function(w,b,fixed_v,X,Y)
    Wg, Bg = np.meshgrid(Wvals, Bvals)
    plt.figure(figsize=(6,5))
    cp = plt.contourf(Wg, Bg, L_grid.T, levels=20, cmap='viridis')
    plt.colorbar(cp)
    plt.title("Loss landscape slice at V=1.0 (Train Set)")
    plt.xlabel("W")
    plt.ylabel("B")
    plt.show()

############################################
# Main Execution
############################################

def main():
    # Generate data and split into train/test
    X, Y = generate_data(n_samples=8, noise=0.05, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
    
    # Symbolic setup on training data
    L_expr, solutions, (W,B,V) = symbolic_setup(X_train, Y_train)
    
    print("Symbolic loss:", L_expr)
    print("Found symbolic solutions:")
    for sol in solutions:
        print(sol)
    
    # Brute force verification on training set
    bf_loss, bf_params = brute_force_search(X_train,Y_train,(-2,2),30)
    print("Brute force global min approx on TRAIN set:", bf_loss, "at params:", bf_params)
    
    # Evaluate symbolic solutions on training set
    best_symbolic_loss = float('inf')
    best_symbolic_params = None
    
    for sol in solutions:
        # Check if W,B,V are in the solution
        if (W in sol) and (B in sol) and (V in sol):
            w_sol = sol[W]
            b_sol = sol[B]
            v_sol = sol[V]
            # Evaluate loss on training set
            L_sol_train = loss_function(w_sol, b_sol, v_sol, X_train, Y_train)
            print(f"Solution on TRAIN set L={L_sol_train}, Params={(w_sol,b_sol,v_sol)}")
            if L_sol_train < best_symbolic_loss:
                best_symbolic_loss = L_sol_train
                best_symbolic_params = (w_sol, b_sol, v_sol)
        else:
            # If solution does not contain all variables, skip
            print("Incomplete solution keys, skipping this solution.")
    
    # Compare best symbolic solution to brute force
    print("Best symbolic solution on TRAIN set:", best_symbolic_loss, "Params:", best_symbolic_params)
    print("Brute force approx best on TRAIN set:", bf_loss, "Params:", bf_params)
    
    # Evaluate the best symbolic solution on TEST set
    if best_symbolic_params is not None:
        w_sol, b_sol, v_sol = best_symbolic_params
        L_test = loss_function(w_sol, b_sol, v_sol, X_test, Y_test)
        print("Test set loss using best symbolic solution:", L_test)
    else:
        print("No complete symbolic solution found to evaluate on TEST set.")
    
    # Optional: Visualize a slice of the loss landscape
    # Note: This is just for intuition and is not strictly necessary.
    visualize_loss_landscape(X_train, Y_train, fixed_v=1.0, param_range=(-2,2), steps=30)

if __name__ == "__main__":
    main()

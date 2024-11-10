import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from scipy.optimize import linprog
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def optimize_subproblem(args):
    cluster_tasks, resources = args
    num_tasks = len(cluster_tasks)
    num_resources = len(resources)
    
    if num_tasks == 0:
        return []
    
    # Objective: Maximize resource utilization
    c = -np.ones(num_tasks * num_resources)  # Negative for maximization
    
    # Constraints
    A_eq = []
    b_eq = []
    
    # Each task is assigned to exactly one resource
    for i in range(num_tasks):
        constraint = np.zeros(num_tasks * num_resources)
        for j in range(num_resources):
            constraint[i * num_resources + j] = 1
        A_eq.append(constraint)
        b_eq.append(1)
    
    # Resource capacity constraints
    A_ub = []
    b_ub = []
    for j in range(num_resources):
        # CPU capacity
        constraint_cpu = np.zeros(num_tasks * num_resources)
        for i in range(num_tasks):
            constraint_cpu[i * num_resources + j] = cluster_tasks.iloc[i]['cpu']
        A_ub.append(constraint_cpu)
        b_ub.append(resources.iloc[j]['cpu_capacity'])
        
        # Memory capacity
        constraint_mem = np.zeros(num_tasks * num_resources)
        for i in range(num_tasks):
            constraint_mem[i * num_resources + j] = cluster_tasks.iloc[i]['memory']
        A_ub.append(constraint_mem)
        b_ub.append(resources.iloc[j]['memory_capacity'])
    
    # Bounds: Assignment variables are binary (0 or 1)
    bounds = [(0, 1) for _ in range(num_tasks * num_resources)]
    
    # Solve the linear programming problem
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs-ipm'
    )
    
    # Process the result
    if result.success:
        x = result.x
        assignments = []
        for i in range(num_tasks):
            for j in range(num_resources):
                if x[i * num_resources + j] > 0.5:
                    assignments.append({
                        'task_id': cluster_tasks.iloc[i]['task_id'],
                        'resource_id': resources.iloc[j]['resource_id']
                    })
        return assignments
    else:
        return []

def main():
    # Number of tasks and resources
    num_tasks = 1000
    num_resources = 100

    # Generate task requirements and deadlines
    tasks = pd.DataFrame({
        'task_id': range(num_tasks),
        'cpu': np.random.randint(1, 10, size=num_tasks),
        'memory': np.random.randint(1, 16, size=num_tasks),
        'deadline': np.random.randint(1, 100, size=num_tasks),
        'priority': np.random.choice(['Low', 'Medium', 'High'], size=num_tasks)
    })

    # Generate resource capacities
    resources = pd.DataFrame({
        'resource_id': range(num_resources),
        'cpu_capacity': np.random.randint(50, 100, size=num_resources),
        'memory_capacity': np.random.randint(200, 500, size=num_resources)
    })

    # Encode priority levels
    tasks['priority_encoded'] = tasks['priority'].map({'Low': 0, 'Medium': 1, 'High': 2})

    # Feature matrix
    X_tasks = tasks[['cpu', 'memory', 'deadline', 'priority_encoded']].values

    # Define the autoencoder model
    input_dim = X_tasks.shape[1]
    encoding_dim = 2  # Dimensionality of the encoding space

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the model
    autoencoder.fit(X_tasks, X_tasks, epochs=50, batch_size=32, shuffle=True, verbose=0)

    # Get the encoded representations
    task_embeddings = encoder.predict(X_tasks)
    tasks['cluster_x'] = task_embeddings[:, 0]
    tasks['cluster_y'] = task_embeddings[:, 1]

    # Clustering Tasks
    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters)
    tasks['cluster'] = kmeans.fit_predict(task_embeddings)

    # Create a list of task clusters
    task_clusters = []
    for cluster_id in range(num_clusters):
        cluster_tasks = tasks[tasks['cluster'] == cluster_id]
        task_clusters.append(cluster_tasks)

    # Prepare arguments for each subproblem
    args_list = [(cluster_tasks, resources) for cluster_tasks in task_clusters]

    # Use multiprocessing Pool
    with Pool(processes=num_clusters) as pool:
        results = pool.map(optimize_subproblem, args_list)

    # Solution Recombination
    assignments = [assignment for sublist in results for assignment in sublist]
    assignments_df = pd.DataFrame(assignments)
    assignments_df = assignments_df.drop_duplicates(subset=['task_id'], keep='first')

    # Evaluate Resource Utilization
    utilization = resources.copy()
    utilization['cpu_used'] = 0
    utilization['memory_used'] = 0

    for idx, assignment in assignments_df.iterrows():
        task = tasks[tasks['task_id'] == assignment['task_id']].iloc[0]
        resource_idx = utilization[utilization['resource_id'] == assignment['resource_id']].index[0]
        utilization.at[resource_idx, 'cpu_used'] += task['cpu']
        utilization.at[resource_idx, 'memory_used'] += task['memory']

    utilization['cpu_utilization'] = utilization['cpu_used'] / utilization['cpu_capacity']
    utilization['memory_utilization'] = utilization['memory_used'] / utilization['memory_capacity']

    # Feedback Loop
    cpu_threshold = 0.7
    memory_threshold = 0.7

    underutilized_resources = utilization[
        (utilization['cpu_utilization'] < cpu_threshold) &
        (utilization['memory_utilization'] < memory_threshold)
    ]

    if not underutilized_resources.empty:
        print("Underutilized resources detected. Adjusting partitions...")
        # Adjust number of clusters or re-cluster tasks
        # For simplicity, we won't implement the adjustment here
    else:
        print("Resource utilization is satisfactory.")

if __name__ == "__main__":
    main()

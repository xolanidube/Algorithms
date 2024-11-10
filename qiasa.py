import numpy as np
import matplotlib.pyplot as plt

# Load real-world city data (coordinates)
cities = {
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Chicago': (41.8781, -87.6298),
    'Houston': (29.7604, -95.3698),
    'Phoenix': (33.4484, -112.0740),
    'Philadelphia': (39.9526, -75.1652),
    'San Antonio': (29.4241, -98.4936),
    'San Diego': (32.7157, -117.1611),
    'Dallas': (32.7767, -96.7970),
    'San Jose': (37.3382, -121.8863)
}

city_names = list(cities.keys())
city_coords = np.array(list(cities.values()))
n_cities = len(cities)

# Compute the distance matrix
def compute_distance_matrix(coords):
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix

distance_matrix = compute_distance_matrix(city_coords)

# QIASA parameters
population_size = 100
max_iterations = 1000
amplitude_decay = 0.9
amplitude_gain = 1.1

# Initialize probability amplitudes
prob_amplitudes = np.ones(population_size) / population_size

# Initialize population
def initialize_population(size, n_cities):
    population = []
    for _ in range(size):
        tour = np.random.permutation(n_cities)
        population.append(tour)
    return population

population = initialize_population(population_size, n_cities)

# Fitness function
def fitness(tour):
    total_distance = 0.0
    for i in range(len(tour)):
        from_city = tour[i]
        to_city = tour[(i + 1) % len(tour)]  # Wrap around to the starting city
        total_distance += distance_matrix[from_city][to_city]
    return total_distance

# Adaptive interference
def adaptive_interference(fitness_values, prob_amplitudes):
    avg_fitness = np.mean(fitness_values)
    new_amplitudes = np.copy(prob_amplitudes)
    for i in range(len(fitness_values)):
        if fitness_values[i] < avg_fitness:
            # Constructive interference
            new_amplitudes[i] *= amplitude_gain
        else:
            # Destructive interference
            new_amplitudes[i] *= amplitude_decay
    # Normalize amplitudes
    new_amplitudes /= np.sum(new_amplitudes)
    return new_amplitudes

# Main optimization loop
best_tour = None
best_distance = float('inf')
fitness_history = []

for iteration in range(max_iterations):
    # Evaluate fitness for the current population
    fitness_values = np.array([fitness(tour) for tour in population])
    
    # Update the best tour found
    min_idx = np.argmin(fitness_values)
    if fitness_values[min_idx] < best_distance:
        best_distance = fitness_values[min_idx]
        best_tour = population[min_idx]
    
    fitness_history.append(best_distance)
    
    # Apply adaptive interference
    prob_amplitudes = adaptive_interference(fitness_values, prob_amplitudes)
    
    # Generate new population based on updated amplitudes
    indices = np.arange(population_size)
    selected_indices = np.random.choice(indices, size=population_size, p=prob_amplitudes)
    new_population = []
    for idx in selected_indices:
        # Copy the selected tour
        tour = np.copy(population[idx])
        # Mutation: Swap two cities with a small probability
        if np.random.rand() < 0.1:
            i, j = np.random.choice(n_cities, 2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]
        new_population.append(tour)
    
    # Update population
    population = new_population
    
    # Print progress every 100 iterations
    if (iteration + 1) % 100 == 0:
        print(f"Iteration {iteration+1}, Best Distance: {best_distance:.2f}")

# Plot the fitness history
plt.plot(fitness_history)
plt.xlabel('Iteration')
plt.ylabel('Best Distance')
plt.title('QIASA Optimization Progress')
plt.show()

# Display the best tour
print("Best tour found:")
for idx in best_tour:
    print(city_names[idx], end=' -> ')
print(city_names[best_tour[0]])  # Return to starting city
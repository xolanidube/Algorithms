import numpy as np
import random
from typing import Callable, List, Tuple

# Optional imports for multiprocessing and plotting
# import multiprocessing as mp
# import matplotlib.pyplot as plt


class QubitIndividual:
    def __init__(self, num_bits: int):
        # Initialize probability amplitudes uniformly
        self.num_bits = num_bits
        self.alpha = np.full(num_bits, 1/np.sqrt(2))
        self.beta = np.full(num_bits, 1/np.sqrt(2))
        self.theta = np.zeros(num_bits)  # Quantum angles
        self.solution = None
        self.fitness = None

    def measure(self):
        # Collapse Q-bit to classical bitstring
        probabilities = self.alpha ** 2
        self.solution = np.random.rand(self.num_bits) > probabilities
        return self.solution.astype(int)

    def update(self, best_solution: np.ndarray):
        # Adaptive quantum rotation gates
        for i in range(self.num_bits):
            if self.solution[i] != best_solution[i]:
                delta_theta = self.get_rotation_angle(self.solution[i], best_solution[i])
                self.theta[i] += delta_theta
                self.apply_rotation(i)

    def apply_rotation(self, index: int):
        # Update probability amplitudes using rotation angles
        cos_theta = np.cos(self.theta[index])
        sin_theta = np.sin(self.theta[index])
        alpha = self.alpha[index]
        beta = self.beta[index]
        self.alpha[index] = alpha * cos_theta - beta * sin_theta
        self.beta[index] = alpha * sin_theta + beta * cos_theta
        # Normalize amplitudes
        norm = np.sqrt(self.alpha[index] ** 2 + self.beta[index] ** 2)
        self.alpha[index] /= norm
        self.beta[index] /= norm

    def get_rotation_angle(self, bit, best_bit):
        # Define adaptive rotation angle
        if bit == 0 and best_bit == 1:
            return np.pi / 180  # Rotate by 1 degree
        elif bit == 1 and best_bit == 0:
            return -np.pi / 180
        else:
            return 0.0


class QIAMO:
    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], float],
        num_bits: int,
        population_size: int = 50,
        max_generations: int = 100,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.01,
    ):
        self.fitness_func = fitness_func
        self.num_bits = num_bits
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population: List[QubitIndividual] = []
        self.best_solution: Tuple[np.ndarray, float] = (None, -np.inf)

    def initialize_population(self):
        self.population = [QubitIndividual(self.num_bits) for _ in range(self.population_size)]

    def evaluate_population(self):
        for individual in self.population:
            solution = individual.measure()
            fitness = self.fitness_func(solution)
            individual.fitness = fitness
            if fitness > self.best_solution[1]:
                self.best_solution = (solution.copy(), fitness)

    def evolve(self):
        self.initialize_population()
        for generation in range(self.max_generations):
            self.evaluate_population()
            new_population = []
            for individual in self.population:
                # Apply adaptive quantum rotation
                individual.update(self.best_solution[0])
                # Apply hybrid operators
                if random.random() < self.crossover_rate:
                    partner = random.choice(self.population)
                    offspring = self.crossover(individual, partner)
                    new_population.append(offspring)
                else:
                    new_population.append(individual)
            # Apply mutation
            for individual in new_population:
                self.mutate(individual)
            self.population = new_population
            print(f"Generation {generation+1}: Best Fitness = {self.best_solution[1]}")
        return self.best_solution

    def crossover(self, parent1: QubitIndividual, parent2: QubitIndividual):
        crossover_point = random.randint(1, self.num_bits - 1)
        child = QubitIndividual(self.num_bits)
        child.alpha = np.concatenate((parent1.alpha[:crossover_point], parent2.alpha[crossover_point:]))
        child.beta = np.concatenate((parent1.beta[:crossover_point], parent2.beta[crossover_point:]))
        return child

    def mutate(self, individual: QubitIndividual):
        for i in range(self.num_bits):
            if random.random() < self.mutation_rate:
                # Swap alpha and beta to simulate mutation
                individual.alpha[i], individual.beta[i] = individual.beta[i], individual.alpha[i]
                # Normalize amplitudes
                norm = np.sqrt(individual.alpha[i] ** 2 + individual.beta[i] ** 2)
                individual.alpha[i] /= norm
                individual.beta[i] /= norm


def fitness_function(solution: np.ndarray) -> float:
    # Sample fitness function: OneMax Problem (maximize number of ones)
    return np.sum(solution)

def deceptive_trap(solution: np.ndarray) -> float:
    k = 5  # Deceptive block size
    total = 0
    for i in range(0, len(solution), k):
        block = solution[i:i+k]
        ones = np.sum(block)
        if ones == k:
            total += k
        else:
            total += k - ones - 1
    return total


if __name__ == "__main__":
    num_bits = 100  # Dimension of the problem
    qiamo = QIAMO(
        fitness_func=fitness_function,
        num_bits=num_bits,
        population_size=50,
        max_generations=100,
        crossover_rate=0.7,
        mutation_rate=0.01,
    )
    best_solution, best_fitness = qiamo.evolve()
    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}")

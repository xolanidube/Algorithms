import unittest
import numpy as np
from typing import List, Dict, Any
import time
import qiasa_advanced 
from qiasa_advanced import load_city_data, QIASAParams, TSP, load_knapsack_data, KnapsackProblem, QIASA, load_job_data, JobScheduling

class TestQIASA(unittest.TestCase):
    def setUp(self):
        self.params = QIASAParams(
            population_size=50,
            max_iterations=100,
            amplitude_decay=0.9,
            amplitude_gain=1.1,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
    def test_tsp_optimization(self):
        # Test TSP with different sizes
        sizes = [10, 20, 30]
        results = []
        
        for size in sizes:
            cities = n_cities=size
            tsp = TSP(cities)
            optimizer = QIASA(tsp, self.params)
            
            start_time = time.time()
            solution, fitness, history = optimizer.optimize()
            end_time = time.time()
            
            results.append({
                'size': size,
                'fitness': fitness,
                'time': end_time - start_time,
                'convergence_rate': (history[0] - history[-1]) / len(history)
            })
            
            # Verify solution validity
            self.assertTrue(tsp.validate(solution))
            
        return results
        
    def test_knapsack_optimization(self):
        # Test Knapsack with different sizes
        sizes = [20, 40, 60]
        results = []
        
        for size in sizes:
            items, capacity = load_knapsack_data(n_items=size)
            knapsack = KnapsackProblem(items, capacity)
            optimizer = QIASA(knapsack, self.params)
            
            start_time = time.time()
            solution, fitness, history = optimizer.optimize()
            end_time = time.time()
            
            results.append({
                'size': size,
                'fitness': fitness,
                'time': end_time - start_time,
                'convergence_rate': (history[-1] - history[0]) / len(history)
            })
            
            # Verify solution validity
            self.assertTrue(knapsack.validate(solution))
            
        return results
        
    def test_job_scheduling_optimization(self):
        # Test Job Scheduling with different sizes
        configs = [
            {'jobs': 15, 'machines': 3},
            {'jobs': 30, 'machines': 5},
            {'jobs': 45, 'machines': 7}
        ]
        results = []
        
        for config in configs:
            jobs, n_machines = load_job_data(
                n_jobs=config['jobs'], 
                n_machines=config['machines']
            )
            scheduling = JobScheduling(jobs, n_machines)
            optimizer = QIASA(scheduling, self.params)
            
            start_time = time.time()
            solution, fitness, history = optimizer.optimize()
            end_time = time.time()
            
            results.append({
                'config': config,
                'fitness': -fitness,  # Convert back to makespan
                'time': end_time - start_time,
                'convergence_rate': (history[0] - history[-1]) / len(history)
            })
            
            # Verify solution validity
            self.assertTrue(scheduling.validate(solution))
            
        return results

def run_benchmark():
    """
    Run comprehensive benchmarks and generate performance report
    """
    test_suite = TestQIASA()
    
    # Run all tests
    tsp_results = test_suite.test_tsp_optimization()
    knapsack_results = test_suite.test_knapsack_optimization()
    scheduling_results = test_suite.test_job_scheduling_optimization()
    
    # Print results
    print("\nQIASA Performance Benchmark Results")
    print("==================================")
    
    print("\nTraveling Salesman Problem:")
    print("--------------------------")
    for result in tsp_results:
        print(f"Size: {result['size']} cities")
        print(f"Best Distance: {result['fitness']:.2f}")
        print(f"Time: {result['time']:.2f} seconds")
        print(f"Convergence Rate: {result['convergence_rate']:.4f}")
        print()
        
    print("\nKnapsack Problem:")
    print("----------------")
    for result in knapsack_results:
        print(f"Size: {result['size']} items")
        print(f"Best Value: {result['fitness']:.2f}")
        print(f"Time: {result['time']:.2f} seconds")
        print(f"Convergence Rate: {result['convergence_rate']:.4f}")
        print()
        
    print("\nJob Scheduling Problem:")
    print("----------------------")
    for result in scheduling_results:
        print(f"Config: {result['config']['jobs']} jobs, {result['config']['machines']} machines")
        print(f"Best Makespan: {result['fitness']:.2f}")
        print(f"Time: {result['time']:.2f} seconds")
        print(f"Convergence Rate: {result['convergence_rate']:.4f}")
        print()

if __name__ == "__main__":
    run_benchmark()
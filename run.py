import numpy as np
import os  # For getting process ID
import time  # To track execution time
from concurrent.futures import ProcessPoolExecutor
from SSA import SSA
from MFO import MFO
from GEA import GEA
from functions import selectFunction
import csv

# Benchmark function indices and algorithms
benchmark_functions = [0, 1, 2, 4, 7, 8, 9, 10, 13]  # Indices of functions in `selectFunction`
algorithms = {
    "SSA": SSA,
    "MFO": MFO,
    "GEA": GEA,
}

# Parameters for optimization
lb = -100         # Lower bound of search space
ub = 100          # Upper bound of search space
dim = 30          # Dimensionality of the problem
N = 50            # Population size
Max_iteration = 1000  # Maximum number of iterations

# Wrapper function to run a single algorithm on a single benchmark function
def run_algorithm(algorithm_name, algorithm, objf_index):
    try:
        # Select the objective function
        objf = selectFunction(objf_index)

        # Run the algorithm
        result = algorithm(
            objf=objf,
            lb=lb,
            ub=ub,
            dim=dim,
            N=N,
            Max_iteration=Max_iteration,
        )

        # Return results along with process ID
        return {
            "algorithm": algorithm_name,
            "benchmark": objf.__name__,
            "best_fitness": result.convergence[-1],
            "execution_time": result.executionTime,
            "pid": os.getpid(),  # Process ID
        }
    except Exception as e:
        return {
            "algorithm": algorithm_name,
            "benchmark": selectFunction(objf_index).__name__,
            "error": str(e),
            "pid": os.getpid(),  # Process ID
        }

# Run all algorithms on all benchmark functions in parallel
def main():
    start_time = time.time()  # Start time
    
    tasks = []
    with ProcessPoolExecutor() as executor:
        for objf_index in benchmark_functions:
            for algorithm_name, algorithm in algorithms.items():
                tasks.append(executor.submit(run_algorithm, algorithm_name, algorithm, objf_index))

        # Gather results
        results = [task.result() for task in tasks]

    # Print results
    for result in results:
        if "error" in result:
            print(
                f"Error in {result['algorithm']} on {result['benchmark']} "
                f"by Process {result['pid']}: {result['error']}"
            )
        else:
            print(
                f"{result['algorithm']} on {result['benchmark']} by Process {result['pid']}: "
                f"Best Fitness = {result['best_fitness']}"
            )

    # Calculate total time taken
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

    # Add the total time to the CSV file
    with open('optimization_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Total Time', f'{total_time:.2f} seconds'])

if __name__ == "__main__":
    main()

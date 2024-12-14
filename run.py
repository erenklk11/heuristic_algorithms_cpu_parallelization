import csv
import numpy as np
import os  # For getting process ID
from concurrent.futures import ProcessPoolExecutor
from SSA import SSA
from MFO import MFO
from GEA import GEA
from functions import selectFunction

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
    tasks = []
    with ProcessPoolExecutor() as executor:
        for objf_index in benchmark_functions:
            for algorithm_name, algorithm in algorithms.items():
                tasks.append(executor.submit(run_algorithm, algorithm_name, algorithm, objf_index))

        # Gather results
        results = [task.result() for task in tasks]

    # Write results to CSV
    csv_filename = "optimization_results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Algorithm", "Benchmark", "Best Fitness", "Process ID", "Error"])
        # Write rows
        for result in results:
            if "error" in result:
                writer.writerow([result['algorithm'], result['benchmark'], None, result['pid'], None, result['error']])
                print(
                    f"Error in {result['algorithm']} on {result['benchmark']} "
                    f"by Process {result['pid']}: {result['error']}"
                )
            else:
                writer.writerow([result['algorithm'], result['benchmark'], result['best_fitness'], result['pid'], None])
                print(
                    f"{result['algorithm']} on {result['benchmark']} by Process {result['pid']}: "
                    f"Best Fitness = {result['best_fitness']}"
                )

    print(f"Results have been written to {csv_filename}")

if __name__ == "__main__":
    main()

import time
import csv
import os  # For getting process ID
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
N = 5000          # Population size
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
            "execution_time": 0,
            "pid": os.getpid(),  # Process ID
        }

# Main function
def main():
    # Start timing the program
    start_time = time.time()

    # List to store results
    results = []

    # List to store unique PIDs
    unique_pids = []

    # Run tasks sequentially
    for objf_index in benchmark_functions:
        for algorithm_name, algorithm in algorithms.items():
            result = run_algorithm(algorithm_name, algorithm, objf_index)
            results.append(result)
            unique_pids.append(result["pid"])

    # Define CSV file name
    csv_file = "optimization_results.csv"

    # Write results to the CSV
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Benchmark", "Best Fitness", "Execution Time (s)", "PID"])
        for result in results:
            if "error" in result:
                writer.writerow(
                    [result["algorithm"], result["benchmark"], "Error", result["execution_time"], result["pid"]]
                )
            else:
                writer.writerow(
                    [result["algorithm"], result["benchmark"], result["best_fitness"], result["execution_time"], result["pid"]]
                )

    # Calculate and print total time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal Program Execution Time: {total_time:.2f} seconds")

    # Append total time and number of CPU cores to the CSV
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Total Time (s):", f"{total_time:.2f}"])

        # Count and write the number of CPU processes used
        unique_pid_count = len(set(unique_pids))
        writer.writerow(["Number of CPU Processes Used:", unique_pid_count])

    # Display the number of unique processes in the console
    print(f"\nNumber of CPU processes used: {unique_pid_count}")


if __name__ == "__main__":
    main()

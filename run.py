import time
import csv
import os  # For getting process ID
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from SSA import SSA
from MFO import MFO
from GEA import GEA
from functions import selectFunction

# Benchmark function indices and algorithms
benchmark_functions = [0, 1, 2, 4, 7, 8, 9, 10, 13]  # Indices of functions in selectFunction
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
def run_algorithm(algorithm_name, algorithm, objf_index, unique_pids):
    try:
        # Start timing for this specific task
        start_time = time.time()

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

        # End timing for this specific task
        end_time = time.time()
        task_time = end_time - start_time

        # Add the current process ID to the unique list
        unique_pids.append(os.getpid())

        # Return results along with process ID and task time
        return {
            "algorithm": algorithm_name,
            "benchmark": objf.__name__,
            "best_fitness": result.convergence[-1],
            "execution_time": task_time,
            "pid": os.getpid(),  # Process ID
        }
    except Exception as e:
        unique_pids.append(os.getpid())
        return {
            "algorithm": algorithm_name,
            "benchmark": selectFunction(objf_index).__name__,
            "error": str(e),
            "execution_time": None,
            "pid": os.getpid(),  # Process ID
        }

# Main function
def main():
    # Start timing the program
    start_time = time.time()

    # Initialize multiprocessing manager for unique PIDs
    manager = Manager()
    unique_pids = manager.list()

    # Run tasks in parallel
    tasks = []
    with ProcessPoolExecutor() as executor:
        for objf_index in benchmark_functions:
            for algorithm_name, algorithm in algorithms.items():
                tasks.append(executor.submit(run_algorithm, algorithm_name, algorithm, objf_index, unique_pids))

        # Gather results
        results = [task.result() for task in tasks]

    # Define CSV file name
    csv_file = "optimization_results_parallel.csv"

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
                    [result["algorithm"], result["benchmark"], result["best_fitness"], f"{result['execution_time']:.2f}", result["pid"]]
                )

    # Calculate and print total time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal Program Execution Time: {total_time:.2f} seconds")

    # Append the total time and number of processes to the CSV
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["\nTotal Time (s): ", f"{total_time:.2f}"])
        unique_pid_count = len(set(unique_pids))
        writer.writerow(["\nNumber of CPU Processes Used: ", unique_pid_count])

    print(f"\nNumber of CPU cores used: {unique_pid_count}\n")


if __name__ == "__main__":
    main()

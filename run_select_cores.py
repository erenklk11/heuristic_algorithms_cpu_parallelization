import time
import csv
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from SSA import SSA
from MFO import MFO
from GEA import GEA
from functions import selectFunction

# Benchmark function indices and algorithms
benchmark_functions = [0, 1, 2, 4, 7, 8, 9, 10, 13]
algorithms = {
    "SSA": SSA,
    "MFO": MFO,
    "GEA": GEA,
}

# Parameters for optimization
lb = -100
ub = 100
dim = 30
N = 5000
Max_iteration = 1000

def get_core_count():
    max_cores = multiprocessing.cpu_count()
    while True:
        try:
            print(f"\nAvailable CPU cores: {max_cores}")
            cores = int(input(f"Enter number of cores to use (1-{max_cores}): "))
            if 1 <= cores <= max_cores:
                return cores
            print(f"Please enter a number between 1 and {max_cores}")
        except ValueError:
            print("Please enter a valid number")

def run_algorithm(algorithm_name, algorithm, objf_index, unique_pids):
    try:
        start_time = time.time()
        objf = selectFunction(objf_index)
        
        result = algorithm(
            objf=objf,
            lb=lb,
            ub=ub,
            dim=dim,
            N=N,
            Max_iteration=Max_iteration,
        )

        end_time = time.time()
        task_time = end_time - start_time
        
        unique_pids.append(os.getpid())
        
        return {
            "algorithm": algorithm_name,
            "benchmark": objf.__name__,
            "best_fitness": result.convergence[-1],
            "execution_time": task_time,
            "pid": os.getpid(),
        }
    except Exception as e:
        unique_pids.append(os.getpid())
        return {
            "algorithm": algorithm_name,
            "benchmark": selectFunction(objf_index).__name__,
            "error": str(e),
            "execution_time": None,
            "pid": os.getpid(),
        }

def update_scalability_data(cores, execution_time):
    scalability_file = "scalability_data.csv"
    file_exists = os.path.isfile(scalability_file)
    
    with open(scalability_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Cores", "Execution Time (s)"])
        writer.writerow([cores, f"{execution_time:.2f}"])

def main():
    cores_to_use = get_core_count()
    start_time = time.time()
    
    manager = Manager()
    unique_pids = manager.list()

    tasks = []
    with ProcessPoolExecutor(max_workers=cores_to_use) as executor:
        for objf_index in benchmark_functions:
            for algorithm_name, algorithm in algorithms.items():
                tasks.append(executor.submit(run_algorithm, algorithm_name, algorithm, objf_index, unique_pids))

        results = [task.result() for task in tasks]

    csv_file = f"optimization_results_{cores_to_use}_cores.csv"

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

    end_time = time.time()
    total_time = end_time - start_time
    unique_pid_count = len(set(unique_pids))

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["Total Time (s)", f"{total_time:.2f}"])
        writer.writerow(["Requested CPU Cores", cores_to_use])
        writer.writerow(["Actual CPU Processes Used", unique_pid_count])

    # Update scalability data
    update_scalability_data(cores_to_use, total_time)

    print(f"\nTotal Program Execution Time: {total_time:.2f} seconds")
    print(f"Number of CPU cores requested: {cores_to_use}")
    print(f"Actual number of CPU processes used: {unique_pid_count}\n")

if __name__ == "__main__":
    main()
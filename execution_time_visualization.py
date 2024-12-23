import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths for the results
single_file = "optimization_results_single.csv"
parallel_file = "optimization_results_parallel.csv"

# Load the CSV data into DataFrames
single_data = pd.read_csv(single_file)
parallel_data = pd.read_csv(parallel_file)

# Merge the data for comparison
merged_data = pd.merge(
    single_data, 
    parallel_data, 
    on=["Algorithm", "Benchmark"], 
    suffixes=('_Single', '_Parallel')
)

# Extract the relevant columns for plotting
algorithms = merged_data["Algorithm"]
benchmarks = merged_data["Benchmark"]
single_times = merged_data["Execution Time (s)_Single"]
parallel_times = merged_data["Execution Time (s)_Parallel"]

# Create a bar chart
x = np.arange(len(benchmarks))  # X-axis locations
bar_width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars for single-threaded and parallelized execution times
rects1 = ax.bar(x - bar_width/2, single_times, bar_width, label='Single-Threaded', color='skyblue')
rects2 = ax.bar(x + bar_width/2, parallel_times, bar_width, label='Parallelized', color='orange')

# Add labels and title
ax.set_xlabel("Benchmark (Algorithm)", fontsize=12)
ax.set_ylabel("Execution Time (s)", fontsize=12)
ax.set_title("Comparison of Execution Times: Single-Threaded vs. Parallelized", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f"{alg}\n({bench})" for alg, bench in zip(algorithms, benchmarks)], rotation=45, ha="right")
ax.legend()

# Add values on top of bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2., 
            height + 0.1, 
            f'{height:.2f}', 
            ha='center', 
            va='bottom', 
            fontsize=9
        )

add_labels(rects1)
add_labels(rects2)

# Adjust layout and show the plot
fig.tight_layout()
plt.show()

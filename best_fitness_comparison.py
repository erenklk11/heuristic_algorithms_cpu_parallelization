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
single_fitness = merged_data["Best Fitness_Single"]
parallel_fitness = merged_data["Best Fitness_Parallel"]

# Calculate the percentage difference
percentage_difference = abs(parallel_fitness - single_fitness) / single_fitness * 100

# Cap the percentage difference at 100% if it exceeds that value
percentage_difference = np.minimum(percentage_difference, 100)

# Create a bar chart
x = np.arange(len(benchmarks))  # X-axis locations
bar_width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars for the percentage difference in fitness values
rects = ax.bar(x, percentage_difference, bar_width, label='Percentage Difference', color='lightcoral')

# Add labels and title
ax.set_xlabel("Benchmark (Algorithm)", fontsize=12)
ax.set_ylabel("Percentage Difference (%)", fontsize=12)
ax.set_title("Comparison of Percentage Difference in Best Fitness: Single-Threaded vs. Parallelized", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f"{alg}\n({bench})" for alg, bench in zip(algorithms, benchmarks)], rotation=45, ha="right")

# Adjust layout and show the plot
fig.tight_layout()
plt.show()

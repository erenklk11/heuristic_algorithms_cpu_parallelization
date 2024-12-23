import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load scalability data
data_file = "scalability_data.csv"
scalability_data = pd.read_csv(data_file)

# Extract columns
cores = scalability_data["Cores"]
execution_times = scalability_data["Execution Time (s)"]

# Compute ideal scalability for comparison
ideal_times = execution_times[0] / cores

# Plot the scalability curve
plt.figure(figsize=(10, 6))
plt.plot(cores, execution_times, marker='o', label='Measured Execution Time', color='blue')

# Add labels, title, and legend
plt.xlabel("Number of Cores", fontsize=12)
plt.ylabel("Execution Time (s)", fontsize=12)
plt.title("Performance Scalability of Parallelized Code", fontsize=14)
plt.xticks(cores)
plt.legend(fontsize=10)

# Annotate points
for i, txt in enumerate(execution_times):
    plt.annotate(f'{txt:.2f}s', (cores[i], execution_times[i]), textcoords="offset points", xytext=(-10, 5), ha='center')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

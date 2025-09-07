import pandas as pd
import matplotlib.pyplot as plt

# Load the benchmarking CSV
df = pd.read_csv("./output/benchmark_full_results.csv")  # Adjust path if needed

# Rename for clarity
df = df.rename(columns={
    "columns": "variables",
    "rows": "samples",
    "runtime": "runtime_seconds"
})

# Convert runtime to minutes
df["runtime_minutes"] = df["runtime_seconds"] / 60

# Group by (variables, samples) and compute mean + std of runtime in minutes
grouped = df.groupby(['variables', 'samples'])['runtime_minutes'].agg(['mean', 'std']).reset_index()

# Plot
plt.figure(figsize=(12, 7))
for var in sorted(grouped['variables'].unique()):
    subset = grouped[grouped['variables'] == var]
    plt.errorbar(
        subset['samples'],             # x-axis: samples
        subset['mean'],               # y-axis: mean runtime in minutes
        yerr=subset['std'],           # error bars = std deviation
        label=f"{var} variables",
        capsize=5,
        marker='o',
        linestyle='-'
    )

# Labels and title
plt.title("Average Runtime vs Samples and Variables", fontsize=14)
plt.xlabel("Number of Samples", fontsize=12)
plt.ylabel("Average Runtime (minutes)", fontsize=12)  # <--- updated y-label
plt.legend(title="Variables")
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()

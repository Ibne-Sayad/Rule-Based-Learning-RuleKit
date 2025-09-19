import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# --- Paths ---
csv_path = "./output/benchmark_full_results.csv"
save_path = "./output/average_runtime_vs_samples_variables.png"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

# --- Load & prep ---
df = pd.read_csv(csv_path)

df = df.rename(columns={
    "columns": "variables",
    "rows": "samples",
    "runtime": "runtime_seconds"
})
df["runtime_minutes"] = df["runtime_seconds"] / 60

grouped = (
    df.groupby(["variables", "samples"])["runtime_minutes"]
      .agg(["mean", "std"])
      .reset_index()
)

# --- Plot ---
plt.figure(figsize=(12, 7))

# Red colormap (avoid very light colors by clipping range)
full_cmap = cm.get_cmap("Reds")
clipped_cmap = colors.LinearSegmentedColormap.from_list(
    "ClippedReds",
    full_cmap(np.linspace(0.3, 1.0, len(grouped["variables"].unique())))
)

# Sorted variables and manual color assignment
vars_sorted = sorted(grouped["variables"].unique())

for i, v in enumerate(vars_sorted):
    subset = grouped[grouped["variables"] == v]
    plt.errorbar(
        subset["samples"],
        subset["mean"],
        yerr=subset["std"],
        label=f"{v} variables",
        capsize=5,
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=6,
        color=clipped_cmap(i / (len(vars_sorted) - 1))  # evenly spaced shades
    )

# Labels and formatting
plt.title("Average Runtime vs Samples and Variables", fontsize=18, pad=12)
plt.xlabel("Number of Samples", fontsize=16, labelpad=10)
plt.ylabel("Average Runtime (minutes)", fontsize=16, labelpad=10)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(title="Variables", fontsize=12, title_fontsize=13, frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Plot saved to: {save_path}")

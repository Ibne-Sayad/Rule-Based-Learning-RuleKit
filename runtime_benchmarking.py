import os
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
from rulefetcher.rulekit_wrapper import RuleKitWrapper

# === Configuration ===
test_data_dir = "./test_data"
arff_output_path = "./data/tcga_dataset_clean.arff"
raw_results = []

# === Extract info from filename ===
def parse_filename(filename):
    match = re.match(r"simulated_(\d+)cols_(\d+)rows_run(\d+)\.csv", filename)
    if match:
        n_cols = int(match.group(1))
        n_rows = int(match.group(2))
        run = int(match.group(3))
        return n_cols, n_rows, run
    return None, None, None

# === Benchmarking wrapper ===
def benchmark_file(file_path, target_column="cluster"):
    wrapper = RuleKitWrapper(file_path, target_column=target_column)
    start = time.perf_counter()
    wrapper.load_and_clean()
    wrapper.write_arff(arff_output_path)
    wrapper.load_arff()
    wrapper.train_all()
    end = time.perf_counter()
    return round(end - start, 2)

# === Main ===
def main():
    print("🔍 Starting benchmarking over test_data...\n")
    for file_name in sorted(os.listdir(test_data_dir)):
        if file_name.endswith(".csv") and "cols" in file_name:
            cols, rows, run = parse_filename(file_name)
            if cols and rows and run:
                file_path = os.path.join(test_data_dir, file_name)
                print(f"⏳ Benchmarking {file_name} ...")
                try:
                    runtime = benchmark_file(file_path)
                    print(f"✅ Done: {cols} cols, {rows} rows, run {run} → {runtime:.2f} sec\n")
                    raw_results.append((cols, rows, run, runtime))
                except Exception as e:
                    print(f"❌ Error processing {file_name}: {e}")
                    raw_results.append((cols, rows, run, None))

    # Convert to DataFrame
    df = pd.DataFrame(raw_results, columns=["columns", "rows", "run", "runtime"])
    df.dropna(inplace=True)
    df.to_csv("./output/benchmark_full_results.csv", index=False)
    print("📄 Full results saved as: ./output/benchmark_full_results.csv")

    # Group by (cols, rows) and compute average
    summary = df.groupby(["columns", "rows"]).agg(
        avg_runtime=("runtime", "mean"),
        std_runtime=("runtime", "std"),
        min_runtime=("runtime", "min"),
        max_runtime=("runtime", "max")
    ).reset_index()

    summary.to_csv("./output/benchmark_runtime_summary.csv", index=False)
    print("📄 Summary table saved as: ./output/benchmark_runtime_summary.csv\n")

    # === Plot ===
    plt.figure(figsize=(10, 6))
    for col in sorted(summary['columns'].unique()):
        subset = summary[summary['columns'] == col]
        plt.plot(subset['rows'], subset['avg_runtime'], marker='o', label=f"{col} cols")

    plt.title("Average Runtime vs Rows and Columns")
    plt.xlabel("Number of Rows")
    plt.ylabel("Average Runtime (seconds)")
    plt.legend(title="Columns")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./output/benchmark_runtime_plot.png")
    plt.show()
    print("📊 Runtime plot saved as: ./output/benchmark_runtime_plot.png")

    # === Print summary table ===
    print("\n📋 Summary Table (Grouped by Columns & Rows):\n")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

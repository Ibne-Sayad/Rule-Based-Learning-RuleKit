import pandas as pd
import numpy as np
import random
import os

# === Configuration ===
input_path = "data/clustered_output - Age_less_40.csv"
output_dir = "test_data"
column_levels = [100, 200, 300, 400, 500]
row_levels = [5000, 10000, 15000, 20000]
repeats_per_combo = 5

# === Load Base Dataset ===
df_base = pd.read_csv(input_path)

# Drop old 'cluster' column if exists
if 'cluster' in df_base.columns:
    df_base = df_base.drop(columns=['cluster'])

# === Function to Simulate Columns ===
def simulate_columns(df, target_cols, target_rows):
    df = df.sample(n=target_rows, replace=True).reset_index(drop=True)
    current_cols = df.shape[1]
    n_to_add = target_cols - current_cols
    original_cols = df.columns.tolist()

    for i in range(n_to_add):
        ref_col = random.choice(original_cols)
        if df[ref_col].dtype == object:
            values = df[ref_col].dropna().unique()
            new_col = np.random.choice(values, size=target_rows)
        else:
            mean, std = df[ref_col].mean(), df[ref_col].std()
            new_col = np.random.normal(mean, std, size=target_rows)
        df[f"sim_{i}_{ref_col}"] = new_col

    # Add new 'cluster' column
    df['cluster'] = df['age_at_initial_pathologic_diagnosis'].apply(lambda x: 1 if x < 40 else 0)

    return df

# === Ensure output folder exists ===
os.makedirs(output_dir, exist_ok=True)

# === Generate All Combinations ===
for cols in column_levels:
    for rows in row_levels:
        for run in range(1, repeats_per_combo + 1):
            print(f"⏳ Generating: {cols} cols, {rows} rows, run {run}")
            df_simulated = simulate_columns(df_base.copy(), cols, rows)
            filename = f"simulated_{cols}cols_{rows}rows_run{run}.csv"
            full_path = os.path.join(output_dir, filename)
            df_simulated.to_csv(full_path, index=False)
            print(f"✅ Saved: {full_path}")

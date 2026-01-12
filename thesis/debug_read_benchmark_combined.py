
import pandas as pd
import sys

file_path = "final_runs/supp_parallelization_benchmark/Data/benchmark_results_combined_20251218_011948.xlsx"

try:
    df = pd.read_excel(file_path)
    print(df.to_string())
except Exception as e:
    print(f"Error reading Excel: {e}")

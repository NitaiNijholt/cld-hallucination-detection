#!/usr/bin/env python3
"""
Run RQ2 analysis pipeline on the preload validation experiment data.
"""
import sys
import os

# Add paths
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./parameter_tuning_experiments/analysis'))

from rq2_data_loader import RQ2DataLoader
import pandas as pd

print("="*80)
print("RQ2 ANALYSIS: Preload Validation Experiment")
print("="*80)
print()

# Initialize data loader with ground_truth mode
print("Step 1: Loading data with GROUND TRUTH hallucination definition...")
print("-"*80)
loader = RQ2DataLoader(
    results_base_dir="parameter_tuning_experiments/results",
    hallucination_mode="ground_truth"  # Use ground truth Classification column
)

# Load data
df = loader.load_all_experiments()
print(f"\n✅ Loaded {len(df)} edges from all experiments")
print(f"   Experiments: {df['experiment_id'].nunique()}")
print(f"   Unique CLDs: {df['cld_name'].nunique()}")
print()

# Filter to only the latest preload experiment
print("Step 2: Filtering to preload validation experiment...")
print("-"*80)
latest_exp = "exp_20251007_234155_60068754"
df_preload = df[df['experiment_id'] == latest_exp].copy()
print(f"   Filtered to experiment: {latest_exp}")
print(f"   Edges: {len(df_preload)}")
print()

# Check hallucination distribution
print("Step 3: Hallucination distribution (Ground Truth)...")
print("-"*80)
halluc_col = loader.get_hallucination_column()
if halluc_col in df_preload.columns:
    halluc_count = df_preload[halluc_col].sum()
    correct_count = (~df_preload[halluc_col]).sum()
    print(f"   Hallucinations (FP): {halluc_count}/{len(df_preload)} ({halluc_count/len(df_preload)*100:.1f}%)")
    print(f"   Correct (TP):        {correct_count}/{len(df_preload)} ({correct_count/len(df_preload)*100:.1f}%)")
else:
    print(f"   ⚠️  Hallucination column '{halluc_col}' not found!")
print()

# Save for RQ2 pipeline
output_file = "parameter_tuning_experiments/analysis/rq2_preload_data.csv"
df_preload.to_csv(output_file, index=False)
print(f"✅ Saved data to: {output_file}")
print(f"   Shape: {df_preload.shape}")
print()

# Show CI metrics availability
print("Step 4: CI Metrics availability...")
print("-"*80)
ci_metrics = loader.get_ci_metric_columns()
for metric in ci_metrics:
    if metric in df_preload.columns:
        non_null = df_preload[metric].notna().sum()
        pct = (non_null / len(df_preload) * 100)
        print(f"   {metric:30s}: {non_null:3d}/{len(df_preload):3d} ({pct:5.1f}%)")
    else:
        print(f"   {metric:30s}: ❌ MISSING")
print()

print("="*80)
print("✅ Data preparation complete! Ready for RQ2 analysis pipeline.")
print("="*80)
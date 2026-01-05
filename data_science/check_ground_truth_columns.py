#!/usr/bin/env python3
"""
Check what ground truth columns are available in the experiment data.
"""

import pandas as pd

excel_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_230722_42be2970/combo_1/prompts_Nitai_C/CLD_test/run1_fac1146e/results_exp_20251007_230722_42be2970_param_combo_nr_1_prompts_Nitai_C_run1_CLD_test_20251007_231255.xlsx"

print(f"Loading: {excel_file}\n")
df = pd.read_excel(excel_file, sheet_name='All Edges')

print(f"Total edges: {len(df)}\n")

# Show all columns
print("📋 Available columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "="*80)
print("🔍 Ground Truth / Corruption Related Columns:")
print("="*80)

# Check for corruption-related columns
corruption_cols = [col for col in df.columns if 'corrupt' in col.lower() or 'spurious' in col.lower()]
for col in corruption_cols:
    count = df[col].notna().sum()
    unique_vals = df[col].dropna().unique()
    print(f"\n  {col}:")
    print(f"    Non-null: {count}/{len(df)}")
    print(f"    Unique values: {unique_vals}")
    if len(unique_vals) <= 10:
        print(f"    Value counts:")
        print(df[col].value_counts().to_string().replace('\n', '\n      '))

# Check classification column
print("\n" + "="*80)
print("🏷️  Classification Column:")
print("="*80)
if 'Classification' in df.columns:
    print(f"\n  Classification:")
    print(f"    Non-null: {df['Classification'].notna().sum()}/{len(df)}")
    print(f"    Value counts:")
    print(df['Classification'].value_counts().to_string().replace('\n', '\n      '))

# Check graph membership
print("\n" + "="*80)
print("📊 Graph Membership:")
print("="*80)
for col in ['In Session Graph', 'In Validation Graph']:
    if col in df.columns:
        print(f"\n  {col}:")
        print(df[col].value_counts().to_string().replace('\n', '\n      '))

print("\n" + "="*80)
print("🎯 Sample Data (first 3 edges):")
print("="*80)
display_cols = ['Source', 'Target', 'Classification', 'Is Corrupted', 'In Session Graph', 'In Validation Graph', 'Aggregate Score']
display_cols = [c for c in display_cols if c in df.columns]
print(df[display_cols].head(3).to_string())
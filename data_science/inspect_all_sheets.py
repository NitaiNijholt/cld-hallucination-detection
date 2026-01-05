#!/usr/bin/env python3
import pandas as pd

excel_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_234155_60068754/combo_1/prompts_Nitai_C/Social_norms_and_obesity_prevalence/run1_5db6738d/results_exp_20251007_234155_60068754_param_combo_nr_1_prompts_Nitai_C_run1_Social_norms_and_obesity_prevalence_20251008_001309.xlsx"

print(f"Inspecting: {excel_file.split('/')[-1]}")
print("="*80)

# Check what sheets exist
xl = pd.ExcelFile(excel_file)
print(f"Available sheets: {xl.sheet_names}\n")

# Check each sheet for CI/judge metrics
target_cols = ['perplexity', 'min_prob', 'max_window_entropy', 'cosine_similarity',
               'judge_perplexity', 'judge_min_prob', 'judge_max_window_entropy',
               'aggregate_score', 'classification']

for sheet_name in xl.sheet_names:
    print(f"\n{'='*80}")
    print(f"SHEET: {sheet_name}")
    print(f"{'='*80}")
    
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}\n")
        
        # Check for target columns
        found_cols = [col for col in target_cols if col in df.columns]
        if found_cols:
            print(f"✅ Found {len(found_cols)} target columns:")
            for col in found_cols:
                non_null = df[col].notna().sum()
                pct = (non_null / len(df) * 100) if len(df) > 0 else 0
                print(f"   {col:25s}: {non_null:3d}/{len(df):3d} ({pct:5.1f}%)")
        else:
            print("❌ No target columns found")
            print("\nAll columns in this sheet:")
            for col in df.columns:
                print(f"   - {col}")
    except Exception as e:
        print(f"⚠️  Error reading sheet: {e}")
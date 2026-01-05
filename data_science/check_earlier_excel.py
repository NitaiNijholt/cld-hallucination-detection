#!/usr/bin/env python3
import pandas as pd

excel_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_230722_42be2970/combo_1/prompts_Nitai_C/CLD_test/run1_fac1146e/results_exp_20251007_230722_42be2970_param_combo_nr_1_prompts_Nitai_C_run1_CLD_test_20251007_231255.xlsx"

print(f"Checking: {excel_file.split('/')[-1]}")
print("="*80)
df = pd.read_excel(excel_file, sheet_name="All Edges")
print(f"✅ 'All Edges' sheet: {len(df)} edges\n")

# Check for CI metrics
ci_metrics = ['perplexity', 'min_prob', 'max_window_entropy', 'cosine_similarity']
print("📊 CI Metrics (Generator):")
for metric in ci_metrics:
    if metric in df.columns:
        non_null = df[metric].notna().sum()
        print(f"   {metric:25s}: {non_null:3d}/{len(df):3d} ({non_null/len(df)*100:5.1f}%)")
    else:
        print(f"   {metric:25s}: ❌ MISSING")

# Check for judge logit metrics
judge_metrics = ['judge_perplexity', 'judge_min_prob', 'judge_max_window_entropy']
print("\n📊 Judge Logit Metrics:")
for metric in judge_metrics:
    if metric in df.columns:
        non_null = df[metric].notna().sum()
        print(f"   {metric:25s}: {non_null:3d}/{len(df):3d} ({non_null/len(df)*100:5.1f}%)")
    else:
        print(f"   {metric:25s}: ❌ MISSING")

print("\n📊 Other Key Columns:")
for col in ['aggregate_score', 'classification', 'judge_verdict']:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"   {col:25s}: {non_null:3d}/{len(df):3d} ({non_null/len(df)*100:5.1f}%)")
    else:
        print(f"   {col:25s}: ❌ MISSING")
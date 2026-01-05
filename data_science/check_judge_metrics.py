#!/usr/bin/env python3
import pandas as pd

excel_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_051110_dfe82577/combo_1/prompts_Nitai_C/CLD_test/run1_519e320e/results_exp_20251007_051110_dfe82577_param_combo_nr_1_prompts_Nitai_C_run1_CLD_test_20251007_051253.xlsx"

# Read the All Edges sheet
df = pd.read_excel(excel_file, sheet_name='All Edges')
print(f"✅ Loaded {len(df)} edges from 'All Edges' sheet")
print()

# Check for all CI metrics including judge logit metrics
ci_metrics = [
    'Perplexity',
    'Min Prob', 
    'Max Window Entropy',
    'Cosine Similarity',
    'Judge Perplexity',
    'Judge Min Prob',
    'Judge Max Window Entropy'
]

print("📊 CI Metrics Status:")
print()
for metric in ci_metrics:
    if metric in df.columns:
        count = df[metric].notna().sum()
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {metric}: {count}/{len(df)} populated")
        if count > 0:
            print(f"      Sample values: {df[metric].dropna().head(3).tolist()}")
    else:
        print(f"  ❌ {metric}: Column NOT FOUND")
    print()

# Show column names to verify
print("\n📋 Available Columns (first 30):")
print(df.columns.tolist()[:30])
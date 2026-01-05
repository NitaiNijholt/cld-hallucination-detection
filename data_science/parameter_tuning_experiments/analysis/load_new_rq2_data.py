#!/usr/bin/env python3
"""
Load the new experiment data with judge logit metrics and prepare it for RQ2 analysis.
"""

import pandas as pd
from pathlib import Path

# Load the new experiment Excel file
excel_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_230722_42be2970/combo_1/prompts_Nitai_C/CLD_test/run1_fac1146e/results_exp_20251007_230722_42be2970_param_combo_nr_1_prompts_Nitai_C_run1_CLD_test_20251007_231255.xlsx"

print(f"Loading data from: {excel_file}")
df = pd.read_excel(excel_file, sheet_name='All Edges')
print(f"✅ Loaded {len(df)} edges")
print()

# Standardize column names to match expected format
column_mapping = {
    'Perplexity': 'perplexity',
    'Min Prob': 'min_prob',
    'Max Window Entropy': 'max_window_entropy',
    'Cosine Similarity': 'cosine_similarity',
    'Judge Perplexity': 'judge_perplexity',
    'Judge Min Prob': 'judge_min_prob',
    'Judge Max Window Entropy': 'judge_max_window_entropy',
    'Aggregate Score': 'aggregate_score',
    'Judge Verdict': 'judge_verdict',
    'Source': 'source',
    'Target': 'target'
}

# Rename columns
df_clean = df.rename(columns=column_mapping)

# Create hallucination label (aggregate_score < 0.5)
df_clean['is_hallucination'] = df_clean['aggregate_score'] < 0.5

# Filter to edges that have CI metrics
ci_cols = ['perplexity', 'min_prob', 'max_window_entropy', 'cosine_similarity',
           'judge_perplexity', 'judge_min_prob', 'judge_max_window_entropy']
has_ci = df_clean[ci_cols].notna().any(axis=1)
df_clean = df_clean[has_ci].copy()

print(f"📊 After filtering to edges with CI metrics: {len(df_clean)} edges")
print()

# Select relevant columns for analysis
analysis_cols = ['source', 'target', 'aggregate_score', 'judge_verdict', 'is_hallucination'] + ci_cols
df_analysis = df_clean[analysis_cols].copy()

# Show summary
print("🔍 Data Summary:")
print(f"   Total edges: {len(df_analysis)}")
print(f"   Hallucinations: {df_analysis['is_hallucination'].sum()} ({df_analysis['is_hallucination'].mean()*100:.1f}%)")
print(f"   Non-hallucinations: {(~df_analysis['is_hallucination']).sum()} ({(~df_analysis['is_hallucination']).mean()*100:.1f}%)")
print()

print("📊 CI Metrics Availability:")
for col in ci_cols:
    count = df_analysis[col].notna().sum()
    print(f"   {col}: {count}/{len(df_analysis)} ({count/len(df_analysis)*100:.1f}%)")
print()

# Save to CSV for RQ2 pipeline
output_file = Path(__file__).parent.parent / 'tables' / 'rq2_standalone_data.csv'
df_analysis.to_csv(output_file, index=False)
print(f"✅ Saved analysis data to: {output_file}")
print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
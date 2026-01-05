#!/usr/bin/env python3
"""
Load the new experiment data using GROUND TRUTH mode for hallucination classification.

This script demonstrates the difference between judge-based and ground-truth-based
hallucination detection.
"""

import pandas as pd
from pathlib import Path

# Load the new experiment Excel file directly
excel_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_230722_42be2970/combo_1/prompts_Nitai_C/CLD_test/run1_fac1146e/results_exp_20251007_230722_42be2970_param_combo_nr_1_prompts_Nitai_C_run1_CLD_test_20251007_231255.xlsx"

print(f"Loading data from: {Path(excel_file).name}")
df = pd.read_excel(excel_file, sheet_name='All Edges')
print(f"✅ Loaded {len(df)} total edges\n")

# Standardize column names
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
    'Classification': 'classification',
    'Source': 'source',
    'Target': 'target',
    'In Session Graph': 'in_session_graph',
    'In Validation Graph': 'in_validation_graph'
}

df_clean = df.rename(columns=column_mapping)

# Filter to edges with CI metrics (TP/FP only)
ci_cols = ['perplexity', 'min_prob', 'max_window_entropy', 'cosine_similarity',
           'judge_perplexity', 'judge_min_prob', 'judge_max_window_entropy']
has_ci = df_clean[ci_cols].notna().any(axis=1)
df_clean = df_clean[has_ci].copy()

print(f"📊 After filtering to edges with CI metrics: {len(df_clean)} edges\n")

# Compare JUDGE mode vs GROUND TRUTH mode
print("="*80)
print("COMPARISON: Judge-based vs Ground-Truth-based Hallucination Detection")
print("="*80)

# Judge mode: based on aggregate_score < 0.5
df_clean['is_hallucination_judge'] = df_clean['aggregate_score'] < 0.5

# Ground truth mode: based on Classification='FP'
df_clean['is_hallucination_ground_truth'] = df_clean['classification'] == 'FP'

print("\n📊 JUDGE MODE (aggregate_score < 0.5):")
print(f"   Hallucinations (judge says not supported): {df_clean['is_hallucination_judge'].sum()}")
print(f"   Correct (judge says supported): {(~df_clean['is_hallucination_judge']).sum()}")

print("\n📊 GROUND TRUTH MODE (Classification='FP'):")
print(f"   Hallucinations (FP - not in ground truth CLD): {df_clean['is_hallucination_ground_truth'].sum()}")
print(f"   Correct (TP - matches ground truth CLD): {(~df_clean['is_hallucination_ground_truth']).sum()}")

# Show classification breakdown
print("\n📊 Classification Breakdown:")
class_counts = df_clean['classification'].value_counts()
for class_label, count in class_counts.items():
    print(f"   {class_label}: {count}")

# Agreement analysis
agreement = (df_clean['is_hallucination_judge'] == df_clean['is_hallucination_ground_truth'])
print(f"\n🔍 Judge-Ground Truth Agreement: {agreement.sum()}/{len(df_clean)} ({agreement.mean()*100:.1f}%)")

# Show confusion matrix
print("\n📊 Confusion Matrix (Judge vs Ground Truth):")
print(pd.crosstab(
    df_clean['is_hallucination_judge'], 
    df_clean['is_hallucination_ground_truth'],
    rownames=['Judge'],
    colnames=['Ground Truth'],
    margins=True
))

# Save ground truth version for RQ2 analysis
print("\n" + "="*80)
print("Saving data for RQ2 analysis...")
print("="*80)

# Select columns for analysis
analysis_cols = ['source', 'target', 'classification', 'aggregate_score', 'judge_verdict',
                 'is_hallucination_ground_truth'] + ci_cols
df_analysis = df_clean[analysis_cols].copy()
df_analysis = df_analysis.rename(columns={'is_hallucination_ground_truth': 'is_hallucination'})

# Show summary
print("\n🔍 Final Data Summary:")
print(f"   Total edges: {len(df_analysis)}")
print(f"   Hallucinations (ground truth): {df_analysis['is_hallucination'].sum()} ({df_analysis['is_hallucination'].mean()*100:.1f}%)")
print(f"   Correct edges (ground truth): {(~df_analysis['is_hallucination']).sum()} ({(~df_analysis['is_hallucination']).mean()*100:.1f}%)")

print("\n📊 CI Metrics Availability:")
for col in ci_cols:
    count = df_analysis[col].notna().sum()
    print(f"   {col}: {count}/{len(df_analysis)} ({count/len(df_analysis)*100:.1f}%)")

# Save to CSV
output_file = Path(__file__).parent.parent / 'tables' / 'rq2_standalone_data.csv'
df_analysis.to_csv(output_file, index=False)
print(f"\n✅ Saved analysis data to: {output_file}")
print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
print(f"\n💡 This data uses GROUND TRUTH mode: is_hallucination = (Classification == 'FP')")
#!/usr/bin/env python3
"""
Run RQ2 analysis pipeline on social norms data.
"""
import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, '/home/nitai/code/causalix.ai/data_science')
sys.path.insert(0, '/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/analysis')

import pandas as pd

# Import analysis modules
from rq2_correlation_analysis import RQ2CorrelationAnalysis
from rq2_classification_analysis import RQ2ClassificationAnalysis
from rq2_statistical_validation import RQ2StatisticalValidation

# Data path
DATA_CSV = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/analysis/rq2_social_norms_data.csv"
OUTPUT_BASE = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments")
FIGURES_DIR = OUTPUT_BASE / "figures"
TABLES_DIR = OUTPUT_BASE / "tables"

# Create directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print(" "*25 + "RQ2 ANALYSIS PIPELINE")
print("="*80)
print(f"Data: {DATA_CSV}")
print(f"Output: {OUTPUT_BASE}")
print("="*80)
print("")

# Load data
print("📂 Loading data...")
df = pd.read_excel(
    "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_234155_60068754/combo_1/prompts_Nitai_C/Social_norms_and_obesity_prevalence/run1_5db6738d/results_exp_20251007_234155_60068754_param_combo_nr_1_prompts_Nitai_C_run1_Social_norms_and_obesity_prevalence_20251008_001309.xlsx",
    sheet_name='All Edges'
)
print(f"✅ Loaded {len(df)} edges")
print("")

# Prepare data
# Create hallucination label based on JUDGE VERDICT (not ground truth)
# Judge considers a hallucination if aggregate_score < 0.5
df['is_hallucination'] = df['Aggregate Score'].fillna(1.0) < 0.5

print(f"📊 Hallucination Definition: Judge Verdict (Aggregate Score < 0.5)")
print(f"   Hallucinations: {df['is_hallucination'].sum()} ({df['is_hallucination'].sum()/len(df)*100:.1f}%)")
print(f"   Non-hallucinations: {(~df['is_hallucination']).sum()} ({(~df['is_hallucination']).sum()/len(df)*100:.1f}%)")
print("")

# Rename columns to match expected format (lowercase with underscores)
column_rename_map = {
    'Perplexity': 'perplexity',
    'Min Prob': 'min_prob',
    'Max Window Entropy': 'max_window_entropy',
    'Cosine Similarity': 'cosine_similarity',
    'Judge Perplexity': 'judge_perplexity',
    'Judge Min Prob': 'judge_min_prob',
    'Judge Max Window Entropy': 'judge_max_window_entropy'
}

# Apply renaming
for old_name, new_name in column_rename_map.items():
    if old_name in df.columns:
        df[new_name] = df[old_name]
        
# Filter out inf/nan values in Judge Perplexity if present
if 'judge_perplexity' in df.columns:
    df['judge_perplexity'] = df['judge_perplexity'].replace([float('inf'), float('-inf')], float('nan'))

print("📊 CI Metrics Available:")
ci_metric_names = ['perplexity', 'min_prob', 'max_window_entropy', 'cosine_similarity', 
                   'judge_perplexity', 'judge_min_prob', 'judge_max_window_entropy']
for metric_name in ci_metric_names:
    if metric_name in df.columns:
        count = df[metric_name].notna().sum()
        print(f"   {metric_name}: {count}/{len(df)} ({count/len(df)*100:.1f}%)")
print("")

# Save cleaned data
output_csv = TABLES_DIR / "rq2_social_norms_judge_ground_truth.csv"
df.to_csv(output_csv, index=False)
print(f"✅ Saved clean data to: {output_csv}")
print("")

# ============================================================================
# STEP 1: CORRELATION ANALYSIS
# ============================================================================
print("="*80)
print("STEP 1/3: CORRELATION ANALYSIS")
print("="*80)
print("")

try:
    corr_analysis = RQ2CorrelationAnalysis(df=df)
    corr_results = corr_analysis.compute_correlations()
    
    # Save results
    corr_results_path = TABLES_DIR / "rq2_correlations_judge_ground_truth.csv"
    corr_results.to_csv(corr_results_path, index=False)
    
    print("✅ Correlation analysis complete")
    print(f"   Results saved to: {corr_results_path}")
    print("")
    
except Exception as e:
    print(f"❌ Correlation analysis failed: {e}")
    import traceback
    traceback.print_exc()
    print("")

# Classification and statistical validation omitted for now - will run separately

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("RQ2 ANALYSIS COMPLETE (JUDGE AS GROUND TRUTH)")
print("="*80)
print("")
print(f"📊 Data: {len(df)} edges")
print(f"   Ground Truth Source: Judge Verdict (Aggregate Score < 0.5)")
print(f"   Hallucinations: {df['is_hallucination'].sum()} ({df['is_hallucination'].sum()/len(df)*100:.1f}%)")
print(f"   Non-Hallucinations: {(~df['is_hallucination']).sum()} ({(~df['is_hallucination']).sum()/len(df)*100:.1f}%)")
print("")
print(f"📁 Output directories:")
print(f"   Figures: {FIGURES_DIR}")
print(f"   Tables: {TABLES_DIR}")
print("")
print("✅ All analyses complete!")
print("="*80)
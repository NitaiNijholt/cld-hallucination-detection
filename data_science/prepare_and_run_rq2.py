#!/usr/bin/env python3
"""
Load preload experiment data and run full RQ2 analysis pipeline.
"""
import sys
import os
sys.path.insert(0, 'parameter_tuning_experiments/analysis')

import pandas as pd
from pathlib import Path

# Direct load from the Excel file
excel_file = "parameter_tuning_experiments/results/exp_20251007_234155_60068754/combo_1/prompts_Nitai_C/Social_norms_and_obesity_prevalence/run1_5db6738d/results_exp_20251007_234155_60068754_param_combo_nr_1_prompts_Nitai_C_run1_Social_norms_and_obesity_prevalence_20251008_001309.xlsx"

print("="*80)
print("RQ2 DATA PREPARATION")
print("="*80)
print(f"\nLoading data from: {excel_file.split('/')[-1]}")

# Load All Edges sheet
df = pd.read_excel(excel_file, sheet_name="All Edges")
print(f"✅ Loaded {len(df)} edges")

# Normalize column names (title case to snake_case with lowercase)
column_mapping = {
    'Source': 'source',
    'Target': 'target',
    'Classification': 'classification',
    'In Session Graph': 'in_session_graph',
    'In Validation Graph': 'in_validation_graph',
    'Judge Verdict': 'judge_verdict',
    'Motivation': 'motivation',
    'Citation': 'citation',
    'Judge Message': 'judge_message',
    'Aggregate Score': 'aggregate_score',
    'Corrected': 'corrected',
    'Correction Action': 'correction_action',
    'Corrected Motivation': 'corrected_motivation',
    'Corrector Message': 'corrector_message',
    'Is Corrupted': 'is_corrupted',
    'Spurious Motivation': 'spurious_motivation',
    'Perplexity': 'perplexity',
    'Min Prob': 'min_prob',
    'Max Window Entropy': 'max_window_entropy',
    'Gen Perplexity': 'gen_perplexity',
    'Gen Min Prob': 'gen_min_prob',
    'Gen Max Window Entropy': 'gen_max_window_entropy',
    'Judge Perplexity': 'judge_perplexity',
    'Judge Min Prob': 'judge_min_prob',
    'Judge Max Window Entropy': 'judge_max_window_entropy',
    'Cosine Similarity': 'cosine_similarity',
    'Gen Cosine Similarity': 'gen_cosine_similarity',
    'Judge Cosine Similarity': 'judge_cosine_similarity',
}

df = df.rename(columns=column_mapping)

# Create hallucination label based on ground truth Classification
# FP = False Positive = Hallucination, TP = True Positive = Correct
df['is_hallucination'] = (df['classification'] == 'FP')

# Add metadata columns
df['experiment_id'] = 'exp_20251007_234155_60068754'
df['cld_name'] = 'Social_norms_and_obesity_prevalence'

print(f"\n📊 Hallucination Distribution (Ground Truth):")
print(f"   TP (Correct):        {(df['classification'] == 'TP').sum():3d} ({(df['classification'] == 'TP').sum()/len(df)*100:5.1f}%)")
print(f"   FP (Hallucination):  {(df['classification'] == 'FP').sum():3d} ({(df['classification'] == 'FP').sum()/len(df)*100:5.1f}%)")

# Save to expected location
output_dir = Path("parameter_tuning_experiments/tables")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "rq2_standalone_data.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Saved data to: {output_file}")
print(f"   Shape: {df.shape}")
print(f"   Columns: {len(df.columns)}")

# Now run the master pipeline
print("\n" + "="*80)
print("RUNNING RQ2 MASTER PIPELINE")
print("="*80)

from rq2_master_pipeline import RQ2MasterPipeline

pipeline = RQ2MasterPipeline(output_base_dir="parameter_tuning_experiments")
pipeline.run_full_pipeline(n_permutations=10000, n_bootstrap=10000)

print("\n" + "="*80)
print("✅ RQ2 ANALYSIS COMPLETE!")
print("="*80)
print(f"\n📁 Outputs saved to:")
print(f"   • Tables: parameter_tuning_experiments/tables/")
print(f"   • Figures: parameter_tuning_experiments/figures/")
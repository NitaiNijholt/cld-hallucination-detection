#!/usr/bin/env python3
"""
Run RQ2 analysis pipeline on social norms data.
"""
import sys
import os
sys.path.insert(0, '/home/nitai/code/causalix.ai/data_science')

from parameter_tuning_experiments.analysis.rq2_data_loader import RQ2DataLoader
import pandas as pd

# Excel file from the experiment
EXCEL_PATH = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_234155_60068754/combo_1/prompts_Nitai_C/Social_norms_and_obesity_prevalence/run1_5db6738d/results_exp_20251007_234155_60068754_param_combo_nr_1_prompts_Nitai_C_run1_Social_norms_and_obesity_prevalence_20251008_001309.xlsx"

print("="*80)
print("RQ2 ANALYSIS PIPELINE - Social Norms Data")
print("="*80)
print("")

# Load the data
print("📂 Loading data from Excel...")
try:
    df = pd.read_excel(EXCEL_PATH, sheet_name='All Edges')
    print(f"✅ Loaded {len(df)} edges")
    print("")
    
    # Save to CSV for easier analysis
    output_csv = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/analysis/rq2_social_norms_data.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved to: {output_csv}")
    print("")
    
    # Show summary statistics
    print("📊 DATA SUMMARY:")
    print(f"   Total edges: {len(df)}")
    print("")
    
    if 'Classification' in df.columns:
        print("   Classification:")
        for cls, count in df['Classification'].value_counts().items():
            print(f"      {cls}: {count} ({count/len(df)*100:.1f}%)")
        print("")
    
    if 'Aggregate Score' in df.columns:
        print("   Aggregate Score (Judge Verdict):")
        print(f"      Mean: {df['Aggregate Score'].mean():.3f}")
        print(f"      Min:  {df['Aggregate Score'].min():.3f}")
        print(f"      Max:  {df['Aggregate Score'].max():.3f}")
        halluc_judge = (df['Aggregate Score'].fillna(1.0) < 0.5).sum()
        print(f"      Hallucinations (< 0.5): {halluc_judge} ({halluc_judge/len(df)*100:.1f}%)")
        print("")
    
    # Check CI metrics availability
    print("   CI Metrics Availability:")
    ci_metrics = {
        'Perplexity': 'Perplexity',
        'Min Prob': 'Min Prob',
        'Max Window Entropy': 'Max Window Entropy',
        'Cosine Similarity': 'Cosine Similarity',
        'Judge Perplexity': 'Judge Perplexity',
        'Judge Min Prob': 'Judge Min Prob',
        'Judge Max Window Entropy': 'Judge Max Window Entropy'
    }
    
    for name, col in ci_metrics.items():
        if col in df.columns:
            count = df[col].notna().sum()
            print(f"      {name}: {count}/{len(df)} ({count/len(df)*100:.1f}%)")
    
    print("")
    print("="*80)
    print("✅ DATA LOADED AND READY FOR ANALYSIS")
    print("="*80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
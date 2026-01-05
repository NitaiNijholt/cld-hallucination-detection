#!/usr/bin/env python3
"""
Check both recent experiments to find one with varied hallucination labels.
"""

import pandas as pd
from pathlib import Path

experiments = [
    "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_230722_42be2970/combo_1/prompts_Nitai_C/CLD_test/run1_fac1146e/results_exp_20251007_230722_42be2970_param_combo_nr_1_prompts_Nitai_C_run1_CLD_test_20251007_231255.xlsx",
]

# Find the second experiment's Excel file
results_dir = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_230948_08783a2b")
excel_files = list(results_dir.rglob("*.xlsx"))
if excel_files:
    experiments.append(str(excel_files[0]))

print(f"Found {len(experiments)} experiment(s) to check\n")

best_exp = None
best_variation = 0

for i, exp_file in enumerate(experiments, 1):
    print(f"{'='*80}")
    print(f"Experiment {i}: {Path(exp_file).name[:60]}...")
    print(f"{'='*80}")
    
    try:
        df = pd.read_excel(exp_file, sheet_name='All Edges')
        
        # Check for aggregate_score column
        if 'Aggregate Score' not in df.columns:
            print("❌ No 'Aggregate Score' column found\n")
            continue
            
        # Create hallucination label
        df['is_hallucination'] = df['Aggregate Score'] < 0.5
        
        # Filter to edges with CI metrics
        ci_cols = ['Perplexity', 'Min Prob', 'Max Window Entropy', 'Cosine Similarity']
        has_ci = df[ci_cols].notna().any(axis=1)
        df_filtered = df[has_ci].copy()
        
        n_total = len(df_filtered)
        n_halluc = df_filtered['is_hallucination'].sum()
        n_non_halluc = (~df_filtered['is_hallucination']).sum()
        halluc_rate = df_filtered['is_hallucination'].mean() * 100
        
        # Check judge metrics
        judge_cols = ['Judge Perplexity', 'Judge Min Prob', 'Judge Max Window Entropy']
        has_judge = df_filtered[judge_cols].notna().any(axis=1).sum()
        
        print(f"📊 Total edges with CI metrics: {n_total}")
        print(f"   Hallucinations: {n_halluc} ({halluc_rate:.1f}%)")
        print(f"   Non-hallucinations: {n_non_halluc} ({100-halluc_rate:.1f}%)")
        print(f"   Edges with judge metrics: {has_judge}/{n_total}")
        
        # Calculate variation (how balanced the classes are)
        variation = min(n_halluc, n_non_halluc)
        print(f"   Variation score: {variation}")
        
        if variation > best_variation:
            best_variation = variation
            best_exp = exp_file
            
        print()
        
    except Exception as e:
        print(f"❌ Error reading file: {e}\n")
        continue

print(f"{'='*80}")
if best_exp and best_variation > 0:
    print(f"✅ Best experiment for RQ2 analysis:")
    print(f"   {best_exp}")
    print(f"   Variation score: {best_variation}")
else:
    print(f"❌ No suitable experiment found with varied hallucination labels")
    print(f"\n💡 Recommendation: Run an experiment with:")
    print(f"   - corruption_rate: 0.0 (no corruption)")
    print(f"   - A larger CLD with more complex relationships")
    print(f"   - This will produce real edges with varied aggregate scores")
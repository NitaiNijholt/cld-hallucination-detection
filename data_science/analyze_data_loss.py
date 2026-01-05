#!/usr/bin/env python3
"""
Analyze Data Loss During Cleaning

Investigate why 407 → 136 (33% retention) in original data
vs 223 → 200 (90% retention) in recent data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Original data
ORIGINAL_FILES = {
    'Depressive_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run1_7a66e382/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251008_181257.xlsx',
    'Depressive_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run2_0d57ed9a/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251008_200816.xlsx',
    'Social_norms_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run1_86b016e1/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Social_norms_and_obesity_prevalence_20251008_213554.xlsx',
    'Social_norms_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run2_d09234b7/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Social_norms_and_obesity_prevalence_20251008_221139.xlsx',
    'Older_persons_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run1_d45241a5/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251008_230700.xlsx',
    'Older_persons_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run2_96605199/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251009_082149.xlsx'
}

# Recent data
RECENT_DIR = Path("parameter_tuning_experiments/results/rq1_base_judging_correctness_20251012_225925")
RECENT_FILES = [
    RECENT_DIR / "judged_Social_norms_and_obesity_prevalence_citations_20251012_230158.xlsx",
    RECENT_DIR / "judged_older_persons_emergency_department_visits_citations_20251012_233401.xlsx",
    RECENT_DIR / "judged_Depressive_symptoms_in_response_to_a_stressor_citations_20251013_001809.xlsx",
]

CI_METRICS = [
    'Gen Perplexity',
    'Gen Max Window Entropy',
    'Gen Min Prob',
    'Gen Cosine Similarity',
    'Judge Perplexity',
    'Judge Max Window Entropy',
    'Judge Min Prob',
    'Judge Cosine Similarity',
    'Aggregate Score'
]

def analyze_data_loss(name, df_generated):
    """Analyze where data is lost during cleaning."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {name}")
    print(f"{'='*80}")
    
    initial_count = len(df_generated)
    print(f"\n1️⃣  Initial: {initial_count} generated edges (TP + FP)")
    
    # Check for missing CI metrics
    X = df_generated[CI_METRICS].copy()
    
    # Count NaN per column
    print(f"\n2️⃣  Missing Values (NaN) per metric:")
    nan_counts = X.isnull().sum()
    for metric, count in nan_counts.items():
        pct = (count / len(X) * 100)
        if count > 0:
            print(f"     {metric:30s}: {count:4d} ({pct:5.1f}%)")
    
    total_with_nan = X.isnull().any(axis=1).sum()
    print(f"\n   📊 Total rows with ANY NaN: {total_with_nan} ({total_with_nan/len(X)*100:.1f}%)")
    
    # Remove NaN
    mask_complete = ~X.isnull().any(axis=1)
    X_no_nan = X[mask_complete]
    after_nan = len(X_no_nan)
    print(f"   ✓ After removing NaN: {after_nan} samples (lost {initial_count - after_nan})")
    
    if after_nan == 0:
        print("\n   ❌ No samples left after removing NaN!")
        return
    
    # Check for inf values
    print(f"\n3️⃣  Infinite Values (inf) per metric:")
    inf_counts = {}
    for col in X_no_nan.columns:
        inf_count = (~np.isfinite(X_no_nan[col])).sum()
        if inf_count > 0:
            pct = (inf_count / len(X_no_nan) * 100)
            print(f"     {col:30s}: {inf_count:4d} ({pct:5.1f}%)")
            inf_counts[col] = inf_count
    
    if not inf_counts:
        print(f"     (None)")
    
    total_with_inf = (~np.isfinite(X_no_nan).all(axis=1)).sum()
    print(f"\n   📊 Total rows with ANY inf: {total_with_inf} ({total_with_inf/len(X_no_nan)*100:.1f}%)")
    
    # Remove inf
    mask_finite = np.isfinite(X_no_nan).all(axis=1)
    X_no_inf = X_no_nan[mask_finite]
    after_inf = len(X_no_inf)
    print(f"   ✓ After removing inf: {after_inf} samples (lost {after_nan - after_inf})")
    
    if after_inf == 0:
        print("\n   ❌ No samples left after removing inf!")
        return
    
    # Check extreme perplexity values
    print(f"\n4️⃣  Extreme Perplexity Values (>= 10):")
    perplexity_cols = [col for col in X_no_inf.columns if 'Perplexity' in col]
    extreme_perplexity_mask = np.zeros(len(X_no_inf), dtype=bool)
    
    for col in perplexity_cols:
        extreme_count = (X_no_inf[col] >= 10).sum()
        if extreme_count > 0:
            pct = (extreme_count / len(X_no_inf) * 100)
            print(f"     {col:30s}: {extreme_count:4d} ({pct:5.1f}%)")
            print(f"       Max value: {X_no_inf[col].max():.4f}")
            extreme_perplexity_mask |= (X_no_inf[col] >= 10)
    
    total_with_extreme = extreme_perplexity_mask.sum()
    print(f"\n   📊 Total rows with ANY extreme perplexity: {total_with_extreme} ({total_with_extreme/len(X_no_inf)*100:.1f}%)")
    
    # Remove extreme perplexity
    X_clean = X_no_inf[~extreme_perplexity_mask]
    after_perplexity = len(X_clean)
    print(f"   ✓ After removing extreme perplexity: {after_perplexity} samples (lost {after_inf - after_perplexity})")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY FOR {name}")
    print(f"{'='*80}")
    print(f"  Initial:                {initial_count:4d} samples")
    print(f"  After removing NaN:     {after_nan:4d} samples (lost {initial_count - after_nan:4d}, {(initial_count - after_nan)/initial_count*100:5.1f}%)")
    print(f"  After removing inf:     {after_inf:4d} samples (lost {after_nan - after_inf:4d}, {(after_nan - after_inf)/after_nan*100 if after_nan > 0 else 0:5.1f}%)")
    print(f"  After removing extreme: {after_perplexity:4d} samples (lost {after_inf - after_perplexity:4d}, {(after_inf - after_perplexity)/after_inf*100 if after_inf > 0 else 0:5.1f}%)")
    print(f"  {'='*30}")
    print(f"  FINAL RETENTION:        {after_perplexity:4d} / {initial_count:4d} = {after_perplexity/initial_count*100:5.1f}%")
    
    return {
        'initial': initial_count,
        'after_nan': after_nan,
        'after_inf': after_inf,
        'after_perplexity': after_perplexity,
        'retention': after_perplexity / initial_count * 100
    }

def main():
    print("=" * 80)
    print("DATA LOSS ANALYSIS: Why are so many samples thrown away?")
    print("=" * 80)
    
    # Analyze original data
    print("\n\n" + "#" * 80)
    print("# ORIGINAL DATA (6 CLDs)")
    print("#" * 80)
    
    all_orig_dfs = []
    for name, filepath in ORIGINAL_FILES.items():
        try:
            df = pd.read_excel(filepath, sheet_name="All Edges")
            df_generated = df[df['Classification'].isin(['TP', 'FP'])].copy()
            all_orig_dfs.append(df_generated)
            analyze_data_loss(name, df_generated)
        except Exception as e:
            print(f"\n⚠️  Could not load {name}: {e}")
    
    # Combined original analysis
    if all_orig_dfs:
        combined_orig = pd.concat(all_orig_dfs, ignore_index=True)
        analyze_data_loss("COMBINED ORIGINAL (6 CLDs)", combined_orig)
    
    # Analyze recent data
    print("\n\n" + "#" * 80)
    print("# RECENT DATA (3 CLDs)")
    print("#" * 80)
    
    all_recent_dfs = []
    for filepath in RECENT_FILES:
        try:
            df = pd.read_excel(filepath, sheet_name="All Edges")
            df_generated = df[df['Classification'].isin(['TP', 'FP'])].copy()
            all_recent_dfs.append(df_generated)
            analyze_data_loss(filepath.name, df_generated)
        except Exception as e:
            print(f"\n⚠️  Could not load {filepath.name}: {e}")
    
    # Combined recent analysis
    if all_recent_dfs:
        combined_recent = pd.concat(all_recent_dfs, ignore_index=True)
        analyze_data_loss("COMBINED RECENT (3 CLDs)", combined_recent)
    
    # Final comparison
    print("\n\n" + "#" * 80)
    print("# COMPARISON: ORIGINAL vs RECENT")
    print("#" * 80)
    
    print("\nOriginal: 407 → 136 (33.4% retention)")
    print("Recent:   223 → 200 (89.7% retention)")
    print("\nKey question: Which metrics are missing in original data?")

if __name__ == "__main__":
    main()

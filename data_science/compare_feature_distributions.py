#!/usr/bin/env python3
"""
Compare Feature Distributions: Original (0.7250) vs Recent (0.5090)

Investigate why the same task on different data yields such different AUC.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Original data (0.7250 AUC)
ORIGINAL_FILES = {
    'Depressive_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run1_7a66e382/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251008_181257.xlsx',
    'Depressive_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run2_0d57ed9a/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251008_200816.xlsx',
    'Social_norms_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run1_86b016e1/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Social_norms_and_obesity_prevalence_20251008_213554.xlsx',
    'Social_norms_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run2_d09234b7/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Social_norms_and_obesity_prevalence_20251008_221139.xlsx',
    'Older_persons_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run1_d45241a5/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251008_230700.xlsx',
    'Older_persons_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run2_96605199/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251009_082149.xlsx'
}

# Recent data (0.5090 AUC)
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

def load_original_data():
    """Load original data (0.7250 AUC)."""
    all_dfs = []
    for name, filepath in ORIGINAL_FILES.items():
        try:
            df = pd.read_excel(filepath, sheet_name="All Edges")
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️  Could not load {name}: {e}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    # Filter to generated edges only (TP + FP)
    df_generated = combined[combined['Classification'].isin(['TP', 'FP'])].copy()
    df_generated['is_hallucination'] = (df_generated['Classification'] == 'FP').astype(int)
    return df_generated

def load_recent_data():
    """Load recent data (0.5090 AUC)."""
    all_dfs = []
    for filepath in RECENT_FILES:
        df = pd.read_excel(filepath, sheet_name="All Edges")
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    # Filter to generated edges only (TP + FP)
    df_generated = combined[combined['Classification'].isin(['TP', 'FP'])].copy()
    df_generated['is_hallucination'] = (df_generated['Classification'] == 'FP').astype(int)
    return df_generated

def clean_data(df):
    """Apply same cleaning as classifier scripts."""
    # Filter to samples with all metrics
    X = df[CI_METRICS].copy()
    y = df['is_hallucination'].copy()
    
    # Remove rows with NaN
    mask_complete = ~X.isnull().any(axis=1)
    X = X[mask_complete]
    y = y[mask_complete]
    
    # Remove inf values
    mask_finite = np.isfinite(X).all(axis=1)
    X = X[mask_finite]
    y = y[mask_finite]
    
    # Remove extreme perplexity values
    perplexity_cols = [col for col in X.columns if 'Perplexity' in col]
    for col in perplexity_cols:
        mask_reasonable = X[col] < 10
        X = X[mask_reasonable]
        y = y[mask_reasonable]
    
    return X, y

def compare_distributions(X_orig, y_orig, X_recent, y_recent):
    """Compare feature distributions between datasets."""
    print("=" * 80)
    print("FEATURE DISTRIBUTION COMPARISON")
    print("=" * 80)
    
    print(f"\nDataset sizes:")
    print(f"  Original: {len(X_orig)} samples")
    print(f"  Recent: {len(X_recent)} samples")
    
    print(f"\nClass balance:")
    print(f"  Original: {y_orig.mean()*100:.1f}% hallucinations")
    print(f"  Recent: {y_recent.mean()*100:.1f}% hallucinations")
    
    print(f"\n{'Metric':<30s} | {'Original Mean':<15s} | {'Recent Mean':<15s} | {'Diff %':<10s}")
    print("-" * 80)
    
    for metric in CI_METRICS:
        orig_mean = X_orig[metric].mean()
        recent_mean = X_recent[metric].mean()
        diff_pct = ((recent_mean - orig_mean) / orig_mean * 100) if orig_mean != 0 else 0
        
        print(f"{metric:<30s} | {orig_mean:>15.4f} | {recent_mean:>15.4f} | {diff_pct:>+9.1f}%")
    
    print("\n" + "=" * 80)
    print("FEATURE RANGES")
    print("=" * 80)
    
    print(f"\n{'Metric':<30s} | {'Original Range':<25s} | {'Recent Range':<25s}")
    print("-" * 85)
    
    for metric in CI_METRICS:
        orig_min, orig_max = X_orig[metric].min(), X_orig[metric].max()
        recent_min, recent_max = X_recent[metric].min(), X_recent[metric].max()
        
        print(f"{metric:<30s} | [{orig_min:7.4f}, {orig_max:7.4f}] | [{recent_min:7.4f}, {recent_max:7.4f}]")
    
    # Analyze separation power
    print("\n" + "=" * 80)
    print("FEATURE SEPARATION POWER (TP vs FP)")
    print("=" * 80)
    
    print(f"\n{'Metric':<30s} | {'Orig Δ Mean':<12s} | {'Recent Δ Mean':<12s} | {'Change':<10s}")
    print("-" * 80)
    
    for metric in CI_METRICS:
        # Original separation
        orig_tp_mean = X_orig.loc[y_orig == 0, metric].mean()
        orig_fp_mean = X_orig.loc[y_orig == 1, metric].mean()
        orig_delta = abs(orig_fp_mean - orig_tp_mean)
        
        # Recent separation
        recent_tp_mean = X_recent.loc[y_recent == 0, metric].mean()
        recent_fp_mean = X_recent.loc[y_recent == 1, metric].mean()
        recent_delta = abs(recent_fp_mean - recent_tp_mean)
        
        change_pct = ((recent_delta - orig_delta) / orig_delta * 100) if orig_delta != 0 else 0
        
        print(f"{metric:<30s} | {orig_delta:>12.4f} | {recent_delta:>12.4f} | {change_pct:>+9.1f}%")
    
    # Check if features are discriminative
    print("\n" + "=" * 80)
    print("DISCRIMINATIVE POWER (Absolute mean difference / std)")
    print("=" * 80)
    
    print(f"\n{'Metric':<30s} | {'Original':<12s} | {'Recent':<12s}")
    print("-" * 60)
    
    for metric in CI_METRICS:
        # Original
        orig_tp_mean = X_orig.loc[y_orig == 0, metric].mean()
        orig_fp_mean = X_orig.loc[y_orig == 1, metric].mean()
        orig_std = X_orig[metric].std()
        orig_disc = abs(orig_fp_mean - orig_tp_mean) / orig_std if orig_std > 0 else 0
        
        # Recent
        recent_tp_mean = X_recent.loc[y_recent == 0, metric].mean()
        recent_fp_mean = X_recent.loc[y_recent == 1, metric].mean()
        recent_std = X_recent[metric].std()
        recent_disc = abs(recent_fp_mean - recent_tp_mean) / recent_std if recent_std > 0 else 0
        
        print(f"{metric:<30s} | {orig_disc:>12.4f} | {recent_disc:>12.4f}")

def main():
    print("=" * 80)
    print("INVESTIGATING WHY 0.7250 AUC COULD NOT BE REPRODUCED")
    print("=" * 80)
    
    print("\nLoading original data (0.7250 AUC)...")
    df_orig = load_original_data()
    print(f"✓ Loaded {len(df_orig)} generated edges")
    
    print("\nLoading recent data (0.5090 AUC)...")
    df_recent = load_recent_data()
    print(f"✓ Loaded {len(df_recent)} generated edges")
    
    print("\nCleaning data...")
    X_orig, y_orig = clean_data(df_orig)
    X_recent, y_recent = clean_data(df_recent)
    print(f"✓ Original: {len(X_orig)} clean samples")
    print(f"✓ Recent: {len(X_recent)} clean samples")
    
    compare_distributions(X_orig, y_orig, X_recent, y_recent)

if __name__ == "__main__":
    main()

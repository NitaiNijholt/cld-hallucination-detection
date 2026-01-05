#!/usr/bin/env python3
"""
Analyze RQ1 Multi-CLD Experiment Results
"""

import pandas as pd
from pathlib import Path
import json

# Base directory
base_dir = Path("parameter_tuning_experiments/results/rq1_multi_cld_20251105_205040")

# Load summary
with open(base_dir / "experiment_summary.json", 'r') as f:
    summary = json.load(f)

print("="*80)
print("RQ1 MULTI-CLD EXPERIMENT RESULTS")
print("="*80)
print(f"Corruption Rate: {summary['corruption_rate']}")
print(f"Generator Model: {summary['generator_model']}")
print(f"Corrector Model: {summary['corrector_model']}")
print()

results_table = []

for cld_result in summary['results']:
    cld_name = cld_result['cld_name']
    output_dir = Path(cld_result['output_dir'])
    
    print(f"\n{'='*80}")
    print(f"CLD: {cld_name}")
    print(f"{'='*80}")
    
    # Find the Excel files
    corrupted_xlsx = list(output_dir.glob("corrupted_*.xlsx"))[0]
    judged_no_corr_xlsx = list(output_dir.glob("judged_no_correction_*.xlsx"))[0]
    corrected_xlsx = list(output_dir.glob("corrected_*.xlsx"))[0]
    
    # Read "All Edges Summary" sheet from each
    print("\n📊 METRICS COMPARISON (All Edges):")
    print("-" * 80)
    
    # Corrupted (baseline)
    df_corrupted = pd.read_excel(corrupted_xlsx, sheet_name="All Edges Summary")
    corrupted_metrics = df_corrupted.set_index('Metric')['Value'].to_dict()
    
    # Judged without correction
    df_judged = pd.read_excel(judged_no_corr_xlsx, sheet_name="All Edges Summary")
    judged_metrics = df_judged.set_index('Metric')['Value'].to_dict()
    
    # Corrected
    df_corrected = pd.read_excel(corrected_xlsx, sheet_name="All Edges Summary")
    corrected_metrics = df_corrected.set_index('Metric')['Value'].to_dict()
    
    # Extract key metrics
    print(f"{'Metric':<30} {'Corrupted':<15} {'Judged Only':<15} {'Corrected':<15} {'Improvement'}")
    print("-" * 95)
    
    for metric in ['Precision', 'Recall', 'F1 Score']:
        corr_val = corrupted_metrics.get(metric, 0.0)
        judg_val = judged_metrics.get(metric, 0.0)
        corr2_val = corrected_metrics.get(metric, 0.0)
        improvement = corr2_val - corr_val
        
        print(f"{metric:<30} {corr_val:<15.3f} {judg_val:<15.3f} {corr2_val:<15.3f} {improvement:+.3f}")
    
    print()
    print(f"{'Metric':<30} {'Corrupted':<15} {'Judged Only':<15} {'Corrected':<15}")
    print("-" * 80)
    
    for metric in ['True Positives (TP)', 'False Positives (FP)', 'False Negatives (FN)']:
        corr_val = corrupted_metrics.get(metric, 0)
        judg_val = judged_metrics.get(metric, 0)
        corr2_val = corrected_metrics.get(metric, 0)
        
        print(f"{metric:<30} {int(corr_val):<15} {int(judg_val):<15} {int(corr2_val):<15}")
    
    # Store for aggregate table
    results_table.append({
        'CLD': cld_name,
        'Corrupted_F1': corrupted_metrics.get('F1 Score', 0.0),
        'Judged_F1': judged_metrics.get('F1 Score', 0.0),
        'Corrected_F1': corrected_metrics.get('F1 Score', 0.0),
        'Improvement': corrected_metrics.get('F1 Score', 0.0) - corrupted_metrics.get('F1 Score', 0.0),
        'Corrupted_Precision': corrupted_metrics.get('Precision', 0.0),
        'Corrected_Precision': corrected_metrics.get('Precision', 0.0),
        'Corrupted_Recall': corrupted_metrics.get('Recall', 0.0),
        'Corrected_Recall': corrected_metrics.get('Recall', 0.0),
    })

# Summary table
print(f"\n{'='*80}")
print("AGGREGATE RESULTS")
print(f"{'='*80}\n")

results_df = pd.DataFrame(results_table)
print("F1 Score Comparison:")
print(results_df[['CLD', 'Corrupted_F1', 'Judged_F1', 'Corrected_F1', 'Improvement']].to_string(index=False))

print("\n\nPrecision Comparison:")
print(results_df[['CLD', 'Corrupted_Precision', 'Corrected_Precision']].to_string(index=False))

print("\n\nRecall Comparison:")
print(results_df[['CLD', 'Corrupted_Recall', 'Corrected_Recall']].to_string(index=False))

print("\n\nAverage Metrics:")
print(f"  Avg Corrupted F1:  {results_df['Corrupted_F1'].mean():.3f}")
print(f"  Avg Judged F1:     {results_df['Judged_F1'].mean():.3f}")
print(f"  Avg Corrected F1:  {results_df['Corrected_F1'].mean():.3f}")
print(f"  Avg Improvement:   {results_df['Improvement'].mean():+.3f}")

print(f"\n{'='*80}")
print("✅ ANALYSIS COMPLETE")
print(f"{'='*80}\n")




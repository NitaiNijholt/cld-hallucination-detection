#!/usr/bin/env python3
"""
RQ1a: Can the judge identify corrupted edges?
Test correlation between corruption status and judge verdicts.
"""

import pandas as pd
from pathlib import Path
import json
import numpy as np

# Base directory
base_dir = Path("parameter_tuning_experiments/results/rq1_multi_cld_20251105_205040")

# Load summary
with open(base_dir / "experiment_summary.json", 'r') as f:
    summary = json.load(f)

print("="*80)
print("RQ1a: CAN THE JUDGE IDENTIFY CORRUPTED EDGES?")
print("="*80)
print("Hypothesis: Judge verdicts should correlate with corruption status")
print("  - Corrupted edges → INCORRECT verdicts")
print("  - Non-corrupted edges → CORRECT verdicts")
print()

all_results = []

for cld_result in summary['results']:
    cld_name = cld_result['cld_name']
    output_dir = Path(cld_result['output_dir'])
    
    print(f"\n{'='*80}")
    print(f"CLD: {cld_name}")
    print(f"{'='*80}")
    
    # Load judged_no_correction data (has both corruption status and judge verdicts)
    judged_xlsx = list(output_dir.glob("judged_no_correction_*.xlsx"))[0]
    df = pd.read_excel(judged_xlsx, sheet_name="All Edges")
    
    # Check what columns we have
    print(f"\nColumns available: {df.columns.tolist()}")
    
    # Filter to causal edges only
    df = df[df['Relationship Type'] != 'NONE'].copy()
    
    print(f"\nTotal causal edges analyzed: {len(df)}")
    
    # Check if we have corruption status
    if 'Is Corrupted' in df.columns:
        print(f"\n✅ 'Is Corrupted' column found!")
        
        # Analyze corruption status
        corruption_status = df['Is Corrupted'].value_counts()
        print(f"\nCorruption Status Distribution:")
        print(corruption_status)
        
        # Analyze judge verdicts by corruption status
        print(f"\n📊 JUDGE VERDICTS BY CORRUPTION STATUS:")
        print("-" * 80)
        
        crosstab = pd.crosstab(
            df['Is Corrupted'], 
            df['Judge Verdict'], 
            margins=True,
            normalize='index'
        )
        print(crosstab)
        
        # Detailed breakdown
        print(f"\n📈 DETAILED BREAKDOWN:")
        for is_corrupted in [True, False]:
            subset = df[df['Is Corrupted'] == is_corrupted]
            if len(subset) == 0:
                continue
                
            label = "CORRUPTED" if is_corrupted else "NON-CORRUPTED"
            print(f"\n{label} Edges (n={len(subset)}):")
            
            verdict_counts = subset['Judge Verdict'].value_counts()
            for verdict, count in verdict_counts.items():
                pct = 100 * count / len(subset)
                print(f"  {verdict:20s}: {count:3d} ({pct:5.1f}%)")
        
        # Create binary judge verdict
        # Simplify judge verdicts: CORRECT/PARTIALLY_CORRECT → Accept, INCORRECT/ERROR → Reject
        df['judge_binary'] = df['Judge Verdict'].apply(
            lambda x: 'Accept' if x in ['CORRECT', 'PARTIALLY_CORRECT'] else 'Reject'
        )
        
        print(f"\n📊 CONTINGENCY TABLE:")
        contingency = pd.crosstab(df['Is Corrupted'], df['judge_binary'], margins=True)
        print(contingency)
        
        # Compute correlation coefficient (point-biserial)
        df['corrupted_numeric'] = df['Is Corrupted'].astype(int)
        df['judge_numeric'] = df['judge_binary'].apply(lambda x: 1 if x == 'Accept' else 0)
        
        # Note: We want corrupted → low judge score (reject)
        # So negative correlation is good
        correlation = df['corrupted_numeric'].corr(df['judge_numeric'])
        
        print(f"\n📊 CORRELATION ANALYSIS:")
        print(f"  Point-biserial correlation: {correlation:.4f}")
        
        if correlation < -0.1:
            print(f"  ✅ Negative correlation (corrupted → reject)")
        elif correlation > 0.1:
            print(f"  ❌ Positive correlation (corrupted → accept) - BAD!")
        else:
            print(f"  ⚠️  Weak correlation")
        
        # Compute accuracy metrics
        print(f"\n📊 JUDGE ACCURACY METRICS:")
        
        # True Positive: Corrupted edge → Reject
        tp = len(df[(df['Is Corrupted'] == True) & (df['judge_binary'] == 'Reject')])
        # False Negative: Corrupted edge → Accept
        fn = len(df[(df['Is Corrupted'] == True) & (df['judge_binary'] == 'Accept')])
        # True Negative: Non-corrupted edge → Accept
        tn = len(df[(df['Is Corrupted'] == False) & (df['judge_binary'] == 'Accept')])
        # False Positive: Non-corrupted edge → Reject
        fp = len(df[(df['Is Corrupted'] == False) & (df['judge_binary'] == 'Reject')])
        
        accuracy = (tp + tn) / len(df)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Accuracy:     {accuracy:.3f}")
        print(f"  Precision:    {precision:.3f} (of rejected edges, % truly corrupted)")
        print(f"  Recall:       {recall:.3f} (of corrupted edges, % detected)")
        print(f"  Specificity:  {specificity:.3f} (of non-corrupted edges, % accepted)")
        print(f"  F1 Score:     {f1:.3f}")
        
        # Store results
        all_results.append({
            'CLD': cld_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1': f1,
            'Correlation': correlation
        })
        
    else:
        print(f"\n❌ 'Is Corrupted' column NOT found")
        print(f"   Available: {df.columns.tolist()}")

# Summary
if all_results:
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS")
    print(f"{'='*80}\n")
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("RQ1a CONCLUSION")
    print(f"{'='*80}")
    
    avg_accuracy = results_df['Accuracy'].mean()
    avg_f1 = results_df['F1'].mean()
    avg_correlation = results_df['Correlation'].mean()
    
    print(f"\nAverage Judge Accuracy: {avg_accuracy:.3f}")
    print(f"Average F1 Score: {avg_f1:.3f}")
    print(f"Average Correlation: {avg_correlation:.3f}")
    
    if avg_accuracy > 0.7 and avg_correlation < -0.3:
        print(f"\n✅ STRONG EVIDENCE: Judge can reliably identify corrupted edges")
    elif avg_accuracy > 0.5 and avg_correlation < -0.1:
        print(f"\n⚠️  MODERATE EVIDENCE: Judge has some ability but not highly accurate")
    else:
        print(f"\n❌ WEAK EVIDENCE: Judge struggles to identify corrupted edges")

print(f"\n{'='*80}")


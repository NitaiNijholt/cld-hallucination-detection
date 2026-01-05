#!/usr/bin/env python3
"""
Analyze Judge Accuracy vs Synthetic Ground Truth
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
print("JUDGE ACCURACY ANALYSIS (vs Synthetic Ground Truth)")
print("="*80)
print(f"Question: Can the judge correctly identify which edges are correct/incorrect?")
print()

for cld_result in summary['results']:
    cld_name = cld_result['cld_name']
    output_dir = Path(cld_result['output_dir'])
    
    print(f"\n{'='*80}")
    print(f"CLD: {cld_name}")
    print(f"{'='*80}")
    
    # Load corrupted data (has ground truth labels)
    corrupted_xlsx = list(output_dir.glob("corrupted_*.xlsx"))[0]
    df_corrupted = pd.read_excel(corrupted_xlsx, sheet_name="All Edges")
    
    # Load judged_no_correction data (has judge verdicts)
    judged_xlsx = list(output_dir.glob("judged_no_correction_*.xlsx"))[0]
    df_judged = pd.read_excel(judged_xlsx, sheet_name="All Edges")
    
    # Filter to causal edges only (exclude NONE)
    df_corrupted = df_corrupted[df_corrupted['Relationship Type'] != 'NONE'].copy()
    df_judged = df_judged[df_judged['Relationship Type'] != 'NONE'].copy()
    
    print(f"\nTotal causal edges: {len(df_judged)}")
    
    # Ground truth: Classification column tells us TP/FP
    # TP = edge is in ground truth (correct)
    # FP = edge is NOT in ground truth (incorrect/spurious)
    df_judged['ground_truth'] = df_judged['Classification'].apply(
        lambda x: 'CORRECT' if x == 'TP' else 'INCORRECT'
    )
    
    # Judge verdict: CORRECT, PARTIALLY_CORRECT, or INCORRECT
    judge_verdicts = df_judged['Judge Verdict'].value_counts()
    print(f"\nJudge Verdicts Distribution:")
    print(judge_verdicts)
    
    # Treat PARTIALLY_CORRECT as a separate category first
    print(f"\n📊 DETAILED JUDGE ANALYSIS:")
    print(f"{'Ground Truth':<20} {'Judge: CORRECT':<20} {'Judge: PARTIAL':<20} {'Judge: INCORRECT':<20}")
    print("-" * 80)
    
    for gt in ['CORRECT', 'INCORRECT']:
        correct_count = len(df_judged[(df_judged['ground_truth'] == gt) & 
                                      (df_judged['Judge Verdict'] == 'CORRECT')])
        partial_count = len(df_judged[(df_judged['ground_truth'] == gt) & 
                                      (df_judged['Judge Verdict'] == 'PARTIALLY_CORRECT')])
        incorrect_count = len(df_judged[(df_judged['ground_truth'] == gt) & 
                                        (df_judged['Judge Verdict'].isin(['INCORRECT', 'ERROR']))])
        
        print(f"{gt:<20} {correct_count:<20} {partial_count:<20} {incorrect_count:<20}")
    
    # Now calculate metrics treating PARTIALLY_CORRECT as CORRECT
    print(f"\n📊 JUDGE CONFUSION MATRIX (treating PARTIALLY_CORRECT as CORRECT):")
    judge_correct_correct = len(df_judged[(df_judged['ground_truth'] == 'CORRECT') & 
                                           (df_judged['Judge Verdict'].isin(['CORRECT', 'PARTIALLY_CORRECT']))])
    judge_correct_incorrect = len(df_judged[(df_judged['ground_truth'] == 'CORRECT') & 
                                            (df_judged['Judge Verdict'].isin(['INCORRECT', 'ERROR']))])
    judge_incorrect_correct = len(df_judged[(df_judged['ground_truth'] == 'INCORRECT') & 
                                            (df_judged['Judge Verdict'].isin(['CORRECT', 'PARTIALLY_CORRECT']))])
    judge_incorrect_incorrect = len(df_judged[(df_judged['ground_truth'] == 'INCORRECT') & 
                                              (df_judged['Judge Verdict'].isin(['INCORRECT', 'ERROR']))])
    
    print(f"{'':20} {'Ground Truth CORRECT':25} {'Ground Truth INCORRECT':25}")
    print("-" * 70)
    print(f"{'Judge: ACCEPT':20} {judge_correct_correct:^25} {judge_incorrect_correct:^25}")
    print(f"{'Judge: REJECT':20} {judge_correct_incorrect:^25} {judge_incorrect_incorrect:^25}")
    
    # Calculate judge accuracy metrics
    total = len(df_judged)
    judge_accuracy = (judge_correct_correct + judge_incorrect_incorrect) / total
    
    # Precision: Of edges judge said are correct, how many really are?
    judge_precision = judge_correct_correct / (judge_correct_correct + judge_incorrect_correct) if (judge_correct_correct + judge_incorrect_correct) > 0 else 0
    
    # Recall: Of edges that are really correct, how many did judge identify?
    judge_recall = judge_correct_correct / (judge_correct_correct + judge_correct_incorrect) if (judge_correct_correct + judge_correct_incorrect) > 0 else 0
    
    # F1 Score
    judge_f1 = 2 * (judge_precision * judge_recall) / (judge_precision + judge_recall) if (judge_precision + judge_recall) > 0 else 0
    
    # Specificity: Of edges that are really incorrect, how many did judge identify?
    judge_specificity = judge_incorrect_incorrect / (judge_incorrect_incorrect + judge_incorrect_correct) if (judge_incorrect_incorrect + judge_incorrect_correct) > 0 else 0
    
    print(f"\n📈 JUDGE PERFORMANCE METRICS:")
    print(f"  Accuracy:     {judge_accuracy:.3f} ({judge_correct_correct + judge_incorrect_incorrect}/{total} correct)")
    print(f"  Precision:    {judge_precision:.3f} (of edges judged correct, % truly correct)")
    print(f"  Recall:       {judge_recall:.3f} (of truly correct edges, % identified by judge)")
    print(f"  Specificity:  {judge_specificity:.3f} (of truly incorrect edges, % identified by judge)")
    print(f"  F1 Score:     {judge_f1:.3f}")
    
    # Compare to baseline (random guessing)
    ground_truth_correct = judge_correct_correct + judge_correct_incorrect
    ground_truth_incorrect = judge_incorrect_correct + judge_incorrect_incorrect
    baseline_accuracy = max(ground_truth_correct, ground_truth_incorrect) / total
    
    print(f"\n  Baseline (majority class): {baseline_accuracy:.3f}")
    print(f"  Improvement over baseline: {judge_accuracy - baseline_accuracy:+.3f}")

print(f"\n{'='*80}")
print("KEY INSIGHT:")
print("="*80)
print("Judge accuracy shows how well the judge can distinguish correct edges")
print("from corrupted/spurious ones WITHOUT making any corrections.")
print("="*80)


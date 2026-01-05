#!/usr/bin/env python3
"""
Compare RQ2 correlation results: Ground Truth CLD vs Judge Verdict
"""
import pandas as pd
import sys

# Load both result sets
gt_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/tables/rq2_correlations.csv"
judge_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/tables/rq2_correlations_judge_ground_truth.csv"

print("="*80)
print("RQ2 CORRELATION ANALYSIS COMPARISON")
print("="*80)
print("")

try:
    df_gt = pd.read_csv(gt_file)
    df_judge = pd.read_csv(judge_file)
    
    print("📊 Ground Truth Source Comparison:")
    print("   1. CLD Ground Truth (Classification == 'FP')")
    print("      - 79/90 hallucinations (87.8%)")
    print("")
    print("   2. Judge Verdict (Aggregate Score < 0.5)")
    print("      - 68/90 hallucinations (75.6%)")
    print("")
    print("="*80)
    print("")
    
    # Merge results
    df_gt = df_gt.rename(columns={
        'pearson_r': 'gt_pearson_r',
        'pearson_p': 'gt_pearson_p',
        'spearman_r': 'gt_spearman_r',
        'spearman_p': 'gt_spearman_p',
        'significant': 'gt_significant'
    })
    
    df_judge = df_judge.rename(columns={
        'pearson_r': 'judge_pearson_r',
        'pearson_p': 'judge_pearson_p',
        'spearman_r': 'judge_spearman_r',
        'spearman_p': 'judge_spearman_p',
        'significant': 'judge_significant'
    })
    
    # Merge on metric
    comparison = pd.merge(
        df_gt[['metric', 'gt_pearson_r', 'gt_pearson_p', 'gt_significant']],
        df_judge[['metric', 'judge_pearson_r', 'judge_pearson_p', 'judge_significant']],
        on='metric',
        how='outer'
    )
    
    print("CORRELATION COMPARISON (Pearson's r):")
    print("")
    print(f"{'Metric':<28} {'GT CLD':<12} {'p-value':<10} {'Judge':<12} {'p-value':<10} {'Sig?':<8}")
    print("-" * 80)
    
    for _, row in comparison.iterrows():
        metric = row['metric']
        gt_r = row['gt_pearson_r']
        gt_p = row['gt_pearson_p']
        judge_r = row['judge_pearson_r']
        judge_p = row['judge_pearson_p']
        
        # Determine significance
        gt_sig = "✓" if row['gt_significant'] else ""
        judge_sig = "✓" if row['judge_significant'] else ""
        sig_str = f"{gt_sig:^4}|{judge_sig:^4}"
        
        print(f"{metric:<28} {gt_r:>11.3f} {gt_p:>9.4f}  {judge_r:>11.3f} {judge_p:>9.4f}  {sig_str:<8}")
    
    print("")
    print("="*80)
    print("KEY FINDINGS:")
    print("="*80)
    print("")
    
    # Identify significant correlations
    sig_gt = comparison[comparison['gt_significant'] == True]
    sig_judge = comparison[comparison['judge_significant'] == True]
    
    if len(sig_gt) == 0:
        print("❌ Ground Truth CLD:")
        print("   NO significant correlations found (all p > 0.05)")
        print("")
    else:
        print("✅ Ground Truth CLD - Significant correlations:")
        for _, row in sig_gt.iterrows():
            print(f"   {row['metric']}: r = {row['gt_pearson_r']:.3f}, p = {row['gt_pearson_p']:.4f}")
        print("")
    
    if len(sig_judge) == 0:
        print("❌ Judge Verdict:")
        print("   NO significant correlations found (all p > 0.05)")
        print("")
    else:
        print("✅ Judge Verdict - Significant correlations:")
        for _, row in sig_judge.iterrows():
            metric = row['metric']
            r = row['judge_pearson_r']
            p = row['judge_pearson_p']
            
            direction = "positive" if r > 0 else "negative"
            strength = "strong" if abs(r) > 0.4 else "moderate" if abs(r) > 0.3 else "weak"
            
            print(f"   ✓ {metric}: r = {r:.3f}, p = {p:.6f}")
            print(f"      → {strength.capitalize()} {direction} correlation")
        print("")
    
    print("="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("")
    
    if len(sig_judge) > 0:
        print("✅ When using JUDGE VERDICT as ground truth:")
        print("   • Judge's own CI metrics (min_prob, max_window_entropy) strongly")
        print("     correlate with the judge's hallucination verdict!")
        print("   • This suggests the judge is internally consistent - its confidence")
        print("     metrics align with its final verdict.")
        print("")
        print("   Interpretation:")
        print("   • Judge Min Prob ↑ → Less likely to call it a hallucination")
        print("   • Judge Max Window Entropy ↑ → More likely to call it a hallucination")
        print("")
    
    if len(sig_gt) == 0:
        print("❌ When using CLD GROUND TRUTH:")
        print("   • NO CI metrics significantly correlate with actual hallucinations")
        print("   • This suggests CI metrics alone may not be sufficient to detect")
        print("     hallucinations when compared against ground truth.")
        print("")
    
    print("="*80)
    
    # Save comparison
    output_path = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/tables/rq2_comparison_gt_vs_judge.csv"
    comparison.to_csv(output_path, index=False)
    print(f"✅ Saved comparison to: {output_path}")
    print("="*80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
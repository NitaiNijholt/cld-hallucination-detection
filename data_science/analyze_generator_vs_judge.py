#!/usr/bin/env python3
"""
Analyze how generator CI metrics correlate with judge verdicts.
"""
import pandas as pd

# Load judge verdict correlations
judge_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/tables/rq2_correlations_judge_ground_truth.csv"
df = pd.read_csv(judge_file)

print("="*80)
print("GENERATOR CI METRICS vs JUDGE VERDICTS")
print("="*80)
print("")

# Separate generator and judge metrics
generator_metrics = df[~df['metric'].str.startswith('judge_')].copy()
judge_metrics = df[df['metric'].str.startswith('judge_')].copy()

print("📊 How do GENERATOR CI metrics correlate with judge verdicts?")
print("")
print(f"{'Metric':<28} {'Pearson r':<12} {'p-value':<12} {'Significant?':<12}")
print("-" * 80)

for _, row in generator_metrics.iterrows():
    metric = row['metric']
    r = row['pearson_r']
    p = row['pearson_p']
    sig = "✓ Yes" if row['significant'] else "✗ No"
    
    # Add interpretation
    if p < 0.1:
        interpretation = " (borderline)" if p >= 0.05 else " (significant)"
    else:
        interpretation = ""
    
    print(f"{metric:<28} {r:>11.3f} {p:>11.6f} {sig:<12}{interpretation}")

print("")
print("="*80)
print("COMPARISON: GENERATOR vs JUDGE CI METRICS")
print("="*80)
print("")

print("Generator CI Metrics (predicting judge verdict):")
print(f"   • Perplexity:        r = {generator_metrics[generator_metrics['metric']=='perplexity']['pearson_r'].values[0]:.3f}, p = {generator_metrics[generator_metrics['metric']=='perplexity']['pearson_p'].values[0]:.4f}")
print(f"   • Min Prob:          r = {generator_metrics[generator_metrics['metric']=='min_prob']['pearson_r'].values[0]:.3f}, p = {generator_metrics[generator_metrics['metric']=='min_prob']['pearson_p'].values[0]:.4f}")
print(f"   • Max Window Entropy: r = {generator_metrics[generator_metrics['metric']=='max_window_entropy']['pearson_r'].values[0]:.3f}, p = {generator_metrics[generator_metrics['metric']=='max_window_entropy']['pearson_p'].values[0]:.4f}")
print(f"   • Cosine Similarity: r = {generator_metrics[generator_metrics['metric']=='cosine_similarity']['pearson_r'].values[0]:.3f}, p = {generator_metrics[generator_metrics['metric']=='cosine_similarity']['pearson_p'].values[0]:.4f}")
print("")

print("Judge CI Metrics (predicting judge verdict):")
print(f"   • Judge Perplexity:  r = {judge_metrics[judge_metrics['metric']=='judge_perplexity']['pearson_r'].values[0]:.3f}, p = {judge_metrics[judge_metrics['metric']=='judge_perplexity']['pearson_p'].values[0]:.6f}")
print(f"   • Judge Min Prob:    r = {judge_metrics[judge_metrics['metric']=='judge_min_prob']['pearson_r'].values[0]:.3f}, p = {judge_metrics[judge_metrics['metric']=='judge_min_prob']['pearson_p'].values[0]:.6f} ✓✓")
print(f"   • Judge Max Window Entropy: r = {judge_metrics[judge_metrics['metric']=='judge_max_window_entropy']['pearson_r'].values[0]:.3f}, p = {judge_metrics[judge_metrics['metric']=='judge_max_window_entropy']['pearson_p'].values[0]:.6f} ✓✓")
print("")

print("="*80)
print("KEY FINDINGS:")
print("="*80)
print("")

# Count significant correlations
gen_sig_count = generator_metrics['significant'].sum()
judge_sig_count = judge_metrics['significant'].sum()

print(f"1. Generator CI Metrics:")
print(f"   • {gen_sig_count}/4 metrics significantly correlate with judge verdict")
print(f"   • Cosine Similarity shows borderline correlation (r = -0.208, p = 0.054)")
print(f"   • Effect sizes are weak (|r| < 0.21)")
print("")

print(f"2. Judge CI Metrics:")
print(f"   • {judge_sig_count}/3 metrics significantly correlate with judge verdict")
print(f"   • Strong correlations: Judge Min Prob (r = 0.341), Judge Max Window Entropy (r = -0.476)")
print(f"   • Effect sizes are moderate to strong (|r| > 0.3)")
print("")

print("="*80)
print("INTERPRETATION:")
print("="*80)
print("")

print("⚠️  Generator CI metrics are WEAK predictors of judge verdicts:")
print("")
print("   • Generator perplexity, min_prob, max_window_entropy:")
print("     → Do NOT significantly correlate with judge verdicts")
print("     → |r| < 0.15, p > 0.17")
print("")
print("   • Generator cosine similarity:")
print("     → Shows weak negative trend (r = -0.208)")
print("     → Borderline significance (p = 0.054)")
print("     → Lower similarity → Judge more likely to call hallucination")
print("")

print("✅ Judge CI metrics are STRONG predictors of judge verdicts:")
print("")
print("   • The judge's own confidence metrics strongly predict its verdict")
print("   • This suggests the judge is internally consistent")
print("   • Judge confidence → Judge verdict alignment")
print("")

print("="*80)
print("CONCLUSION:")
print("="*80)
print("")

print("The judge's verdict is primarily driven by ITS OWN confidence metrics,")
print("not the generator's CI metrics. This suggests:")
print("")
print("1. The judge evaluates edges independently of generator confidence")
print("2. Generator CI metrics don't transfer well to judge predictions")
print("3. Judge confidence is a better predictor of judge behavior than generator confidence")
print("")
print("Implication: Generator CI metrics alone are insufficient to predict")
print("whether the judge will accept or reject an edge.")
print("")
print("="*80)
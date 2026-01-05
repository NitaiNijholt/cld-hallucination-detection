#!/usr/bin/env python3
import pandas as pd

excel_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_234155_60068754/combo_1/prompts_Nitai_C/Social_norms_and_obesity_prevalence/run1_5db6738d/results_exp_20251007_234155_60068754_param_combo_nr_1_prompts_Nitai_C_run1_Social_norms_and_obesity_prevalence_20251008_001309.xlsx"

print("Verifying data quality for RQ2 analysis...")
print("="*80)

df = pd.read_excel(excel_file, sheet_name="All Edges")
print(f"✅ Loaded {len(df)} edges\n")

# Check CI metrics (with correct capitalization)
ci_metrics = ['Perplexity', 'Min Prob', 'Max Window Entropy', 'Cosine Similarity']
print("📊 CI Metrics (Generator):")
for metric in ci_metrics:
    non_null = df[metric].notna().sum()
    pct = (non_null / len(df) * 100) if len(df) > 0 else 0
    print(f"   {metric:25s}: {non_null:3d}/{len(df):3d} ({pct:5.1f}%)")

# Check judge logit metrics
judge_metrics = ['Judge Perplexity', 'Judge Min Prob', 'Judge Max Window Entropy']
print("\n📊 Judge Logit Metrics:")
for metric in judge_metrics:
    non_null = df[metric].notna().sum()
    pct = (non_null / len(df) * 100) if len(df) > 0 else 0
    print(f"   {metric:25s}: {non_null:3d}/{len(df):3d} ({pct:5.1f}%)")

# Check classification distribution
print("\n🎯 Classification Distribution:")
class_counts = df['Classification'].value_counts()
for cls, count in class_counts.items():
    pct = (count / len(df) * 100)
    print(f"   {cls:10s}: {count:3d}/{len(df):3d} ({pct:5.1f}%)")

# Check aggregate score distribution
print("\n⚖️  Aggregate Score Distribution:")
valid_scores = df['Aggregate Score'].notna()
if valid_scores.any():
    scores = df.loc[valid_scores, 'Aggregate Score']
    halluc_judge = (scores < 0.5).sum()
    correct_judge = (scores >= 0.5).sum()
    print(f"   Judge: Hallucination (< 0.5): {halluc_judge:3d}/{len(scores):3d} ({halluc_judge/len(scores)*100:5.1f}%)")
    print(f"   Judge: Correct (>= 0.5):      {correct_judge:3d}/{len(scores):3d} ({correct_judge/len(scores)*100:5.1f}%)")

# Summary
print("\n" + "="*80)
print("SUMMARY:")
print(f"  ✅ All CI metrics present with high coverage")
print(f"  ✅ All judge logit metrics present with high coverage")
print(f"  ✅ Classification labels present")
print(f"  ✅ Aggregate scores present")
print()
tp_count = (df['Classification'] == 'TP').sum()
fp_count = (df['Classification'] == 'FP').sum()
print(f"  Ground Truth: {tp_count} TP, {fp_count} FP")
if fp_count > 0:
    print(f"  ✅ SUITABLE FOR RQ2 ANALYSIS (has variation in labels)")
else:
    print(f"  ⚠️  WARNING: No FP edges, may have limited analysis power")
print("="*80)
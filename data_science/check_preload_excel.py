#!/usr/bin/env python3
import pandas as pd

excel_files = [
    "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_234410_10d61c06/combo_1/prompts_Nitai_C/Social_norms_and_obesity_prevalence/run1_ddce5c21/results_exp_20251007_234410_10d61c06_param_combo_nr_1_prompts_Nitai_C_run1_Social_norms_and_obesity_prevalence_20251008_001425.xlsx",
    "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_234155_60068754/combo_1/prompts_Nitai_C/Social_norms_and_obesity_prevalence/run1_5db6738d/results_exp_20251007_234155_60068754_param_combo_nr_1_prompts_Nitai_C_run1_Social_norms_and_obesity_prevalence_20251008_001309.xlsx"
]

for i, excel_file in enumerate(excel_files, 1):
    print("="*80)
    print(f"FILE {i}: ...{excel_file[-80:]}")
    print("="*80)
    print()
    
    df = pd.read_excel(excel_file, sheet_name="All Edges")
    print(f"✅ Loaded 'All Edges' sheet: {len(df)} edges")
    print()
    
    # Check for CI metrics
    ci_metrics = ['perplexity', 'min_prob', 'max_window_entropy', 'cosine_similarity']
    print("📊 CI Metrics (Generator):")
    for metric in ci_metrics:
        if metric in df.columns:
            non_null = df[metric].notna().sum()
            pct = (non_null / len(df) * 100) if len(df) > 0 else 0
            print(f"   {metric:25s}: {non_null:3d}/{len(df):3d} ({pct:5.1f}%)")
        else:
            print(f"   {metric:25s}: ❌ MISSING")
    
    # Check for judge logit metrics
    judge_metrics = ['judge_perplexity', 'judge_min_prob', 'judge_max_window_entropy']
    print()
    print("📊 Judge Logit Metrics:")
    for metric in judge_metrics:
        if metric in df.columns:
            non_null = df[metric].notna().sum()
            pct = (non_null / len(df) * 100) if len(df) > 0 else 0
            print(f"   {metric:25s}: {non_null:3d}/{len(df):3d} ({pct:5.1f}%)")
        else:
            print(f"   {metric:25s}: ❌ MISSING")
    
    # Check for classification (ground truth)
    if 'classification' in df.columns:
        print()
        print("🎯 Ground Truth Classification:")
        class_counts = df['classification'].value_counts()
        for cls, count in class_counts.items():
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            print(f"   {cls:10s}: {count:3d}/{len(df):3d} ({pct:5.1f}%)")
    else:
        print()
        print("⚠️  'classification' column not found")
    
    # Check for aggregate_score (judge verdict)
    if 'aggregate_score' in df.columns:
        print()
        print("⚖️  Judge Aggregate Score:")
        valid_scores = df['aggregate_score'].notna()
        if valid_scores.any():
            judge_halluc = ((df['aggregate_score'] < 0.5) & valid_scores).sum()
            judge_correct = ((df['aggregate_score'] >= 0.5) & valid_scores).sum()
            total_valid = valid_scores.sum()
            print(f"   Hallucination (< 0.5): {judge_halluc:3d}/{total_valid:3d} ({judge_halluc/total_valid*100:5.1f}%)")
            print(f"   Correct (>= 0.5):      {judge_correct:3d}/{total_valid:3d} ({judge_correct/total_valid*100:5.1f}%)")
        else:
            print("   ⚠️  No valid aggregate scores")
    else:
        print()
        print("⚠️  'aggregate_score' column not found")
    
    print()
    print()
#!/usr/bin/env python3
"""
Quick check of the preload experiment Excel files to verify all metrics are present.
"""
import pandas as pd
import glob

# Find the most recent Excel files from the preload experiments
excel_files = sorted(glob.glob(
    "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_202510072341*/combo_1/prompts_Nitai_C/Social_norms_and_obesity_prevalence/run1_*/results_*.xlsx"
), reverse=True)

print(f"Found {len(excel_files)} Excel files from preload experiments\n")

for i, excel_file in enumerate(excel_files[:2], 1):  # Check the 2 most recent
    print(f"{'='*80}")
    print(f"FILE {i}: {excel_file.split('/')[-1]}")
    print(f"{'='*80}\n")
    
    try:
        # Load the "All Edges" sheet
        df = pd.read_excel(excel_file, sheet_name="All Edges")
        print(f"✅ Loaded 'All Edges' sheet: {len(df)} edges\n")
        
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
        print("\n📊 Judge Logit Metrics:")
        for metric in judge_metrics:
            if metric in df.columns:
                non_null = df[metric].notna().sum()
                pct = (non_null / len(df) * 100) if len(df) > 0 else 0
                print(f"   {metric:25s}: {non_null:3d}/{len(df):3d} ({pct:5.1f}%)")
            else:
                print(f"   {metric:25s}: ❌ MISSING")
        
        # Check for classification (ground truth)
        if 'classification' in df.columns:
            print("\n🎯 Ground Truth Classification:")
            class_counts = df['classification'].value_counts()
            for cls, count in class_counts.items():
                pct = (count / len(df) * 100) if len(df) > 0 else 0
                print(f"   {cls:10s}: {count:3d}/{len(df):3d} ({pct:5.1f}%)")
        else:
            print("\n⚠️  'classification' column not found")
        
        # Check for aggregate_score (judge verdict)
        if 'aggregate_score' in df.columns:
            print("\n⚖️  Judge Aggregate Score:")
            judge_halluc = (df['aggregate_score'] < 0.5).sum()
            judge_correct = (df['aggregate_score'] >= 0.5).sum()
            print(f"   Hallucination (< 0.5): {judge_halluc:3d}/{len(df):3d} ({judge_halluc/len(df)*100:5.1f}%)")
            print(f"   Correct (>= 0.5):      {judge_correct:3d}/{len(df):3d} ({judge_correct/len(df)*100:5.1f}%)")
        else:
            print("\n⚠️  'aggregate_score' column not found")
        
        print()
    
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}\n")

print("="*80)
print("SUMMARY: Check if we have:")
print("  1. All CI metrics (including cosine_similarity) ✓")
print("  2. All judge logit metrics ✓")
print("  3. Mix of TP and FP edges (not 100% hallucinations) ?")
print("="*80)
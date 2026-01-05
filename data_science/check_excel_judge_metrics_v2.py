#!/usr/bin/env python3
"""
Check if judge logit metrics are populated in the latest Excel output.
"""
import pandas as pd
import sys

excel_path = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_233430_b4acc355/combo_1/prompts_Nitai_C/CLD_test/run1_a50a9fb5/results_exp_20251007_233430_b4acc355_param_combo_nr_1_prompts_Nitai_C_run1_CLD_test_20251007_233606.xlsx"

print(f"📊 Checking Excel file for judge logit metrics...")
print(f"File: {excel_path}\n")

try:
    df = pd.read_excel(excel_path, sheet_name='All Edges')
    print(f"✅ Loaded 'All Edges' sheet: {len(df)} rows\n")
    
    # Check for judge logit metric columns
    judge_metrics = ['judge_perplexity', 'judge_min_prob', 'judge_max_window_entropy']
    
    print("🔍 Judge Logit Metrics Availability:")
    for metric in judge_metrics:
        if metric in df.columns:
            non_null_count = df[metric].notna().sum()
            percentage = (non_null_count / len(df)) * 100 if len(df) > 0 else 0
            print(f"   ✅ {metric}: {non_null_count}/{len(df)} populated ({percentage:.1f}%)")
            if non_null_count > 0:
                print(f"      Sample values: {df[metric].dropna().head(3).tolist()}")
        else:
            print(f"   ❌ {metric}: Column not found")
    
    # Check classification distribution
    print("\n🏷️  Classification Distribution:")
    if 'Classification' in df.columns:
        class_counts = df['Classification'].value_counts()
        print(f"   {class_counts.to_dict()}")
        
        # Calculate hallucination rate based on FP
        fp_count = (df['Classification'] == 'FP').sum()
        tp_count = (df['Classification'] == 'TP').sum()
        halluc_rate = (fp_count / len(df)) * 100 if len(df) > 0 else 0
        print(f"   Hallucination rate (FP): {fp_count}/{len(df)} = {halluc_rate:.1f}%")
        print(f"   Correct edges (TP): {tp_count}/{len(df)} = {(tp_count/len(df)*100):.1f}%")
    else:
        print("   ⚠️  'Classification' column not found")
    
    # Check aggregate_score distribution
    print("\n🎯 Aggregate Score (Judge Verdict):")
    if 'aggregate_score' in df.columns:
        agg_score_stats = df['aggregate_score'].describe()
        print(f"   {agg_score_stats}")
        
        # Calculate hallucination rate based on aggregate_score < 0.5
        judge_halluc_count = (df['aggregate_score'].fillna(1.0) < 0.5).sum()
        judge_halluc_rate = (judge_halluc_count / len(df)) * 100 if len(df) > 0 else 0
        print(f"\n   Judge hallucinations (aggregate_score < 0.5): {judge_halluc_count}/{len(df)} = {judge_halluc_rate:.1f}%")
    else:
        print("   ⚠️  'aggregate_score' column not found")
    
    # Check for CI metrics
    print("\n🧪 Context-Insensitive Metrics:")
    ci_metrics = ['perplexity', 'min_prob', 'max_window_entropy', 'cosine_similarity']
    for metric in ci_metrics:
        if metric in df.columns:
            non_null_count = df[metric].notna().sum()
            percentage = (non_null_count / len(df)) * 100 if len(df) > 0 else 0
            print(f"   ✅ {metric}: {non_null_count}/{len(df)} populated ({percentage:.1f}%)")
        else:
            print(f"   ❌ {metric}: Column not found")
    
    print("\n" + "="*60)
    print("✅ SUMMARY:")
    
    # Judge metrics check
    judge_metrics_populated = all(
        (df[m].notna().sum() > 0) if m in df.columns else False 
        for m in judge_metrics
    )
    if judge_metrics_populated:
        print("   ✅ Judge logit metrics are populated!")
    else:
        print("   ❌ Judge logit metrics are NOT populated")
    
    # Classification check
    if 'Classification' in df.columns:
        fp_rate = (df['Classification'] == 'FP').sum() / len(df) * 100
        if 10 <= fp_rate <= 90:
            print(f"   ✅ Good classification mix: {fp_rate:.1f}% FP")
        elif fp_rate == 100:
            print(f"   ⚠️  100% FP - all hallucinations!")
        elif fp_rate == 0:
            print(f"   ⚠️  0% FP - no hallucinations!")
        else:
            print(f"   ⚙️  Classification: {fp_rate:.1f}% FP")
    
    print("="*60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

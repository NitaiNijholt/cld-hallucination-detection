#!/usr/bin/env python3
"""
Verify the social norms experiment results before running analysis.
"""
import pandas as pd
import sys

excel_path = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_234155_60068754/combo_1/prompts_Nitai_C/Social_norms_and_obesity_prevalence/run1_5db6738d/results_exp_20251007_234155_60068754_param_combo_nr_1_prompts_Nitai_C_run1_Social_norms_and_obesity_prevalence_20251008_001309.xlsx"

print(f"📊 Verifying Social Norms Experiment Results")
print(f"File: {excel_path}")
print("="*80)
print("")

try:
    df = pd.read_excel(excel_path, sheet_name='All Edges')
    print(f"✅ Loaded 'All Edges' sheet: {len(df)} rows")
    print("")
    
    # Check for judge logit metric columns
    print("🔍 Judge Logit Metrics:")
    judge_metrics = ['Judge Perplexity', 'Judge Min Prob', 'Judge Max Window Entropy']
    
    for metric in judge_metrics:
        if metric in df.columns:
            non_null_count = df[metric].notna().sum()
            percentage = (non_null_count / len(df)) * 100 if len(df) > 0 else 0
            print(f"   ✅ {metric}: {non_null_count}/{len(df)} populated ({percentage:.1f}%)")
            if non_null_count > 0:
                print(f"      Range: [{df[metric].min():.3f}, {df[metric].max():.3f}]")
                print(f"      Mean: {df[metric].mean():.3f}")
        else:
            print(f"   ❌ {metric}: Column not found")
    
    print("")
    
    # Check generator CI metrics
    print("🧪 Generator CI Metrics:")
    gen_metrics = ['Perplexity', 'Min Prob', 'Max Window Entropy', 'Cosine Similarity']
    for m in gen_metrics:
        if m in df.columns:
            count = df[m].notna().sum()
            print(f"   ✅ {m}: {count}/{len(df)} = {count/len(df)*100:.1f}%")
        else:
            print(f"   ❌ {m}: Column not found")
    
    print("")
    
    # Check classification distribution
    print("🏷️  Classification Distribution:")
    if 'Classification' in df.columns:
        class_counts = df['Classification'].value_counts()
        print(f"   {class_counts.to_dict()}")
        print("")
        
        # Calculate rates
        fp_count = (df['Classification'] == 'FP').sum()
        tp_count = (df['Classification'] == 'TP').sum()
        fn_count = (df['Classification'] == 'FN').sum()
        tn_count = (df['Classification'] == 'TN').sum()
        
        print(f"   TP (True Positive - Correct):     {tp_count:3d} ({tp_count/len(df)*100:5.1f}%)")
        print(f"   FP (False Positive - Hallucination): {fp_count:3d} ({fp_count/len(df)*100:5.1f}%)")
        print(f"   FN (False Negative - Missed):     {fn_count:3d} ({fn_count/len(df)*100:5.1f}%)")
        print(f"   TN (True Negative - Correct):     {tn_count:3d} ({tn_count/len(df)*100:5.1f}%)")
        
        halluc_rate = (fp_count / len(df)) * 100 if len(df) > 0 else 0
        print(f"   → Hallucination rate (FP): {halluc_rate:.1f}%")
    else:
        print("   ⚠️  'Classification' column not found")
    
    print("")
    
    # Check aggregate_score distribution
    print("🎯 Aggregate Score (Judge Verdict):")
    if 'Aggregate Score' in df.columns:
        agg_score_stats = df['Aggregate Score'].describe()
        print(f"   Count: {agg_score_stats['count']:.0f}")
        print(f"   Mean:  {agg_score_stats['mean']:.3f}")
        print(f"   Min:   {agg_score_stats['min']:.3f}")
        print(f"   Max:   {agg_score_stats['max']:.3f}")
        
        # Calculate hallucination rate based on aggregate_score < 0.5
        judge_halluc_count = (df['Aggregate Score'].fillna(1.0) < 0.5).sum()
        judge_halluc_rate = (judge_halluc_count / len(df)) * 100 if len(df) > 0 else 0
        print(f"   → Judge hallucinations (score < 0.5): {judge_halluc_count}/{len(df)} = {judge_halluc_rate:.1f}%")
    else:
        print("   ⚠️  'Aggregate Score' column not found")
    
    print("")
    print("="*80)
    print("✅ SUMMARY:")
    print("")
    
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
        tp_rate = (df['Classification'] == 'TP').sum() / len(df) * 100
        if 10 <= fp_rate <= 90:
            print(f"   ✅ Good classification mix: {tp_rate:.1f}% TP, {fp_rate:.1f}% FP")
        elif fp_rate == 100:
            print(f"   ⚠️  100% FP - all hallucinations!")
        elif fp_rate == 0:
            print(f"   ⚠️  0% FP - no hallucinations!")
        else:
            print(f"   ⚙️  Classification: {tp_rate:.1f}% TP, {fp_rate:.1f}% FP")
    
    # Overall readiness
    print("")
    if judge_metrics_populated and 'Classification' in df.columns:
        print("   ✅ READY FOR RQ2 ANALYSIS!")
    else:
        print("   ⚠️  Some metrics missing - analysis may be limited")
    
    print("="*80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
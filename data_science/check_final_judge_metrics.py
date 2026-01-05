#!/usr/bin/env python3
import pandas as pd

excel_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_230722_42be2970/combo_1/prompts_Nitai_C/CLD_test/run1_fac1146e/results_exp_20251007_230722_42be2970_param_combo_nr_1_prompts_Nitai_C_run1_CLD_test_20251007_231255.xlsx"

# Read the All Edges sheet
df = pd.read_excel(excel_file, sheet_name='All Edges')
print(f"✅ Loaded {len(df)} edges from 'All Edges' sheet")
print()

# Check for all CI metrics including judge logit metrics
ci_metrics = [
    'Perplexity',
    'Min Prob',
    'Max Window Entropy',
    'Cosine Similarity',
    'Judge Perplexity',
    'Judge Min Prob',
    'Judge Max Window Entropy'
]

print("📊 CI Metrics Status:")
print()
for metric in ci_metrics:
    if metric in df.columns:
        count = df[metric].notna().sum()
        status = "✅" if count > 0 else "❌"
        
        # Show sample values for populated metrics
        if count > 0:
            sample_values = df[metric].dropna().head(3).tolist()
            print(f"  {status} {metric}: {count}/{len(df)} populated")
            print(f"      Sample values: {sample_values}")
        else:
            print(f"  {status} {metric}: {count}/{len(df)} populated")
    else:
        print(f"  ❌ {metric}: Column NOT FOUND")
    print()

print()
print(f"🔍 Summary:")
print(f"   Total edges: {len(df)}")
print(f"   Edges with Judge Perplexity: {df['Judge Perplexity'].notna().sum() if 'Judge Perplexity' in df.columns else 0}")
print(f"   Edges with Cosine Similarity: {df['Cosine Similarity'].notna().sum() if 'Cosine Similarity' in df.columns else 0}")
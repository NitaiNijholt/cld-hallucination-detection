#!/usr/bin/env python3
"""Investigate extreme perplexity values in Older Persons CLD."""

import pandas as pd
import numpy as np
from pathlib import Path

excel_file = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/rq1_base_generation_original_20251014_004654/result_excel_path_20251014_010206.xlsx")

print("="*80)
print("INVESTIGATING EXTREME PERPLEXITY VALUES - Older Persons CLD")
print("="*80)

df = pd.read_excel(excel_file, sheet_name=0)

# Find extreme perplexity values
extreme = df[df['Gen Perplexity'] > 100].copy()

print(f"\nTotal edges: {len(df)}")
print(f"Edges with extreme perplexity (>100): {len(extreme)}")

if len(extreme) > 0:
    print("\nExtreme perplexity edges:")
    print("-" * 80)
    
    for idx, row in extreme.head(5).iterrows():
        print(f"\nEdge #{idx}:")
        print(f"  Source: {row.get('Source', 'N/A')}")
        print(f"  Target: {row.get('Target', 'N/A')}")
        print(f"  Relationship: {row.get('Relationship', 'N/A')}")
        print(f"  Gen Perplexity: {row['Gen Perplexity']:.2e}")
        print(f"  Gen Min Prob: {row.get('Gen Min Prob', 'N/A')}")
        print(f"  Gen Max Window Entropy: {row.get('Gen Max Window Entropy', 'N/A')}")
        print(f"  Classification: {row.get('Classification', 'N/A')}")
        
        # Check if there's a pattern with very low token probabilities
        if pd.notna(row.get('Gen Min Prob')):
            print(f"  ⚠️  Gen Min Prob is: {row['Gen Min Prob']:.2e}")

# Distribution of perplexity values
print("\n" + "="*80)
print("PERPLEXITY DISTRIBUTION")
print("="*80)
perp_values = df['Gen Perplexity'].dropna()
print(f"Count: {len(perp_values)}")
print(f"Mean: {perp_values.mean():.4f}")
print(f"Median: {perp_values.median():.4f}")
print(f"Std: {perp_values.std():.4f}")
print(f"Min: {perp_values.min():.4f}")
print(f"Max: {perp_values.max():.2e}")
print(f"\nPercentiles:")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {np.percentile(perp_values, p):.4f}")

# Check if it's a specific token with near-zero probability
print("\n" + "="*80)
print("HYPOTHESIS: Extreme perplexity caused by single token with near-zero probability")
print("="*80)
print("\nThe issue is likely that one or more tokens in the sequence has a")
print("log probability close to -inf (e.g., -200), which when exponentiated")
print("in the perplexity calculation, produces an overflow.")
print("\nFIX NEEDED: Cap perplexity values at a reasonable threshold (e.g., 10)")
print("at the source in logit_metrics.py compute_perplexity() function.")

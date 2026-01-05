#!/usr/bin/env python3
"""Quick verification script to check if CI metrics were added to files."""

import pandas as pd
from pathlib import Path

# Check one file
test_file = Path("../final_runs/RQ1a_gt_lit_correctness/depressive/run_1/judged_Depressive_symptoms_in_response_to_a_stressor_baseline_20251116_143935_20251116_144026.xlsx")

print("=" * 80)
print("VERIFYING CI METRICS IN EXCEL FILE")
print("=" * 80)
print(f"File: {test_file.name}\n")

try:
    df = pd.read_excel(test_file, sheet_name='All Edges')
    
    print(f"Total edges: {len(df)}\n")
    
    # Check key columns
    check_cols = {
        'Gen Cosine Similarity': 'Generator cosine similarity',
        'Judge Cosine Similarity': 'Judge cosine similarity',
        'Gen Perplexity': 'Generator perplexity',
        'Judge Perplexity': 'Judge perplexity',
    }
    
    print("Column Status:")
    for col, desc in check_cols.items():
        if col in df.columns:
            filled = df[col].notna().sum()
            pct = 100 * filled / len(df)
            status = "✓" if filled > 0 else "✗"
            print(f"  {status} {desc:30s}: {filled:3d}/{len(df)} ({pct:5.1f}%)")
        else:
            print(f"  ✗ {desc:30s}: COLUMN MISSING")
    
    # Show sample
    if 'Gen Cosine Similarity' in df.columns:
        print("\nSample values (first 3 edges):")
        sample_cols = ['Source', 'Target', 'Gen Cosine Similarity', 'Judge Cosine Similarity']
        print(df[sample_cols].head(3).to_string(index=False))
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

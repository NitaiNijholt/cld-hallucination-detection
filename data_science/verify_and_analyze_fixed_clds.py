#!/usr/bin/env python3
"""
Verify CI metrics in the newly generated CLDs and run ensemble classifier analysis.
"""

import pandas as pd
from pathlib import Path
import sys

# Paths to the 3 newly generated Excel files
DATA_DIR = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/rq1_base_generation_original_20251014_004654")

DATA_FILES = [
    DATA_DIR / "result_excel_path_20251014_004801.xlsx",  # Social Norms
    DATA_DIR / "result_excel_path_20251014_010206.xlsx",  # Older Persons
    DATA_DIR / "result_excel_path_20251014_020108.xlsx",  # Depressive Symptoms
]

# Expected CI metrics (9 Gen + 9 Judge = 18 total)
EXPECTED_CI_METRICS = [
    'Gen Perplexity', 'Gen Min Prob', 'Gen Max Window Entropy', 'Gen Cosine Similarity',
    'Gen Mean Token Prob', 'Gen Prob Variance', 'Gen Prob Std', 'Gen Max Prob Diff', 'Gen Token Prob Slope',
    'Judge Perplexity', 'Judge Min Prob', 'Judge Max Window Entropy', 'Judge Cosine Similarity',
    'Judge Mean Token Prob', 'Judge Prob Variance', 'Judge Prob Std', 'Judge Max Prob Diff', 'Judge Token Prob Slope'
]

print("="*80)
print("VERIFYING CI METRICS IN NEWLY GENERATED CLDs")
print("="*80)

all_verified = True

for i, excel_file in enumerate(DATA_FILES, 1):
    cld_name = excel_file.stem.replace("result_excel_path_", "CLD_")
    print(f"\n{i}. {excel_file.name}")
    print(f"   File size: {excel_file.stat().st_size / 1024:.1f} KB")
    
    if not excel_file.exists():
        print(f"   ❌ FILE NOT FOUND!")
        all_verified = False
        continue
    
    try:
        df = pd.read_excel(excel_file, sheet_name=0)
        print(f"   Total rows: {len(df)}")
        print(f"   Total columns: {len(df.columns)}")
        
        # Check for CI metrics
        found_metrics = [m for m in EXPECTED_CI_METRICS if m in df.columns]
        print(f"\n   CI Metrics: {len(found_metrics)}/{len(EXPECTED_CI_METRICS)} found")
        
        if len(found_metrics) < 9:  # At least Gen metrics should be there
            print(f"   ❌ MISSING CI METRICS!")
            all_verified = False
        else:
            print(f"   ✅ CI metrics present")
            
            # Check for extreme values in perplexity
            if 'Gen Perplexity' in df.columns:
                perp_values = df['Gen Perplexity'].dropna()
                if len(perp_values) > 0:
                    max_perp = perp_values.max()
                    print(f"   Gen Perplexity range: [{perp_values.min():.4f}, {max_perp:.4f}]")
                    if max_perp > 100:
                        print(f"   ⚠️  WARNING: Extreme perplexity values detected!")
                        all_verified = False
                    else:
                        print(f"   ✅ Perplexity values look reasonable")
        
        # Check for TP/FP classification
        if 'Classification' in df.columns:
            class_counts = df['Classification'].value_counts()
            print(f"   Classification counts: {dict(class_counts)}")
            tp_count = (df['Classification'] == 'TP').sum()
            fp_count = (df['Classification'] == 'FP').sum()
            print(f"   Available for classifier: TP={tp_count}, FP={fp_count}")
        
    except Exception as e:
        print(f"   ❌ ERROR reading file: {e}")
        all_verified = False

print("\n" + "="*80)
if all_verified:
    print("✅ ALL FILES VERIFIED - CI METRICS LOOK GOOD!")
    print("="*80)
    print("\nReady to run ensemble classifier analysis!")
    sys.exit(0)
else:
    print("❌ VERIFICATION FAILED - ISSUES DETECTED")
    print("="*80)
    sys.exit(1)

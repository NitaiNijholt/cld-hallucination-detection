#!/usr/bin/env python3
"""Check if CI metrics are present in a recent Excel export."""

import pandas as pd
from pathlib import Path

# Path to the recent Excel file
excel_path = Path(__file__).parent / "results/rq1_base_generation_original_20251012_052918/result_excel_path_20251012_061310.xlsx"

print(f"Checking Excel file: {excel_path}")
print("=" * 80)

# Read the Excel file
df = pd.read_excel(excel_path, sheet_name=0)

print(f"\n📊 Total rows: {len(df)}")
print(f"📊 Total columns: {len(df.columns)}")

# Expected CI metrics
expected_ci_metrics = [
    "Perplexity",
    "Min Prob",
    "Max Window Entropy",
    "Mean Token Prob",
    "Prob Variance",
    "Prob Std",
    "Max Prob Diff",
    "Token Prob Slope",
    "Gen Perplexity",
    "Gen Min Prob",
    "Gen Max Window Entropy",
    "Gen Mean Token Prob",
    "Gen Prob Variance",
    "Gen Prob Std",
    "Gen Max Prob Diff",
    "Gen Token Prob Slope",
    "Cosine Similarity",
    "Gen Cosine Similarity",
]

print(f"\n🔍 Checking for CI metrics:")
print("-" * 80)

found_metrics = []
missing_metrics = []

for metric in expected_ci_metrics:
    if metric in df.columns:
        # Check if the column has any non-null values
        non_null_count = df[metric].notna().sum()
        found_metrics.append(metric)
        print(f"✅ {metric:30s} - Found ({non_null_count}/{len(df)} non-null values)")
    else:
        missing_metrics.append(metric)
        print(f"❌ {metric:30s} - MISSING")

print("\n" + "=" * 80)
print(f"Summary: {len(found_metrics)}/{len(expected_ci_metrics)} metrics found")

if missing_metrics:
    print(f"\n⚠️  Missing metrics:")
    for metric in missing_metrics:
        print(f"   - {metric}")
else:
    print("\n✅ All CI metrics are present!")

# Show a sample row with CI metrics
print("\n" + "=" * 80)
print("Sample data (first row with CI metrics):")
print("-" * 80)

ci_cols = [col for col in df.columns if col in expected_ci_metrics]
if ci_cols and len(df) > 0:
    sample = df[ci_cols].iloc[0]
    for col, val in sample.items():
        print(f"{col:30s}: {val}")
else:
    print("No CI metric data found")

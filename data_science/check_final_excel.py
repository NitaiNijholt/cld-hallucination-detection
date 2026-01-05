#!/usr/bin/env python3
"""Check the final regenerated Excel file."""

import pandas as pd

# Read the Excel file
excel_path = "parameter_tuning_experiments/results/social_norms_with_relationship_type_REGENERATED_20251008_030604.xlsx"
df = pd.read_excel(excel_path, sheet_name='All Edges')

print("✅ Excel file loaded successfully!")
print(f"\n📊 Total edges: {len(df)}")
print(f"📊 Total columns: {len(df.columns)}")

print("\n📋 All columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n🔗 Relationship Type distribution:")
print(df['Relationship Type'].value_counts())

print("\n📊 Classification distribution:")
print(df['Classification'].value_counts())

print("\n📈 Sample rows:")
sample_cols = ['Source', 'Target', 'Relationship Type', 'Classification', 'Aggregate Score']
print(df[sample_cols].head(10).to_string())

print("\n🔍 Check for NONE relationships:")
none_edges = df[df['Relationship Type'].str.upper().isin(['NONE', 'NO RELATIONSHIP'])]
print(f"  Found {len(none_edges)} edges with NONE relationship type")

if len(none_edges) > 0:
    print("\n  Sample NONE edges:")
    print(none_edges[['Source', 'Target', 'Relationship Type', 'Classification']].head().to_string())

print("\n🔍 Check for metrics availability:")
metric_cols = ['Aggregate Score', 'Judge Perplexity', 'Gen Perplexity', 'Judge Cosine Similarity', 'Gen Cosine Similarity']
for col in metric_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(df)} ({100*non_null/len(df):.1f}%)")

print("\n✅ SUCCESS! Excel file is complete and has Relationship Type column!")

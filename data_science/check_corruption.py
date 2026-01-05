#!/usr/bin/env python3
"""Check corruption status in Phase 1 output Excel file"""
import pandas as pd

file_path = 'parameter_tuning_experiments/results/rq1_phase1_20251010_010857/corrupted_b458c955-6f86-4321-9443-8d29e57a923b_cr50_20251010_011100.xlsx'

# Check Params sheet
print("="*80)
print("PARAMS SHEET")
print("="*80)
df_params = pd.read_excel(file_path, sheet_name='Params')
if 'corrupted_edges_count' in df_params.columns:
    print(f"corrupted_edges_count: {df_params['corrupted_edges_count'].iloc[0]}")
else:
    print("❌ No 'corrupted_edges_count' column found")

# Check Edge sheet
print("\n" + "="*80)
print("EDGE SHEET")
print("="*80)
df_edge = pd.read_excel(file_path, sheet_name='Edge')
print(f"Total edges: {len(df_edge)}")

if 'is_corrupted' in df_edge.columns:
    corrupted_count = df_edge['is_corrupted'].sum()
    print(f"✅ Corrupted edges (is_corrupted=True): {corrupted_count}")
    print(f"✅ Clean edges (is_corrupted=False): {len(df_edge) - corrupted_count}")
    
    if corrupted_count > 0:
        print("\nSample corrupted edges:")
        print(df_edge[df_edge['is_corrupted']==True][['source', 'target', 'is_corrupted']].head(10))
else:
    print("❌ No 'is_corrupted' column found")



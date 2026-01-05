#!/usr/bin/env python3
"""
Extract ground truth edges (non-corrupted) from RQ1 experiments for Deep Research testing.
"""

import pandas as pd
import json

# Most recent Social norms experiment
EXCEL_FILE = "parameter_tuning_experiments/results/exp_20251010_235745_274dcb4d/combo_1/prompts_Nitai_C/Social_norms_and_obesity_prevalence/run1_030b89e0/results_exp_20251010_235745_274dcb4d_param_combo_nr_1_prompts_Nitai_C_run1_Social_norms_and_obesity_prevalence_20251011_000447.xlsx"

print("="*100)
print("EXTRACTING GROUND TRUTH EDGES FOR DEEP RESEARCH")
print("="*100)
print(f"\nReading: {EXCEL_FILE.split('/')[-1]}\n")

# Load data
df = pd.read_excel(EXCEL_FILE, sheet_name='All Edges')

print(f"Total edges: {len(df)}")

# Since this is a non-corrupted experiment file, all edges are ground truth
# Filter for validation set edges only
gt_df = df[df['In Validation Graph'] == 'Yes'].copy()
print(f"Edges in validation graph: {len(gt_df)}")

# Get TP, FP, TN, FN edges
tp_edges = gt_df[gt_df['Classification'] == 'TP']
fp_edges = gt_df[gt_df['Classification'] == 'FP']
tn_edges = gt_df[gt_df['Classification'] == 'TN']
fn_edges = gt_df[gt_df['Classification'] == 'FN']

print(f"\nBreakdown:")
print(f"  TP: {len(tp_edges)}")
print(f"  FP: {len(fp_edges)}")
print(f"  TN: {len(tn_edges)}")
print(f"  FN: {len(fn_edges)}")

# Extract edges for Deep Research test
def create_edge_dict(row):
    return {
        "Source": row['Source'],
        "Target": row['Target'],
        "Motivation": row['Motivation'] if pd.notna(row['Motivation']) else "",
        "Classification": row['Classification'],
        "Expected": f"Classification: {row['Classification']}"
    }

# Extract ALL validation edges
test_edges = {
    "TP": [create_edge_dict(row) for _, row in tp_edges.iterrows()],
    "FP": [create_edge_dict(row) for _, row in fp_edges.iterrows()],
    "TN": [create_edge_dict(row) for _, row in tn_edges.iterrows()],
    "FN": [create_edge_dict(row) for _, row in fn_edges.iterrows()]
}

print(f"\n" + "="*100)
print("SELECTED TEST EDGES:")
print("="*100)

for classification, edges in test_edges.items():
    print(f"\n{classification} ({len(edges)} edges):")
    for i, edge in enumerate(edges, 1):
        print(f"  {i}. {edge['Source']} → {edge['Target']}")
        if edge['Motivation']:
            print(f"     Motivation: {edge['Motivation'][:100]}...")

# Save to JSON
output_file = "deep_research_ground_truth_edges.json"
with open(output_file, 'w') as f:
    json.dump(test_edges, f, indent=2)

print(f"\n✅ Saved to: {output_file}")
print(f"\nTotal edges to test: {sum(len(edges) for edges in test_edges.values())}")


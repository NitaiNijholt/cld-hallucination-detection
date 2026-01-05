#!/usr/bin/env python3
"""
Extract validation edges from all 3 CLDs in exp_20251010_235745_274dcb4d/combo_1
"""

import pandas as pd
import json
from pathlib import Path

# Excel files for the 3 CLDs
EXCEL_FILES = [
    {
        "name": "Social_norms_and_obesity_prevalence",
        "path": "parameter_tuning_experiments/results/exp_20251010_235745_274dcb4d/combo_1/prompts_Nitai_C/Social_norms_and_obesity_prevalence/run1_030b89e0/results_exp_20251010_235745_274dcb4d_param_combo_nr_1_prompts_Nitai_C_run1_Social_norms_and_obesity_prevalence_20251011_000447.xlsx"
    },
    {
        "name": "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults",
        "path": "parameter_tuning_experiments/results/exp_20251010_235745_274dcb4d/combo_1/prompts_Nitai_C/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run1_0f955b75/results_exp_20251010_235745_274dcb4d_param_combo_nr_1_prompts_Nitai_C_run1_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251011_000158.xlsx"
    },
    {
        "name": "older_persons_emergency_department_visits_and_interactionstitled_spreadsheet",
        "path": "parameter_tuning_experiments/results/exp_20251010_235745_274dcb4d/combo_1/prompts_Nitai_C/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run1_16b23155/results_exp_20251010_235745_274dcb4d_param_combo_nr_1_prompts_Nitai_C_run1_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251011_003421.xlsx"
    }
]

print("="*100)
print("EXTRACTING VALIDATION EDGES FROM ALL 3 CLDS")
print("="*100)

all_edges = []

for cld_info in EXCEL_FILES:
    print(f"\n{'='*100}")
    print(f"CLD: {cld_info['name']}")
    print(f"{'='*100}")
    
    # Load data
    df = pd.read_excel(cld_info['path'], sheet_name='All Edges')
    
    print(f"Total edges: {len(df)}")
    
    # Filter for validation set edges only
    val_edges = df[df['In Validation Graph'] == 'Yes'].copy()
    print(f"Edges in validation graph: {len(val_edges)}")
    
    # Get TP, FP, TN, FN breakdown
    tp_count = len(val_edges[val_edges['Classification'] == 'TP'])
    fp_count = len(val_edges[val_edges['Classification'] == 'FP'])
    tn_count = len(val_edges[val_edges['Classification'] == 'TN'])
    fn_count = len(val_edges[val_edges['Classification'] == 'FN'])
    
    print(f"  TP: {tp_count}")
    print(f"  FP: {fp_count}")
    print(f"  TN: {tn_count}")
    print(f"  FN: {fn_count}")
    
    # Add edges to list
    for _, row in val_edges.iterrows():
        edge = {
            "CLD": cld_info['name'],
            "Source": row['Source'],
            "Target": row['Target'],
            "Motivation": row['Motivation'] if pd.notna(row['Motivation']) else "",
            "Classification": row['Classification'],
            "Expected": f"Classification: {row['Classification']}"
        }
        all_edges.append(edge)
        print(f"  {row['Classification']}: {row['Source']} → {row['Target']}")

print(f"\n{'='*100}")
print(f"TOTAL VALIDATION EDGES ACROSS ALL CLDs: {len(all_edges)}")
print(f"{'='*100}")

# Group by classification
by_class = {}
for edge in all_edges:
    cls = edge['Classification']
    if cls not in by_class:
        by_class[cls] = []
    by_class[cls].append(edge)

print("\nBreakdown by classification:")
for cls in ['TP', 'FP', 'TN', 'FN']:
    if cls in by_class:
        print(f"  {cls}: {len(by_class[cls])} edges")

# Save to JSON
output_file = "deep_research_all_validation_edges.json"
with open(output_file, 'w') as f:
    json.dump(by_class, f, indent=2)

print(f"\n✅ Saved to: {output_file}")
print(f"\nTotal edges to test: {len(all_edges)}")

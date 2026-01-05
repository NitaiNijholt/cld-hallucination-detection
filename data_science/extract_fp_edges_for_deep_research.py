#!/usr/bin/env python3
"""
Extract FALSE POSITIVE (FP) edges from experiment results for deep research testing.
FP edges are those that were generated but are NOT in the ground truth.
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
print("EXTRACTING FALSE POSITIVE (FP) EDGES FROM EXPERIMENT RESULTS")
print("="*100)

all_fp_edges = []

for cld_info in EXCEL_FILES:
    print(f"\n{'='*100}")
    print(f"CLD: {cld_info['name']}")
    print(f"{'='*100}")
    
    # Load data
    df = pd.read_excel(cld_info['path'], sheet_name='All Edges')
    
    print(f"Total edges: {len(df)}")
    
    # Filter for FP edges (edges in generated graph but NOT in ground truth)
    fp_edges = df[df['Classification'] == 'FP'].copy()
    print(f"FP edges found: {len(fp_edges)}")
    
    if len(fp_edges) == 0:
        print("  (No FP edges in this CLD)")
        continue
    
    # Add edges to list
    for _, row in fp_edges.iterrows():
        # Try to get motivation if it exists
        motivation = ""
        if 'Motivation' in row and pd.notna(row['Motivation']):
            motivation = row['Motivation']
        elif 'LLM Motivation' in row and pd.notna(row['LLM Motivation']):
            motivation = row['LLM Motivation']
        else:
            # Generate a basic motivation for FP edges
            motivation = f"{row['Source']} was hypothesized to cause {row['Target']}, but this relationship is NOT supported by the ground truth causal model."
        
        edge = {
            "CLD": cld_info['name'],
            "Source": row['Source'],
            "Target": row['Target'],
            "Motivation": motivation,
            "Classification": "FP",
            "Expected": "Classification: FP"
        }
        all_fp_edges.append(edge)
        print(f"  FP: {row['Source']} → {row['Target']}")

print(f"\n{'='*100}")
print(f"TOTAL FP EDGES ACROSS ALL CLDs: {len(all_fp_edges)}")
print(f"{'='*100}")

if len(all_fp_edges) > 0:
    # Save to JSON
    output_json = "deep_research_fp_edges.json"
    with open(output_json, 'w') as f:
        json.dump({"FP": all_fp_edges}, f, indent=2)
    print(f"\n✅ Saved JSON to: {output_json}")
    
    # Save to CSV
    output_csv = "deep_research_fp_edges.csv"
    df_fp = pd.DataFrame(all_fp_edges)
    df_fp.to_csv(output_csv, index=False)
    print(f"✅ Saved CSV to: {output_csv}")
    
    print(f"\nTotal FP edges to test: {len(all_fp_edges)}")
else:
    print("\n⚠️  No FP edges found in any CLD!")





















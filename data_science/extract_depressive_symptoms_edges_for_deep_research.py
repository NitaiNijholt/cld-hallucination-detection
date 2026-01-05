#!/usr/bin/env python3
"""
Extract TP, FP, and FN edges from Depressive Symptoms CLD for deep research testing.
Excludes TN (True Negatives) as requested.
"""

import pandas as pd
import json
from pathlib import Path

# Depressive Symptoms CLD results file (NEW - Oct 12, 2025)
DEPRESSIVE_SYMPTOMS_FILE = (
    "parameter_tuning_experiments/results/rq1_base_generation_original_20251012_200603/"
    "result_excel_path_20251012_212700.xlsx"
)

def extract_edges():
    """Extract TP, FP, and FN edges (exclude TN)."""
    
    print("=" * 80)
    print("EXTRACTING EDGES FROM DEPRESSIVE SYMPTOMS CLD")
    print("=" * 80)
    print()
    
    # Read the Excel file
    df = pd.read_excel(DEPRESSIVE_SYMPTOMS_FILE, sheet_name='All Edges')
    
    print(f"Total edges in file: {len(df)}")
    print()
    
    # Count by classification
    classification_counts = df['Classification'].value_counts()
    print("Classification breakdown:")
    for cls, count in classification_counts.items():
        print(f"  {cls}: {count}")
    print()
    
    # Extract TP, FP, FN (but NOT TN)
    edges_to_extract = df[df['Classification'].isin(['TP', 'FP', 'FN'])].copy()
    
    print(f"Extracting {len(edges_to_extract)} edges (TP + FP + FN only)")
    print(f"  TP: {sum(edges_to_extract['Classification'] == 'TP')}")
    print(f"  FP: {sum(edges_to_extract['Classification'] == 'FP')}")
    print(f"  FN: {sum(edges_to_extract['Classification'] == 'FN')}")
    print(f"  TN: 0 (excluded)")
    print()
    
    # Create edge list
    all_edges = []
    
    for _, row in edges_to_extract.iterrows():
        edge = {
            "CLD": "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults",
            "Source": row['Source'],
            "Target": row['Target'],
            "Motivation": row['Motivation'] if pd.notna(row['Motivation']) else "",
            "Classification": row['Classification'],
            "Expected": f"Classification: {row['Classification']}"
        }
        all_edges.append(edge)
    
    # Save to JSON
    output_json = "deep_research_depressive_symptoms_edges.json"
    with open(output_json, 'w') as f:
        json.dump({
            "description": "TP, FP, and FN edges from Depressive Symptoms in Response to a Stressor in Healthy Adults CLD",
            "cld_name": "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults",
            "total_edges": len(all_edges),
            "TP": sum(1 for e in all_edges if e['Classification'] == 'TP'),
            "FP": sum(1 for e in all_edges if e['Classification'] == 'FP'),
            "FN": sum(1 for e in all_edges if e['Classification'] == 'FN'),
            "edges": all_edges
        }, f, indent=2)
    
    print(f"✓ Saved to: {output_json}")
    
    # Save to CSV
    output_csv = "deep_research_depressive_symptoms_edges.csv"
    pd.DataFrame(all_edges).to_csv(output_csv, index=False)
    print(f"✓ Saved to: {output_csv}")
    print()
    
    # Show sample edges by classification
    print("Sample edges by classification:")
    for cls in ['TP', 'FP', 'FN']:
        cls_edges = [e for e in all_edges if e['Classification'] == cls]
        if cls_edges:
            print(f"\n{cls} examples:")
            for edge in cls_edges[:2]:
                print(f"  {edge['Source']} → {edge['Target']}")
    
    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    
    return output_json, output_csv

if __name__ == "__main__":
    extract_edges()





















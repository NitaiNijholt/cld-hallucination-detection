#!/usr/bin/env python3
"""
Quick script to regenerate Excel from the most recent session.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules import export_edges_comparison_to_excel, CausalDiscovery

# Session ID from the Social Norms experiment with all metrics
SESSION_ID = "7c149637-f87d-4e93-bc7c-77c50f62788a"

# Validation files
VALIDATION_VARS = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ground_truth_clds_for_experiments/Social_norms_and_obesity_prevalence_vars_data.json"
VALIDATION_EDGES = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ground_truth_clds_for_experiments/Social_norms_and_obesity_prevalence_edges_data.json"

# Output file
OUTPUT_EXCEL = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/social_norms_with_relationship_type_REGENERATED.xlsx"

print(f"🆔 Session ID: {SESSION_ID}")
print(f"📂 Output: {OUTPUT_EXCEL}")

# Create minimal CausalDiscovery object
print("\n🔧 Creating discovery object...")
disco = CausalDiscovery(
    target_variable="dummy",
    temporal_scale="dummy",
    spatial_scale="dummy",
    yaml_path="/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/prompts_Nitai_C.yaml",
    generator_config={"provider": "openai", "model": "gpt-4o"}
)

# Export to Excel
print("📊 Exporting to Excel...")
export_edges_comparison_to_excel(
    discovery=disco,
    session_ids=[SESSION_ID],
    validation_vars_json_path=VALIDATION_VARS,
    validation_edges_json_path=VALIDATION_EDGES,
    output_filename=OUTPUT_EXCEL
)

print(f"\n✅ Excel file regenerated successfully!")
print(f"📁 File: {OUTPUT_EXCEL}")

# Quick sanity check
import pandas as pd
df = pd.read_excel(OUTPUT_EXCEL, sheet_name='All Edges')
print(f"\n📈 Quick Stats:")
print(f"   Total edges: {len(df)}")
print(f"   Columns: {len(df.columns)}")

if 'Relationship Type' in df.columns:
    print(f"\n🔗 Relationship Types:")
    print(df['Relationship Type'].value_counts())
else:
    print("\n⚠️  'Relationship Type' column NOT found!")
    
if 'Classification' in df.columns:
    print(f"\n📊 Classifications:")
    print(df['Classification'].value_counts())

# Check for NONE relationships
if 'Relationship Type' in df.columns:
    none_edges = df[df['Relationship Type'].str.upper().isin(['NONE', 'NO RELATIONSHIP'])]
    if len(none_edges) > 0:
        print(f"\n⚠️  Found {len(none_edges)} edges with NONE relationship type")
        print(f"   These are shown in the Excel with their relationship type!")

print(f"\n✅ SUCCESS!")

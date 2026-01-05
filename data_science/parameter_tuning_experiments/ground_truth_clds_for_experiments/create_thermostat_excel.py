#!/usr/bin/env python3
"""
Create Excel file for the thermostat heating system CLD.
"""

import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Load the JSON files
with open(BASE_DIR / "thermostat_heating_system_vars_data.json", 'r') as f:
    vars_data = json.load(f)

with open(BASE_DIR / "thermostat_heating_system_edges_data.json", 'r') as f:
    edges_data = json.load(f)

with open(BASE_DIR / "thermostat_heating_system_context_data.json", 'r') as f:
    context_data = json.load(f)

# Create Variable_definitions DataFrame
var_df = pd.DataFrame([
    {"Variable": v["name"], "Definition": v["definition"]}
    for v in vars_data
])

# Create Variable_links DataFrame
edges_df = pd.DataFrame([
    {"Source": e["source"], "Target": e["target"], "Polarity": e["polarity"]}
    for e in edges_data
])

# Create Context DataFrame
context_df = pd.DataFrame([
    {"Field": k, "Value": v}
    for k, v in context_data.items()
])

# Add metadata
metadata_df = pd.DataFrame([
    {"Field": "Source", "Value": "Physics Textbooks - Sterman (2000), Incropera et al. (2007), ASHRAE (2017)"},
    {"Field": "Domain", "Value": "Thermal Physics / Control Systems"},
    {"Field": "Physics Principles", "Value": "Newton's Law of Cooling, First Law of Thermodynamics, Feedback Control Theory"},
    {"Field": "Total Variables", "Value": len(vars_data)},
    {"Field": "Total Edges", "Value": len(edges_data)},
    {"Field": "Positive Edges", "Value": sum(1 for e in edges_data if e["polarity"] == "positive")},
    {"Field": "Negative Edges", "Value": sum(1 for e in edges_data if e["polarity"] == "negative")}
])

# Combine context and metadata
context_full = pd.concat([context_df, metadata_df], ignore_index=True)

# Write to Excel
output_file = BASE_DIR / "thermostat_heating_system.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    var_df.to_excel(writer, sheet_name='Variable_definitions', index=False)
    edges_df.to_excel(writer, sheet_name='Variable_links', index=False)
    context_full.to_excel(writer, sheet_name='Context', index=False)

print(f"✅ Created Excel file: {output_file}")
print(f"   Variables: {len(vars_data)}")
print(f"   Edges: {len(edges_data)}")










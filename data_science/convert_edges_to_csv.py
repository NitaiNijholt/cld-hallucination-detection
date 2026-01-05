#!/usr/bin/env python3
"""
Convert validation edges JSON to CSV format
"""

import pandas as pd
import json

# Load JSON
with open('deep_research_all_validation_edges.json', 'r') as f:
    edges_by_class = json.load(f)

# Flatten to list of edges
all_edges = []
for classification, edges in edges_by_class.items():
    for edge in edges:
        all_edges.append({
            'CLD': edge['CLD'],
            'Classification': classification,
            'Source': edge['Source'],
            'Target': edge['Target'],
            'Motivation': edge['Motivation'],
            'Expected': edge['Expected']
        })

# Convert to DataFrame
df = pd.DataFrame(all_edges)

# Save to CSV
output_file = 'deep_research_all_validation_edges.csv'
df.to_csv(output_file, index=False)

print(f"✅ Saved {len(df)} edges to: {output_file}")
print(f"\nBreakdown by CLD:")
for cld in df['CLD'].unique():
    cld_df = df[df['CLD'] == cld]
    print(f"  {cld}: {len(cld_df)} edges")
    for cls in ['TP', 'FP', 'TN', 'FN']:
        count = len(cld_df[cld_df['Classification'] == cls])
        if count > 0:
            print(f"    {cls}: {count}")

#!/usr/bin/env python3
"""
Convert Alzheimer's expert CLD to standard ground truth format.
"""

import pandas as pd
from pathlib import Path

def convert_alzheimers_cld():
    """Convert the Alzheimer's CLD to standard format."""
    
    print("="*80)
    print("CONVERTING ALZHEIMER'S EXPERT CLD TO STANDARD FORMAT")
    print("="*80)
    
    # Load the connections
    df = pd.read_excel('alzheimers_cld_supplementary.xlsx', sheet_name='Connections')
    
    # Skip header rows and extract actual data
    # The data starts after row 0 (which has column names)
    df = df[df['Unnamed: 1'].notna()].copy()  # Keep rows with origin
    df = df[df['Unnamed: 2'].notna()].copy()  # Keep rows with polarity
    
    print(f"\nTotal connections found: {len(df)}")
    
    # Map to standard format
    standard_df = pd.DataFrame({
        'Source': df['Unnamed: 1'],
        'Target': df['Unnamed: 3'],
        'Polarity': df['Unnamed: 2'],
        'Citations': df['Unnamed: 4']
    })
    
    # Map polarity to relationship type
    polarity_map = {
        '+': 'POSITIVE',
        '-': 'NEGATIVE'
    }
    
    standard_df['Relationship Type'] = standard_df['Polarity'].map(polarity_map)
    
    # Check for any unmapped polarities
    unmapped = standard_df[standard_df['Relationship Type'].isna()]
    if len(unmapped) > 0:
        print(f"\n⚠️  Warning: {len(unmapped)} edges with unmapped polarity:")
        print(unmapped[['Source', 'Target', 'Polarity']].to_string())
    
    # Remove rows with missing relationship type
    standard_df = standard_df[standard_df['Relationship Type'].notna()].copy()
    
    # Create final format (matching ground truth CLDs)
    final_df = standard_df[['Source', 'Target', 'Relationship Type', 'Citations']].copy()
    
    print(f"\nValid causal edges: {len(final_df)}")
    print(f"\nRelationship type distribution:")
    print(final_df['Relationship Type'].value_counts())
    
    # Show unique variables
    sources = set(final_df['Source'].unique())
    targets = set(final_df['Target'].unique())
    all_vars = sources | targets
    
    print(f"\nTotal unique variables: {len(all_vars)}")
    print(f"\nVariable list:")
    for var in sorted(all_vars):
        print(f"  - {var}")
    
    # Show sample edges
    print(f"\n{'='*80}")
    print("SAMPLE EDGES (first 10):")
    print(f"{'='*80}")
    print(final_df.head(10).to_string(max_colwidth=60))
    
    # Save to Excel in standard format
    output_file = Path('parameter_tuning_experiments/ground_truth_clds_for_experiments/Alzheimers_disease_expert_validated.xlsx')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main edges sheet
        final_df.to_excel(writer, sheet_name='Edges', index=False)
        
        # Create a Variables sheet (just list of unique variables)
        variables_df = pd.DataFrame({
            'Variable': sorted(all_vars)
        })
        variables_df.to_excel(writer, sheet_name='Variables', index=False)
        
        # Create a Metadata sheet
        metadata = pd.DataFrame({
            'Field': [
                'Source',
                'Domain',
                'Expert Validation',
                'Method',
                'Citation',
                'Total Variables',
                'Total Edges',
                'Positive Edges',
                'Negative Edges'
            ],
            'Value': [
                'Alzheimer\'s Disease Multicausality CLD',
                'Neuroscience / Geriatrics',
                'Group Model Building with 17 experts',
                'Systems thinking / Causal Loop Diagram',
                'Uleman et al. (2021) GeroScience, https://doi.org/10.1007/s11357-020-00228-7',
                len(all_vars),
                len(final_df),
                len(final_df[final_df['Relationship Type'] == 'POSITIVE']),
                len(final_df[final_df['Relationship Type'] == 'NEGATIVE'])
            ]
        })
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ CONVERSION COMPLETE")
    print(f"{'='*80}")
    print(f"Output saved to: {output_file}")
    print(f"\nThis CLD can now be used as expert ground truth for:")
    print(f"  - Step 4: Literature-based validation")
    print(f"  - Comparing LLM-generated CLDs against expert consensus")
    print(f"  - Testing judge performance on expert-validated relationships")
    print()
    
    return final_df, output_file


def load_variables_sheet():
    """Load the Variables sheet to get variable details."""
    df_vars = pd.read_excel('alzheimers_cld_supplementary.xlsx', sheet_name='Variables')
    
    print("\n" + "="*80)
    print("VARIABLE DEFINITIONS")
    print("="*80)
    print(f"Shape: {df_vars.shape}")
    print(f"\nFirst 10 variables:")
    print(df_vars.head(10).to_string(max_colwidth=70))


if __name__ == "__main__":
    convert_alzheimers_cld()
    load_variables_sheet()




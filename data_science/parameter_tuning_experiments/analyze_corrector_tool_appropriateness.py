import pandas as pd
import numpy as np

# Load files
base = pd.read_excel("parameter_tuning_experiments/results/rq1_phase0_generation_20251113_173550/result_excel_path_20251113_174500.xlsx", sheet_name="All Edges")
corrupted = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251113_181347/corrupted_6bd82e7d_20251113_181531.xlsx", sheet_name="All Edges")
corrected = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251113_181347/corrected_b4a635ec_20251113_201447.xlsx", sheet_name="All Edges")

print("="*80)
print("CORRECTOR TOOL APPROPRIATENESS ANALYSIS")
print("="*80)

# Get corrupted edges
corrupted_edges = corrupted[corrupted['Is Corrupted'] == True].copy()

# For each corrupted edge, find its state in corrected file
def get_correction_info(row, corrected_df):
    match = corrected_df[(corrected_df['Source'] == row['Source']) & (corrected_df['Target'] == row['Target'])]
    if len(match) > 0:
        return pd.Series({
            'corrected_action': match.iloc[0].get('Correction Action', None),
            'was_corrected': match.iloc[0].get('Corrected', False)
        })
    return pd.Series({'corrected_action': None, 'was_corrected': False})

correction_info = corrupted_edges.apply(lambda row: get_correction_info(row, corrected), axis=1)
merged = pd.concat([corrupted_edges, correction_info], axis=1)

print(f"\nTotal corrupted edges: {len(merged)}")
print(f"Edges that received correction: {merged['was_corrected'].sum()}")

# Analyze by corruption type
print("\n" + "="*80)
print("CORRUPTION TYPE ANALYSIS")
print("="*80)

# Determine corruption type for each edge
def identify_corruption_type(row, base_df):
    src = row['Source']
    tgt = row['Target']
    corrupt_type = row['Relationship Type']
    
    # Find in base
    base_edge = base_df[(base_df['Source'] == src) & (base_df['Target'] == tgt)]
    
    if len(base_edge) == 0:
        return 'unknown'
    
    base_type = base_edge.iloc[0]['Relationship Type']
    
    # Determine corruption type
    if base_type == 'NONE' and corrupt_type in ['POSITIVE', 'NEGATIVE']:
        return 'spurious_edge'
    elif base_type in ['POSITIVE', 'NEGATIVE'] and corrupt_type in ['POSITIVE', 'NEGATIVE'] and base_type != corrupt_type:
        return 'polarity_flip'
    elif base_type == corrupt_type:
        # Same type, so likely motivation corruption
        return 'motivation_corruption'
    else:
        return 'other'

merged['corruption_type'] = merged.apply(lambda row: identify_corruption_type(row, base), axis=1)

# Analyze by corruption type
print("\nCorruption Type Distribution:")
print(merged['corruption_type'].value_counts())

print("\n" + "="*80)
print("CORRECTOR ACTIONS BY CORRUPTION TYPE")
print("="*80)

for corr_type in ['spurious_edge', 'polarity_flip', 'motivation_corruption']:
    subset = merged[merged['corruption_type'] == corr_type]
    if len(subset) == 0:
        continue
    
    print(f"\n{corr_type.upper().replace('_', ' ')} ({len(subset)} edges):")
    print(f"  Edges corrected: {subset['was_corrected'].sum()}")
    
    action_counts = subset['corrected_action'].value_counts()
    for action, count in action_counts.items():
        pct = count/len(subset)*100
        print(f"    {action}: {count} ({pct:.1f}%)")
    
    # Check if appropriate tool was used
    print(f"\n  Expected tool vs Actual:")
    if corr_type == 'spurious_edge':
        print(f"    Expected: remove (or change_type to NONE)")
        removes = subset['corrected_action'].isin(['remove']).sum()
        print(f"    Actual removes: {removes}/{len(subset)} ({removes/len(subset)*100:.1f}%)")
    elif corr_type == 'polarity_flip':
        print(f"    Expected: flip_polarity")
        flips = subset['corrected_action'].isin(['flip_polarity']).sum()
        print(f"    Actual flips: {flips}/{len(subset)} ({flips/len(subset)*100:.1f}%)")
    elif corr_type == 'motivation_corruption':
        print(f"    Expected: revise")
        revises = subset['corrected_action'].isin(['revise']).sum()
        print(f"    Actual revises: {revises}/{len(subset)} ({revises/len(subset)*100:.1f}%)")

# Summary
print("\n" + "="*80)
print("SUMMARY: TOOL APPROPRIATENESS")
print("="*80)

spurious = merged[merged['corruption_type'] == 'spurious_edge']
flips = merged[merged['corruption_type'] == 'polarity_flip']
motivation = merged[merged['corruption_type'] == 'motivation_corruption']

if len(spurious) > 0:
    spurious_removed = spurious['corrected_action'].isin(['remove']).sum()
    print(f"\nSpurious edges (should remove): {spurious_removed}/{len(spurious)} removed correctly ({spurious_removed/len(spurious)*100:.1f}%)")

if len(flips) > 0:
    flips_flipped = flips['corrected_action'].isin(['flip_polarity']).sum()
    print(f"Polarity flips (should flip): {flips_flipped}/{len(flips)} flipped correctly ({flips_flipped/len(flips)*100:.1f}%)")

if len(motivation) > 0:
    motivation_revised = motivation['corrected_action'].isin(['revise']).sum()
    print(f"Motivation corruption (should revise): {motivation_revised}/{len(motivation)} revised correctly ({motivation_revised/len(motivation)*100:.1f}%)")

print("\n" + "="*80)

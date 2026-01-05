#!/usr/bin/env python3
"""Analyze RQ1a Run 1 Results"""

import pandas as pd
import numpy as np

print("=" * 80)
print("ANALYZING RQ1a CORRECTNESS JUDGING RESULTS - Run 1")
print("=" * 80)

base_path = '/home/nitai/code/causalix.ai/final_runs/RQ1a_judging_correctness_GT_depresive/run_1/'

# Load all three files
baseline = pd.read_excel(base_path + 'judged_Depressive_symptoms_in_response_to_a_stressor_baseline_20251113_235504.xlsx')
cot = pd.read_excel(base_path + 'judged_Depressive_symptoms_in_response_to_a_stressor_chain_of_thought_20251113_235829.xlsx')
mechanistic = pd.read_excel(base_path + 'judged_Depressive_symptoms_in_response_to_a_stressor_mechanistic_20251114_000058.xlsx')

print(f"\n📊 DATASET INFO")
print(f"Total edges: {len(baseline)}")

print("\n" + "=" * 80)
print("1. VERDICT DISTRIBUTIONS")
print("=" * 80)

for name, df in [('Baseline', baseline), ('Chain-of-Thought', cot), ('Mechanistic', mechanistic)]:
    print(f"\n{name}:")
    if 'Judge Verdict' in df.columns:
        counts = df['Judge Verdict'].value_counts()
        total_judged = df['Judge Verdict'].notna().sum()
        for verdict, count in counts.items():
            pct = 100 * count / len(df)
            print(f"  {verdict}: {count:3d} ({pct:5.1f}%)")
        print(f"  Total judged: {total_judged}/{len(df)}")

print("\n" + "=" * 80)
print("2. SCORE DISTRIBUTIONS")
print("=" * 80)

for name, df in [('Baseline', baseline), ('Chain-of-Thought', cot), ('Mechanistic', mechanistic)]:
    if 'Aggregate Score' in df.columns:
        scores = df['Aggregate Score'].dropna()
        print(f"\n{name}:")
        print(f"  Mean:   {scores.mean():.3f}")
        print(f"  Median: {scores.median():.3f}")
        print(f"  Std:    {scores.std():.3f}")
        print(f"  Min:    {scores.min():.3f}")
        print(f"  Max:    {scores.max():.3f}")

print("\n" + "=" * 80)
print("3. AGREEMENT ANALYSIS")
print("=" * 80)

# Create edge IDs for comparison
baseline['edge_id'] = baseline['Source'] + ' -> ' + baseline['Target']
cot['edge_id'] = cot['Source'] + ' -> ' + cot['Target']
mechanistic['edge_id'] = mechanistic['Source'] + ' -> ' + mechanistic['Target']

# Merge on edge_id
comparison = baseline[['edge_id', 'Judge Verdict']].merge(
    cot[['edge_id', 'Judge Verdict']],
    on='edge_id',
    suffixes=('_baseline', '_cot')
).merge(
    mechanistic[['edge_id', 'Judge Verdict']],
    on='edge_id'
).rename(columns={'Judge Verdict': 'mechanistic'})

print(f"\nTotal edges compared: {len(comparison)}")

# Calculate agreement rates
baseline_vs_cot = (comparison['Judge Verdict_baseline'] == comparison['Judge Verdict_cot']).sum()
baseline_vs_mech = (comparison['Judge Verdict_baseline'] == comparison['mechanistic']).sum()
cot_vs_mech = (comparison['Judge Verdict_cot'] == comparison['mechanistic']).sum()

print(f"\nPairwise Agreement:")
print(f"  Baseline vs CoT:         {baseline_vs_cot:3d}/{len(comparison)} ({100*baseline_vs_cot/len(comparison):5.1f}%)")
print(f"  Baseline vs Mechanistic: {baseline_vs_mech:3d}/{len(comparison)} ({100*baseline_vs_mech/len(comparison):5.1f}%)")
print(f"  CoT vs Mechanistic:      {cot_vs_mech:3d}/{len(comparison)} ({100*cot_vs_mech/len(comparison):5.1f}%)")

# Full agreement
full_agreement = ((comparison['Judge Verdict_baseline'] == comparison['Judge Verdict_cot']) & 
                  (comparison['Judge Verdict_baseline'] == comparison['mechanistic'])).sum()
print(f"\nAll 3 prompts agree: {full_agreement}/{len(comparison)} ({100*full_agreement/len(comparison):.1f}%)")

# Disagreements
disagreements = comparison[
    (comparison['Judge Verdict_baseline'] != comparison['Judge Verdict_cot']) |
    (comparison['Judge Verdict_baseline'] != comparison['mechanistic'])
]
print(f"Disagreements: {len(disagreements)} edges ({100*len(disagreements)/len(comparison):.1f}%)")

print("\n" + "=" * 80)
print("4. DISAGREEMENT PATTERNS")
print("=" * 80)

if len(disagreements) > 0:
    print(f"\nShowing first 10 disagreements:")
    for idx, row in disagreements.head(10).iterrows():
        print(f"\n  Edge: {row['edge_id']}")
        print(f"    Baseline:    {row['Judge Verdict_baseline']}")
        print(f"    CoT:         {row['Judge Verdict_cot']}")
        print(f"    Mechanistic: {row['mechanistic']}")

print("\n" + "=" * 80)
print("5. PARSE ERROR CHECK (CoT)")
print("=" * 80)

# Check if CoT had any parse errors
if 'Judge Verdict' in cot.columns:
    error_count = (cot['Judge Verdict'] == 'ERROR').sum()
    print(f"\nChain-of-Thought parse errors: {error_count}/{len(cot)} ({100*error_count/len(cot):.1f}%)")
    
    if error_count > 0:
        print("\nFirst 3 error examples:")
        errors = cot[cot['Judge Verdict'] == 'ERROR']
        for idx, row in errors.head(3).iterrows():
            print(f"\n  Edge: {row['Source']} -> {row['Target']}")
            if 'Judge Message' in row:
                msg = str(row.get('Judge Message', ''))[:200]
                print(f"  Message: {msg}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
Key Findings:
- Total edges judged: {len(baseline)} per prompt variant
- Agreement: {100*full_agreement/len(comparison):.1f}% full agreement across all 3 prompts
- CoT parse errors: {(cot['Judge Verdict'] == 'ERROR').sum()} (parser test result)
- Most lenient: Check verdict distributions above
- Most strict: Check verdict distributions above
""")

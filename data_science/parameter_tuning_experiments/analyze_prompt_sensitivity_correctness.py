#!/usr/bin/env python3
"""
Analyze Prompt Sensitivity Results - Correctness Judging
"""

import pandas as pd
import numpy as np

# Use correct filenames with timestamps
base_path = '/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/rq1_prompt_sensitivity_correctness_20251113_233234/'

baseline = pd.read_excel(base_path + 'judged_Depressive_symptoms_in_response_to_a_stressor_baseline_20251113_233346.xlsx')
cot = pd.read_excel(base_path + 'judged_Depressive_symptoms_in_response_to_a_stressor_chain_of_thought_20251113_233710.xlsx')
mechanistic = pd.read_excel(base_path + 'judged_Depressive_symptoms_in_response_to_a_stressor_mechanistic_20251113_233946.xlsx')

print('=== DATASET INFO ===')
print(f'Total edges: {len(baseline)}')

print('\n=== VERDICT DISTRIBUTIONS ===')
for name, df in [('Baseline', baseline), ('Chain-of-Thought', cot), ('Mechanistic', mechanistic)]:
    if 'Judge Verdict' in df.columns:
        print(f'\n{name}:')
        counts = df['Judge Verdict'].value_counts()
        for verdict, count in counts.items():
            pct = 100 * count / len(df)
            print(f'  {verdict}: {count} ({pct:.1f}%)')
        print(f'  Total judged: {df["Judge Verdict"].notna().sum()}/{len(df)}')

print('\n=== VERDICT AGREEMENT ANALYSIS ===')
# Compare verdicts across prompts
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

print(f'\nTotal edges compared: {len(comparison)}')

# Calculate agreement rates
baseline_vs_cot = (comparison['Judge Verdict_baseline'] == comparison['Judge Verdict_cot']).sum()
baseline_vs_mech = (comparison['Judge Verdict_baseline'] == comparison['mechanistic']).sum()
cot_vs_mech = (comparison['Judge Verdict_cot'] == comparison['mechanistic']).sum()

print(f'\nPairwise Agreement:')
print(f'  Baseline vs CoT: {baseline_vs_cot}/{len(comparison)} ({100*baseline_vs_cot/len(comparison):.1f}%)')
print(f'  Baseline vs Mechanistic: {baseline_vs_mech}/{len(comparison)} ({100*baseline_vs_mech/len(comparison):.1f}%)')
print(f'  CoT vs Mechanistic: {cot_vs_mech}/{len(comparison)} ({100*cot_vs_mech/len(comparison):.1f}%)')

# Full agreement (all 3 agree)
full_agreement = ((comparison['Judge Verdict_baseline'] == comparison['Judge Verdict_cot']) & 
                  (comparison['Judge Verdict_baseline'] == comparison['mechanistic'])).sum()
print(f'\nAll 3 prompts agree: {full_agreement}/{len(comparison)} ({100*full_agreement/len(comparison):.1f}%)')

# Where do they disagree?
disagreements = comparison[
    (comparison['Judge Verdict_baseline'] != comparison['Judge Verdict_cot']) |
    (comparison['Judge Verdict_baseline'] != comparison['mechanistic'])
]
print(f'\nDisagreements: {len(disagreements)} edges ({100*len(disagreements)/len(comparison):.1f}%)')

if len(disagreements) > 0:
    print('\n=== EXAMPLES OF DISAGREEMENTS ===')
    print('\nShowing first 10 disagreement patterns:')
    for idx, row in disagreements.head(10).iterrows():
        print(f'\nEdge: {row["edge_id"]}')
        print(f'  Baseline: {row["Judge Verdict_baseline"]}')
        print(f'  CoT: {row["Judge Verdict_cot"]}')
        print(f'  Mechanistic: {row["mechanistic"]}')

# Save comparison
output_file = base_path + 'prompt_comparison_analysis.csv'
comparison.to_csv(output_file, index=False)
print(f'\n✅ Saved detailed comparison to: {output_file}')

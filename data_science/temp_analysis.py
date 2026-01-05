import pandas as pd

# Check ground truth
gt_file = 'parameter_tuning_experiments/ground_truth_clds_for_experiments/Social_norms_and_obesity_prevalence.xlsx'
gt_df = pd.read_excel(gt_file)

print('GROUND TRUTH:')
print(f'Total edges: {len(gt_df)}')
print('\nEdges by relationship type:')
print(gt_df['relationship'].value_counts())

# Check corrupted (before correction)
corrupted_file = 'parameter_tuning_experiments/results/rq1_corruption_experiment_20251105_140909/judged_no_correction_aaa235c9_20251105_141523.xlsx'
cor_df = pd.read_excel(corrupted_file, sheet_name='All Edges')

print('\n' + '='*60)
print('\nCORRUPTED (Phase 2a - before correction):')
print(f'Total edges: {len(cor_df)}')
print('\nEdges by relationship type:')
print(cor_df['Relationship Type'].value_counts())

# Check corrected
corrected_file = 'parameter_tuning_experiments/results/rq1_corruption_experiment_20251105_140909/corrected_8d804169_20251105_143343.xlsx'
corr_df = pd.read_excel(corrected_file, sheet_name='All Edges')

print('\n' + '='*60)
print('\nCORRECTED (Phase 2b - after agent correction):')
print(f'Total edges: {len(corr_df)}')
print('\nEdges by relationship type:')
print(corr_df['Relationship Type'].value_counts())

print('\n' + '='*60)
print('\nSUMMARY:')
gt_causal = len(gt_df[gt_df['relationship'] != 'NONE'])
cor_causal = len(cor_df[cor_df['Relationship Type'] != 'NONE'])
corr_causal = len(corr_df[corr_df['Relationship Type'] != 'NONE'])

print(f'Ground truth has {gt_causal} causal edges')
print(f'Corrupted had {cor_causal} causal edges')
print(f'Corrected has {corr_causal} causal edges')
print(f'\nAgent removed {cor_causal - corr_causal} causal edges')

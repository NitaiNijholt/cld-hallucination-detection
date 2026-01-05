import pandas as pd

# Load the files
df_before = pd.read_excel('parameter_tuning_experiments/results/rq1_corruption_experiment_20251104_032531/judged_no_correction_d315026d_20251104_130713.xlsx', sheet_name='All Edges')
df_after = pd.read_excel('parameter_tuning_experiments/results/rq1_corruption_experiment_20251104_032531/corrected_34d738d6_20251105_010956.xlsx', sheet_name='All Edges')

print('=' * 100)
print('ANALYZING INCORRECTLY REMOVED EDGES')
print('=' * 100)

# Find edges that were INCORRECT before correction
incorrect_before = df_before[df_before['Judge Verdict'] == 'INCORRECT']
print(f'\n📋 Total edges with INCORRECT verdict before correction: {len(incorrect_before)}')
print(f'   - Actually corrupted (TP): {incorrect_before["Is Corrupted"].sum()}')
print(f'   - Actually clean (FP - FALSE POSITIVES): {(~incorrect_before["Is Corrupted"]).sum()}')

# Find which of these were removed
removed_edges = []
for idx, row_before in incorrect_before.iterrows():
    src = row_before['Source']
    tgt = row_before['Target']
    is_corrupted = row_before['Is Corrupted']
    
    # Check if edge still exists after correction
    match = df_after[(df_after['Source'] == src) & (df_after['Target'] == tgt)]
    
    if len(match) == 0:
        # Edge was removed
        removed_edges.append({
            'Source': src,
            'Target': tgt,
            'Is Corrupted': is_corrupted,
            'Judge Verdict': row_before['Judge Verdict'],
            'Judge Message': row_before['Judge Message']
        })
    elif match.iloc[0]['Relationship Type'] is None or pd.isna(match.iloc[0]['Relationship Type']):
        # Edge was removed (marked as None type)
        removed_edges.append({
            'Source': src,
            'Target': tgt,
            'Is Corrupted': is_corrupted,
            'Judge Verdict': row_before['Judge Verdict'],
            'Judge Message': row_before['Judge Message']
        })

removed_df = pd.DataFrame(removed_edges)

if len(removed_df) > 0:
    print(f'\n📊 Edges removed by corrector: {len(removed_df)}')
    print(f'   - Actually corrupted (CORRECT removal): {removed_df["Is Corrupted"].sum()}')
    print(f'   - Actually clean (INCORRECT removal): {(~removed_df["Is Corrupted"]).sum()}')
    
    # Show FALSE NEGATIVE REMOVALS (clean edges incorrectly removed)
    false_removals = removed_df[~removed_df['Is Corrupted']]
    if len(false_removals) > 0:
        print(f'\n\n❌ {len(false_removals)} CLEAN EDGES WERE INCORRECTLY REMOVED:')
        print('=' * 100)
        for idx, row in false_removals.head(10).iterrows():
            print(f'\n{idx+1}. {row["Source"]} → {row["Target"]}')
            print(f'   Is Corrupted: {row["Is Corrupted"]} (FALSE - this is a valid edge!)')
            print(f'   Judge Verdict: {row["Judge Verdict"]}')
            print(f'   Judge Message: {row["Judge Message"][:200]}...')
else:
    print('\n✓ No edges were removed')

print('\n' + '=' * 100)
print('CONCLUSION')
print('=' * 100)
print(f'''
The corrector is blindly trusting the judge's INCORRECT verdicts and removing edges.
However, the judge has {(~incorrect_before["Is Corrupted"]).sum()} FALSE POSITIVES (incorrectly flagging clean edges).
These false positives are being removed by the corrector, which damages CLD accuracy.

RECOMMENDATION:
1. Don't immediately remove edges with INCORRECT verdicts
2. Try to REVISE the motivation first
3. Only remove if revision fails or if multiple judges agree it's incorrect
4. Consider using a confidence threshold for removal decisions
''')

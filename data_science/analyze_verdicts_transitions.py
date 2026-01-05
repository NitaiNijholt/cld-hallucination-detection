import pandas as pd
import sys

# Load both files
df_before = pd.read_excel('parameter_tuning_experiments/results/rq1_corruption_experiment_20251104_032531/judged_no_correction_d315026d_20251104_130713.xlsx', sheet_name='All Edges')
df_after = pd.read_excel('parameter_tuning_experiments/results/rq1_corruption_experiment_20251104_032531/corrected_568d4702_20251105_002727.xlsx', sheet_name='All Edges')

print('=' * 80)
print('3. VERDICT TRANSITIONS ANALYSIS')
print('=' * 80)

# Create comparison by matching on Source and Target
merged = []
for idx_before, row_before in df_before.iterrows():
    src = row_before['Source']
    tgt = row_before['Target']
    
    # Find matching row in after
    match = df_after[(df_after['Source'] == src) & (df_after['Target'] == tgt)]
    if len(match) > 0:
        row_after = match.iloc[0]
        if row_before['Judge Verdict'] != row_after['Judge Verdict']:
            merged.append({
                'Source': src,
                'Target': tgt,
                'Before': row_before['Judge Verdict'],
                'After': row_after['Judge Verdict'],
                'Is Corrupted': row_before['Is Corrupted'],
                'Was Corrected': row_after.get('Corrected', None),
                'Correction Action': row_after.get('Correction Action', None)
            })

print(f'\nTotal verdict changes: {len(merged)}/{len(df_before)} ({100*len(merged)/len(df_before):.1f}%)')

if len(merged) > 0:
    print('\n📋 Detailed verdict transitions:')
    trans_df = pd.DataFrame(merged)
    print(trans_df.to_string(index=False))
    
    print('\n\n📊 Transition summary:')
    summary = trans_df.groupby(['Before', 'After']).size().reset_index(name='Count')
    print(summary.to_string(index=False))

print('\n\n🔍 Which edges were actually corrected?')
print(f'Edges with Corrected=True: {df_after["Corrected"].sum() if "Corrected" in df_after.columns else "N/A"}')
print(f'Non-null Correction Actions: {df_after["Correction Action"].notna().sum() if "Correction Action" in df_after.columns else "N/A"}')

if 'Correction Action' in df_after.columns and df_after['Correction Action'].notna().sum() > 0:
    print(f'\nCorrection Action distribution:')
    print(df_after['Correction Action'].value_counts())

print('\n' + '=' * 80)
print('4. DETAILED ANALYSIS OF INCORRECT VERDICTS')
print('=' * 80)

# Analyze INCORRECT verdicts before correction
incorrect_before = df_before[df_before['Judge Verdict'] == 'INCORRECT']
print(f'\n🔍 43 INCORRECT verdicts BEFORE correction:')
print(f'  - Actually corrupted (TP): {incorrect_before["Is Corrupted"].sum()}/{len(incorrect_before)}')
print(f'  - Actually clean (FP): {(~incorrect_before["Is Corrupted"]).sum()}/{len(incorrect_before)}')

print(f'\n❓ WHY were these 43 edges NOT corrected?')
print('   The corrector should have acted on them since they match the selection criteria...')

print('\n\n📋 Sample INCORRECT edges:')
sample_incorrect = incorrect_before[['Source', 'Target', 'Is Corrupted', 'Judge Message', 'Motivation']].head(5)
print(sample_incorrect.to_string(index=False))

print('\n' + '=' * 80)
print('5. ANALYZING PARTIALLY_CORRECT VERDICTS')
print('=' * 80)

partial_before = df_before[df_before['Judge Verdict'] == 'PARTIALLY_CORRECT']
print(f'\n🔍 21 PARTIALLY_CORRECT verdicts BEFORE correction:')
print(f'  - Actually corrupted: {partial_before["Is Corrupted"].sum()}/{len(partial_before)}')
print(f'  - Actually clean: {(~partial_before["Is Corrupted"]).sum()}/{len(partial_before)}')

print(f'\n💡 Should we include PARTIALLY_CORRECT in corrector selection criteria?')

print('\n' + '=' * 80)
print('6. RECOMMENDATION: SHOULD CORRECTOR ACT ON THESE VERDICTS?')
print('=' * 80)

print('\nCurrent criteria selects:')
print('  ✅ INCORRECT (43 edges)')
print('  ❌ PARTIALLY_CORRECT (21 edges) - NOT selected')
print('  ❌ ERROR (1 edge) - NOT selected')

print('\n💡 Analysis suggests:')
print('  - INCORRECT: Should be corrected (currently selected) ✓')
print('  - PARTIALLY_CORRECT: Contains mixed results')
print(f'    • {partial_before["Is Corrupted"].sum()} were actually corrupted')
print(f'    • {(~partial_before["Is Corrupted"]).sum()} were actually clean')
print('    → Should we correct these? Maybe with a different strategy?')
print('  - ERROR: Likely should be corrected (currently NOT selected)')

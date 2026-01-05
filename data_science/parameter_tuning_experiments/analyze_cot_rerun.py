#!/usr/bin/env python3
"""
Analyze CoT Re-run Results (max_tokens=3000)
"""
import pandas as pd

print("=" * 80)
print("ANALYZING CoT RE-RUN RESULTS (max_tokens=3000)")
print("=" * 80)

base_path = 'final_runs/RQ1a_judging_correctness_GT_depresive/run_1/'

# Load baseline, mechanistic (from original), and NEW CoT
baseline = pd.read_excel(base_path + 'judged_Depressive_symptoms_in_response_to_a_stressor_baseline_20251113_235504.xlsx')
mechanistic = pd.read_excel(base_path + 'judged_Depressive_symptoms_in_response_to_a_stressor_mechanistic_20251114_000058.xlsx')
cot_new = pd.read_excel(base_path + 'judged_Depressive_symptoms_in_response_to_a_stressor_chain_of_thought_20251114_001142.xlsx')

print(f"\n📊 DATASET INFO")
print(f"Total edges: {len(cot_new)}")

print("\n" + "=" * 80)
print("1. CoT VERDICT DISTRIBUTION (NEW with max_tokens=3000)")
print("=" * 80)

if 'Judge Verdict' in cot_new.columns:
    counts = cot_new['Judge Verdict'].value_counts()
    total_judged = cot_new['Judge Verdict'].notna().sum()
    print(f"\nChain-of-Thought (NEW):")
    for verdict, count in counts.items():
        pct = 100 * count / len(cot_new)
        print(f"  {verdict}: {count:3d} ({pct:5.1f}%)")
    print(f"  Total judged: {total_judged}/{len(cot_new)}")
    
    # Check for parse errors
    error_count = (cot_new['Judge Verdict'] == 'ERROR').sum()
    print(f"\n🎯 Parse error rate: {error_count}/{len(cot_new)} ({100*error_count/len(cot_new):.1f}%)")
    
    if error_count == 0:
        print("✅ SUCCESS! No parse errors with max_tokens=3000")
    else:
        print(f"⚠️  Still have {error_count} parse errors")

print("\n" + "=" * 80)
print("2. COMPARISON: All 3 Prompts")
print("=" * 80)

print("\nBaseline:")
b_counts = baseline['Judge Verdict'].value_counts()
for verdict, count in b_counts.items():
    pct = 100 * count / len(baseline)
    print(f"  {verdict}: {count:3d} ({pct:5.1f}%)")

print("\nChain-of-Thought (NEW max_tokens=3000):")
c_counts = cot_new['Judge Verdict'].value_counts()
for verdict, count in c_counts.items():
    pct = 100 * count / len(cot_new)
    print(f"  {verdict}: {count:3d} ({pct:5.1f}%)")

print("\nMechanistic:")
m_counts = mechanistic['Judge Verdict'].value_counts()
for verdict, count in m_counts.items():
    pct = 100 * count / len(mechanistic)
    print(f"  {verdict}: {count:3d} ({pct:5.1f}%)")

print("\n" + "=" * 80)
print("3. SCORE DISTRIBUTIONS")
print("=" * 80)

for name, df in [('Baseline', baseline), ('Chain-of-Thought (NEW)', cot_new), ('Mechanistic', mechanistic)]:
    if 'Aggregate Score' in df.columns:
        scores = df['Aggregate Score'].dropna()
        print(f"\n{name}:")
        print(f"  Mean:   {scores.mean():.3f}")
        print(f"  Median: {scores.median():.3f}")
        print(f"  Std:    {scores.std():.3f}")

print("\n" + "=" * 80)
print("4. AGREEMENT RATES")
print("=" * 80)

# Create edge IDs
baseline['edge_id'] = baseline['Source'] + ' -> ' + baseline['Target']
cot_new['edge_id'] = cot_new['Source'] + ' -> ' + cot_new['Target']
mechanistic['edge_id'] = mechanistic['Source'] + ' -> ' + mechanistic['Target']

# Merge
comparison = baseline[['edge_id', 'Judge Verdict']].merge(
    cot_new[['edge_id', 'Judge Verdict']],
    on='edge_id',
    suffixes=('_baseline', '_cot')
).merge(
    mechanistic[['edge_id', 'Judge Verdict']],
    on='edge_id'
).rename(columns={'Judge Verdict': 'mechanistic'})

# Agreement rates
baseline_vs_cot = (comparison['Judge Verdict_baseline'] == comparison['Judge Verdict_cot']).sum()
baseline_vs_mech = (comparison['Judge Verdict_baseline'] == comparison['mechanistic']).sum()
cot_vs_mech = (comparison['Judge Verdict_cot'] == comparison['mechanistic']).sum()

print(f"\nPairwise Agreement:")
print(f"  Baseline vs CoT (NEW):   {baseline_vs_cot:3d}/{len(comparison)} ({100*baseline_vs_cot/len(comparison):5.1f}%)")
print(f"  Baseline vs Mechanistic: {baseline_vs_mech:3d}/{len(comparison)} ({100*baseline_vs_mech/len(comparison):5.1f}%)")
print(f"  CoT vs Mechanistic:      {cot_vs_mech:3d}/{len(comparison)} ({100*cot_vs_mech/len(comparison):5.1f}%)")

# Full agreement
full_agreement = ((comparison['Judge Verdict_baseline'] == comparison['Judge Verdict_cot']) & 
                  (comparison['Judge Verdict_baseline'] == comparison['mechanistic'])).sum()
print(f"\nAll 3 prompts agree: {full_agreement}/{len(comparison)} ({100*full_agreement/len(comparison):.1f}%)")

disagreements = comparison[
    (comparison['Judge Verdict_baseline'] != comparison['Judge Verdict_cot']) |
    (comparison['Judge Verdict_baseline'] != comparison['mechanistic'])
]
print(f"Disagreements: {len(disagreements)} edges ({100*len(disagreements)/len(comparison):.1f}%)")

print("\n" + "=" * 80)
print("5. EXAMPLE DISAGREEMENTS")
print("=" * 80)

# Show a few examples
if len(disagreements) > 0:
    print(f"\nShowing first 5 disagreements:\n")
    for i, row in disagreements.head(5).iterrows():
        print(f"Edge: {row['edge_id']}")
        print(f"  Baseline:    {row['Judge Verdict_baseline']}")
        print(f"  CoT (NEW):   {row['Judge Verdict_cot']}")
        print(f"  Mechanistic: {row['mechanistic']}")
        print()

print("\n" + "=" * 80)
print("✅ FINAL SUMMARY")
print("=" * 80)

error_count_val = (cot_new['Judge Verdict'] == 'ERROR').sum() if 'Judge Verdict' in cot_new.columns else 0
mean_score = cot_new['Aggregate Score'].mean() if 'Aggregate Score' in cot_new.columns else 0

print(f"""
Results with max_tokens=3000:
- Parse errors: {error_count_val} (vs 59 with max_tokens=250)
- Mean score: {mean_score:.3f}
- Full 3-way agreement: {100*full_agreement/len(comparison):.1f}%
- Baseline vs CoT agreement: {100*baseline_vs_cot/len(comparison):.1f}%
- CoT vs Mechanistic agreement: {100*cot_vs_mech/len(comparison):.1f}%
- Total disagreements: {len(disagreements)} edges ({100*len(disagreements)/len(comparison):.1f}%)

✅ All 3 prompt variants are now successfully analyzed!
""")

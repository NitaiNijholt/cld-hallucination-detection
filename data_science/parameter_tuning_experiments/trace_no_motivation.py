import pandas as pd

# Load all 3 phases
base = pd.read_excel("parameter_tuning_experiments/results/rq1_phase0_generation_20251113_173550/result_excel_path_20251113_174500.xlsx", sheet_name="All Edges")
corrupted = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251113_181347/corrupted_6bd82e7d_20251113_181531.xlsx", sheet_name="All Edges")
judged = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251113_181347/judged_no_correction_2daff4b7_20251113_182648.xlsx", sheet_name="All Edges")

print("="*80)
print("WHERE DID NO_MOTIVATION EDGES COME FROM?")
print("="*80)

# Check if motivations exist in base
base['has_motivation'] = base['Motivation'].notna() & (base['Motivation'].str.strip() != '')
print(f"\nBase CLD:")
print(f"  Total edges: {len(base)}")
print(f"  With motivation: {base['has_motivation'].sum()}")
print(f"  Without motivation: {(~base['has_motivation']).sum()}")

# By edge type in base
print(f"\nBase - Motivation by edge type:")
for rel_type in ['POSITIVE', 'NEGATIVE', 'NONE']:
    edges = base[base['Relationship Type'] == rel_type]
    with_mot = edges['has_motivation'].sum()
    print(f"  {rel_type}: {with_mot}/{len(edges)} have motivation ({with_mot/len(edges)*100:.1f}%)")

# Check judged file for NO_MOTIVATION verdict
print(f"\nJudged (Phase 2a):")
no_mot_edges = judged[judged['Judge Verdict'] == 'NO_MOTIVATION']
print(f"  Edges with NO_MOTIVATION verdict: {len(no_mot_edges)}")

# What types are they?
if len(no_mot_edges) > 0:
    print(f"\n  NO_MOTIVATION edges by type:")
    for rel_type in no_mot_edges['Relationship Type'].unique():
        count = (no_mot_edges['Relationship Type'] == rel_type).sum()
        print(f"    {rel_type}: {count}")
    
    # Are they corrupted?
    if 'Is Corrupted' in no_mot_edges.columns:
        corrupted_count = no_mot_edges['Is Corrupted'].sum() if no_mot_edges['Is Corrupted'].dtype == bool else (no_mot_edges['Is Corrupted'] == True).sum()
        print(f"\n  NO_MOTIVATION edges that are corrupted: {corrupted_count}/{len(no_mot_edges)}")

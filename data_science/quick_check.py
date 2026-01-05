import pandas as pd

df = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251105_140909/corrected_672a19cd_20251105_155600.xlsx")

print("=== CORRECTED CLD SUMMARY ===")
print("Total edges:", len(df))

print("\nRelationship types:")
for rt in ['POSITIVE', 'NEGATIVE', 'NONE']:
    count = (df['relationship_type'] == rt).sum()
    print(f"  {rt}: {count}")

print("\n=== EDGE-LEVEL METRICS ===")
causal_edges = df[df['relationship_type'].isin(['POSITIVE', 'NEGATIVE'])]
print("Causal edges:", len(causal_edges))

if 'edge_level_f1' in df.columns:
    avg_f1 = causal_edges['edge_level_f1'].mean()
    print(f"Average F1 score: {avg_f1:.3f}")

if 'edge_level_match' in df.columns:
    matches = causal_edges['edge_level_match'].sum()
    print(f"Matches: {matches}/{len(causal_edges)}")

print("\n=== CORRECTION ACTIONS ===")
if 'corrected' in df.columns:
    print("Corrected edges:", df['corrected'].sum())
if 'added_by_corrector' in df.columns:
    print("Added by corrector:", df['added_by_corrector'].sum())

print("\n=== SAMPLE CAUSAL EDGES ===")
print(causal_edges[['source', 'target', 'relationship_type', 'edge_level_f1']].head(10).to_string(index=False))

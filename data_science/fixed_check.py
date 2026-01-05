import pandas as pd

df = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251105_140909/corrected_672a19cd_20251105_155600.xlsx")

print("=== CORRECTED CLD SUMMARY ===")
print("Total edges:", len(df))
print("\nColumns:", list(df.columns))

print("\nRelationship types:")
for rt in ['POSITIVE', 'NEGATIVE', 'NONE']:
    count = (df['Relationship Type'] == rt).sum()
    print(f"  {rt}: {count}")

print("\n=== CLASSIFICATION (EDGE-LEVEL MATCH) ===")
if 'Classification' in df.columns:
    print(df['Classification'].value_counts())

print("\n=== IN SESSION vs VALIDATION GRAPH ===")
if 'In Session Graph' in df.columns and 'In Validation Graph' in df.columns:
    in_session = df['In Session Graph'].sum()
    in_validation = df['In Validation Graph'].sum()
    print(f"In Session Graph: {in_session}")
    print(f"In Validation Graph: {in_validation}")

print("\n=== SAMPLE CAUSAL EDGES ===")
causal_edges = df[df['Relationship Type'].isin(['POSITIVE', 'NEGATIVE'])]
print(f"\nTotal causal edges: {len(causal_edges)}")
print(causal_edges[['Source', 'Target', 'Relationship Type', 'Classification', 'Judge Verdict']].head(15).to_string(index=False))

print("\n=== GROUND TRUTH COMPARISON ===")
# Read the ground truth
gt_df = pd.read_excel("parameter_tuning_experiments/ground_truth_clds_for_experiments/Social_norms_and_obesity_prevalence.xlsx")
print(f"Ground truth causal edges: {len(gt_df[gt_df['Relationship Type'].isin(['POSITIVE', 'NEGATIVE'])])}")

# Show matches
if 'Classification' in df.columns:
    tp = (df['Classification'] == 'TP').sum()
    fp = (df['Classification'] == 'FP').sum()
    fn = (df['Classification'] == 'FN').sum()
    tn = (df['Classification'] == 'TN').sum()
    
    print(f"\nConfusion Matrix:")
    print(f"  TP (True Positives): {tp}")
    print(f"  FP (False Positives): {fp}")
    print(f"  TN (True Negatives): {tn}")
    print(f"  FN (False Negatives): {fn}")
    
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        print(f"\nPrecision: {precision:.3f}")
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
        print(f"Recall: {recall:.3f}")
    if (tp + fp) > 0 and (tp + fn) > 0:
        f1 = 2 * precision * recall / (precision + recall)
        print(f"F1 Score: {f1:.3f}")

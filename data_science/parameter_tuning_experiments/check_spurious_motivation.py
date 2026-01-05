import pandas as pd

corrupted = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251113_181347/corrupted_6bd82e7d_20251113_181531.xlsx", sheet_name="All Edges")

# Get corrupted edges
corrupted_edges = corrupted[corrupted['Is Corrupted'] == True].copy()

print("="*80)
print(f"CORRUPTION ANALYSIS - {len(corrupted_edges)} CORRUPTED EDGES")
print("="*80)

# Check for spurious_motivation column
if 'Spurious Motivation' in corrupted_edges.columns:
    has_spurious_mot = corrupted_edges['Spurious Motivation'].notna() & (corrupted_edges['Spurious Motivation'].str.strip() != '')
    print(f"\nEdges WITH Spurious Motivation column filled: {has_spurious_mot.sum()}/{len(corrupted_edges)}")
    print(f"Edges WITHOUT Spurious Motivation column: {(~has_spurious_mot).sum()}/{len(corrupted_edges)}")
    
    # Show examples with and without
    print("\n" + "="*80)
    print("EXAMPLES WITH Spurious Motivation:")
    print("="*80)
    with_spur = corrupted_edges[has_spurious_mot].head(3)
    for idx, row in with_spur.iterrows():
        print(f"\n{row['Source']} -[{row['Relationship Type']}]-> {row['Target']}")
        print(f"  Spurious Motivation: {str(row['Spurious Motivation'])[:150]}...")
    
    print("\n" + "="*80)
    print("EXAMPLES WITHOUT Spurious Motivation (stored in main Motivation field):")
    print("="*80)
    without_spur = corrupted_edges[~has_spurious_mot].head(5)
    for idx, row in without_spur.iterrows():
        print(f"\n{row['Source']} -[{row['Relationship Type']}]-> {row['Target']}")
        print(f"  Motivation: {str(row['Motivation'])[:150]}...")
        print(f"  Is Corrupted: {row['Is Corrupted']}")

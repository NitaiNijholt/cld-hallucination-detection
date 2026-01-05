"""
Compare Simple (add/remove only) vs Sophisticated (4 tools) Corrector Results

Analyzes both corrected CLDs against the base CLD to compute recovery metrics.
"""

import pandas as pd
import sys

print("="*80)
print("SIMPLE vs SOPHISTICATED CORRECTOR COMPARISON")
print("="*80)

# File paths
base_file = "parameter_tuning_experiments/results/rq1_phase0_generation_20251113_173550/result_excel_path_20251113_174500.xlsx"
corrupted_file = "parameter_tuning_experiments/results/rq1_corruption_experiment_20251113_181347/corrupted_6bd82e7d_20251113_181531.xlsx"
sophisticated_file = "parameter_tuning_experiments/results/rq1_corruption_experiment_20251113_181347/corrected_b4a635ec_20251113_201447.xlsx"
simple_file = "parameter_tuning_experiments/results/rq1_simple_corrector_20251113/simple_corrected_2163de07_20251113_223851.xlsx"

# Load data
print("\n📂 Loading files...")
base = pd.read_excel(base_file, sheet_name="All Edges")
corrupted = pd.read_excel(corrupted_file, sheet_name="All Edges")
sophisticated = pd.read_excel(sophisticated_file, sheet_name="All Edges")
simple = pd.read_excel(simple_file, sheet_name="All Edges")

# Get causal edges (POSITIVE/NEGATIVE)
def get_causal_edges(df):
    causal = df[df['Relationship Type'].isin(['POSITIVE', 'NEGATIVE'])]
    return set(zip(causal['Source'], causal['Target']))

base_edges = get_causal_edges(base)
corrupted_edges = get_causal_edges(corrupted)
sophisticated_edges = get_causal_edges(sophisticated)
simple_edges = get_causal_edges(simple)

print(f"  Base CLD: {len(base_edges)} causal edges")
print(f"  Corrupted: {len(corrupted_edges)} causal edges")
print(f"  Sophisticated corrector: {len(sophisticated_edges)} causal edges")
print(f"  Simple corrector: {len(simple_edges)} causal edges")

# Calculate metrics for both correctors
def calculate_metrics(corrected_edges, base_edges, name):
    print(f"\n{'='*80}")
    print(f"{name.upper()}")
    print(f"{'='*80}")
    
    tp = len(corrected_edges.intersection(base_edges))
    fp = len(corrected_edges - base_edges)
    fn = len(base_edges - corrected_edges)
    
    # TN is not meaningful in edge recovery (infinite non-edges)
    # So we focus on TP, FP, FN
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"\n📊 RECOVERY METRICS (vs Base CLD):")
    print(f"  TP (base edges kept):     {tp}")
    print(f"  FP (spurious edges kept): {fp}")
    print(f"  FN (base edges removed):  {fn}")
    print()
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    
    return {"name": name, "tp": tp, "fp": fp, "fn": fn, 
            "precision": precision, "recall": recall, "f1": f1}

# Calculate for both
sophisticated_metrics = calculate_metrics(sophisticated_edges, base_edges, "Sophisticated Corrector (4 tools)")
simple_metrics = calculate_metrics(simple_edges, base_edges, "Simple Corrector (2 tools)")

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

print(f"\n{'Metric':<20} {'Sophisticated':<15} {'Simple':<15} {'Difference':<15}")
print("-" * 65)
print(f"{'Precision':<20} {sophisticated_metrics['precision']:.3f}{'':<11} {simple_metrics['precision']:.3f}{'':<11} {simple_metrics['precision']-sophisticated_metrics['precision']:+.3f}")
print(f"{'Recall':<20} {sophisticated_metrics['recall']:.3f}{'':<11} {simple_metrics['recall']:.3f}{'':<11} {simple_metrics['recall']-sophisticated_metrics['recall']:+.3f}")
print(f"{'F1 Score':<20} {sophisticated_metrics['f1']:.3f}{'':<11} {simple_metrics['f1']:.3f}{'':<11} {simple_metrics['f1']-sophisticated_metrics['f1']:+.3f}")

# Interpretation
print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")

f1_diff = simple_metrics['f1'] - sophisticated_metrics['f1']
if abs(f1_diff) < 0.05:
    print("\n✅ SIMILAR PERFORMANCE")
    print(f"   F1 difference: {f1_diff:+.3f} (< 0.05 threshold)")
    print("   → Sophisticated tools (revise, flip, change_type) add NO significant value")
    print("   → Simple add/remove is sufficient for this task")
elif f1_diff > 0.05:
    print("\n✅ SIMPLE CORRECTOR BETTER")
    print(f"   F1 difference: {f1_diff:+.3f}")
    print("   → Sophisticated tools may be causing errors")
    print("   → Simple add/remove is more reliable")
else:
    print("\n⚠️ SOPHISTICATED CORRECTOR BETTER")
    print(f"   F1 difference: {f1_diff:+.3f}")
    print("   → Sophisticated tools do add value")
    print("   → Complex corrections (revise/flip/change) improve results")

# Action distribution
print(f"\n{'='*80}")
print("ACTION DISTRIBUTION")
print(f"{'='*80}")

def count_actions(df, name):
    print(f"\n{name}:")
    corrected = df[df['Corrected'] == True]
    if len(corrected) > 0:
        actions = corrected['Correction Action'].value_counts()
        total = len(corrected)
        for action, count in actions.items():
            pct = count/total*100
            print(f"  {action}: {count}/{total} ({pct:.1f}%)")
    else:
        print("  No corrections applied")

count_actions(sophisticated, "Sophisticated Corrector")
count_actions(simple, "Simple Corrector")

print(f"\n{'='*80}")
print("✅ COMPARISON COMPLETE")
print(f"{'='*80}")

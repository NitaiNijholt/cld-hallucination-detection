#!/usr/bin/env python3
"""
Analyze Corruption Recovery: How well does correction recover the base CLD?
"""

import pandas as pd
from pathlib import Path
import json

# Base directory
base_dir = Path("parameter_tuning_experiments/results/rq1_multi_cld_20251105_205040")

# Load summary
with open(base_dir / "experiment_summary.json", 'r') as f:
    summary = json.load(f)

print("="*80)
print("CORRUPTION RECOVERY ANALYSIS")
print("="*80)
print("Question: How well does the corrector recover the original base CLD?")
print("(Base CLD = Synthetic Ground Truth from Phase 0)")
print()

for cld_result in summary['results']:
    cld_name = cld_result['cld_name']
    output_dir = Path(cld_result['output_dir'])
    
    print(f"\n{'='*80}")
    print(f"CLD: {cld_name}")
    print(f"{'='*80}")
    
    # Load base, corrupted, and corrected Excel files
    base_xlsx = list(output_dir.glob("result_excel_path_*.xlsx"))[0]  # Phase 0
    corrupted_xlsx = list(output_dir.glob("corrupted_*.xlsx"))[0]
    corrected_xlsx = list(output_dir.glob("corrected_*.xlsx"))[0]
    
    # Read All Edges sheets
    df_base = pd.read_excel(base_xlsx, sheet_name="All Edges")
    df_corrupted = pd.read_excel(corrupted_xlsx, sheet_name="All Edges")
    df_corrected = pd.read_excel(corrected_xlsx, sheet_name="All Edges")
    
    # Filter to causal edges only
    df_base = df_base[df_base['Relationship Type'] != 'NONE'].copy()
    df_corrupted = df_corrupted[df_corrupted['Relationship Type'] != 'NONE'].copy()
    df_corrected = df_corrected[df_corrected['Relationship Type'] != 'NONE'].copy()
    
    print(f"\n📊 EDGE COUNTS:")
    print(f"  Base (Phase 0):      {len(df_base)} causal edges")
    print(f"  Corrupted (Phase 1): {len(df_corrupted)} causal edges")
    print(f"  Corrected (Phase 2b): {len(df_corrected)} causal edges")
    
    # Create edge signatures for matching
    def edge_sig(row):
        return f"{row['Source']}|{row['Target']}|{row['Relationship Type']}"
    
    base_edges = set(df_base.apply(edge_sig, axis=1))
    corrupted_edges = set(df_corrupted.apply(edge_sig, axis=1))
    corrected_edges = set(df_corrected.apply(edge_sig, axis=1))
    
    # Analysis: What happened during corruption?
    print(f"\n🔍 CORRUPTION ANALYSIS:")
    added_by_corruption = corrupted_edges - base_edges
    print(f"  Spurious edges added: {len(added_by_corruption)}")
    
    # Check for polarity flips
    base_pairs = {f"{row['Source']}|{row['Target']}" for _, row in df_base.iterrows()}
    corrupted_pairs = {f"{row['Source']}|{row['Target']}" for _, row in df_corrupted.iterrows()}
    
    polarity_flips = 0
    for _, base_row in df_base.iterrows():
        pair = f"{base_row['Source']}|{base_row['Target']}"
        if pair in corrupted_pairs:
            # Find in corrupted
            corr_row = df_corrupted[
                (df_corrupted['Source'] == base_row['Source']) & 
                (df_corrupted['Target'] == base_row['Target'])
            ]
            if len(corr_row) > 0:
                if base_row['Relationship Type'] != corr_row.iloc[0]['Relationship Type']:
                    polarity_flips += 1
    
    print(f"  Polarity flips: {polarity_flips}")
    print(f"  Total corruptions: {len(added_by_corruption) + polarity_flips}")
    
    # Analysis: What happened during correction?
    print(f"\n🔧 CORRECTION ANALYSIS:")
    removed_by_correction = corrupted_edges - corrected_edges
    added_by_correction = corrected_edges - corrupted_edges
    
    print(f"  Edges removed: {len(removed_by_correction)}")
    print(f"  Edges added: {len(added_by_correction)}")
    
    # Recovery metrics
    print(f"\n📈 RECOVERY METRICS (vs Base CLD):")
    
    # True Positives: Base edges correctly kept in corrected
    tp_recovery = len(base_edges & corrected_edges)
    
    # False Positives: Spurious edges still in corrected
    fp_recovery = len(corrected_edges - base_edges)
    
    # False Negatives: Base edges removed by corrector
    fn_recovery = len(base_edges - corrected_edges)
    
    # True Negatives: Spurious edges correctly removed
    # (This is tricky - we need to know all possible spurious edges)
    # For now, use the ones that were added during corruption
    tn_recovery = len(added_by_corruption - corrected_edges)
    
    precision_recovery = tp_recovery / (tp_recovery + fp_recovery) if (tp_recovery + fp_recovery) > 0 else 0
    recall_recovery = tp_recovery / (tp_recovery + fn_recovery) if (tp_recovery + fn_recovery) > 0 else 0
    f1_recovery = 2 * (precision_recovery * recall_recovery) / (precision_recovery + recall_recovery) if (precision_recovery + recall_recovery) > 0 else 0
    
    print(f"  TP (base edges kept):          {tp_recovery}")
    print(f"  FP (spurious edges kept):      {fp_recovery}")
    print(f"  FN (base edges removed):       {fn_recovery}")
    print(f"  TN (spurious edges removed):   {tn_recovery}")
    print()
    print(f"  Precision (recovery):  {precision_recovery:.3f}")
    print(f"  Recall (recovery):     {recall_recovery:.3f}")
    print(f"  F1 Score (recovery):   {f1_recovery:.3f}")
    
    # Compare to corrupted state
    tp_corrupted = len(base_edges & corrupted_edges)
    fp_corrupted = len(corrupted_edges - base_edges)
    fn_corrupted = len(base_edges - corrupted_edges)
    
    precision_corrupted = tp_corrupted / (tp_corrupted + fp_corrupted) if (tp_corrupted + fp_corrupted) > 0 else 0
    recall_corrupted = tp_corrupted / (tp_corrupted + fn_corrupted) if (tp_corrupted + fn_corrupted) > 0 else 0
    f1_corrupted = 2 * (precision_corrupted * recall_corrupted) / (precision_corrupted + recall_corrupted) if (precision_corrupted + recall_corrupted) > 0 else 0
    
    print(f"\n📊 COMPARISON:")
    print(f"{'Metric':<20} {'Corrupted':<15} {'Corrected':<15} {'Improvement':<15}")
    print("-" * 65)
    print(f"{'Precision':<20} {precision_corrupted:<15.3f} {precision_recovery:<15.3f} {precision_recovery - precision_corrupted:+.3f}")
    print(f"{'Recall':<20} {recall_corrupted:<15.3f} {recall_recovery:<15.3f} {recall_recovery - recall_corrupted:+.3f}")
    print(f"{'F1 Score':<20} {f1_corrupted:<15.3f} {f1_recovery:<15.3f} {f1_recovery - f1_corrupted:+.3f}")
    
    print(f"\n💡 INTERPRETATION:")
    if f1_recovery > f1_corrupted:
        print(f"  ✅ Corrector successfully recovered towards base CLD")
    elif f1_recovery < f1_corrupted:
        print(f"  ❌ Corrector made things worse")
    else:
        print(f"  ⚠️  No improvement")

print(f"\n{'='*80}")
print("✅ ANALYSIS COMPLETE")
print(f"{'='*80}\n")




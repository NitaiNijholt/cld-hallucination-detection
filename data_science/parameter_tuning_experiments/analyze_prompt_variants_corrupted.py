#!/usr/bin/env python3
"""
RQ1 Prompt Variants Analysis - Corrupted CLD

Analyzes judge performance across multiple prompt variants on a corrupted CLD.
Dynamically discovers variant names from filenames.

Compares:
- Judge performance (Precision, Recall, F1, Accuracy) for each prompt
- Inter-prompt agreement
- Verdict distributions

Usage:
    python analyze_prompt_variants_corrupted.py <judged_xlsx> [<judged_xlsx> ...] [--output-dir DIR]

Example:
    python analyze_prompt_variants_corrupted.py \\
        judged_*_baseline_*.xlsx \\
        judged_*_cot_*.xlsx \\
        judged_*_mechanistic_v2_*.xlsx \\
        --output-dir analysis/
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import re
from itertools import combinations

def extract_variant_name(filename: str) -> str:
    """
    Extract variant name from judged Excel filename.
    
    Patterns:
    - judged_CLD_NAME_VARIANT_TIMESTAMP.xlsx → VARIANT
    - judged_citation_CLD_NAME_VARIANT_TIMESTAMP.xlsx → VARIANT
    
    Examples:
    - judged_Depressive_symptoms_baseline_20251114_010000_20251114_010100.xlsx → baseline
    - judged_citation_Social_norms_mechanistic_v2_20251114_020000.xlsx → mechanistic_v2
    """
    # Remove .xlsx extension
    name = Path(filename).stem
    
    # Remove leading "judged_" or "judged_citation_"
    if name.startswith('judged_citation_'):
        name = name[len('judged_citation_'):]
    elif name.startswith('judged_'):
        name = name[len('judged_'):]
    
    # Pattern: CLD_NAME_VARIANT_TIMESTAMP_TIMESTAMP or CLD_NAME_VARIANT_TIMESTAMP
    # Split by underscores and remove parts that look like timestamps
    parts = name.split('_')
    
    # Filter out timestamp patterns (8 digits followed by 6 digits)
    filtered_parts = []
    for part in parts:
        # Skip if it looks like a date (YYYYMMDD - 8 digits starting with 19 or 20)
        if re.match(r'^(19|20)\d{6}$', part):
            continue
        # Skip if it looks like a time (HHMMSS - 6 digits)
        if re.match(r'^\d{6}$', part):
            continue
        filtered_parts.append(part)
    
    # The last remaining part should be the variant name
    if filtered_parts:
        return filtered_parts[-1]
    
    return "unknown"

def load_metrics_from_excel(excel_path: Path) -> dict:
    """Extract precision, recall, F1 from Excel summary sheet."""
    try:
        # Try multiple possible sheet names
        sheet_names_to_try = ['Metrics', 'Cited Edges Only Summary', 'All Edges Summary', 'Summary']
        df = None
        
        for sheet_name in sheet_names_to_try:
            try:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                break
            except:
                continue
        
        if df is None:
            return {}
        
        # Find rows with the key metrics
        metrics = {}
        
        # Look for precision, recall, F1 in the Metric column
        # Priority: exact matches first, then with 'edge', then with 'node'
        for _, row in df.iterrows():
            metric_name = str(row.get('Metric', '')).lower().strip()
            value = row.get('Value', None)
            
            # Try exact matches first (edge-level metrics)
            if metric_name == 'precision':
                metrics['precision'] = float(value) if value is not None and pd.notna(value) else 0.0
            elif metric_name == 'recall':
                metrics['recall'] = float(value) if value is not None and pd.notna(value) else 0.0
            elif metric_name == 'f1 score':
                metrics['f1'] = float(value) if value is not None and pd.notna(value) else 0.0
            # Fallback: look for 'edge' keyword
            elif 'precision' in metric_name and 'edge' in metric_name and 'precision' not in metrics:
                metrics['precision'] = float(value) if value is not None and pd.notna(value) else 0.0
            elif 'recall' in metric_name and 'edge' in metric_name and 'recall' not in metrics:
                metrics['recall'] = float(value) if value is not None and pd.notna(value) else 0.0
            elif 'f1' in metric_name and 'edge' in metric_name and 'f1' not in metrics:
                metrics['f1'] = float(value) if value is not None and pd.notna(value) else 0.0
        
        # Calculate F1 if not found
        if 'f1' not in metrics and 'precision' in metrics and 'recall' in metrics:
            p, r = metrics['precision'], metrics['recall']
            if p + r > 0:
                metrics['f1'] = 2 * (p * r) / (p + r)
            else:
                metrics['f1'] = 0.0
        
        return metrics
    
    except Exception as e:
        print(f"Error loading {excel_path}: {e}")
        return {}

def extract_judge_performance(excel_path: Path) -> dict:
    """
    Extract judge performance metrics from judged Excel file.
    
    Compares:
    - Ground Truth: Is Corrupted column (True = corrupted, False = clean)
    - Judge Prediction: Judge Message aggregate_verdict (INCORRECT = judge says corrupted, CORRECT = judge says clean)
    
    Returns:
    - precision: Of edges flagged as incorrect, how many were actually corrupted
    - recall: Of actually corrupted edges, how many did judge flag as incorrect
    - f1: Harmonic mean of precision and recall
    - accuracy: Overall % of correct judgments
    - confusion matrix: TP, TN, FP, FN
    """
    try:
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        
        # Extract verdict from Judge Message JSON
        verdicts = []
        for _, row in df.iterrows():
            judge_msg = row.get('Judge Message', '')
            is_corrupted = row.get('Is Corrupted', False)
            
            try:
                judge_data = json.loads(judge_msg)
                aggregate_verdict = judge_data.get('aggregate_verdict', 'UNKNOWN')
            except:
                aggregate_verdict = 'UNKNOWN'
            
            verdicts.append({
                'is_corrupted': is_corrupted,
                'judge_verdict': aggregate_verdict
            })
        
        verdicts_df = pd.DataFrame(verdicts)
        
        # Map to binary:
        # Ground truth: True (corrupted) = 1, False (clean) = 0
        # Judge: INCORRECT = 1 (flagged as problem), CORRECT = 0 (accepted), PARTIALLY_CORRECT = 0.5
        # Fill NaN values with False before converting to int
        verdicts_df['ground_truth'] = verdicts_df['is_corrupted'].fillna(False).astype(int)
        
        # For strict evaluation, treat PARTIALLY_CORRECT/Partially supported as INCORRECT (flagged)
        # Support both old citation-based format and new correctness format
        verdict_mapping = {
            # New correctness format
            'INCORRECT': 1,
            'PARTIALLY_CORRECT': 1,
            'CORRECT': 0,
            'UNKNOWN': 0,
            # Old citation-based format
            'Not supported': 1,
            'Partially supported': 1,
            'Supported': 0,
            # Fallback
            'ERROR': 0
        }
        verdicts_df['judge_pred'] = verdicts_df['judge_verdict'].map(verdict_mapping).fillna(0).astype(int)
        
        # Calculate confusion matrix
        tp = ((verdicts_df['ground_truth'] == 1) & (verdicts_df['judge_pred'] == 1)).sum()
        tn = ((verdicts_df['ground_truth'] == 0) & (verdicts_df['judge_pred'] == 0)).sum()
        fp = ((verdicts_df['ground_truth'] == 0) & (verdicts_df['judge_pred'] == 1)).sum()
        fn = ((verdicts_df['ground_truth'] == 1) & (verdicts_df['judge_pred'] == 0)).sum()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(verdicts_df) if len(verdicts_df) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'n_total': len(verdicts_df),
            'n_corrupted': int(verdicts_df['ground_truth'].sum()),
            'n_clean': int((verdicts_df['ground_truth'] == 0).sum())
        }
    
    except Exception as e:
        print(f"Error extracting judge performance from {excel_path}: {e}")
        return {}

def analyze_corruption_score_correlation(excel_path: Path) -> dict:
    """
    Analyze correlation between Is Corrupted (binary) and Aggregate Score (continuous).
    
    Uses point-biserial correlation to measure how well judge scores distinguish
    corrupted vs clean edges.
    
    Returns dict with:
    - point_biserial_r: correlation coefficient
    - p_value: statistical significance
    - corrupted_mean_score: mean score for corrupted edges
    - clean_mean_score: mean score for clean edges
    - score_diff: difference in means
    """
    from scipy import stats
    
    try:
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        
        # Extract scores and corruption status
        scores = []
        for _, row in df.iterrows():
            is_corrupted = row.get('Is Corrupted', False)
            aggregate_score = row.get('Aggregate Score', np.nan)
            
            # Try to parse from Judge Message if not directly available
            if pd.isna(aggregate_score):
                judge_msg = row.get('Judge Message', '')
                try:
                    judge_data = json.loads(judge_msg)
                    aggregate_score = judge_data.get('aggregate_score', np.nan)
                except:
                    pass
            
            scores.append({
                'is_corrupted': 1 if is_corrupted else 0,
                'aggregate_score': aggregate_score
            })
        
        scores_df = pd.DataFrame(scores)
        scores_df = scores_df[scores_df['aggregate_score'].notna()]
        
        if len(scores_df) < 2:
            return {}
        
        # Calculate point-biserial correlation
        r_pb, p_value = stats.pointbiserialr(
            scores_df['is_corrupted'], 
            scores_df['aggregate_score']
        )
        
        # Descriptive statistics
        corrupted_scores = scores_df[scores_df['is_corrupted'] == 1]['aggregate_score']
        clean_scores = scores_df[scores_df['is_corrupted'] == 0]['aggregate_score']
        
        return {
            'point_biserial_r': r_pb,
            'p_value': p_value,
            'n_total': len(scores_df),
            'n_corrupted': len(corrupted_scores),
            'n_clean': len(clean_scores),
            'corrupted_mean_score': corrupted_scores.mean() if len(corrupted_scores) > 0 else 0,
            'corrupted_std_score': corrupted_scores.std() if len(corrupted_scores) > 0 else 0,
            'clean_mean_score': clean_scores.mean() if len(clean_scores) > 0 else 0,
            'clean_std_score': clean_scores.std() if len(clean_scores) > 0 else 0,
            'score_diff': (corrupted_scores.mean() - clean_scores.mean()) if len(corrupted_scores) > 0 and len(clean_scores) > 0 else 0
        }
    
    except Exception as e:
        print(f"Error analyzing corruption-score correlation from {excel_path}: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(
        description='Analyze judge performance across multiple prompt variants on corrupted CLD',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('judged_files', nargs='+', type=str,
                        help='One or more judged Excel files to analyze')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for analysis results (default: same as first judged file)')
    
    args = parser.parse_args()
    
    # Convert to Path objects and validate
    judged_files = [Path(f) for f in args.judged_files]
    for file_path in judged_files:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = judged_files[0].parent
    
    # Extract variant names from filenames
    excel_files = {}
    for file_path in judged_files:
        variant_name = extract_variant_name(file_path.name)
        excel_files[variant_name] = file_path
    
    # Sort variants by name for consistent ordering
    sorted_variants = sorted(excel_files.keys())
    
    print("="*80)
    print("RQ1: PROMPT VARIANTS ANALYSIS - CORRUPTED CLD")
    print("="*80)
    print(f"Analyzing {len(excel_files)} prompt variant(s):")
    for variant_name in sorted_variants:
        print(f"  - {variant_name.upper()}: {excel_files[variant_name].name}")
    print()
    
    # Extract judge performance for each prompt
    print("="*80)
    print("JUDGE PERFORMANCE ANALYSIS")
    print("="*80)
    print("Analyzing judge ability to detect synthetic corruptions...")
    print("Comparing judge verdicts (CORRECT/INCORRECT) against ground truth (Is Corrupted)\n")
    
    judge_results = {}
    
    for prompt_name in sorted_variants:
        label = prompt_name.upper()
        excel_path = excel_files[prompt_name]
        
        print(f"\n{label}")
        print(f"  File: {excel_path.name}")
        judge_perf = extract_judge_performance(excel_path)
        judge_results[prompt_name] = judge_perf
        
        if judge_perf:
            print(f"  Dataset: {judge_perf['n_corrupted']} corrupted + {judge_perf['n_clean']} clean = {judge_perf['n_total']} edges")
            print(f"  Confusion Matrix:")
            print(f"    TP (Correctly flagged corrupted): {judge_perf['tp']}")
            print(f"    TN (Correctly accepted clean):    {judge_perf['tn']}")
            print(f"    FP (Incorrectly flagged clean):   {judge_perf['fp']}")
            print(f"    FN (Missed corrupted):            {judge_perf['fn']}")
            print(f"  Metrics:")
            print(f"    Precision: {judge_perf['precision']:.4f} (of flagged edges, % actually corrupted)")
            print(f"    Recall:    {judge_perf['recall']:.4f} (of corrupted edges, % detected)")
            print(f"    F1 Score:  {judge_perf['f1']:.4f} (harmonic mean)")
            print(f"    Accuracy:  {judge_perf['accuracy']:.4f} (overall % correct)")
        else:
            print("  ✗ Failed to extract judge performance")
    
    # Corruption-Score Correlation Analysis
    print()
    print("="*80)
    print("CORRUPTION-SCORE CORRELATION ANALYSIS")
    print("="*80)
    print("Analyzing correlation between Is Corrupted (binary) and Aggregate Score (continuous)")
    print("Point-biserial correlation measures judge's ability to distinguish corrupted vs clean edges\n")
    
    corruption_score_results = {}
    for prompt_name in sorted_variants:
        label = prompt_name.upper()
        excel_path = excel_files[prompt_name]
        
        print(f"\n{label}")
        corr_result = analyze_corruption_score_correlation(excel_path)
        corruption_score_results[prompt_name] = corr_result
        
        if corr_result:
            print(f"  Point-biserial correlation (r_pb): {corr_result['point_biserial_r']:.4f}")
            print(f"  P-value: {corr_result['p_value']:.6f}")
            
            if corr_result['p_value'] < 0.001:
                sig = "*** (highly significant)"
            elif corr_result['p_value'] < 0.01:
                sig = "** (very significant)"
            elif corr_result['p_value'] < 0.05:
                sig = "* (significant)"
            else:
                sig = "ns (not significant)"
            print(f"  Significance: {sig}")
            
            print(f"  Corrupted edges (n={corr_result['n_corrupted']}): mean score = {corr_result['corrupted_mean_score']:.4f} ± {corr_result['corrupted_std_score']:.4f}")
            print(f"  Clean edges (n={corr_result['n_clean']}): mean score = {corr_result['clean_mean_score']:.4f} ± {corr_result['clean_std_score']:.4f}")
            print(f"  Mean difference: {corr_result['score_diff']:.4f}")
            
            # Interpret correlation strength
            r = abs(corr_result['point_biserial_r'])
            if r > 0.7:
                interpretation = "Strong - Judge scores highly discriminate corruptions"
            elif r > 0.5:
                interpretation = "Moderate - Judge scores moderately discriminate corruptions"
            elif r > 0.3:
                interpretation = "Weak - Judge scores weakly discriminate corruptions"
            else:
                interpretation = "Very weak - Judge scores poorly discriminate corruptions"
            print(f"  Interpretation: {interpretation}")
        else:
            print("  ✗ Failed to extract correlation")
    
    # Create comparison table
    print()
    print("="*80)
    print("PROMPT COMPARISON TABLE")
    print("="*80)
    
    comparison_data = []
    for prompt_name in sorted_variants:
        perf = judge_results.get(prompt_name, {})
        comparison_data.append({
            'Prompt': prompt_name.upper(),
            'Precision': perf.get('precision', 0),
            'Recall': perf.get('recall', 0),
            'F1 Score': perf.get('f1', 0),
            'Accuracy': perf.get('accuracy', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Calculate inter-prompt agreement
    print()
    print("="*80)
    print("INTER-PROMPT AGREEMENT")
    print("="*80)
    
    # Load verdict data for each prompt
    prompt_verdicts = {}
    for prompt_name, excel_path in excel_files.items():
        try:
            df = pd.read_excel(excel_path, sheet_name='All Edges')
            verdicts = []
            for _, row in df.iterrows():
                edge_id = f"{row['Source']}|{row['Target']}"
                judge_msg = row.get('Judge Message', '')
                try:
                    judge_data = json.loads(judge_msg)
                    verdict = judge_data.get('aggregate_verdict', 'UNKNOWN')
                except:
                    verdict = 'UNKNOWN'
                verdicts.append({'edge_id': edge_id, 'verdict': verdict})
            prompt_verdicts[prompt_name] = pd.DataFrame(verdicts).set_index('edge_id')
        except Exception as e:
            print(f"Error loading verdicts from {prompt_name}: {e}")
    
    # Calculate pairwise agreement for ALL variant pairs
    prompt_pairs = list(combinations(sorted_variants, 2))
    agreement_results = []
    
    for p1, p2 in prompt_pairs:
        if p1 in prompt_verdicts and p2 in prompt_verdicts:
            df1 = prompt_verdicts[p1]
            df2 = prompt_verdicts[p2]
            
            # Merge on edge_id
            merged = df1.join(df2, lsuffix='_1', rsuffix='_2', how='inner')
            
            # Calculate agreement
            total = len(merged)
            exact_match = (merged['verdict_1'] == merged['verdict_2']).sum()
            agreement_rate = exact_match / total if total > 0 else 0
            
            agreement_results.append({
                'Prompt 1': p1.upper(),
                'Prompt 2': p2.upper(),
                'Agreement Rate': agreement_rate,
                'Exact Matches': exact_match,
                'Total Edges': total
            })
            
            print(f"\n{p1.upper()} vs {p2.upper()}:")
            print(f"  Agreement Rate: {agreement_rate:.4f} ({exact_match}/{total})")
    
    # Calculate full N-way agreement (all prompts must agree)
    if len(sorted_variants) >= 2:
        # Merge all prompt verdicts
        all_dfs = []
        for variant in sorted_variants:
            if variant in prompt_verdicts:
                df = prompt_verdicts[variant].copy()
                df.columns = [f'verdict_{variant}']
                all_dfs.append(df)
        
        if len(all_dfs) >= 2:
            merged_all = all_dfs[0]
            for df in all_dfs[1:]:
                merged_all = merged_all.join(df)
            
            # Check where all verdicts are equal
            total = len(merged_all)
            verdict_cols = [f'verdict_{v}' for v in sorted_variants if v in prompt_verdicts]
            
            # All agree when all columns have the same value
            all_agree = merged_all[verdict_cols].nunique(axis=1) == 1
            all_agree_count = all_agree.sum()
            full_agreement_rate = all_agree_count / total if total > 0 else 0
            
            print(f"\nFull {len(sorted_variants)}-way agreement:")
            print(f"  Agreement Rate: {full_agreement_rate:.4f} ({all_agree_count}/{total})")
    
    print()
    print("="*80)
    print("PROMPT PERFORMANCE RANKING")
    print("="*80)
    
    # Rank prompts by F1 score
    ranked = sorted(comparison_data, key=lambda x: x['F1 Score'], reverse=True)
    print("\nRanked by F1 Score (Judge Accuracy):")
    for i, prompt_data in enumerate(ranked, 1):
        print(f"  {i}. {prompt_data['Prompt']}: F1={prompt_data['F1 Score']:.4f}, "
              f"Precision={prompt_data['Precision']:.4f}, Recall={prompt_data['Recall']:.4f}")
    
    # Identify best prompt
    best_prompt = ranked[0]
    print(f"\n🏆 Best performing prompt: {best_prompt['Prompt']}")
    print(f"   F1 Score: {best_prompt['F1 Score']:.4f}")
    
    
    # Save results
    output_file = output_dir / "prompt_variants_analysis.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main comparison sheet
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
        
        # Detailed judge performance sheet
        if judge_results:
            judge_perf_data = []
            for prompt_name in sorted_variants:
                if prompt_name in judge_results and judge_results[prompt_name]:
                    perf = judge_results[prompt_name]
                    judge_perf_data.append({
                        'Prompt': prompt_name.upper(),
                        'Excel File': excel_files[prompt_name].name,
                        'Precision': perf.get('precision', 0),
                        'Recall': perf.get('recall', 0),
                        'F1 Score': perf.get('f1', 0),
                        'Accuracy': perf.get('accuracy', 0),
                        'TP': perf.get('tp', 0),
                        'TN': perf.get('tn', 0),
                        'FP': perf.get('fp', 0),
                        'FN': perf.get('fn', 0),
                        'N Corrupted': perf.get('n_corrupted', 0),
                        'N Clean': perf.get('n_clean', 0),
                        'N Total': perf.get('n_total', 0)
                    })
            
            if judge_perf_data:
                judge_perf_df = pd.DataFrame(judge_perf_data)
                judge_perf_df.to_excel(writer, sheet_name='Judge Performance', index=False)
        
        # Inter-prompt agreement sheet
        if agreement_results:
            agreement_df = pd.DataFrame(agreement_results)
            agreement_df.to_excel(writer, sheet_name='Inter-Prompt Agreement', index=False)
        
        # Corruption-Score Correlation sheet
        if corruption_score_results:
            corr_data = []
            for prompt_name in sorted_variants:
                if prompt_name in corruption_score_results and corruption_score_results[prompt_name]:
                    result = corruption_score_results[prompt_name]
                    corr_data.append({
                        'Prompt': prompt_name.upper(),
                        'Point-Biserial r': result.get('point_biserial_r', 0),
                        'P-value': result.get('p_value', 1.0),
                        'Corrupted Mean Score': result.get('corrupted_mean_score', 0),
                        'Clean Mean Score': result.get('clean_mean_score', 0),
                        'Score Difference': result.get('score_diff', 0),
                        'N Corrupted': result.get('n_corrupted', 0),
                        'N Clean': result.get('n_clean', 0),
                        'N Total': result.get('n_total', 0)
                    })
            
            if corr_data:
                corr_df = pd.DataFrame(corr_data)
                corr_df.to_excel(writer, sheet_name='Corruption-Score Correlation', index=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Create visualization
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: F1 Score comparison across prompts
    ax1 = axes[0]
    prompts = [item['Prompt'] for item in comparison_data]
    f1_scores = [item['F1 Score'] for item in comparison_data]
    
    # Generate colors dynamically based on number of prompts
    import matplotlib.cm as cm
    n_prompts = len(comparison_data)
    if n_prompts <= 10:
        # Use tab10 colormap for up to 10 prompts (distinct colors)
        colors = [cm.tab10(i) for i in range(n_prompts)]
    else:
        # Use viridis for many prompts
        colors = [cm.viridis(i/n_prompts) for i in range(n_prompts)]
    
    bars = ax1.bar(prompts, f1_scores, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('Judge F1 Score by Prompt Variant', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(prompts, rotation=15, ha='right')
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: All metrics comparison
    ax2 = axes[1]
    metrics_comparison = comparison_df.set_index('Prompt')[['Precision', 'Recall', 'F1 Score', 'Accuracy']]
    metrics_comparison.plot(kind='bar', ax=ax2, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('All Metrics by Prompt Variant', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15, ha='right')
    ax2.legend(title='Metric', fontsize=9, loc='lower right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_dir / "prompt_variants_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_file}")
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results directory: {output_dir}")
    print(f"  - {output_file.name}")
    print(f"  - {plot_file.name}")
    print()

if __name__ == "__main__":
    main()

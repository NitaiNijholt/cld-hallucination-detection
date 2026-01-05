#!/usr/bin/env python3
"""
RQ1a_ground_truth: TP/FP Analysis - Single Run

Analyzes judge performance on clean (non-corrupted) CLDs by comparing:
- Ground Truth: Edge Classification (TP vs FP)
- Judge Prediction: Verdict (CORRECT vs INCORRECT)

Research Question: Can judges distinguish TP from FP edges?

Usage:
    python analyze_rq1a_ground_truth_tp_fp.py <judged_xlsx> [<judged_xlsx> ...] [--output-dir DIR]

Example:
    python analyze_rq1a_ground_truth_tp_fp.py \\
        final_runs/RQ1a_gt_lit_correctness/depressive/run_1/judged_*_baseline_*.xlsx \\
        final_runs/RQ1a_gt_lit_correctness/depressive/run_1/judged_*_cot_*.xlsx \\
        final_runs/RQ1a_gt_lit_correctness/depressive/run_1/judged_*_mechanistic_*.xlsx \\
        --output-dir final_runs/RQ1a_gt_lit_correctness/depressive/run_1/analysis
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
from scipy import stats

def extract_variant_name(filepath: str) -> str:
    """Extract CLD_run_prompt identifier from file path.
    
    Example: 
    Input: .../depressive/run_1/judged_Depressive_..._baseline_....xlsx
    Output: depressive_run1_baseline
    """
    path = Path(filepath)
    
    # Extract CLD from parent directory (e.g., 'depressive', 'social_norms', 'emergency_department')
    cld = path.parent.parent.name
    
    # Extract run number from parent directory (e.g., 'run_1' -> 'run1')
    run = path.parent.name.replace('_', '')
    
    # Extract prompt type from filename (baseline, cot, mechanistic)
    filename = path.stem
    if 'baseline' in filename.lower():
        prompt = 'baseline'
    elif 'cot' in filename.lower():
        prompt = 'cot'
    elif 'mechanistic' in filename.lower():
        prompt = 'mechanistic'
    else:
        prompt = 'unknown'
    
    return f"{cld}_{run}_{prompt}"

def extract_tp_fp_performance(excel_path: Path) -> dict:
    """
    Extract judge performance on TP vs FP vs FN edges.
    
    Ground Truth: Classification column (TP, FP, FN, TN)
    - TP = edge in both generated & ground truth (TRUE POSITIVE) - correct relationship
    - FP = edge in generated but NOT in ground truth (FALSE POSITIVE) - hallucinated relationship
    - FN = edge in ground truth but generator said NONE (FALSE NEGATIVE) - missed relationship with hallucinated "no relationship" reasoning
    - TN = edge proposed as NONE and not in ground truth (TRUE NEGATIVE) - correct rejection
    
    Judge Prediction: Judge verdict from Judge Message
    - CORRECT = judge says edge/reasoning is correct
    - INCORRECT = judge says edge/reasoning is wrong
    - PARTIALLY_CORRECT = judge says edge/reasoning is partially correct
    
    **CRITICAL INSIGHT**: FN (NONE) edges have hallucinated reasoning!
    - When generator says "NONE" for a relationship that DOES exist in ground truth
    - It provides reasoning for why there's no relationship
    - That reasoning is HALLUCINATED because the relationship actually exists!
    - This is a hallucination of the "no relationship" justification
    
    Expected: 
    - Judges should give lower scores (INCORRECT) to FP edges (wrong relationship)
    - Judges should give lower scores (INCORRECT) to FN edges (missed relationship + hallucinated justification)
    
    Returns dict with:
    - TP/FP/FN verdict distributions
    - Confusion matrix treating both FP and FN as hallucinations
    - Point-biserial correlation
    - Score means by class
    - FN-specific reasoning quality metrics
    """
    try:
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        
        # Analyze ALL edges that were part of the generation/judging process
        # TP = correct relationship (correct)
        # FP = hallucinated relationship (hallucination)
        # FN = missed relationship with "no relationship" reasoning (hallucination!)
        # TN = correctly rejected (correct)
        df = df[df['Classification'].isin(['TP', 'FP', 'FN', 'TN'])].copy()
        
        if len(df) == 0:
            return None
        
        # Extract verdicts and scores
        results = []
        for _, row in df.iterrows():
            classification = row.get('Classification', 'UNKNOWN')
            # Hallucinations include both FP (wrong relationship) and FN (missed relationship with wrong reasoning)
            is_hallucination = (classification in ['FP', 'FN'])
            
            judge_msg = row.get('Judge Message', '')
            aggregate_score = row.get('Aggregate Score', 0.5)
            
            try:
                judge_data = json.loads(judge_msg)
                verdict = judge_data.get('aggregate_verdict', 'UNKNOWN')
            except:
                verdict = 'UNKNOWN'
            
            results.append({
                'classification': classification,
                'is_hallucination': is_hallucination,
                'verdict': verdict,
                'score': float(aggregate_score) if pd.notna(aggregate_score) else 0.5
            })
        
        results_df = pd.DataFrame(results)
        
        # Binary labels for confusion matrix
        # Ground truth: 0 = Correct (TP or TN), 1 = Hallucination (FP or FN)
        # Prediction: 0 = CORRECT (accept), 1 = INCORRECT (reject)
        results_df['y_true'] = results_df['is_hallucination'].astype(int)
        
        verdict_mapping = {
            'CORRECT': 0,
            'PARTIALLY_CORRECT': 0,  # Treat as accepted
            'INCORRECT': 1,
            'UNKNOWN': 0
        }
        results_df['y_pred'] = results_df['verdict'].map(verdict_mapping).fillna(0).astype(int)
        
        # Confusion matrix (detecting hallucinations = FP + FN)
        tp_detect = ((results_df['y_true'] == 1) & (results_df['y_pred'] == 1)).sum()  # Correctly flagged hallucination
        tn_detect = ((results_df['y_true'] == 0) & (results_df['y_pred'] == 0)).sum()  # Correctly accepted correct edge
        fp_detect = ((results_df['y_true'] == 0) & (results_df['y_pred'] == 1)).sum()  # Wrongly flagged correct edge
        fn_detect = ((results_df['y_true'] == 1) & (results_df['y_pred'] == 0)).sum()  # Missed hallucination
        
        # Metrics
        precision = tp_detect / (tp_detect + fp_detect) if (tp_detect + fp_detect) > 0 else 0.0
        recall = tp_detect / (tp_detect + fn_detect) if (tp_detect + fn_detect) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp_detect + tn_detect) / len(results_df) if len(results_df) > 0 else 0.0
        
        # Point-biserial correlation: is_hallucination (binary) vs score (continuous)
        # Expected: negative correlation (lower scores for hallucinations)
        r_pb, p_value = stats.pointbiserialr(results_df['is_hallucination'], results_df['score'])
        
        # Score statistics by class
        tp_scores = results_df[results_df['classification'] == 'TP']['score']
        tn_scores = results_df[results_df['classification'] == 'TN']['score']
        fp_scores = results_df[results_df['classification'] == 'FP']['score']
        fn_scores = results_df[results_df['classification'] == 'FN']['score']
        
        correct_scores = results_df[results_df['classification'].isin(['TP', 'TN'])]['score']
        hallucination_scores = results_df[results_df['classification'].isin(['FP', 'FN'])]['score']
        
        return {
            'total_edges': len(results_df),
            'tp_edges': int((results_df['classification'] == 'TP').sum()),
            'tn_edges': int((results_df['classification'] == 'TN').sum()),
            'fp_edges': int((results_df['classification'] == 'FP').sum()),
            'fn_edges': int((results_df['classification'] == 'FN').sum()),
            'hallucination_edges': int((results_df['classification'].isin(['FP', 'FN'])).sum()),
            'confusion_matrix': {
                'tp': int(tp_detect),  # Correctly flagged hallucination (FP or FN)
                'tn': int(tn_detect),  # Correctly accepted correct edge (TP or TN)
                'fp': int(fp_detect),  # Wrongly flagged correct edge as hallucination
                'fn': int(fn_detect)   # Missed hallucination (accepted FP or FN as correct)
            },
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            },
            'correlation': {
                'r_pb': r_pb,
                'p_value': p_value,
                'interpretation': 'negative' if r_pb < 0 else 'positive'
            },
            'scores': {
                'tp_mean': float(tp_scores.mean()) if len(tp_scores) > 0 else 0.0,
                'tp_std': float(tp_scores.std()) if len(tp_scores) > 0 else 0.0,
                'tn_mean': float(tn_scores.mean()) if len(tn_scores) > 0 else 0.0,
                'tn_std': float(tn_scores.std()) if len(tn_scores) > 0 else 0.0,
                'fp_mean': float(fp_scores.mean()) if len(fp_scores) > 0 else 0.0,
                'fp_std': float(fp_scores.std()) if len(fp_scores) > 0 else 0.0,
                'fn_mean': float(fn_scores.mean()) if len(fn_scores) > 0 else 0.0,
                'fn_std': float(fn_scores.std()) if len(fn_scores) > 0 else 0.0,
                'correct_mean': float(correct_scores.mean()) if len(correct_scores) > 0 else 0.0,
                'correct_std': float(correct_scores.std()) if len(correct_scores) > 0 else 0.0,
                'hallucination_mean': float(hallucination_scores.mean()) if len(hallucination_scores) > 0 else 0.0,
                'hallucination_std': float(hallucination_scores.std()) if len(hallucination_scores) > 0 else 0.0,
                'mean_diff': float(correct_scores.mean() - hallucination_scores.mean()) if len(correct_scores) > 0 and len(hallucination_scores) > 0 else 0.0
            },
            'verdict_distributions': {
                'tp': results_df[results_df['classification'] == 'TP']['verdict'].value_counts().to_dict(),
                'tn': results_df[results_df['classification'] == 'TN']['verdict'].value_counts().to_dict(),
                'fp': results_df[results_df['classification'] == 'FP']['verdict'].value_counts().to_dict(),
                'fn': results_df[results_df['classification'] == 'FN']['verdict'].value_counts().to_dict()
            }
        }
    
    except Exception as e:
        print(f"Error analyzing {excel_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_inter_prompt_agreement(excel_files: dict) -> dict:
    """Calculate agreement rates between different prompts."""
    agreements = {}
    
    # Load verdicts from all prompts
    all_verdicts = {}
    for prompt_name, excel_path in excel_files.items():
        try:
            df = pd.read_excel(excel_path, sheet_name='All Edges')
            df = df[df['Classification'].isin(['TP', 'FP', 'FN', 'TN'])].copy()
            
            # Create edge identifier (src->tgt)
            df['edge_id'] = df['Source Node'] + ' -> ' + df['Target Node']
            
            verdicts = {}
            for _, row in df.iterrows():
                edge_id = row['edge_id']
                judge_msg = row.get('Judge Message', '')
                try:
                    judge_data = json.loads(judge_msg)
                    verdict = judge_data.get('aggregate_verdict', 'UNKNOWN')
                except:
                    verdict = 'UNKNOWN'
                verdicts[edge_id] = verdict
            
            all_verdicts[prompt_name] = verdicts
        except:
            continue
    
    # Calculate pairwise agreement
    prompt_names = list(all_verdicts.keys())
    for i, prompt1 in enumerate(prompt_names):
        for prompt2 in prompt_names[i+1:]:
            # Find common edges
            edges1 = set(all_verdicts[prompt1].keys())
            edges2 = set(all_verdicts[prompt2].keys())
            common_edges = edges1 & edges2
            
            if len(common_edges) == 0:
                continue
            
            # Count agreements
            agreements_count = sum(
                1 for edge in common_edges
                if all_verdicts[prompt1][edge] == all_verdicts[prompt2][edge]
            )
            
            agreement_rate = agreements_count / len(common_edges)
            agreements[f"{prompt1}_vs_{prompt2}"] = {
                'agreement_rate': agreement_rate,
                'common_edges': len(common_edges),
                'agreements': agreements_count
            }
    
    # Calculate n-way agreement (all prompts agree)
    if len(prompt_names) >= 2:
        all_edges = set.intersection(*[set(all_verdicts[p].keys()) for p in prompt_names])
        if len(all_edges) > 0:
            n_way_agreements = sum(
                1 for edge in all_edges
                if len(set(all_verdicts[p][edge] for p in prompt_names)) == 1
            )
            agreements['all_prompts'] = {
                'agreement_rate': n_way_agreements / len(all_edges),
                'common_edges': len(all_edges),
                'agreements': n_way_agreements
            }
    
    return agreements

def create_visualizations(results: dict, output_dir: Path):
    """Create visualization plots."""
    prompts = list(results.keys())
    n_prompts = len(prompts)
    
    # Set up colors (dynamic)
    if n_prompts <= 10:
        colors = plt.cm.tab10(range(n_prompts))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_prompts))
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Panel 1: F1 Scores
    ax1 = axes[0, 0]
    f1_scores = [results[p]['metrics']['f1'] for p in prompts]
    bars = ax1.bar(range(n_prompts), f1_scores, color=colors)
    ax1.set_xticks(range(n_prompts))
    ax1.set_xticklabels([p.upper() for p in prompts], rotation=15, ha='right')
    ax1.set_ylabel('F1 Score', fontweight='bold')
    ax1.set_title('F1 Score: Detecting FP Edges', fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, f1 + 0.02, f'{f1:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    # Panel 2: Precision vs Recall
    ax2 = axes[0, 1]
    precisions = [results[p]['metrics']['precision'] for p in prompts]
    recalls = [results[p]['metrics']['recall'] for p in prompts]
    for i, prompt in enumerate(prompts):
        ax2.scatter(recalls[i], precisions[i], s=200, color=colors[i], 
                   label=prompt.upper(), alpha=0.7, edgecolors='black', linewidth=1.5)
    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision vs Recall (FP Detection)', fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    
    # Panel 3: Point-Biserial Correlations
    ax3 = axes[1, 0]
    correlations = [results[p]['correlation']['r_pb'] for p in prompts]
    bars = ax3.barh(range(n_prompts), correlations, color=colors)
    ax3.set_yticks(range(n_prompts))
    ax3.set_yticklabels([p.upper() for p in prompts])
    ax3.set_xlabel('Point-Biserial r (Score vs FP)', fontweight='bold')
    ax3.set_title('Correlation: Judge Score vs FP Status', fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax3.grid(axis='x', alpha=0.3)
    for i, (bar, r) in enumerate(zip(bars, correlations)):
        sig = '***' if results[prompts[i]]['correlation']['p_value'] < 0.001 else \
              '**' if results[prompts[i]]['correlation']['p_value'] < 0.01 else \
              '*' if results[prompts[i]]['correlation']['p_value'] < 0.05 else ''
        ax3.text(r + (0.02 if r >= 0 else -0.02), bar.get_y() + bar.get_height()/2, 
                f'{r:.3f}{sig}', ha='left' if r >= 0 else 'right', va='center', fontsize=10)
    
    # Panel 4: Score Distributions (TP/TN/FP/FN) - NEW: includes FN!
    ax4 = axes[1, 1]
    width = 0.2
    x = np.arange(n_prompts)
    tp_means = [results[p]['scores']['tp_mean'] for p in prompts]
    tn_means = [results[p]['scores']['tn_mean'] for p in prompts]
    fp_means = [results[p]['scores']['fp_mean'] for p in prompts]
    fn_means = [results[p]['scores']['fn_mean'] for p in prompts]
    
    bars1 = ax4.bar(x - 1.5*width, tp_means, width, label='TP (Correct)', color='darkgreen', alpha=0.8)
    bars2 = ax4.bar(x - 0.5*width, tn_means, width, label='TN (Correct)', color='lightgreen', alpha=0.8)
    bars3 = ax4.bar(x + 0.5*width, fp_means, width, label='FP (Hallucination)', color='red', alpha=0.8)
    bars4 = ax4.bar(x + 1.5*width, fn_means, width, label='FN (Hallucination)', color='darkred', alpha=0.8)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels([p.upper() for p in prompts], rotation=15, ha='right')
    ax4.set_ylabel('Mean Aggregate Score', fontweight='bold')
    ax4.set_title('Score Distributions: TP/TN/FP/FN Edges', fontweight='bold')
    ax4.set_ylim([0, 1])
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    
    # Panel 5: Stacked Bar - Confusion Matrix Breakdown (TP/TN/FP/FN counts)
    ax5 = axes[2, 0]
    tp_counts = [results[p]['confusion_matrix']['tp'] for p in prompts]
    tn_counts = [results[p]['confusion_matrix']['tn'] for p in prompts]
    fp_counts = [results[p]['confusion_matrix']['fp'] for p in prompts]
    fn_counts = [results[p]['confusion_matrix']['fn'] for p in prompts]
    
    x = np.arange(n_prompts)
    ax5.bar(x, tp_counts, label='TP (Correctly Flagged Hall.)', color='darkgreen', alpha=0.8)
    ax5.bar(x, tn_counts, bottom=tp_counts, label='TN (Correctly Accepted)', color='lightgreen', alpha=0.8)
    ax5.bar(x, fp_counts, bottom=np.array(tp_counts)+np.array(tn_counts), 
            label='FP (Wrongly Flagged)', color='orange', alpha=0.8)
    ax5.bar(x, fn_counts, bottom=np.array(tp_counts)+np.array(tn_counts)+np.array(fp_counts),
            label='FN (Missed Hallucination)', color='darkred', alpha=0.8)
    
    ax5.set_xticks(x)
    ax5.set_xticklabels([p.upper() for p in prompts], rotation=15, ha='right')
    ax5.set_ylabel('Number of Edges', fontweight='bold')
    ax5.set_title('Confusion Matrix Breakdown (Hallucination Detection)', fontweight='bold')
    ax5.legend(fontsize=8, loc='upper left')
    ax5.grid(axis='y', alpha=0.3)
    
    # Panel 6: Hallucinations Caught vs Missed (TP+FP detection vs FN missed)
    ax6 = axes[2, 1]
    caught = [results[p]['confusion_matrix']['tp'] for p in prompts]  # Correctly flagged hallucinations
    missed = [results[p]['confusion_matrix']['fn'] for p in prompts]  # Missed hallucinations
    total_hall = [results[p]['hallucination_edges'] for p in prompts]
    catch_rate = [c/t if t > 0 else 0 for c, t in zip(caught, total_hall)]
    
    x = np.arange(n_prompts)
    width = 0.35
    bars1 = ax6.bar(x - width/2, caught, width, label='Caught (TP)', color='green', alpha=0.8)
    bars2 = ax6.bar(x + width/2, missed, width, label='Missed (FN)', color='red', alpha=0.8)
    
    # Add catch rate as text
    for i, rate in enumerate(catch_rate):
        ax6.text(i, max(caught[i], missed[i]) + 2, f'{rate*100:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax6.set_xticks(x)
    ax6.set_xticklabels([p.upper() for p in prompts], rotation=15, ha='right')
    ax6.set_ylabel('Number of Hallucinations', fontweight='bold')
    ax6.set_title('Hallucinations: Caught vs Missed', fontweight='bold')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tp_fp_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot saved to: {output_dir / 'tp_fp_comparison.png'}")

def main():
    parser = argparse.ArgumentParser(description='RQ1a_ground_truth: TP/FP Analysis')
    parser.add_argument('judged_files', nargs='+', help='Judged Excel files (one per prompt variant)')
    parser.add_argument('--output-dir', type=str, help='Output directory for analysis results')
    args = parser.parse_args()
    
    # Extract variant names and organize files
    excel_files = {}
    for file_path in args.judged_files:
        variant_name = extract_variant_name(file_path)
        excel_files[variant_name] = Path(file_path)
    
    sorted_variants = sorted(excel_files.keys())
    
    print("=" * 80)
    print("RQ1a_ground_truth: TP/FP ANALYSIS")
    print("=" * 80)
    print(f"Analyzing {len(sorted_variants)} prompt variant(s):")
    for variant in sorted_variants:
        print(f"  - {variant.upper()}: {excel_files[variant].name}")
    print()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use parent directory of first file
        output_dir = Path(args.judged_files[0]).parent / 'analysis'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each prompt
    print("=" * 80)
    print("ANALYZING JUDGE PERFORMANCE (TP vs FP)")
    print("=" * 80)
    
    results = {}
    for variant in sorted_variants:
        print(f"\n{variant.upper()}:")
        print(f"  File: {excel_files[variant].name}")
        
        perf = extract_tp_fp_performance(excel_files[variant])
        if perf is None:
            print("  ❌ Analysis failed")
            continue
        
        results[variant] = perf
        
        print(f"  Dataset: {perf['tp_edges']} TP + {perf['tn_edges']} TN + {perf['fp_edges']} FP + {perf['fn_edges']} FN = {perf['total_edges']} edges")
        print(f"  Hallucinations (FP+FN): {perf['hallucination_edges']}")
        print(f"  Confusion Matrix (Hallucination Detection):")
        print(f"    TP (Correctly flagged hallucination): {perf['confusion_matrix']['tp']}")
        print(f"    TN (Correctly accepted correct edge): {perf['confusion_matrix']['tn']}")
        print(f"    FP (Wrongly flagged correct edge):    {perf['confusion_matrix']['fp']}")
        print(f"    FN (Missed hallucination):            {perf['confusion_matrix']['fn']}")
        print(f"  Metrics:")
        print(f"    Precision: {perf['metrics']['precision']:.4f}")
        print(f"    Recall:    {perf['metrics']['recall']:.4f}")
        print(f"    F1 Score:  {perf['metrics']['f1']:.4f}")
        print(f"    Accuracy:  {perf['metrics']['accuracy']:.4f}")
        print(f"  Point-Biserial Correlation:")
        print(f"    r_pb: {perf['correlation']['r_pb']:.4f} (p={perf['correlation']['p_value']:.6f})")
        print(f"  Score Means:")
        print(f"    Correct edges (TP+TN):      {perf['scores']['correct_mean']:.4f} ± {perf['scores']['correct_std']:.4f}")
        print(f"    Hallucinations (FP+FN):     {perf['scores']['hallucination_mean']:.4f} ± {perf['scores']['hallucination_std']:.4f}")
        print(f"    Difference (Correct-Hall):  {perf['scores']['mean_diff']:.4f}")
    
    if len(results) == 0:
        print("\n❌ No results to analyze")
        return 1
    
    # Inter-prompt agreement
    print(f"\n{'='*80}")
    print("INTER-PROMPT AGREEMENT")
    print("=" * 80)
    agreements = calculate_inter_prompt_agreement(excel_files)
    for key, value in agreements.items():
        print(f"{key}: {value['agreement_rate']:.4f} ({value['agreements']}/{value['common_edges']})")
    
    # Generate visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    create_visualizations(results, output_dir)
    
    # Save results to Excel
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Create summary DataFrame
    summary_data = []
    for variant in sorted_variants:
        if variant not in results:
            continue
        r = results[variant]
        summary_data.append({
            'Prompt': variant.upper(),
            'Total Edges': r['total_edges'],
            'TP Edges': r['tp_edges'],
            'TN Edges': r['tn_edges'],
            'FP Edges': r['fp_edges'],
            'FN Edges': r['fn_edges'],
            'Hallucinations': r['hallucination_edges'],
            'Precision': r['metrics']['precision'],
            'Recall': r['metrics']['recall'],
            'F1 Score': r['metrics']['f1'],
            'Accuracy': r['metrics']['accuracy'],
            'Point-Biserial r': r['correlation']['r_pb'],
            'P-value': r['correlation']['p_value'],
            'Correct Score Mean': r['scores']['correct_mean'],
            'Hallucination Score Mean': r['scores']['hallucination_mean'],
            'Score Difference (Correct-Hall)': r['scores']['mean_diff']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to Excel with multiple sheets
    excel_path = output_dir / 'tp_fp_analysis.xlsx'
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Agreement matrix
        if agreements:
            agreement_data = []
            for key, value in agreements.items():
                agreement_data.append({
                    'Comparison': key,
                    'Agreement Rate': value['agreement_rate'],
                    'Agreements': value['agreements'],
                    'Common Edges': value['common_edges']
                })
            agreement_df = pd.DataFrame(agreement_data)
            agreement_df.to_excel(writer, sheet_name='Inter-Prompt Agreement', index=False)
    
    print(f"✓ Results saved to: {excel_path}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results directory: {output_dir}")
    print(f"  - tp_fp_analysis.xlsx")
    print(f"  - tp_fp_comparison.png")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

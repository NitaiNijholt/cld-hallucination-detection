#!/usr/bin/env python3
"""
RQ1a Aggregate Analysis - Corruption Detection Across All Runs

Aggregates judge performance across:
- All 3 CLDs (Depressive symptoms, Social norms, Emergency department)
- All 3 judge prompts (Baseline, Mechanistic, SCE)
- Multiple runs per CLD
- All corruption types (Spurious edges, Polarity flips, Motivation corruption)

Produces:
- Aggregate metrics (precision, recall, F1, accuracy)
- Corruption-score correlations
- Breakdown by corruption type
- Thesis-ready tables and LaTeX output

Usage:
    python analyze_rq1a_aggregate.py [--base_dir PATH]

Example:
    python analyze_rq1a_aggregate.py --base_dir /home/nitai/code/causalix.ai/final_runs/RQ1a_corruption_detection_correctness
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def collect_all_run_results(base_dir: Path) -> dict:
    """
    Collect results from all run directories.
    
    Expected structure:
    base_dir/
        {cld_name}/
            run_{N}/
                judged_{cld}_baseline_*.xlsx
                judged_{cld}_mechanistic_*.xlsx
                judged_{cld}_sce_*.xlsx
    
    Returns dict with structure:
        {
            'cld_name': {
                'run_N': {
                    'baseline': Path(...),
                    'mechanistic': Path(...),
                    'sce': Path(...)
                }
            }
        }
    """
    results = {}
    
    # CLD name mappings
    cld_mappings = {
        'depressive': 'Depressive symptoms',
        'social_norms': 'Social norms',
        'emergency_department': 'Emergency department'
    }
    
    print("="*80)
    print("COLLECTING RUN RESULTS")
    print("="*80)
    print(f"Base directory: {base_dir}\n")
    
    for cld_dir in base_dir.iterdir():
        if not cld_dir.is_dir():
            continue
        
        cld_name = cld_dir.name
        if cld_name not in cld_mappings:
            continue
        
        results[cld_name] = {}
        
        # Find all run subdirectories
        run_dirs = sorted([d for d in cld_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
        
        for run_dir in run_dirs:
            run_name = run_dir.name
            
            # Find judged Excel files for each prompt
            baseline_files = list(run_dir.glob("judged_*_baseline_*.xlsx"))
            mechanistic_files = list(run_dir.glob("judged_*_mechanistic_*.xlsx"))
            sce_files = list(run_dir.glob("judged_*_sce_*.xlsx"))
            
            if baseline_files and mechanistic_files and sce_files:
                results[cld_name][run_name] = {
                    'baseline': baseline_files[0],
                    'mechanistic': mechanistic_files[0],
                    'sce': sce_files[0]
                }
                print(f"  ✓ {cld_name}/{run_name}: Found all 3 prompt files")
            else:
                print(f"  ✗ {cld_name}/{run_name}: Missing files (baseline={len(baseline_files)}, mechanistic={len(mechanistic_files)}, sce={len(sce_files)})")
    
    # Summary
    print("\nSummary:")
    total_runs = sum(len(runs) for runs in results.values())
    print(f"  Total CLDs: {len(results)}")
    print(f"  Total runs: {total_runs}")
    for cld_name, runs in results.items():
        print(f"    {cld_name}: {len(runs)} runs")
    
    return results


def extract_judge_performance_from_file(excel_path: Path) -> dict:
    """
    Extract judge performance metrics from a single judged Excel file.
    
    Returns metrics for corruption detection:
    - precision, recall, F1, accuracy
    - confusion matrix (TP, TN, FP, FN)
    - corruption-score correlation
    """
    try:
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        
        # Extract verdicts and scores
        data = []
        for _, row in df.iterrows():
            is_corrupted = row.get('Is Corrupted', False)
            aggregate_score = row.get('Aggregate Score', np.nan)
            corruption_type = row.get('Corruption Type', '')
            corruption_subtype = row.get('Corruption Subtype', '')
            
            # Parse judge message
            judge_msg = row.get('Judge Message', '')
            try:
                judge_data = json.loads(judge_msg)
                aggregate_verdict = judge_data.get('aggregate_verdict', 'UNKNOWN')
                if pd.isna(aggregate_score):
                    aggregate_score = judge_data.get('aggregate_score', np.nan)
            except:
                aggregate_verdict = 'UNKNOWN'
            
            data.append({
                'is_corrupted': is_corrupted,
                'aggregate_score': aggregate_score,
                'aggregate_verdict': aggregate_verdict,
                'corruption_type': corruption_type,
                'corruption_subtype': corruption_subtype
            })
        
        df_data = pd.DataFrame(data)
        
        # Binary mapping for classification
        df_data['ground_truth'] = df_data['is_corrupted'].fillna(False).astype(int)
        
        verdict_mapping = {
            'INCORRECT': 1,
            'PARTIALLY_CORRECT': 1,
            'CORRECT': 0,
            'UNKNOWN': 0,
            'Not supported': 1,
            'Partially supported': 1,
            'Supported': 0,
            'ERROR': 0
        }
        df_data['judge_pred'] = df_data['aggregate_verdict'].map(verdict_mapping).fillna(0).astype(int)
        
        # Calculate confusion matrix
        tp = ((df_data['ground_truth'] == 1) & (df_data['judge_pred'] == 1)).sum()
        tn = ((df_data['ground_truth'] == 0) & (df_data['judge_pred'] == 0)).sum()
        fp = ((df_data['ground_truth'] == 0) & (df_data['judge_pred'] == 1)).sum()
        fn = ((df_data['ground_truth'] == 1) & (df_data['judge_pred'] == 0)).sum()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(df_data) if len(df_data) > 0 else 0.0
        
        # Point-biserial correlation (corruption vs score)
        scores_clean = df_data[df_data['aggregate_score'].notna()].copy()
        if len(scores_clean) >= 2:
            r_pb, p_value = stats.pointbiserialr(
                scores_clean['ground_truth'],
                scores_clean['aggregate_score']
            )
        else:
            r_pb, p_value = 0.0, 1.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'n_total': len(df_data),
            'n_corrupted': int(df_data['ground_truth'].sum()),
            'n_clean': int((df_data['ground_truth'] == 0).sum()),
            'point_biserial_r': r_pb,
            'pb_p_value': p_value,
            'file_path': str(excel_path)
        }
    
    except Exception as e:
        print(f"Error extracting from {excel_path}: {e}")
        return {}


def extract_corruption_type_performance(excel_path: Path) -> dict:
    """
    Extract judge performance broken down by corruption type.
    
    Returns dict with corruption types as keys:
        {
            'spurious': {'tp': X, 'fn': Y, 'recall': Z, ...},
            'polarity': {...},
            'motivation': {...}
        }
    """
    try:
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        
        # Extract data
        data = []
        for _, row in df.iterrows():
            is_corrupted = row.get('Is Corrupted', False)
            corruption_subtype = row.get('Corruption Subtype', '')
            
            # Parse judge verdict
            judge_msg = row.get('Judge Message', '')
            try:
                judge_data = json.loads(judge_msg)
                aggregate_verdict = judge_data.get('aggregate_verdict', 'UNKNOWN')
            except:
                aggregate_verdict = 'UNKNOWN'
            
            if is_corrupted and corruption_subtype:
                data.append({
                    'corruption_subtype': corruption_subtype,
                    'aggregate_verdict': aggregate_verdict
                })
        
        df_data = pd.DataFrame(data)
        
        if len(df_data) == 0:
            return {}
        
        # Map verdicts to binary
        verdict_mapping = {
            'INCORRECT': 1,
            'PARTIALLY_CORRECT': 1,
            'CORRECT': 0,
            'UNKNOWN': 0
        }
        df_data['judge_detected'] = df_data['aggregate_verdict'].map(verdict_mapping).fillna(0).astype(int)
        
        # Calculate metrics per corruption type
        results = {}
        for corruption_type in df_data['corruption_subtype'].unique():
            if not corruption_type:
                continue
            
            subset = df_data[df_data['corruption_subtype'] == corruption_type]
            n_total = len(subset)
            n_detected = subset['judge_detected'].sum()
            recall = n_detected / n_total if n_total > 0 else 0.0
            
            results[corruption_type] = {
                'n_total': n_total,
                'n_detected': int(n_detected),
                'n_missed': n_total - int(n_detected),
                'recall': recall
            }
        
        return results
    
    except Exception as e:
        print(f"Error extracting corruption type performance from {excel_path}: {e}")
        return {}


def aggregate_by_cld_and_prompt(all_results: dict) -> pd.DataFrame:
    """
    Aggregate metrics across runs for each CLD/prompt combination.
    
    Returns DataFrame with columns:
    - CLD, Prompt, N Runs, Mean Precision, Mean Recall, Mean F1, Mean Accuracy,
      Mean Point-Biserial r, Std Precision, Std Recall, Std F1, etc.
    """
    aggregate_data = []
    
    prompt_labels = {
        'baseline': 'Baseline',
        'mechanistic': 'Mechanistic',
        'sce': 'SCE'
    }
    
    for cld_name, runs in all_results.items():
        for prompt_name in ['baseline', 'mechanistic', 'sce']:
            # Collect metrics from all runs
            metrics_list = []
            
            for run_name, files in runs.items():
                if prompt_name in files:
                    metrics = extract_judge_performance_from_file(files[prompt_name])
                    if metrics:
                        metrics_list.append(metrics)
            
            if not metrics_list:
                continue
            
            # Calculate aggregate statistics
            n_runs = len(metrics_list)
            
            aggregate_data.append({
                'CLD': cld_name,
                'Prompt': prompt_labels[prompt_name],
                'N Runs': n_runs,
                'Mean Precision': np.mean([m['precision'] for m in metrics_list]),
                'Std Precision': np.std([m['precision'] for m in metrics_list]),
                'Mean Recall': np.mean([m['recall'] for m in metrics_list]),
                'Std Recall': np.std([m['recall'] for m in metrics_list]),
                'Mean F1': np.mean([m['f1'] for m in metrics_list]),
                'Std F1': np.std([m['f1'] for m in metrics_list]),
                'Mean Accuracy': np.mean([m['accuracy'] for m in metrics_list]),
                'Std Accuracy': np.std([m['accuracy'] for m in metrics_list]),
                'Mean Point-Biserial r': np.mean([m['point_biserial_r'] for m in metrics_list]),
                'Std Point-Biserial r': np.std([m['point_biserial_r'] for m in metrics_list]),
                'Total Edges': int(np.mean([m['n_total'] for m in metrics_list])),
                'Total Corrupted': int(np.mean([m['n_corrupted'] for m in metrics_list])),
                'Total Clean': int(np.mean([m['n_clean'] for m in metrics_list]))
            })
    
    return pd.DataFrame(aggregate_data)


def aggregate_by_corruption_type(all_results: dict) -> pd.DataFrame:
    """
    Aggregate detection performance by corruption type across all runs and CLDs.
    
    Returns DataFrame with columns:
    - CLD, Prompt, Corruption Type, N Total, N Detected, N Missed, Recall
    """
    aggregate_data = []
    
    prompt_labels = {
        'baseline': 'Baseline',
        'mechanistic': 'Mechanistic',
        'sce': 'SCE'
    }
    
    for cld_name, runs in all_results.items():
        for prompt_name in ['baseline', 'mechanistic', 'sce']:
            # Collect corruption type metrics from all runs
            corruption_metrics = {}
            
            for run_name, files in runs.items():
                if prompt_name in files:
                    run_metrics = extract_corruption_type_performance(files[prompt_name])
                    
                    for corruption_type, metrics in run_metrics.items():
                        if corruption_type not in corruption_metrics:
                            corruption_metrics[corruption_type] = []
                        corruption_metrics[corruption_type].append(metrics)
            
            # Aggregate per corruption type
            for corruption_type, metrics_list in corruption_metrics.items():
                if not metrics_list:
                    continue
                
                total_n = sum(m['n_total'] for m in metrics_list)
                total_detected = sum(m['n_detected'] for m in metrics_list)
                total_missed = sum(m['n_missed'] for m in metrics_list)
                avg_recall = np.mean([m['recall'] for m in metrics_list])
                
                aggregate_data.append({
                    'CLD': cld_name,
                    'Prompt': prompt_labels[prompt_name],
                    'Corruption Type': corruption_type,
                    'Total Corrupted': total_n,
                    'Detected': total_detected,
                    'Missed': total_missed,
                    'Recall': avg_recall,
                    'N Runs': len(metrics_list)
                })
    
    return pd.DataFrame(aggregate_data)


def generate_thesis_table(aggregate_df: pd.DataFrame, corruption_type_df: pd.DataFrame) -> str:
    """
    Generate LaTeX table for thesis.
    
    Format:
    - Rows: CLD x Prompt combinations
    - Columns: Precision, Recall, F1, Point-Biserial r
    """
    latex_lines = []
    
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{RQ1a: Judge Performance on Synthetic Corruption Detection}")
    latex_lines.append("\\label{tab:rq1a_corruption_detection}")
    latex_lines.append("\\begin{tabular}{llcccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{CLD} & \\textbf{Judge Prompt} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{r$_{pb}$} \\\\")
    latex_lines.append("\\midrule")
    
    # Group by CLD
    for cld in aggregate_df['CLD'].unique():
        cld_data = aggregate_df[aggregate_df['CLD'] == cld].sort_values('Prompt')
        
        for idx, row in cld_data.iterrows():
            cld_name = row['CLD'] if idx == cld_data.index[0] else ''
            prompt_name = row['Prompt']
            precision = row['Mean Precision']
            recall = row['Mean Recall']
            f1 = row['Mean F1']
            r_pb = row['Mean Point-Biserial r']
            
            latex_lines.append(f"{cld_name} & {prompt_name} & {precision:.3f} & {recall:.3f} & {f1:.3f} & {r_pb:.3f} \\\\")
        
        latex_lines.append("\\midrule")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def create_visualizations(aggregate_df: pd.DataFrame, corruption_type_df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualizations."""
    
    # Figure 1: Overall performance by prompt across CLDs
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: F1 Score by Prompt and CLD
    ax1 = axes[0, 0]
    pivot_f1 = aggregate_df.pivot(index='CLD', columns='Prompt', values='Mean F1')
    pivot_f1.plot(kind='bar', ax=ax1, alpha=0.8, edgecolor='black')
    ax1.set_title('F1 Score by Prompt and CLD', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_xlabel('CLD', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.legend(title='Prompt', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15, ha='right')
    
    # Plot 2: Point-Biserial Correlation by Prompt and CLD
    ax2 = axes[0, 1]
    pivot_pb = aggregate_df.pivot(index='CLD', columns='Prompt', values='Mean Point-Biserial r')
    pivot_pb.plot(kind='bar', ax=ax2, alpha=0.8, edgecolor='black', color=['#e74c3c', '#3498db', '#9b59b6'])
    ax2.set_title('Point-Biserial Correlation by Prompt and CLD', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Point-Biserial r', fontsize=12)
    ax2.set_xlabel('CLD', fontsize=12)
    ax2.legend(title='Prompt', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15, ha='right')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot 3: Precision vs Recall
    ax3 = axes[1, 0]
    for prompt in aggregate_df['Prompt'].unique():
        subset = aggregate_df[aggregate_df['Prompt'] == prompt]
        ax3.scatter(subset['Mean Recall'], subset['Mean Precision'], 
                   label=prompt, s=100, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_title('Precision vs Recall by Prompt', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Recall', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax3.legend(title='Prompt', fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    # Plot 4: Detection by Corruption Type
    ax4 = axes[1, 1]
    if not corruption_type_df.empty:
        # Average across CLDs for each prompt/corruption type
        corruption_avg = corruption_type_df.groupby(['Prompt', 'Corruption Type'])['Recall'].mean().unstack()
        corruption_avg.plot(kind='bar', ax=ax4, alpha=0.8, edgecolor='black')
        ax4.set_title('Detection Recall by Corruption Type', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Recall', fontsize=12)
        ax4.set_xlabel('Prompt', fontsize=12)
        ax4.set_ylim(0, 1.0)
        ax4.legend(title='Corruption Type', fontsize=9)
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    output_file = output_dir / "rq1a_aggregate_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='RQ1a Aggregate Analysis')
    parser.add_argument('--base_dir', type=str, 
                       default='/home/nitai/code/causalix.ai/final_runs/RQ1a_corruption_detection_correctness',
                       help='Base directory containing CLD run subdirectories')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        sys.exit(1)
    
    print("="*80)
    print("RQ1a: AGGREGATE ANALYSIS - CORRUPTION DETECTION")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base directory: {base_dir}\n")
    
    # Step 1: Collect all results
    all_results = collect_all_run_results(base_dir)
    
    if not all_results:
        print("\nError: No results found!")
        sys.exit(1)
    
    # Step 2: Aggregate by CLD and Prompt
    print("\n" + "="*80)
    print("AGGREGATING BY CLD AND PROMPT")
    print("="*80)
    aggregate_df = aggregate_by_cld_and_prompt(all_results)
    print(aggregate_df.to_string(index=False))
    
    # Step 3: Aggregate by Corruption Type
    print("\n" + "="*80)
    print("AGGREGATING BY CORRUPTION TYPE")
    print("="*80)
    corruption_type_df = aggregate_by_corruption_type(all_results)
    print(corruption_type_df.to_string(index=False))
    
    # Step 4: Generate thesis table
    print("\n" + "="*80)
    print("GENERATING THESIS TABLE (LaTeX)")
    print("="*80)
    latex_table = generate_thesis_table(aggregate_df, corruption_type_df)
    print(latex_table)
    
    # Step 5: Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_dir / f"aggregate_analysis_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    output_excel = output_dir / "rq1a_aggregate_results.xlsx"
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        aggregate_df.to_excel(writer, sheet_name='CLD and Prompt Aggregate', index=False)
        corruption_type_df.to_excel(writer, sheet_name='Corruption Type Breakdown', index=False)
    
    print(f"\n✓ Excel results saved: {output_excel}")
    
    # Save LaTeX table
    latex_file = output_dir / "thesis_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved: {latex_file}")
    
    # Step 6: Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_visualizations(aggregate_df, corruption_type_df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"  - {output_excel.name}")
    print(f"  - {latex_file.name}")
    print(f"  - rq1a_aggregate_analysis.png")
    print()


if __name__ == "__main__":
    main()

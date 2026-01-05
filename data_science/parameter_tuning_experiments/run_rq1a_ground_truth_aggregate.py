#!/usr/bin/env python3
"""
RQ1a_ground_truth: Enhanced Aggregate Analysis - TP/FP Detection Across All Runs

Comprehensive analysis of judge ability to distinguish TP from FP edges across:
- Multiple CLDs (depressive, social_norms, emergency_department)
- Multiple runs per CLD (typically 3)
- Multiple prompt variants (baseline, mechanistic, cot, etc.)

Includes:
- AUC-ROC metrics
- Statistical significance testing (Friedman, Wilcoxon)
- Confidence intervals and effect sizes
- Per-CLD and per-prompt aggregation
- Enhanced visualizations

Usage:
    python run_rq1a_ground_truth_aggregate.py [--base_dir PATH]
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
from scipy.stats import shapiro, levene, friedmanchisquare, wilcoxon
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn for AUC-ROC
try:
    from sklearn.metrics import roc_auc_score, roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. AUC-ROC metrics will be skipped.")
    SKLEARN_AVAILABLE = False

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)


def collect_all_run_results(base_dir: Path) -> dict:
    """Collect results from all run directories."""
    results = {}
    
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
        run_dirs = sorted([d for d in cld_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
        
        for run_dir in run_dirs:
            run_name = run_dir.name
            
            # Auto-discover judged files
            judged_files = list(run_dir.glob("judged_*.xlsx"))
            
            # Group by variant name
            variant_files = {}
            for file_path in judged_files:
                # Extract variant name (e.g., baseline, cot, mechanistic_v4)
                name_parts = file_path.stem.split('_')
                # Find variant by looking for known patterns or taking last non-timestamp part
                variant = None
                for i in range(len(name_parts) - 1, -1, -1):
                    part = name_parts[i]
                    # Skip timestamps
                    if len(part) == 6 and part.isdigit():
                        continue
                    if len(part) == 8 and part.isdigit():
                        continue
                    # This should be the variant
                    variant = part
                    break
                
                if variant:
                    variant_files[variant] = file_path
            
            if len(variant_files) >= 2:  # At least 2 prompts
                results[cld_name][run_name] = variant_files
                variants_str = ', '.join(variant_files.keys())
                print(f"  ✓ {cld_name}/{run_name}: Found {len(variant_files)} variants: {variants_str}")
            else:
                print(f"  ✗ {cld_name}/{run_name}: Insufficient files ({len(variant_files)} variants)")
    
    total_runs = sum(len(runs) for runs in results.values())
    print(f"\nSummary: {len(results)} CLDs, {total_runs} total runs")
    return results


def calculate_confidence_interval(data: np.ndarray, confidence=0.95) -> Tuple[float, float]:
    """Calculate confidence interval for a dataset."""
    if len(data) < 2:
        return (np.nan, np.nan)
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return (mean - ci, mean + ci)


def extract_tp_fp_performance(excel_path: Path) -> dict:
    """
    Extract TP/FP detection performance.
    
    Returns dict with:
    - Basic metrics: precision, recall, F1, accuracy
    - AUC-ROC (if sklearn available)
    - Point-biserial correlation
    - Confusion matrix
    - Score distributions
    """
    try:
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        
        # Only analyze generated edges (TP and FP)
        df = df[df['Classification'].isin(['TP', 'FP'])].copy()
        
        if len(df) == 0:
            return None
        
        # Extract data
        data = []
        for _, row in df.iterrows():
            classification = row.get('Classification', 'UNKNOWN')
            is_fp = (classification == 'FP')
            aggregate_score = row.get('Aggregate Score', np.nan)
            
            judge_msg = row.get('Judge Message', '')
            try:
                judge_data = json.loads(judge_msg)
                aggregate_verdict = judge_data.get('aggregate_verdict', 'UNKNOWN')
                if pd.isna(aggregate_score):
                    aggregate_score = judge_data.get('aggregate_score', np.nan)
            except:
                aggregate_verdict = 'UNKNOWN'
            
            data.append({
                'is_fp': is_fp,
                'aggregate_score': aggregate_score,
                'aggregate_verdict': aggregate_verdict
            })
        
        df_data = pd.DataFrame(data)
        df_data['ground_truth'] = df_data['is_fp'].astype(int)  # 1=FP, 0=TP
        
        verdict_mapping = {
            'INCORRECT': 1, 'PARTIALLY_CORRECT': 0, 'CORRECT': 0,
            'UNKNOWN': 0, 'ERROR': 0
        }
        df_data['judge_pred'] = df_data['aggregate_verdict'].map(verdict_mapping).fillna(0).astype(int)
        
        # Remove rows with NaN scores
        df_valid = df_data[df_data['aggregate_score'].notna()].copy()
        
        if len(df_valid) == 0:
            return None
        
        # Confusion matrix (detecting FPs)
        tp = ((df_valid['ground_truth'] == 1) & (df_valid['judge_pred'] == 1)).sum()
        tn = ((df_valid['ground_truth'] == 0) & (df_valid['judge_pred'] == 0)).sum()
        fp = ((df_valid['ground_truth'] == 0) & (df_valid['judge_pred'] == 1)).sum()
        fn = ((df_valid['ground_truth'] == 1) & (df_valid['judge_pred'] == 0)).sum()
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(df_valid) if len(df_valid) > 0 else 0.0
        
        # AUC-ROC
        auc_roc = np.nan
        if SKLEARN_AVAILABLE and len(df_valid) > 10:
            try:
                auc_roc = roc_auc_score(df_valid['ground_truth'], df_valid['aggregate_score'])
            except:
                pass
        
        # Point-biserial correlation
        r_pb, p_value = stats.pointbiserialr(df_valid['is_fp'], df_valid['aggregate_score'])
        
        # Score statistics
        tp_scores = df_valid[~df_valid['is_fp']]['aggregate_score']
        fp_scores = df_valid[df_valid['is_fp']]['aggregate_score']
        
        return {
            'total_edges': len(df_valid),
            'tp_edges': int((~df_valid['is_fp']).sum()),
            'fp_edges': int(df_valid['is_fp'].sum()),
            'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
            'metrics': {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy),
                'auc_roc': float(auc_roc) if not np.isnan(auc_roc) else None
            },
            'correlation': {
                'r_pb': float(r_pb),
                'p_value': float(p_value)
            },
            'scores': {
                'tp_mean': float(tp_scores.mean()) if len(tp_scores) > 0 else np.nan,
                'tp_std': float(tp_scores.std()) if len(tp_scores) > 0 else np.nan,
                'fp_mean': float(fp_scores.mean()) if len(fp_scores) > 0 else np.nan,
                'fp_std': float(fp_scores.std()) if len(fp_scores) > 0 else np.nan
            }
        }
    
    except Exception as e:
        print(f"Error analyzing {excel_path}: {e}")
        return None


def aggregate_results(all_results: dict) -> pd.DataFrame:
    """Aggregate performance metrics across all runs."""
    rows = []
    
    for cld_name, runs in all_results.items():
        for run_name, variant_files in runs.items():
            for variant, file_path in variant_files.items():
                perf = extract_tp_fp_performance(file_path)
                if perf is None:
                    continue
                
                rows.append({
                    'CLD': cld_name,
                    'Run': run_name,
                    'Prompt': variant,
                    'Total Edges': perf['total_edges'],
                    'TP Edges': perf['tp_edges'],
                    'FP Edges': perf['fp_edges'],
                    'Precision': perf['metrics']['precision'],
                    'Recall': perf['metrics']['recall'],
                    'F1': perf['metrics']['f1'],
                    'Accuracy': perf['metrics']['accuracy'],
                    'AUC-ROC': perf['metrics'].get('auc_roc'),
                    'Point-Biserial r': perf['correlation']['r_pb'],
                    'P-value': perf['correlation']['p_value'],
                    'TP Score Mean': perf['scores']['tp_mean'],
                    'FP Score Mean': perf['scores']['fp_mean']
                })
    
    return pd.DataFrame(rows)


def perform_statistical_tests(df: pd.DataFrame) -> dict:
    """Perform statistical significance tests."""
    results = {}
    
    # Get unique prompts
    prompts = sorted(df['Prompt'].unique())
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    
    # Test if there are differences between prompts (Friedman test)
    if len(prompts) >= 3:
        print("\nFriedman Test (comparing all prompts):")
        
        # Prepare data for Friedman test (matched samples)
        f1_by_prompt = {}
        for prompt in prompts:
            f1_by_prompt[prompt] = df[df['Prompt'] == prompt]['F1'].values
        
        # Find common length (minimum across prompts)
        min_len = min(len(vals) for vals in f1_by_prompt.values())
        if min_len >= 3:
            f1_arrays = [vals[:min_len] for vals in f1_by_prompt.values()]
            stat, p_value = friedmanchisquare(*f1_arrays)
            print(f"  F1 Score: χ²={stat:.4f}, p={p_value:.6f}")
            results['friedman_f1'] = {'statistic': stat, 'p_value': p_value}
        else:
            print("  Insufficient matched samples for Friedman test")
    
    # Pairwise comparisons (Wilcoxon signed-rank test)
    if len(prompts) >= 2:
        print("\nPairwise Comparisons (Wilcoxon signed-rank):")
        
        pairwise_results = []
        for i, prompt1 in enumerate(prompts):
            for prompt2 in prompts[i+1:]:
                f1_1 = df[df['Prompt'] == prompt1]['F1'].values
                f1_2 = df[df['Prompt'] == prompt2]['F1'].values
                
                min_len = min(len(f1_1), len(f1_2))
                if min_len >= 3:
                    stat, p_value = wilcoxon(f1_1[:min_len], f1_2[:min_len])
                    
                    # Calculate effect size (Cohen's d)
                    mean_diff = np.mean(f1_1[:min_len]) - np.mean(f1_2[:min_len])
                    pooled_std = np.sqrt((np.var(f1_1[:min_len]) + np.var(f1_2[:min_len])) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    print(f"  {prompt1} vs {prompt2}:")
                    print(f"    W={stat:.4f}, p={p_value:.6f}, Cohen's d={cohens_d:.4f}")
                    
                    pairwise_results.append({
                        'Comparison': f"{prompt1} vs {prompt2}",
                        'Metric': 'F1',
                        'Statistic': stat,
                        'P-value': p_value,
                        'Cohens d': cohens_d
                    })
        
        results['pairwise'] = pd.DataFrame(pairwise_results)
    
    return results


def create_enhanced_visualizations(df: pd.DataFrame, output_dir: Path, stat_results: dict):
    """Create comprehensive visualization plots."""
    
    prompts = sorted(df['Prompt'].unique())
    n_prompts = len(prompts)
    
    # Dynamic colors
    if n_prompts <= 10:
        colors = plt.cm.tab10(range(n_prompts))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_prompts))
    
    prompt_colors = {prompt: colors[i] for i, prompt in enumerate(prompts)}
    
    # Create 6-panel figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: F1 Scores by Prompt (box plot)
    ax1 = fig.add_subplot(gs[0, 0])
    f1_data = [df[df['Prompt'] == p]['F1'].values for p in prompts]
    bp = ax1.boxplot(f1_data, labels=[p.upper() for p in prompts], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1 Score Distribution by Prompt', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=15)
    
    # Panel 2: Precision vs Recall (scatter)
    ax2 = fig.add_subplot(gs[0, 1])
    for prompt in prompts:
        prompt_data = df[df['Prompt'] == prompt]
        ax2.scatter(prompt_data['Recall'], prompt_data['Precision'],
                   s=100, color=prompt_colors[prompt], label=prompt.upper(),
                   alpha=0.6, edgecolors='black', linewidth=1)
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision vs Recall (FP Detection)', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Panel 3: Point-Biserial Correlation
    ax3 = fig.add_subplot(gs[1, 0])
    corr_data = df.groupby('Prompt')['Point-Biserial r'].agg(['mean', 'std']).reset_index()
    corr_data = corr_data.sort_values('mean')
    bars = ax3.barh(range(len(corr_data)), corr_data['mean'].values,
                    color=[prompt_colors[p] for p in corr_data['Prompt']])
    ax3.set_yticks(range(len(corr_data)))
    ax3.set_yticklabels([p.upper() for p in corr_data['Prompt']])
    ax3.set_xlabel('Point-Biserial r (Score vs FP)', fontsize=12, fontweight='bold')
    ax3.set_title('Correlation: Judge Score vs FP Status', fontsize=14, fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax3.grid(axis='x', alpha=0.3)
    
    # Add error bars
    for i, (bar, row) in enumerate(zip(bars, corr_data.itertuples())):
        ax3.errorbar(row.mean, i, xerr=row.std, color='black', capsize=5, capthick=2)
    
    # Panel 4: Score Distributions (TP vs FP)
    ax4 = fig.add_subplot(gs[1, 1])
    x = np.arange(n_prompts)
    width = 0.35
    
    tp_means = df.groupby('Prompt')['TP Score Mean'].mean().values
    fp_means = df.groupby('Prompt')['FP Score Mean'].mean().values
    
    bars1 = ax4.bar(x - width/2, tp_means, width, label='TP Edges (Correct)', color='green', alpha=0.7)
    bars2 = ax4.bar(x + width/2, fp_means, width, label='FP Edges (Hallucinations)', color='red', alpha=0.7)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels([p.upper() for p in prompts], rotation=15, ha='right')
    ax4.set_ylabel('Mean Aggregate Score', fontsize=12, fontweight='bold')
    ax4.set_title('Score Distributions: TP vs FP Edges', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 1])
    ax4.legend(fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    
    # Panel 5: Per-CLD Performance
    ax5 = fig.add_subplot(gs[2, 0])
    cld_pivot = df.pivot_table(values='F1', index='CLD', columns='Prompt', aggfunc='mean')
    sns.heatmap(cld_pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                ax=ax5, cbar_kws={'label': 'F1 Score'})
    ax5.set_title('F1 Score by CLD and Prompt', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Prompt', fontsize=12, fontweight='bold')
    ax5.set_ylabel('CLD', fontsize=12, fontweight='bold')
    
    # Panel 6: Pairwise Effect Sizes (if available)
    ax6 = fig.add_subplot(gs[2, 1])
    if 'pairwise' in stat_results and len(stat_results['pairwise']) > 0:
        pairwise_df = stat_results['pairwise']
        pairwise_df = pairwise_df.sort_values('Cohens d', ascending=True)
        
        bars = ax6.barh(range(len(pairwise_df)), pairwise_df['Cohens d'].values,
                       color=['green' if d > 0 else 'red' for d in pairwise_df['Cohens d']])
        ax6.set_yticks(range(len(pairwise_df)))
        ax6.set_yticklabels(pairwise_df['Comparison'].values, fontsize=10)
        ax6.set_xlabel("Cohen's d (standardized mean difference)", fontsize=12, fontweight='bold')
        ax6.set_title('Pairwise Prompt Comparison: F1 Score Differences', fontsize=14, fontweight='bold')
        ax6.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax6.grid(axis='x', alpha=0.3)
        
        # Add significance stars
        for i, row in pairwise_df.iterrows():
            p_val = row['P-value']
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            d_val = row['Cohens d']
            ax6.text(d_val + (0.05 if d_val >= 0 else -0.05), i,
                    sig, ha='left' if d_val >= 0 else 'right', va='center', fontsize=12, fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Insufficient data for pairwise comparisons',
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.axis('off')
    
    plt.savefig(output_dir / 'rq1a_ground_truth_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: {output_dir / 'rq1a_ground_truth_visualization.png'}")


def generate_summary_tables(df: pd.DataFrame, output_dir: Path):
    """Generate summary tables in LaTeX format."""
    
    # Table 1: Prompt Performance Summary
    prompt_summary = df.groupby('Prompt').agg({
        'Precision': ['mean', 'std'],
        'Recall': ['mean', 'std'],
        'F1': ['mean', 'std'],
        'Point-Biserial r': ['mean', 'std']
    }).round(3)
    
    latex_table1 = prompt_summary.to_latex()
    with open(output_dir / 'table_prompt_summary.tex', 'w') as f:
        f.write(latex_table1)
    
    # Table 2: Per-CLD Performance
    cld_summary = df.pivot_table(
        values=['Precision', 'Recall', 'F1'],
        index='CLD',
        columns='Prompt',
        aggfunc='mean'
    ).round(3)
    
    latex_table2 = cld_summary.to_latex()
    with open(output_dir / 'table_cld_performance.tex', 'w') as f:
        f.write(latex_table2)
    
    print(f"✓ LaTeX tables saved:")
    print(f"  - {output_dir / 'table_prompt_summary.tex'}")
    print(f"  - {output_dir / 'table_cld_performance.tex'}")


def main():
    parser = argparse.ArgumentParser(description='RQ1a_ground_truth: Enhanced Aggregate Analysis')
    parser.add_argument('--base_dir', type=str,
                       default='/home/nitai/code/causalix.ai/final_runs/RQ1a_gt_lit_correctness',
                       help='Base directory containing CLD subdirectories')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return 1
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_dir / f'enhanced_analysis_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("RQ1a_ground_truth: ENHANCED AGGREGATE ANALYSIS")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Collect all results
    all_results = collect_all_run_results(base_dir)
    
    if not all_results:
        print("\nError: No results found")
        return 1
    
    # Aggregate data
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)
    df = aggregate_results(all_results)
    print(f"Total observations: {len(df)}")
    print(f"Prompts: {', '.join(sorted(df['Prompt'].unique()))}")
    print(f"CLDs: {', '.join(sorted(df['CLD'].unique()))}")
    
    # Save raw aggregate data
    df.to_excel(output_dir / 'rq1a_ground_truth_aggregate.xlsx', index=False)
    print(f"\n✓ Aggregate data saved: {output_dir / 'rq1a_ground_truth_aggregate.xlsx'}")
    
    # Statistical tests
    stat_results = perform_statistical_tests(df)
    
    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_enhanced_visualizations(df, output_dir, stat_results)
    
    # LaTeX tables
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)
    generate_summary_tables(df, output_dir)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print("\nMean F1 Score by Prompt:")
    print(df.groupby('Prompt')['F1'].agg(['mean', 'std', 'count']))
    
    print("\nMean Point-Biserial Correlation by Prompt:")
    print(df.groupby('Prompt')['Point-Biserial r'].agg(['mean', 'std']))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results directory: {output_dir}")
    print("Files generated:")
    print("  - rq1a_ground_truth_aggregate.xlsx")
    print("  - rq1a_ground_truth_visualization.png")
    print("  - table_prompt_summary.tex")
    print("  - table_cld_performance.tex")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

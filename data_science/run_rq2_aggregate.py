#!/usr/bin/env python3
"""
RQ2 Aggregate Analysis: Cross-CLD meta-analysis
Aggregates results from multiple individual RQ2 analyses to draw general conclusions.
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from scipy import stats

def find_rq2_analyses(experiment_dir: Path, metric_source: str = "generator", 
                      hallucination_def: str = "ground_truth_FP") -> List[Path]:
    """Find all RQ2 analysis directories matching the criteria."""
    # Match directories like: rq2_analysis_TIMESTAMP_EXPID_..._runN_gen_gt
    metric_src = metric_source[:3]  # 'gen' or 'jud'
    halluc_def_short = 'gt' if 'ground_truth' in hallucination_def else 'js'
    suffix = f"{metric_src}_{halluc_def_short}"
    all_analyses = sorted(experiment_dir.glob("rq2_analysis_*"))
    analyses = [a for a in all_analyses if a.name.endswith(suffix)]
    return analyses

def load_analysis_metadata(analysis_dir: Path) -> Dict:
    """Load metadata from an RQ2 analysis directory."""
    metadata_file = analysis_dir / "analysis_metadata.json"
    if not metadata_file.exists():
        return None
    
    with open(metadata_file, 'r') as f:
        return json.load(f)

def load_analysis_summary(analysis_dir: Path) -> Dict:
    """Load complete summary JSON from an RQ2 analysis directory."""
    summary_files = list(analysis_dir.glob("rq2_complete_summary_*.json"))
    if not summary_files:
        return None
    
    with open(summary_files[0], 'r') as f:
        return json.load(f)

def aggregate_correlations(analyses: List[Path]) -> pd.DataFrame:
    """Aggregate correlation results across all analyses."""
    all_corrs = []
    
    for analysis_dir in analyses:
        metadata = load_analysis_metadata(analysis_dir)
        if not metadata:
            continue
        
        corr_file = analysis_dir / "tables" / "rq2_correlations.csv"
        if not corr_file.exists():
            continue
        
        df = pd.read_csv(corr_file)
        # Remove metadata columns if present
        df = df[[c for c in df.columns if not c.startswith('meta_')]]
        
        df['cld_name'] = metadata['cld_name']
        df['run_number'] = metadata['run_number']
        all_corrs.append(df)
    
    if not all_corrs:
        return pd.DataFrame()
    
    combined = pd.concat(all_corrs, ignore_index=True)
    return combined

def aggregate_auc_scores(analyses: List[Path]) -> pd.DataFrame:
    """Aggregate AUC scores across all analyses."""
    all_aucs = []
    
    for analysis_dir in analyses:
        metadata = load_analysis_metadata(analysis_dir)
        if not metadata:
            continue
        
        auc_file = analysis_dir / "tables" / "rq2_roc_auc.csv"
        if not auc_file.exists():
            continue
        
        df = pd.read_csv(auc_file)
        df = df[[c for c in df.columns if not c.startswith('meta_')]]
        
        df['cld_name'] = metadata['cld_name']
        df['run_number'] = metadata['run_number']
        all_aucs.append(df)
    
    if not all_aucs:
        return pd.DataFrame()
    
    combined = pd.concat(all_aucs, ignore_index=True)
    return combined

def aggregate_thresholds(analyses: List[Path]) -> pd.DataFrame:
    """Aggregate optimal thresholds across all analyses."""
    all_thresh = []
    
    for analysis_dir in analyses:
        metadata = load_analysis_metadata(analysis_dir)
        if not metadata:
            continue
        
        thresh_file = analysis_dir / "tables" / "rq2_optimal_thresholds.csv"
        if not thresh_file.exists():
            continue
        
        df = pd.read_csv(thresh_file)
        df = df[[c for c in df.columns if not c.startswith('meta_')]]
        
        df['cld_name'] = metadata['cld_name']
        df['run_number'] = metadata['run_number']
        all_thresh.append(df)
    
    if not all_thresh:
        return pd.DataFrame()
    
    combined = pd.concat(all_thresh, ignore_index=True)
    return combined

def compute_meta_statistics(df: pd.DataFrame, value_col: str, group_col: str = 'metric') -> pd.DataFrame:
    """Compute meta-statistics (mean, std, CI) across datasets."""
    meta_stats = df.groupby(group_col)[value_col].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('n', 'count')
    ]).reset_index()
    
    # Compute 95% CI
    meta_stats['ci_lower'] = meta_stats['mean'] - 1.96 * meta_stats['std'] / np.sqrt(meta_stats['n'])
    meta_stats['ci_upper'] = meta_stats['mean'] + 1.96 * meta_stats['std'] / np.sqrt(meta_stats['n'])
    
    return meta_stats

def plot_aggregated_correlations(corr_df: pd.DataFrame, output_dir: Path):
    """Plot aggregated correlation results across datasets."""
    if corr_df.empty:
        return
    
    metrics = corr_df['metric'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Correlation coefficients by dataset
    pivot = corr_df.pivot_table(values='pearson_r', index='metric', columns='cld_name')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
                ax=axes[0], cbar_kws={'label': 'Pearson r'})
    axes[0].set_title('Correlation with Hallucinations\nAcross Datasets', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('CI Metric', fontsize=10)
    
    # Plot 2: Meta-analysis (mean + CI)
    meta_stats = compute_meta_statistics(corr_df, 'pearson_r', 'metric')
    meta_stats = meta_stats.sort_values('mean', ascending=True)
    
    y_pos = np.arange(len(meta_stats))
    axes[1].barh(y_pos, meta_stats['mean'], color='steelblue', alpha=0.7)
    axes[1].errorbar(meta_stats['mean'], y_pos, 
                     xerr=[meta_stats['mean'] - meta_stats['ci_lower'],
                           meta_stats['ci_upper'] - meta_stats['mean']],
                     fmt='none', color='black', capsize=5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(meta_stats['metric'])
    axes[1].set_xlabel('Mean Pearson r (95% CI)', fontsize=10)
    axes[1].set_title('Meta-Analysis: Average Correlation\nAcross All Datasets', fontsize=12, fontweight='bold')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregate_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: aggregate_correlations.png")

def plot_aggregated_auc(auc_df: pd.DataFrame, output_dir: Path):
    """Plot aggregated AUC scores across datasets."""
    if auc_df.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: AUC by dataset
    pivot = auc_df.pivot_table(values='roc_auc', index='metric', columns='cld_name')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.3, vmax=0.9,
                ax=axes[0], cbar_kws={'label': 'AUC'})
    axes[0].set_title('ROC AUC Scores\nAcross Datasets', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('CI Metric', fontsize=10)
    axes[0].axhline(y=0, color='black', linewidth=2)
    
    # Plot 2: Meta-analysis (mean + CI)
    meta_stats = compute_meta_statistics(auc_df, 'roc_auc', 'metric')
    meta_stats = meta_stats.sort_values('mean', ascending=True)
    
    y_pos = np.arange(len(meta_stats))
    colors = ['green' if m >= 0.7 else 'orange' if m >= 0.6 else 'steelblue' 
              for m in meta_stats['mean']]
    axes[1].barh(y_pos, meta_stats['mean'], color=colors, alpha=0.7)
    axes[1].errorbar(meta_stats['mean'], y_pos,
                     xerr=[meta_stats['mean'] - meta_stats['ci_lower'],
                           meta_stats['ci_upper'] - meta_stats['mean']],
                     fmt='none', color='black', capsize=5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(meta_stats['metric'])
    axes[1].set_xlabel('Mean AUC (95% CI)', fontsize=10)
    axes[1].set_title('Meta-Analysis: Average AUC\nAcross All Datasets', fontsize=12, fontweight='bold')
    axes[1].axvline(0.5, color='red', linestyle='--', linewidth=1, label='Chance')
    axes[1].axvline(0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good')
    axes[1].legend(loc='lower right')
    axes[1].set_xlim(0.3, 0.9)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregate_auc_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: aggregate_auc_scores.png")

def plot_aggregated_f1(thresh_df: pd.DataFrame, output_dir: Path):
    """Plot aggregated F1 scores across datasets."""
    if thresh_df.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: F1 by dataset
    pivot = thresh_df.pivot_table(values='f1', index='metric', columns='cld_name')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=0.6,
                ax=axes[0], cbar_kws={'label': 'F1 Score'})
    axes[0].set_title('Optimal F1 Scores\nAcross Datasets', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('CI Metric', fontsize=10)
    
    # Plot 2: Meta-analysis (mean + CI)
    meta_stats = compute_meta_statistics(thresh_df, 'f1', 'metric')
    meta_stats = meta_stats.sort_values('mean', ascending=True)
    
    y_pos = np.arange(len(meta_stats))
    axes[1].barh(y_pos, meta_stats['mean'], color='coral', alpha=0.7)
    axes[1].errorbar(meta_stats['mean'], y_pos,
                     xerr=[meta_stats['mean'] - meta_stats['ci_lower'],
                           meta_stats['ci_upper'] - meta_stats['mean']],
                     fmt='none', color='black', capsize=5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(meta_stats['metric'])
    axes[1].set_xlabel('Mean F1 Score (95% CI)', fontsize=10)
    axes[1].set_title('Meta-Analysis: Average F1 Score\nAcross All Datasets', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, 0.6)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregate_f1_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: aggregate_f1_scores.png")

def generate_aggregate_report(analyses: List[Path], corr_df: pd.DataFrame, 
                              auc_df: pd.DataFrame, thresh_df: pd.DataFrame,
                              output_dir: Path, experiment_id: str, 
                              metric_source: str, hallucination_def: str):
    """Generate comprehensive aggregate report."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"rq2_aggregate_report_{timestamp}.md"
    
    # Compute meta-statistics
    corr_meta = compute_meta_statistics(corr_df, 'pearson_r', 'metric')
    auc_meta = compute_meta_statistics(auc_df, 'roc_auc', 'metric')
    f1_meta = compute_meta_statistics(thresh_df, 'f1', 'metric')
    
    # Statistical significance testing
    # Test if mean correlation is significantly different from zero (one-sample t-test)
    corr_meta['t_stat'] = corr_meta['mean'] / (corr_meta['std'] / np.sqrt(corr_meta['n']))
    corr_meta['p_value'] = 2 * (1 - stats.t.cdf(np.abs(corr_meta['t_stat']), corr_meta['n'] - 1))
    corr_meta['significant'] = corr_meta['p_value'] < 0.05
    
    # Test if mean AUC is significantly different from chance (0.5)
    auc_meta['t_stat'] = (auc_meta['mean'] - 0.5) / (auc_meta['std'] / np.sqrt(auc_meta['n']))
    auc_meta['p_value'] = 2 * (1 - stats.t.cdf(np.abs(auc_meta['t_stat']), auc_meta['n'] - 1))
    auc_meta['significant'] = auc_meta['p_value'] < 0.05
    
    # Find best metric overall
    best_auc_metric = auc_meta.loc[auc_meta['mean'].idxmax()]
    best_corr_metric = corr_meta.loc[corr_meta['mean'].abs().idxmax()]
    
    # Get dataset info
    total_edges = 0
    total_hallucinations = 0
    cld_names = set()
    runs = set()
    
    for analysis_dir in analyses:
        metadata = load_analysis_metadata(analysis_dir)
        if metadata:
            summary = load_analysis_summary(analysis_dir)
            if summary:
                total_edges += summary['data']['total_edges']
                total_hallucinations += summary['data']['hallucinations']
            cld_names.add(metadata['cld_name'])
            runs.add(metadata['run_number'])
    
    with open(report_path, 'w') as f:
        f.write(f"# RQ2 Aggregate Meta-Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Experiment ID:** {experiment_id}  \n")
        f.write(f"**CI Metric Source:** {metric_source}  \n")
        f.write(f"**Hallucination Definition:** {hallucination_def}  \n\n")
        
        f.write(f"---\n\n")
        
        f.write(f"## Executive Summary\n\n")
        f.write(f"### Dataset Coverage\n")
        f.write(f"- **Total Analyses:** {len(analyses)}\n")
        f.write(f"- **Unique CLDs:** {len(cld_names)}\n")
        f.write(f"- **Runs per CLD:** {len(runs)}\n")
        f.write(f"- **Total Edges Analyzed:** {total_edges:,}\n")
        if total_edges > 0:
            f.write(f"- **Total Hallucinations:** {total_hallucinations} ({100*total_hallucinations/total_edges:.1f}%)\n\n")
        else:
            f.write(f"- **Total Hallucinations:** {total_hallucinations}\n\n")
        
        f.write(f"### Best Performing Metrics (Cross-Dataset)\n\n")
        f.write(f"**Best AUC:** {best_auc_metric['metric']}  \n")
        f.write(f"- Mean AUC: {best_auc_metric['mean']:.3f} (95% CI: [{best_auc_metric['ci_lower']:.3f}, {best_auc_metric['ci_upper']:.3f}])  \n")
        f.write(f"- Range: [{best_auc_metric['min']:.3f}, {best_auc_metric['max']:.3f}]  \n\n")
        
        f.write(f"**Strongest Correlation:** {best_corr_metric['metric']}  \n")
        f.write(f"- Mean |r|: {abs(best_corr_metric['mean']):.3f} (95% CI: [{best_corr_metric['ci_lower']:.3f}, {best_corr_metric['ci_upper']:.3f}])  \n")
        f.write(f"- Range: [{best_corr_metric['min']:.3f}, {best_corr_metric['max']:.3f}]  \n\n")
        
        f.write(f"---\n\n")
        
        f.write(f"## Detailed Meta-Analysis Results\n\n")
        
        f.write(f"### 1. Correlation Analysis (Pearson r)\n\n")
        f.write(f"Mean correlation with hallucinations across all datasets:\n\n")
        corr_meta_sorted = corr_meta.sort_values('mean', key=lambda x: abs(x), ascending=False)
        f.write(f"| Metric | Mean r | 95% CI | Std Dev | p-value | Sig. |\n")
        f.write(f"|--------|--------|--------|---------|---------|------|\n")
        for _, row in corr_meta_sorted.iterrows():
            sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
            f.write(f"| {row['metric']:25s} | {row['mean']:+.3f} | [{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}] | "
                   f"{row['std']:.3f} | {row['p_value']:.3f} | {sig_marker} |\n")
        f.write(f"\n")
        f.write(f"*Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant*\n\n")
        
        f.write(f"### 2. Classification Performance (AUC)\n\n")
        f.write(f"Mean ROC AUC across all datasets (tested against chance=0.5):\n\n")
        auc_meta_sorted = auc_meta.sort_values('mean', ascending=False)
        f.write(f"| Metric | Mean AUC | 95% CI | Std Dev | p-value | Sig. |\n")
        f.write(f"|--------|----------|--------|---------|---------|------|\n")
        for _, row in auc_meta_sorted.iterrows():
            sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
            f.write(f"| {row['metric']:25s} | {row['mean']:.3f} | [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] | "
                   f"{row['std']:.3f} | {row['p_value']:.3f} | {sig_marker} |\n")
        f.write(f"\n")
        f.write(f"*Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant (vs. chance)*\n\n")
        
        f.write(f"### 3. Optimal Threshold Performance (F1 Score)\n\n")
        f.write(f"Mean F1 score at optimal thresholds:\n\n")
        f1_meta_sorted = f1_meta.sort_values('mean', ascending=False)
        f.write(f"| Metric | Mean F1 | 95% CI | Std Dev | Range |\n")
        f.write(f"|--------|---------|--------|---------|-------|\n")
        for _, row in f1_meta_sorted.iterrows():
            f.write(f"| {row['metric']:25s} | {row['mean']:.3f} | [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] | "
                   f"{row['std']:.3f} | [{row['min']:.3f}, {row['max']:.3f}] |\n")
        f.write(f"\n")
        
        f.write(f"---\n\n")
        
        f.write(f"## Interpretation\n\n")
        f.write(f"### Answer to RQ2\n\n")
        if best_auc_metric['mean'] > 0.7 and best_auc_metric['ci_lower'] > 0.6:
            f.write(f"**YES** - CI metrics can reliably predict hallucinations across diverse domains.\n\n")
        elif best_auc_metric['mean'] > 0.6:
            f.write(f"**PARTIALLY** - CI metrics show moderate predictive power, but with substantial variance across domains.\n\n")
        else:
            f.write(f"**NO** - CI metrics do not show consistent predictive power across domains.\n\n")
        
        f.write(f"### Key Findings\n\n")
        
        # Statistical significance summary
        sig_corr = corr_meta[corr_meta['significant']]['metric'].tolist()
        sig_auc = auc_meta[auc_meta['significant']]['metric'].tolist()
        
        f.write(f"**Statistically Significant Predictors:**\n")
        f.write(f"- **Correlations (r ≠ 0):** {len(sig_corr)}/{len(corr_meta)} metrics significant\n")
        if sig_corr:
            f.write(f"  - Significant: {', '.join(sig_corr)}\n")
        f.write(f"- **Classification (AUC > chance):** {len(sig_auc)}/{len(auc_meta)} metrics significant\n")
        if sig_auc:
            f.write(f"  - Significant: {', '.join(sig_auc)}\n")
        f.write(f"\n")
        
        # Find consistently good metrics
        good_metrics = auc_meta[(auc_meta['mean'] > 0.65) & (auc_meta['significant'])]['metric'].tolist()
        if good_metrics:
            f.write(f"**Consistently Effective Metrics (AUC > 0.65 & significant):**\n")
            for metric in good_metrics:
                auc_row = auc_meta[auc_meta['metric'] == metric].iloc[0]
                f.write(f"- **{metric}**: AUC = {auc_row['mean']:.3f} ± {auc_row['std']:.3f} (p={auc_row['p_value']:.3f})\n")
            f.write(f"\n")
        
        # Stability analysis
        f.write(f"**Stability Across Domains:**\n")
        stable_metrics = auc_meta[auc_meta['std'] < 0.1]['metric'].tolist()
        if stable_metrics:
            f.write(f"- Low variance metrics (std < 0.1): {', '.join(stable_metrics)}\n")
        variable_metrics = auc_meta[auc_meta['std'] > 0.15]['metric'].tolist()
        if variable_metrics:
            f.write(f"- High variance metrics (std > 0.15): {', '.join(variable_metrics)}\n")
        f.write(f"\n")
        
        f.write(f"### Limitations\n\n")
        f.write(f"- Analysis based on {len(cld_names)} unique CLDs (limited domain diversity)\n")
        f.write(f"- Ground truth quality varies by CLD completeness\n")
        f.write(f"- CI metrics alone may be insufficient for production deployment (max F1 = {f1_meta['mean'].max():.3f})\n\n")
        
        f.write(f"---\n\n")
        
        f.write(f"## Files Generated\n\n")
        f.write(f"- `aggregate_correlations.png` - Correlation heatmap and meta-analysis\n")
        f.write(f"- `aggregate_auc_scores.png` - AUC heatmap and meta-analysis\n")
        f.write(f"- `aggregate_f1_scores.png` - F1 score heatmap and meta-analysis\n")
        f.write(f"- `aggregate_correlations.csv` - Full correlation data\n")
        f.write(f"- `aggregate_auc_scores.csv` - Full AUC data\n")
        f.write(f"- `aggregate_thresholds.csv` - Full threshold data\n")
        f.write(f"- `aggregate_meta_statistics.csv` - Summary statistics\n\n")
    
    print(f"✓ Saved: {report_path.name}")
    return report_path

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate RQ2 results across multiple analyses (cross-CLD meta-analysis)"
    )
    parser.add_argument(
        "experiment_id",
        help="Experiment ID (e.g., exp_20251008_180446_a0d1b2de)"
    )
    parser.add_argument(
        "--metric-source",
        choices=["generator", "judge"],
        default="generator",
        help="CI metric source to aggregate (default: generator)"
    )
    parser.add_argument(
        "--hallucination-def",
        choices=["ground_truth_FP", "judge_aggregate_low"],
        default="ground_truth_FP",
        help="Hallucination definition used (default: ground_truth_FP)"
    )
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("RQ2 AGGREGATE META-ANALYSIS")
    print("=" * 100)
    print()
    
    # Find all RQ2 analysis directories
    rq2_dir = Path("parameter_tuning_experiments/rq2_analyses")
    
    analyses = find_rq2_analyses(rq2_dir, args.metric_source, args.hallucination_def)
    
    # Filter by experiment ID (match on shortened version: exp_YYYYMMDD_HHMMSS -> YYYYMMDD_HH)
    exp_id_short = args.experiment_id.replace('exp_', '')[:12]  # e.g., "20251008_180"
    analyses = [a for a in analyses if exp_id_short in a.name]
    
    if not analyses:
        print(f"❌ No RQ2 analyses found for experiment: {args.experiment_id}")
        print(f"   Metric source: {args.metric_source}")
        print(f"   Hallucination def: {args.hallucination_def}")
        return 1
    
    print(f"📁 Found {len(analyses)} RQ2 analyses to aggregate:")
    for analysis_dir in analyses:
        metadata = load_analysis_metadata(analysis_dir)
        if metadata:
            print(f"   • {metadata['cld_name'][:60]} ({metadata['run_number']})")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = rq2_dir / f"rq2_aggregate_{timestamp}_{args.experiment_id[:15]}_{args.metric_source[:3]}_{args.hallucination_def[:2]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Output directory: {output_dir.name}")
    print()
    
    # Aggregate data
    print("🔄 Aggregating correlation data...")
    corr_df = aggregate_correlations(analyses)
    if not corr_df.empty:
        corr_df.to_csv(output_dir / "aggregate_correlations.csv", index=False)
        print(f"   ✓ {len(corr_df)} correlation records aggregated")
    
    print("🔄 Aggregating AUC data...")
    auc_df = aggregate_auc_scores(analyses)
    if not auc_df.empty:
        auc_df.to_csv(output_dir / "aggregate_auc_scores.csv", index=False)
        print(f"   ✓ {len(auc_df)} AUC records aggregated")
    
    print("🔄 Aggregating threshold data...")
    thresh_df = aggregate_thresholds(analyses)
    if not thresh_df.empty:
        thresh_df.to_csv(output_dir / "aggregate_thresholds.csv", index=False)
        print(f"   ✓ {len(thresh_df)} threshold records aggregated")
    print()
    
    # Compute and save meta-statistics
    print("📊 Computing meta-statistics...")
    meta_stats = {
        'correlations': compute_meta_statistics(corr_df, 'pearson_r', 'metric'),
        'auc': compute_meta_statistics(auc_df, 'roc_auc', 'metric'),
        'f1': compute_meta_statistics(thresh_df, 'f1', 'metric')
    }
    
    with pd.ExcelWriter(output_dir / "aggregate_meta_statistics.xlsx") as writer:
        meta_stats['correlations'].to_excel(writer, sheet_name='Correlations', index=False)
        meta_stats['auc'].to_excel(writer, sheet_name='AUC', index=False)
        meta_stats['f1'].to_excel(writer, sheet_name='F1', index=False)
    print(f"   ✓ Meta-statistics saved")
    print()
    
    # Generate visualizations
    print("📈 Generating aggregate visualizations...")
    plot_aggregated_correlations(corr_df, output_dir)
    plot_aggregated_auc(auc_df, output_dir)
    plot_aggregated_f1(thresh_df, output_dir)
    print()
    
    # Generate comprehensive report
    print("📝 Generating aggregate report...")
    report_path = generate_aggregate_report(
        analyses, corr_df, auc_df, thresh_df, output_dir,
        args.experiment_id, args.metric_source, args.hallucination_def
    )
    print()
    
    print("=" * 100)
    print("✅ RQ2 AGGREGATE ANALYSIS COMPLETE!")
    print("=" * 100)
    print()
    print(f"📁 All outputs saved to: {output_dir}/")
    print(f"📄 Main report: {report_path.name}")
    print()
    
    # Display key findings
    best_auc = meta_stats['auc'].loc[meta_stats['auc']['mean'].idxmax()]
    print(f"🏆 Best performing metric: {best_auc['metric']}")
    print(f"   Mean AUC: {best_auc['mean']:.3f} (95% CI: [{best_auc['ci_lower']:.3f}, {best_auc['ci_upper']:.3f}])")
    print()

if __name__ == "__main__":
    import sys
    sys.exit(main())

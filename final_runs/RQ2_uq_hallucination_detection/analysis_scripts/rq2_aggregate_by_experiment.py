#!/usr/bin/env python3
"""
RQ2 Aggregate Analysis by Experiment Type
Aggregates individual file results within each experiment type (citation, correctness).
Provides per-prompt-type breakdowns and statistical meta-analysis.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from rq2_paths import rq2_dirs

def load_batch_results(batch_dir: Path):
    """Load results from batch processing."""
    results_file = batch_dir / "batch_results.json"
    
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data

def load_individual_results(batch_dir: Path):
    """Load all individual metric_statistics.csv files."""
    all_data = []
    
    # Find all subdirectories with metric_statistics.csv
    for stats_file in batch_dir.glob("*/metric_statistics.csv"):
        df = pd.read_csv(stats_file)
        
        # Load metadata
        metadata_file = stats_file.parent / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Add metadata to each row
            for key, value in metadata.items():
                df[f'meta_{key}'] = value
            
            all_data.append(df)
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)

def aggregate_by_experiment(df, experiment_type):
    """Aggregate results for a single experiment type."""
    exp_df = df[df['meta_experiment_type'] == experiment_type].copy()
    
    if len(exp_df) == 0:
        return None
    
    # Pool mechanistic and mechanistic_original together
    # mechanistic_lit remains separate
    exp_df['meta_prompt_type_pooled'] = exp_df['meta_prompt_type'].copy()
    exp_df.loc[exp_df['meta_prompt_type'] == 'mechanistic_original', 'meta_prompt_type_pooled'] = 'mechanistic'
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_type.upper()}")
    print(f"{'='*80}")
    
    # Overall stats
    print(f"\nOverall Statistics:")
    print(f"  Total files analyzed: {exp_df['meta_filename'].nunique()}")
    print(f"  Total measurements: {len(exp_df)}")
    
    # By CLD
    print(f"\n  By CLD:")
    for cld in exp_df['meta_cld'].unique():
        count = len(exp_df[exp_df['meta_cld'] == cld])
        print(f"    {cld:25s}: {count} measurements")
    
    # By Prompt Type
    print(f"\n  By Prompt Type (after pooling mechanistic+mechanistic_original):")
    for prompt in sorted(exp_df['meta_prompt_type_pooled'].unique()):
        count = len(exp_df[exp_df['meta_prompt_type_pooled'] == prompt])
        print(f"    {prompt:25s}: {count} measurements")
    
    # Aggregate metrics
    metrics = exp_df['metric'].unique()
    
    aggregate_results = []
    
    for metric in metrics:
        metric_df = exp_df[exp_df['metric'] == metric].copy()
        
        # Overall aggregates
        agg = {
            'experiment_type': experiment_type,
            'metric': metric,
            'n_files': len(metric_df),
            'mean_auc': metric_df['auc'].dropna().mean(),
            'std_auc': metric_df['auc'].dropna().std(),
            'median_auc': metric_df['auc'].dropna().median(),
            'mean_correlation_r': metric_df['correlation_r'].dropna().mean(),
            'mean_cohens_d': metric_df['cohens_d'].dropna().mean(),
            'pct_significant': (metric_df['significant'] == True).sum() / len(metric_df) * 100,
        }
        
        # Test if mean AUC is significantly > 0.5 (one-sample t-test)
        auc_values = metric_df['auc'].dropna()
        if len(auc_values) >= 3:
            t_stat, p_value = stats.ttest_1samp(auc_values, 0.5)
            agg['ttest_vs_chance_t'] = float(t_stat)
            agg['ttest_vs_chance_p'] = float(p_value)
            agg['significantly_above_chance'] = (p_value < 0.05) and (t_stat > 0)
        else:
            agg['ttest_vs_chance_t'] = None
            agg['ttest_vs_chance_p'] = None
            agg['significantly_above_chance'] = False
        
        # Per-prompt-type breakdown (using pooled prompt types)
        for prompt in metric_df['meta_prompt_type_pooled'].unique():
            prompt_df = metric_df[metric_df['meta_prompt_type_pooled'] == prompt]
            agg[f'auc_{prompt}'] = prompt_df['auc'].dropna().mean()
            agg[f'n_{prompt}'] = len(prompt_df)
        
        # Per-CLD breakdown
        for cld in metric_df['meta_cld'].unique():
            cld_df = metric_df[metric_df['meta_cld'] == cld]
            agg[f'auc_{cld}'] = cld_df['auc'].dropna().mean()
            agg[f'n_{cld}'] = len(cld_df)
        
        aggregate_results.append(agg)
    
    agg_df = pd.DataFrame(aggregate_results)
    
    # Sort by mean AUC descending
    agg_df = agg_df.sort_values('mean_auc', ascending=False)
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"METRIC PERFORMANCE SUMMARY: {experiment_type.upper()}")
    print(f"{'='*80}")
    print(f"\n{'Metric':<30} {'Mean AUC':>10} {'Std':>8} {'Sig%':>8} {'Above Chance':>15}")
    print(f"{'-'*80}")
    
    for _, row in agg_df.iterrows():
        sig_marker = "***" if row['significantly_above_chance'] else ""
        print(f"{row['metric']:<30} {row['mean_auc']:>10.3f} {row['std_auc']:>8.3f} "
              f"{row['pct_significant']:>7.1f}% {sig_marker:>15}")
    
    return agg_df

def compare_prompt_types(df, experiment_type):
    """Compare performance across prompt types within an experiment."""
    exp_df = df[df['meta_experiment_type'] == experiment_type].copy()
    
    if len(exp_df) == 0:
        return None
    
    # Pool mechanistic and mechanistic_original together
    exp_df['meta_prompt_type_pooled'] = exp_df['meta_prompt_type'].copy()
    exp_df.loc[exp_df['meta_prompt_type'] == 'mechanistic_original', 'meta_prompt_type_pooled'] = 'mechanistic'
    
    print(f"\n{'='*80}")
    print(f"PROMPT TYPE COMPARISON: {experiment_type.upper()}")
    print(f"{'='*80}")
    
    # Get unique metrics and prompts (using pooled prompt types)
    metrics = exp_df['metric'].unique()
    prompts = exp_df['meta_prompt_type_pooled'].unique()
    
    comparison_data = []
    
    for metric in metrics:
        metric_df = exp_df[exp_df['metric'] == metric]
        
        for prompt in prompts:
            prompt_df = metric_df[metric_df['meta_prompt_type_pooled'] == prompt]
            
            if len(prompt_df) > 0:
                comparison_data.append({
                    'metric': metric,
                    'prompt_type': prompt,
                    'n_files': len(prompt_df),
                    'mean_auc': prompt_df['auc'].dropna().mean(),
                    'std_auc': prompt_df['auc'].dropna().std(),
                    'mean_corr': prompt_df['correlation_r'].dropna().mean(),
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print(f"\nAUC by Prompt Type:")
    
    pivot_auc = comparison_df.pivot_table(
        index='metric',
        columns='prompt_type',
        values='mean_auc',
        aggfunc='mean'
    )
    
    print(pivot_auc.to_string())
    
    return comparison_df

def create_visualizations(agg_df, experiment_type, output_dir):
    """Create visualizations for the experiment."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar plot of mean AUC by metric
    plt.figure(figsize=(10, 6))
    
    # Sort by AUC
    plot_data = agg_df.sort_values('mean_auc', ascending=True)
    
    colors = ['green' if x else 'gray' for x in plot_data['significantly_above_chance']]
    
    plt.barh(range(len(plot_data)), plot_data['mean_auc'], color=colors, alpha=0.7)
    plt.yticks(range(len(plot_data)), plot_data['metric'])
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Chance (0.5)')
    plt.xlabel('Mean AUC', fontsize=12)
    plt.title(f'CI Metric Performance: {experiment_type.upper()}\n(Green = Significantly > Chance)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(fig_dir / f'{experiment_type}_auc_barplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n   ✅ Saved: {experiment_type}_auc_barplot.png")

def generate_report(agg_df, comparison_df, experiment_type, output_dir):
    """Generate markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_path = output_dir / f"{experiment_type}_aggregate_report.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# RQ2 Aggregate Analysis: {experiment_type.upper()}\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write("---\n\n")
        
        f.write("## Overall Metric Performance\n\n")
        f.write("| Metric | Mean AUC | Std | Median AUC | Mean r | Mean Cohen's d | % Sig | Above Chance |\n")
        f.write("|--------|----------|-----|------------|--------|----------------|-------|---------------|\n")
        
        for _, row in agg_df.iterrows():
            sig_marker = "✓" if row['significantly_above_chance'] else "✗"
            f.write(f"| {row['metric']} | {row['mean_auc']:.3f} | {row['std_auc']:.3f} | "
                   f"{row['median_auc']:.3f} | {row['mean_correlation_r']:.3f} | "
                   f"{row['mean_cohens_d']:.3f} | {row['pct_significant']:.1f}% | {sig_marker} |\n")
        
        f.write("\n---\n\n")
        
        if comparison_df is not None:
            f.write("## Performance by Prompt Type\n\n")
            
            pivot = comparison_df.pivot_table(
                index='metric',
                columns='prompt_type',
                values='mean_auc',
                aggfunc='mean'
            )
            
            # Manually format as markdown table
            prompt_types = pivot.columns.tolist()
            f.write("| Metric | " + " | ".join(prompt_types) + " |\n")
            f.write("|" + "|".join(["-" * 8 for _ in range(len(prompt_types) + 1)]) + "|\n")
            
            for metric in pivot.index:
                values = [f"{pivot.loc[metric, pt]:.3f}" if not pd.isna(pivot.loc[metric, pt]) else "N/A" 
                         for pt in prompt_types]
                f.write(f"| {metric} | " + " | ".join(values) + " |\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        f.write("## Interpretation\n\n")
        
        # Best metric
        best = agg_df.iloc[0]
        f.write(f"**Best performing metric:** {best['metric']} (Mean AUC = {best['mean_auc']:.3f})\n\n")
        
        # Count above chance
        above_chance = (agg_df['significantly_above_chance'] == True).sum()
        total = len(agg_df)
        f.write(f"**Metrics significantly above chance:** {above_chance}/{total}\n\n")
    
    print(f"   ✅ Saved: {experiment_type}_aggregate_report.md")

def main():
    print("="*80)
    print("RQ2 AGGREGATE ANALYSIS BY EXPERIMENT")
    print("="*80)
    
    # Find the most recent batch directory
    analyses_dir, _unused_output = rq2_dirs()
    batch_dirs = list(analyses_dir.glob("rq2_batch_*"))
    
    if not batch_dirs:
        print("❌ No batch results found!")
        print("   Run rq2_simple_batch_analyzer.py first")
        sys.exit(1)
    
    batch_dir = max(batch_dirs, key=lambda p: p.stat().st_mtime)
    print(f"\n📁 Loading batch results from: {batch_dir.name}")
    
    # Load all individual results
    df = load_individual_results(batch_dir)
    
    if df is None:
        print("❌ No individual results found!")
        sys.exit(1)
    
    print(f"   Loaded {len(df)} metric measurements from {df['meta_filename'].nunique()} files")
    
    # Create output directory for aggregates
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = analyses_dir / f"rq2_aggregates_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir.name}")
    
    # Aggregate by experiment type
    experiment_types = df['meta_experiment_type'].unique()
    
    all_aggregates = {}
    all_comparisons = {}
    
    for exp_type in experiment_types:
        if exp_type == 'corruption':
            print(f"\nSkipping {exp_type} (no CI metrics available)")
            continue
        
        # Aggregate
        agg_df = aggregate_by_experiment(df, exp_type)
        
        if agg_df is not None:
            all_aggregates[exp_type] = agg_df
            
            # Compare prompt types
            comp_df = compare_prompt_types(df, exp_type)
            all_comparisons[exp_type] = comp_df
            
            # Create experiment-specific output directory
            exp_output_dir = output_dir / exp_type
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save aggregate results
            agg_df.to_csv(exp_output_dir / f"{exp_type}_aggregate_metrics.csv", index=False)
            if comp_df is not None:
                comp_df.to_csv(exp_output_dir / f"{exp_type}_prompt_comparison.csv", index=False)
            
            # Create visualizations
            create_visualizations(agg_df, exp_type, exp_output_dir)
            
            # Generate report
            generate_report(agg_df, comp_df, exp_type, exp_output_dir)
    
    # Save combined aggregates
    if all_aggregates:
        combined_df = pd.concat([
            df.assign(experiment_type=exp) 
            for exp, df in all_aggregates.items()
        ], ignore_index=True)
        
        combined_df.to_csv(output_dir / "all_experiments_aggregate.csv", index=False)
        print(f"\n✅ Saved combined aggregates: all_experiments_aggregate.csv")
    
    print("\n" + "="*80)
    print("AGGREGATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


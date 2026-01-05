#!/usr/bin/env python3
"""
RQ2 Grand Aggregate Analysis
Creates comprehensive cross-experiment analysis combining citation and correctness experiments.
Provides overall meta-analysis with statistical validation.
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

def load_all_individual_results(batch_dir: Path):
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

def bootstrap_ci_blocks(data, n_boot=10000, alpha=0.05):
    """Compute bootstrap CI for mean of block-level data."""
    data = np.array(data)
    n = len(data)
    if n < 2:
        return np.nan, np.nan
        
    means = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
        
    return np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])

def grand_meta_analysis(df):
    """Perform meta-analysis using Block-Level Aggregation (CLD x Run)."""
    
    # Pool mechanistic and mechanistic_original together
    df['meta_prompt_type_pooled'] = df['meta_prompt_type'].copy()
    df.loc[df['meta_prompt_type'] == 'mechanistic_original', 'meta_prompt_type_pooled'] = 'mechanistic'
    
    print("\n" + "="*80)
    print("GRAND META-ANALYSIS: BLOCK-LEVEL (CLD x RUN)")
    print("="*80)
    
    # Per-metric meta-analysis
    metrics = df['metric'].unique()
    
    meta_results = []
    
    for metric in metrics:
        metric_df = df[df['metric'] == metric].copy()
        
        # 1. Aggregate to Block Level (CLD, Run)
        # We take the mean AUC for each block
        block_df = metric_df.groupby(['meta_cld', 'meta_run'])['auc'].mean().reset_index()
        block_aucs = block_df['auc'].dropna()
        
        n_blocks = len(block_aucs)
        
        # Overall statistics (on blocks)
        mean_block_auc = float(block_aucs.mean())
        std_block_auc = float(block_aucs.std())
        median_block_auc = float(block_aucs.median())
        min_block_auc = float(block_aucs.min())
        max_block_auc = float(block_aucs.max())
        
        # 2. Non-parametric test on Blocks (Wilcoxon Signed-Rank vs 0.5)
        if n_blocks >= 6:  # Wilcoxon needs some sample size
            try:
                # Test if distribution is different from 0.5 (two-sided)
                # Note: wilcoxon tests x - y, so we pass (block_aucs - 0.5)
                w_stat, p_value = stats.wilcoxon(block_aucs - 0.5, alternative='two-sided')
                # Check direction: is it above 0.5?
                median_diff = np.median(block_aucs - 0.5)
                above_chance = bool((p_value < 0.05) and (median_diff > 0))
            except Exception as e:
                print(f"Wilcoxon failed for {metric}: {e}")
                w_stat, p_value, above_chance = None, None, False
        else:
            w_stat, p_value, above_chance = None, None, False
            
        # 3. Bootstrap CI over Blocks
        ci_lower, ci_upper = bootstrap_ci_blocks(block_aucs)
        
        # Also compute correlation stats (average r per block)
        block_corr_df = metric_df.groupby(['meta_cld', 'meta_run'])['correlation_r'].mean().reset_index()
        block_corrs = block_corr_df['correlation_r'].dropna()
        mean_block_corr = float(block_corrs.mean()) if len(block_corrs) > 0 else np.nan
        std_block_corr = float(block_corrs.std()) if len(block_corrs) > 0 else np.nan
        
        result = {
            'metric': metric,
            'n_measurements': len(metric_df),
            'n_files': metric_df['meta_filename'].nunique(),
            'n_blocks': n_blocks,
            'mean_auc': mean_block_auc,
            'std_auc': std_block_auc,
            'median_auc': median_block_auc,
            'min_auc': min_block_auc,
            'max_auc': max_block_auc,
            'mean_correlation_r': mean_block_corr,
            'std_correlation_r': std_block_corr,
            
            # Statistical test results
            'wilcoxon_stat': w_stat,
            'ttest_p': p_value,  # Keeping key name for compatibility with report gen
            'above_chance': above_chance,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            
            # Keep breakdowns for reporting
            'pct_significant': float((metric_df['significant'] == True).sum() / len(metric_df) * 100),
        }
        
        # Breakdown by experiment type
        for exp_type in metric_df['meta_experiment_type'].unique():
            exp_df = metric_df[metric_df['meta_experiment_type'] == exp_type]
            exp_auc = exp_df['auc'].dropna()
            result[f'auc_{exp_type}'] = float(exp_auc.mean()) if len(exp_auc) > 0 else None
            result[f'n_{exp_type}'] = len(exp_df)
        
        # Breakdown by CLD
        for cld in metric_df['meta_cld'].unique():
            cld_df = metric_df[metric_df['meta_cld'] == cld]
            cld_auc = cld_df['auc'].dropna()
            result[f'auc_cld_{cld}'] = float(cld_auc.mean()) if len(cld_auc) > 0 else None
        
        meta_results.append(result)
    
    meta_df = pd.DataFrame(meta_results)
    meta_df = meta_df.sort_values('mean_auc', ascending=False)
    
    return meta_df

def print_summary_table(meta_df):
    """Print summary table of grand meta-analysis."""
    print("\n" + "="*80)
    print("METRIC PERFORMANCE RANKING")
    print("="*80)
    print(f"\n{'Metric':<30} {'Mean AUC':>10} {'95% CI':>20} {'p-value':>10} {'Sig':>5}")
    print("-" * 80)
    
    for _, row in meta_df.iterrows():
        ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]" if row['ci_lower'] is not None else "N/A"
        p_str = f"{row['ttest_p']:.4f}" if row['ttest_p'] is not None else "N/A"
        sig_str = "***" if row['above_chance'] else ""
        
        print(f"{row['metric']:<30} {row['mean_auc']:>10.3f} {ci_str:>20} {p_str:>10} {sig_str:>5}")

def create_grand_visualizations(meta_df, df, output_dir):
    """Create comprehensive visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Forest plot of AUC with 95% CI
    plt.figure(figsize=(10, 8))
    
    plot_data = meta_df.sort_values('mean_auc', ascending=True)
    y_pos = range(len(plot_data))
    
    for i, (_, row) in enumerate(plot_data.iterrows()):
        color = 'green' if row['above_chance'] else 'gray'
        plt.errorbar(
            row['mean_auc'], i,
            xerr=[[row['mean_auc'] - row['ci_lower']], [row['ci_upper'] - row['mean_auc']]],
            fmt='o', color=color, ecolor=color, capsize=5, markersize=8
        )
    
    plt.yticks(y_pos, plot_data['metric'])
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Chance (0.5)')
    plt.xlabel('Mean AUC (with 95% CI)', fontsize=12)
    plt.title('CI Metrics Performance: Grand Meta-Analysis\n(Green = Significantly > Chance)', 
             fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(fig_dir / 'grand_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n   ✅ Saved: grand_forest_plot.png")
    
    # 2. Comparison across experiments
    plt.figure(figsize=(12, 6))
    
    comparison_data = []
    for _, row in meta_df.iterrows():
        for exp_type in ['citation', 'correctness']:
            auc_col = f'auc_{exp_type}'
            if auc_col in row and row[auc_col] is not None:
                comparison_data.append({
                    'Metric': row['metric'],
                    'Experiment': exp_type,
                    'AUC': row[auc_col]
                })
    
    comp_df = pd.DataFrame(comparison_data)
    
    pivot = comp_df.pivot(index='Metric', columns='Experiment', values='AUC')
    pivot.plot(kind='bar', figsize=(12, 6), width=0.8)
    
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Chance')
    plt.xlabel('CI Metric', fontsize=12)
    plt.ylabel('Mean AUC', fontsize=12)
    plt.title('CI Metrics Performance by Experiment Type', fontsize=14, fontweight='bold')
    plt.legend(title='Experiment')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(fig_dir / 'experiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: experiment_comparison.png")

def generate_grand_report(meta_df, df, output_dir):
    """Generate comprehensive markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_path = output_dir / "grand_aggregate_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# RQ2 Grand Aggregate Analysis: All Experiments\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write("---\n\n")
        
        f.write("## Executive Summary\n\n")
        
        # Dataset summary
        f.write(f"**Total Measurements:** {len(df)}\n")
        f.write(f"**Total Files:** {df['meta_filename'].nunique()}\n")
        f.write(f"**Experiments:** {', '.join(df['meta_experiment_type'].unique())}\n")
        f.write(f"**CLDs:** {', '.join(df['meta_cld'].unique())}\n")
        f.write(f"**Prompt Types (pooled):** {', '.join(sorted(df['meta_prompt_type_pooled'].unique()))}\n\n")
        
        # Key findings
        best = meta_df.iloc[0]
        above_chance = (meta_df['above_chance'] == True).sum()
        
        f.write("### Key Findings\n\n")
        f.write(f"1. **Best performing metric:** {best['metric']} (Mean AUC = {best['mean_auc']:.3f})\n")
        f.write(f"2. **Metrics significantly above chance:** {above_chance}/{len(meta_df)}\n")
        f.write(f"3. **Overall discrimination:** ")
        
        if best['mean_auc'] > 0.7:
            f.write("Good (AUC > 0.7)\n")
        elif best['mean_auc'] > 0.6:
            f.write("Moderate (0.6 < AUC < 0.7)\n")
        else:
            f.write("Weak (AUC < 0.6)\n")
        
        f.write("\n---\n\n")
        
        f.write("## Overall Metric Performance\n\n")
        f.write("| Metric | Mean AUC | 95% CI | Std | Median | Min | Max | Mean r | % Sig | Above Chance |\n")
        f.write("|--------|----------|--------|-----|--------|-----|-----|--------|-------|---------------|\n")
        
        for _, row in meta_df.iterrows():
            ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]" if row['ci_lower'] is not None else "N/A"
            sig_marker = "✓" if row['above_chance'] else "✗"
            
            f.write(f"| {row['metric']} | {row['mean_auc']:.3f} | {ci_str} | {row['std_auc']:.3f} | "
                   f"{row['median_auc']:.3f} | {row['min_auc']:.3f} | {row['max_auc']:.3f} | "
                   f"{row['mean_correlation_r']:.3f} | {row['pct_significant']:.1f}% | {sig_marker} |\n")
        
        f.write("\n---\n\n")
        
        f.write("## Performance by Experiment Type\n\n")
        f.write("| Metric | Citation | Correctness |\n")
        f.write("|--------|----------|-------------|\n")
        
        for _, row in meta_df.iterrows():
            citation_auc = f"{row['auc_citation']:.3f}" if 'auc_citation' in row and row['auc_citation'] is not None else "N/A"
            correctness_auc = f"{row['auc_correctness']:.3f}" if 'auc_correctness' in row and row['auc_correctness'] is not None else "N/A"
            f.write(f"| {row['metric']} | {citation_auc} | {correctness_auc} |\n")
        
        f.write("\n---\n\n")
        
        f.write("## Statistical Interpretation\n\n")
        
        f.write("### One-Sample Wilcoxon Tests (Block Mean AUC vs. Chance = 0.5)\n\n")
        
        for _, row in meta_df.iterrows():
            if row['ttest_p'] is not None:
                sig_level = ""
                if row['ttest_p'] < 0.001:
                    sig_level = " (p < 0.001, ***)"
                elif row['ttest_p'] < 0.01:
                    sig_level = " (p < 0.01, **)"
                elif row['ttest_p'] < 0.05:
                    sig_level = " (p < 0.05, *)"
                
                interpretation = "significantly above chance" if row['above_chance'] else "not significantly different from chance"
                
                # Using 'ttest_p' key for compatibility but it holds Wilcoxon p-value now
                w_stat = row.get('wilcoxon_stat', row.get('ttest_t', float('nan')))
                if w_stat is None: w_stat = float('nan')
                
                f.write(f"- **{row['metric']}**: W={w_stat:.1f}, p={row['ttest_p']:.4f}{sig_level} → {interpretation}\n")
        
        f.write("\n---\n\n")
        
        f.write("## Research Question Answer\n\n")
        f.write("**RQ2: Can context-insensitive (CI) metrics predict hallucinations?**\n\n")
        
        if above_chance > 0:
            f.write(f"**Answer: Partially Yes.** {above_chance}/{len(meta_df)} CI metrics show significantly above-chance performance. ")
            f.write(f"The best metric ({best['metric']}) achieves Mean AUC = {best['mean_auc']:.3f}, ")
            f.write(f"indicating {('good' if best['mean_auc'] > 0.7 else 'moderate' if best['mean_auc'] > 0.6 else 'weak')} discrimination ability.\n\n")
            
            f.write("**Practical Implications:**\n")
            f.write("- CI metrics can be used as lightweight hallucination detectors\n")
            f.write("- They are not perfect but better than random guessing\n")
            f.write("- Combining multiple metrics (ensemble) may improve performance\n")
        else:
            f.write("**Answer: No.** None of the CI metrics show significantly above-chance performance ")
            f.write("in detecting hallucinations across all experiments.\n")
    
    print(f"   ✅ Saved: grand_aggregate_report.md")

def main():
    print("="*80)
    print("RQ2 GRAND AGGREGATE ANALYSIS")
    print("="*80)
    
    # Find the most recent batch directory
    analyses_dir, _unused_output = rq2_dirs()
    batch_dirs = list(analyses_dir.glob("rq2_batch_*"))
    
    if not batch_dirs:
        print("❌ No batch results found!")
        sys.exit(1)
    
    batch_dir = max(batch_dirs, key=lambda p: p.stat().st_mtime)
    print(f"\n📁 Loading batch results from: {batch_dir.name}")
    
    # Load all individual results
    df = load_all_individual_results(batch_dir)
    
    if df is None:
        print("❌ No individual results found!")
        sys.exit(1)
    
    print(f"   Loaded {len(df)} measurements from {df['meta_filename'].nunique()} files")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = analyses_dir / f"rq2_grand_aggregate_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir.name}")
    
    # Perform grand meta-analysis
    meta_df = grand_meta_analysis(df)
    
    # Print summary
    print_summary_table(meta_df)
    
    # Save results as Excel
    meta_df.to_excel(output_dir / "grand_meta_analysis.xlsx", index=False, engine='openpyxl')
    print(f"\n✅ Saved: grand_meta_analysis.xlsx")
    
    # Create visualizations
    create_grand_visualizations(meta_df, df, output_dir)
    
    # Generate report
    generate_grand_report(meta_df, df, output_dir)
    
    print("\n" + "="*80)
    print("GRAND AGGREGATE ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



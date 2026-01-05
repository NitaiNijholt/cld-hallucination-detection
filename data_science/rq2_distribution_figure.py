#!/usr/bin/env python3
"""
RQ2 Distribution Figure Generator

Generates a readable 4x4 grid figure showing:
- Rows: 4 UQ metrics (Gen Perplexity, Gen Min Prob, Gen Max Window Entropy, Gen Cosine Similarity)
- Columns: 3 CLDs

Author: Generated for thesis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from rq2_data_preparation import load_rq2_combined_data

# Define metrics
CI_METRICS = [
    'Gen Perplexity',
    'Gen Min Prob', 
    'Gen Max Window Entropy',
    'Gen Cosine Similarity'
]

# Pretty names for display
METRIC_LABELS = {
    'Gen Perplexity': 'Perplexity',
    'Gen Min Prob': 'Min Probability',
    'Gen Max Window Entropy': 'Max Window Entropy',
    'Gen Cosine Similarity': 'Cosine Similarity'
}

CLD_LABELS = {
    'depressive': 'Depressive',
    'emergency_department': 'Emergency Dept',
    'social_norms': 'Social Norms'
}


def rank_biserial_correlation(group1, group2, U=None):
    """
    Calculate rank-biserial correlation (r) - non-parametric effect size for Mann-Whitney U.
    
    Formula: r = 1 - (2U)/(n1*n2)
    Where U = count of (group1_i > group2_j) pairs from mannwhitneyu(group1, group2)
    
    Interpretation (same thresholds as Pearson r):
    - |r| < 0.10: negligible
    - 0.10 <= |r| < 0.30: small
    - 0.30 <= |r| < 0.50: medium
    - |r| >= 0.50: large
    
    Returns value in [-1, 1] where:
    - Positive r: group2 tends to have higher values (group1 "loses")
    - Negative r: group1 tends to have higher values (group1 "wins")
    
    In our context (group1=correct, group2=hallucination):
    - Positive r: hallucinations have higher metric values
    - Negative r: correct edges have higher metric values
    """
    n1, n2 = len(group1), len(group2)
    if U is None:
        # Calculate U if not provided
        U, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Rank-biserial correlation: r = 1 - (2U)/(n1*n2)
    r = 1 - (2 * U) / (n1 * n2)
    return r


def generate_distribution_figure(df, output_path):
    """Generate a clean 4x3 distribution figure with block-level effect summaries."""
    
    clds = sorted(df['cld'].unique())
    n_metrics = len(CI_METRICS)
    n_cols = len(clds)  # CLDs only
    
    # Create figure with adjusted size for readability
    fig, axes = plt.subplots(n_metrics, n_cols, figsize=(13, 12))
    
    # Colors
    correct_color = '#27ae60'  # Green
    halluc_color = '#e74c3c'   # Red
    
    for row_idx, metric in enumerate(CI_METRICS):
        metric_label = METRIC_LABELS.get(metric, metric)
        
        for col_idx, cld in enumerate(clds):
            ax = axes[row_idx, col_idx]
            
            cld_df = df[df['cld'] == cld]
            correct_data = cld_df[cld_df['is_hallucination'] == False][metric].dropna()
            halluc_data = cld_df[cld_df['is_hallucination'] == True][metric].dropna()
            
            if len(correct_data) > 0 and len(halluc_data) > 0:
                # Clip to 99th percentile to handle outliers
                combined = pd.concat([correct_data, halluc_data])
                q01, q99 = combined.quantile(0.01), combined.quantile(0.99)
                
                correct_clipped = correct_data[(correct_data >= q01) & (correct_data <= q99)]
                halluc_clipped = halluc_data[(halluc_data >= q01) & (halluc_data <= q99)]
                
                # Create bins
                all_clipped = pd.concat([correct_clipped, halluc_clipped])
                bins = np.histogram_bin_edges(all_clipped, bins=25)
                
                # Plot histograms
                ax.hist(correct_clipped, bins=bins, alpha=0.6, label=f'Correct (n={len(correct_data)})',
                       color=correct_color, density=True, edgecolor='white', linewidth=0.3)
                ax.hist(halluc_clipped, bins=bins, alpha=0.6, label=f'Halluc (n={len(halluc_data)})',
                       color=halluc_color, density=True, edgecolor='white', linewidth=0.3)
                
                # Block-level effect summary (unit = CLD×run blocks, n=3 runs per CLD)
                block_rs = []
                for run in sorted(cld_df['run'].dropna().unique()):
                    run_df = cld_df[cld_df['run'] == run]
                    correct_run = run_df[run_df['is_hallucination'] == False][metric].dropna()
                    halluc_run = run_df[run_df['is_hallucination'] == True][metric].dropna()
                    if len(correct_run) >= 2 and len(halluc_run) >= 2:
                        try:
                            U_stat, _ = stats.mannwhitneyu(correct_run, halluc_run, alternative='two-sided')
                            r_run = rank_biserial_correlation(correct_run.values, halluc_run.values, U=U_stat)
                            block_rs.append(float(r_run))
                        except Exception:
                            pass
                
                if block_rs:
                    r_mean = float(np.mean(block_rs))
                    r_min = float(np.min(block_rs))
                    r_max = float(np.max(block_rs))
                    abs_r = abs(r_mean)
                    if abs_r < 0.10:
                        effect_label = 'negl.'
                    elif abs_r < 0.30:
                        effect_label = 'small'
                    elif abs_r < 0.50:
                        effect_label = 'med.'
                    else:
                        effect_label = 'large'
                    ax.text(
                        0.98, 0.98,
                        f"blocks r={r_mean:.2f} [{r_min:.2f},{r_max:.2f}] ({effect_label})",
                        transform=ax.transAxes, fontsize=8, ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85)
                    )
                else:
                    ax.text(
                        0.98, 0.98,
                        "blocks: n/a",
                        transform=ax.transAxes, fontsize=8, ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85)
                    )
                
                # Labels
                ax.set_xlabel(metric_label, fontsize=9)
                ax.set_ylabel('Density', fontsize=9)
                ax.tick_params(axis='both', labelsize=8)
                
                # Legend only on first row
                if row_idx == 0:
                    ax.legend(fontsize=7, loc='upper left')
                
            else:
                ax.text(0.5, 0.5, 'Insufficient\ndata', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
            
            # Title: CLD name (only on first row)
            if row_idx == 0:
                ax.set_title(CLD_LABELS.get(cld, cld), fontsize=11, fontweight='bold', pad=10)
            
            # Y-axis label: metric name (only on first column)
            if col_idx == 0:
                ax.set_ylabel(f'{metric_label}\nDensity', fontsize=9, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.35)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")


def main():
    """Main entry point."""
    print("="*80)
    print("RQ2 DISTRIBUTION FIGURE GENERATOR")
    print("="*80)
    
    # Load data
    print("\nLoading RQ2 combined data...")
    df = load_rq2_combined_data()
    
    if df is None or df.empty:
        print("ERROR: Could not load data")
        return 1
    
    print(f"Loaded {len(df)} edges from {df['cld'].nunique()} CLDs")
    print(f"Hallucinations: {df['is_hallucination'].sum()} ({df['is_hallucination'].mean():.1%})")
    
    # Output path
    output_dir = Path(__file__).parent.parent / 'final_runs' / 'RQ2_uq_hallucination_detection'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'phase6_distributions_with_effect_sizes.png'
    
    # Generate figure
    print(f"\nGenerating distribution figure...")
    generate_distribution_figure(df, output_path)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


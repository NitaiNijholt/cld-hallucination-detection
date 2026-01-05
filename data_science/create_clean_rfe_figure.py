#!/usr/bin/env python3
"""
Create a clean, publication-ready RFE figure for thesis.
Focuses on the key findings: Permutation Importance with statistical significance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11


def create_clean_rfe_figure():
    """Create a clean 2-panel figure for RFE analysis."""
    
    # Data from the original analysis (extracted from rigorous_feature_selection output)
    # Permutation importance data (sorted by importance)
    perm_data = {
        'feature': [
            'Gen Mean Token Prob',
            'Gen Mean Token Prob Slope',
            'Gen Min Window Entropy',
            'Gen Prob Std',
            'Gen Min Prob',
            'Gen Max Window Entropy',
            'Gen Perplexity',
            'Gen Max Prob Diff',
            'Gen Cosine Similarity'
        ],
        'mean_importance': [-0.043, -0.019, -0.010, -0.004, 0.001, 0.030, 0.044, 0.074, 0.142],
        'ci_lower': [-0.061, -0.039, -0.029, -0.022, -0.017, 0.012, 0.024, 0.054, 0.115],
        'ci_upper': [-0.025, 0.001, 0.009, 0.014, 0.019, 0.048, 0.064, 0.094, 0.169],
        'significant': [True, False, False, False, False, True, True, True, True]
    }
    
    perm_df = pd.DataFrame(perm_data)
    perm_df = perm_df.sort_values('mean_importance', ascending=True).reset_index(drop=True)
    
    # Feature selection frequency (out of 5 folds)
    selection_freq = {
        'Gen Cosine Similarity': 5,
        'Gen Perplexity': 5,
        'Gen Max Prob Diff': 4,
        'Gen Max Window Entropy': 3,
        'Gen Min Prob': 2,
        'Gen Prob Std': 2,
        'Gen Min Window Entropy': 1,
        'Gen Mean Token Prob Slope': 1,
        'Gen Mean Token Prob': 0
    }
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========================================================================
    # Panel A: Permutation Importance with 95% CI
    # ========================================================================
    ax1 = axes[0]
    
    y_pos = np.arange(len(perm_df))
    colors = ['#2ecc71' if sig else '#95a5a6' for sig in perm_df['significant']]
    
    # Plot bars
    bars = ax1.barh(y_pos, perm_df['mean_importance'], color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1)
    
    # Add error bars (95% CI)
    for i, row in perm_df.iterrows():
        ax1.plot([row['ci_lower'], row['ci_upper']], [i, i], 
                color='black', linewidth=2, solid_capstyle='round')
        # Add caps
        cap_height = 0.15
        ax1.plot([row['ci_lower'], row['ci_lower']], [i-cap_height, i+cap_height], 
                color='black', linewidth=2)
        ax1.plot([row['ci_upper'], row['ci_upper']], [i-cap_height, i+cap_height], 
                color='black', linewidth=2)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(perm_df['feature'], fontsize=11)
    ax1.set_xlabel('Permutation Importance (AUC Drop)', fontweight='bold')
    ax1.set_title('Feature Importance via Permutation Testing\n(Green = $p < 0.05$)', 
                  fontweight='bold', fontsize=14)
    ax1.axvline(0, color='black', linestyle='-', linewidth=1)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_xlim([-0.08, 0.20])
    
    # Add significance count annotation
    n_sig = perm_df['significant'].sum()
    ax1.text(0.95, 0.05, f'{n_sig}/9 features significant', 
             transform=ax1.transAxes, fontsize=11, ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========================================================================
    # Panel B: Feature Selection Stability
    # ========================================================================
    ax2 = axes[1]
    
    # Sort by frequency
    sorted_feats = sorted(selection_freq.items(), key=lambda x: -x[1])
    feats = [f[0] for f in sorted_feats]
    counts = [f[1] for f in sorted_feats]
    
    y_pos2 = np.arange(len(feats))
    colors2 = ['#2ecc71' if c >= 3 else '#f39c12' if c >= 2 else '#e74c3c' for c in counts]
    
    bars2 = ax2.barh(y_pos2, counts, color=colors2, alpha=0.8, 
                     edgecolor='black', linewidth=1)
    
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(feats, fontsize=11)
    ax2.set_xlabel('Selection Frequency (out of 5 folds)', fontweight='bold')
    ax2.set_title('Cross-Validation Feature Selection Stability\n(Green = selected in $\\geq$60% of folds)', 
                  fontweight='bold', fontsize=14)
    ax2.axvline(3, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_xlim([0, 5.5])
    ax2.set_xticks([0, 1, 2, 3, 4, 5])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Stable ($\\geq$3/5 folds)'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Moderate (2/5 folds)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Unstable ($\\leq$1/5 folds)')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = Path('/home/nitai/code/causalix.ai/final_runs/RQ2_uq_hallucination_detection/rfe_clean_figure.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Clean RFE figure saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_clean_rfe_figure()












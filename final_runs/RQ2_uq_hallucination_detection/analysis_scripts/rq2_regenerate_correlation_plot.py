#!/usr/bin/env python3
"""
RQ2: Regenerate correlation plot with:
1. Only generator metrics (no judge metrics)
2. Shorter CLD names
3. Check that all 63 files are included
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

from rq2_paths import rq2_dirs
# CLD name mappings for shorter display
CLD_NAME_MAP = {
    'emergency_department': 'Emergency dept',
    'social_norms': 'Social norms',
    'depressive': 'Depressive symptoms'
}

# Generator metrics only (no judge metrics)
GENERATOR_METRICS = [
    'Gen Perplexity',
    'Gen Min Prob',
    'Gen Max Window Entropy',
    'Gen Cosine Similarity'
]

def load_file_list():
    """Load the list of 63 files."""
    _analyses_dir, output_dir = rq2_dirs()
    with open(output_dir / "rq2_files_63_list.json", 'r') as f:
        return json.load(f)

def compute_correlations_per_file(filepath):
    """Compute Pearson correlations for generator metrics in a single file."""
    try:
        df = pd.read_excel(filepath)
        
        # Check for required columns
        if 'Classification' not in df.columns:
            return None
        
        # Define hallucination: FP or FN
        df['is_hallucination'] = df['Classification'].isin(['FP', 'FN']).astype(int)
        
        correlations = {}
        for metric in GENERATOR_METRICS:
            if metric in df.columns:
                # Remove NaN values
                valid_data = df[[metric, 'is_hallucination']].dropna()
                if len(valid_data) > 10:  # Need sufficient data points
                    r, p = stats.pearsonr(valid_data[metric], valid_data['is_hallucination'])
                    correlations[metric] = {
                        'pearson_r': r,
                        'p_value': p,
                        'n': len(valid_data)
                    }
        
        return correlations if correlations else None
        
    except Exception as e:
        print(f"  Warning: Could not process {filepath.name}: {e}")
        return None

def aggregate_correlations_by_cld(file_list):
    """Aggregate correlations by CLD."""
    results = []
    
    print(f"\n🔄 Processing {len(file_list)} files...")
    processed_count = 0
    
    for file_info in file_list:
        filepath = Path(file_info['filepath'])
        
        if not filepath.exists():
            continue
        
        correlations = compute_correlations_per_file(filepath)
        if correlations:
            processed_count += 1
            cld = file_info['cld']
            
            for metric, corr_data in correlations.items():
                results.append({
                    'metric': metric,
                    'cld_name': cld,
                    'pearson_r': corr_data['pearson_r'],
                    'p_value': corr_data['p_value'],
                    'n': corr_data['n'],
                    'run': file_info['run'],
                    'prompt_type': file_info['prompt_type'],
                    'experiment_type': file_info['experiment_type']
                })
    
    print(f"✓ Successfully processed {processed_count} / {len(file_list)} files")
    
    if not results:
        raise ValueError("No correlation data computed!")
    
    return pd.DataFrame(results)

def compute_meta_statistics(df, value_col='pearson_r', group_col='metric'):
    """Compute meta-statistics (mean, std, CI) across datasets."""
    meta_stats = df.groupby(group_col)[value_col].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('n_files', 'count')
    ]).reset_index()
    
    # Compute total number of edges
    n_edges = df.groupby(group_col)['n'].sum().reset_index()
    n_edges.columns = [group_col, 'n_edges']
    meta_stats = meta_stats.merge(n_edges, on=group_col)
    
    # Compute 95% CI
    meta_stats['ci_lower'] = meta_stats['mean'] - 1.96 * meta_stats['std'] / np.sqrt(meta_stats['n_files'])
    meta_stats['ci_upper'] = meta_stats['mean'] + 1.96 * meta_stats['std'] / np.sqrt(meta_stats['n_files'])
    
    return meta_stats

def plot_correlations(corr_df, output_path):
    """Plot correlation heatmap and meta-analysis."""
    
    # Apply CLD name mapping for shorter labels
    corr_df['cld_display'] = corr_df['cld_name'].map(CLD_NAME_MAP)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Correlation coefficients by dataset (heatmap)
    pivot = corr_df.pivot_table(values='pearson_r', index='metric', columns='cld_display')
    
    # Reorder rows to match TABLE_1 (descending by mean correlation)
    row_order = corr_df.groupby('metric')['pearson_r'].mean().sort_values(ascending=False).index
    pivot = pivot.loc[row_order]
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
                ax=axes[0], cbar_kws={'label': 'Pearson r'},
                vmin=-0.3, vmax=0.3)
    axes[0].set_title('Correlations with Hallucinations\nAcross Datasets', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('CI Metric', fontsize=10)
    axes[0].tick_params(axis='x', rotation=0)
    
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
    # Add n_edges to labels
    labels = [f"{row['metric']} (n={int(row['n_edges'])})" for _, row in meta_stats.iterrows()]
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel('Mean Pearson r (95% CI)', fontsize=10)
    axes[1].set_title('Meta-Analysis: Average Correlation\nAcross All Datasets', fontsize=12, fontweight='bold')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].set_xlim([-0.1, 0.2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(meta_stats[['metric', 'mean', 'ci_lower', 'ci_upper', 'n_files', 'n_edges']].to_string(index=False))

def main():
    print("=" * 80)
    print("RQ2: REGENERATE CORRELATION PLOT")
    print("=" * 80)
    print("Changes:")
    print("  • Filter out judge metrics (generator metrics only)")
    print("  • Use shorter CLD names")
    print("  • Verify all 63 files are processed")
    print()
    
    # Load file list
    file_list = load_file_list()
    print(f"📁 Loaded {len(file_list)} files from rq2_files_63_list.json")
    
    # Aggregate correlations
    corr_df = aggregate_correlations_by_cld(file_list)
    
    # Save aggregated data
    output_dir = Path('final_runs/RQ2_uq_hallucination_detection')
    corr_df.to_csv(output_dir / 'aggregate_correlations_regenerated.csv', index=False)
    print(f"✓ Saved: aggregate_correlations_regenerated.csv")
    
    # Create plot
    plot_correlations(corr_df, output_dir / 'aggregate_correlations.png')
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()


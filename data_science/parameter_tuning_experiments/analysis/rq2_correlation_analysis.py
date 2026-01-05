"""
RQ2 Correlation Analysis

Computes correlations between context-insensitive metrics and hallucination labels.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class RQ2CorrelationAnalysis:
    """Correlation analysis for RQ2."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Separate generator and judge CI metrics for clarity
        self.generator_ci_metrics = [
            'perplexity', 'min_prob', 'max_window_entropy', 'cosine_similarity'
        ]
        self.judge_ci_metrics = [
            'judge_perplexity', 'judge_min_prob', 'judge_max_window_entropy', 'judge_cosine_similarity',
            'aggregate_score'  # Judge's aggregate verdict score
        ]
        self.ci_metrics = self.generator_ci_metrics + self.judge_ci_metrics
        
        # Filter to edges with CI metrics
        available_metrics = [m for m in self.ci_metrics if m in df.columns]
        self.df_filtered = df.dropna(subset=available_metrics, how='all').copy()
        
        print(f"Initialized with {len(df)} edges, {len(self.df_filtered)} have CI metrics")
        
    def compute_correlations(self) -> pd.DataFrame:
        """Compute correlation coefficients for all CI metrics."""
        
        results = []
        
        for metric in self.ci_metrics:
            if metric not in self.df_filtered.columns:
                print(f"Warning: {metric} not in data, skipping")
                continue
                
            valid_data = self.df_filtered[[metric, 'is_hallucination']].dropna()
            
            if len(valid_data) < 3:
                print(f"Warning: Only {len(valid_data)} samples for {metric}, skipping")
                continue
            
            X = valid_data[metric]
            y = valid_data['is_hallucination'].astype(int)
            
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(X, y)
            
            # Spearman correlation (for non-linear relationships)
            spearman_r, spearman_p = stats.spearmanr(X, y)
            
            results.append({
                'metric': metric,
                'n_samples': len(valid_data),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'significant': pearson_p < 0.05
            })
        
        self.correlation_results = pd.DataFrame(results)
        return self.correlation_results
    
    def group_comparison(self) -> pd.DataFrame:
        """Compare CI metric distributions between hallucination/non-hallucination."""
        
        results = []
        
        for metric in self.ci_metrics:
            if metric not in self.df_filtered.columns:
                continue
                
            valid_data = self.df_filtered[[metric, 'is_hallucination']].dropna()
            
            if len(valid_data) < 3:
                continue
            
            halluc = valid_data[valid_data['is_hallucination']][metric]
            non_halluc = valid_data[~valid_data['is_hallucination']][metric]
            
            if len(halluc) < 2 or len(non_halluc) < 2:
                continue
            
            # t-test
            t_stat, t_p = stats.ttest_ind(halluc, non_halluc, equal_var=False)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((halluc.std()**2 + non_halluc.std()**2) / 2)
            cohens_d = (halluc.mean() - non_halluc.mean()) / pooled_std if pooled_std > 0 else 0
            
            results.append({
                'metric': metric,
                'halluc_mean': halluc.mean(),
                'halluc_std': halluc.std(),
                'halluc_n': len(halluc),
                'non_halluc_mean': non_halluc.mean(),
                'non_halluc_std': non_halluc.std(),
                'non_halluc_n': len(non_halluc),
                't_statistic': t_stat,
                't_p_value': t_p,
                'cohens_d': cohens_d,
                'significant': t_p < 0.05
            })
        
        self.group_comparison_results = pd.DataFrame(results)
        return self.group_comparison_results
    
    def visualize_distributions(self, output_dir: str = "figures"):
        """Create distribution comparison plots."""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate grid size to accommodate all metrics
        n_metrics = len(self.ci_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.ci_metrics):
            if metric not in self.df_filtered.columns:
                axes[i].text(0.5, 0.5, f'{metric}\nNo data available', 
                           ha='center', va='center')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                continue
                
            valid_data = self.df_filtered[[metric, 'is_hallucination']].dropna()
            
            if len(valid_data) < 3:
                axes[i].text(0.5, 0.5, f'Insufficient data\n(n={len(valid_data)})', 
                           ha='center', va='center')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                continue
            
            halluc = valid_data[valid_data['is_hallucination']][metric]
            non_halluc = valid_data[~valid_data['is_hallucination']][metric]
            
            # Plot histograms
            axes[i].hist(non_halluc, alpha=0.6, label='Non-hallucination', bins=15, 
                        color='green', edgecolor='black')
            axes[i].hist(halluc, alpha=0.6, label='Hallucination', bins=15, 
                        color='red', edgecolor='black')
            
            axes[i].set_xlabel(metric.replace('_', ' ').title())
            axes[i].set_ylabel('Count')
            axes[i].legend()
            axes[i].set_title(f'{metric.replace("_", " ").title()}\n' +
                            f'(n_halluc={len(halluc)}, n_non={len(non_halluc)})')
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rq2_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_dir}/rq2_distributions.png")
    
    def visualize_scatter_plots(self, output_dir: str = "figures"):
        """Create scatter plots with regression lines."""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate grid size to accommodate all metrics
        n_metrics = len(self.ci_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.ci_metrics):
            if metric not in self.df_filtered.columns:
                axes[i].text(0.5, 0.5, f'{metric}\nNo data', ha='center', va='center')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                continue
            
            valid_data = self.df_filtered[[metric, 'is_hallucination']].dropna()
            
            if len(valid_data) < 3:
                axes[i].text(0.5, 0.5, f'Insufficient data', ha='center', va='center')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                continue
            
            X = valid_data[metric]
            y = valid_data['is_hallucination'].astype(int)
            
            # Add jitter to y for visibility
            y_jittered = y + np.random.normal(0, 0.02, len(y))
            
            # Scatter plot
            axes[i].scatter(X, y_jittered, alpha=0.5, s=30, 
                          c=y, cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
            
            # Add regression line
            if len(X) > 1:
                z = np.polyfit(X, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(X.min(), X.max(), 100)
                axes[i].plot(x_line, p(x_line), "b-", alpha=0.8, linewidth=2, 
                           label='Linear fit')
            
            axes[i].set_xlabel(metric.replace('_', ' ').title())
            axes[i].set_ylabel('Hallucination (0=No, 1=Yes)')
            axes[i].set_ylim([-0.1, 1.1])
            axes[i].legend()
            
            # Add correlation to plot
            if hasattr(self, 'correlation_results'):
                corr_row = self.correlation_results[self.correlation_results['metric'] == metric]
                if len(corr_row) > 0:
                    r = corr_row['pearson_r'].values[0]
                    p = corr_row['pearson_p'].values[0]
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    axes[i].text(0.05, 0.95, f'r = {r:.3f}{sig}\np = {p:.4f}',
                               transform=axes[i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rq2_scatter_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_dir}/rq2_scatter_plots.png")


def main():
    """Test/demo correlation analysis."""
    import sys
    
    # Load data
    data_file = Path(__file__).parent.parent / 'tables' / 'rq2_standalone_data.csv'
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} edges\n")
    
    # Run analysis
    analyzer = RQ2CorrelationAnalysis(df)
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Compute correlations
    corr_results = analyzer.compute_correlations()
    print("\nCorrelation Results:")
    print(corr_results.to_string(index=False))
    
    # Group comparison
    print("\n" + "="*60)
    print("GROUP COMPARISON (t-tests)")
    print("="*60)
    group_results = analyzer.group_comparison()
    print("\nGroup Comparison:")
    print(group_results[['metric', 'halluc_mean', 'non_halluc_mean', 
                         't_p_value', 'cohens_d', 'significant']].to_string(index=False))
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    output_dir = Path(__file__).parent.parent / 'figures'
    analyzer.visualize_distributions(str(output_dir))
    analyzer.visualize_scatter_plots(str(output_dir))
    
    # Save results
    tables_dir = Path(__file__).parent.parent / 'tables'
    tables_dir.mkdir(exist_ok=True)
    
    corr_results.to_csv(tables_dir / 'rq2_correlations.csv', index=False)
    group_results.to_csv(tables_dir / 'rq2_group_comparison.csv', index=False)
    
    print(f"\n✓ Saved: {tables_dir}/rq2_correlations.csv")
    print(f"✓ Saved: {tables_dir}/rq2_group_comparison.csv")
    
    print("\n" + "="*60)
    print("✅ CORRELATION ANALYSIS COMPLETE!")
    print("="*60)
    

if __name__ == "__main__":
    main()
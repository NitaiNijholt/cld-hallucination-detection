"""
RQ2 Statistical Validation

Performs rigorous statistical validation of RQ2 findings:
- Permutation tests for correlation significance
- Bootstrap confidence intervals for AUC scores
- Multiple comparison correction (Holm-Bonferroni)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class RQ2StatisticalValidation:
    """Statistical validation for RQ2 findings."""
    
    def __init__(self, df: pd.DataFrame, n_permutations: int = 10000, n_bootstrap: int = 10000):
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
        self.n_permutations = n_permutations
        self.n_bootstrap = n_bootstrap
        
        # Filter to edges with CI metrics
        available_metrics = [m for m in self.ci_metrics if m in df.columns]
        self.df_filtered = df.dropna(subset=available_metrics, how='all').copy()
        
        print(f"Initialized with {len(df)} edges, {len(self.df_filtered)} have CI metrics")
        print(f"Using {n_permutations:,} permutations and {n_bootstrap:,} bootstrap samples\n")
        
    def permutation_test_correlation(self, metric: str) -> Dict:
        """
        Permutation test for correlation significance.
        
        H0: There is no relationship between metric and hallucination
        HA: There is a relationship between metric and hallucination
        """
        
        valid_data = self.df_filtered[[metric, 'is_hallucination']].dropna()
        
        if len(valid_data) < 10:
            return None
        
        X = valid_data[metric].values
        y = valid_data['is_hallucination'].values.astype(int)
        
        # Compute observed correlation
        observed_r, _ = stats.pearsonr(X, y)
        
        # Permutation test
        permuted_correlations = []
        
        print(f"Running permutation test for {metric}...")
        for i in range(self.n_permutations):
            y_permuted = np.random.permutation(y)
            r_perm, _ = stats.pearsonr(X, y_permuted)
            permuted_correlations.append(r_perm)
        
        permuted_correlations = np.array(permuted_correlations)
        
        # Compute p-value (two-tailed)
        p_value = np.mean(np.abs(permuted_correlations) >= np.abs(observed_r))
        
        return {
            'metric': metric,
            'observed_r': observed_r,
            'permutation_p_value': p_value,
            'null_distribution': permuted_correlations,
            'significant_permutation': p_value < 0.05
        }
    
    def bootstrap_auc_ci(self, metric: str, confidence: float = 0.95) -> Dict:
        """
        Bootstrap confidence interval for AUC score.
        
        Estimates the sampling variability of the AUC score.
        """
        
        valid_data = self.df_filtered[[metric, 'is_hallucination']].dropna()
        
        if len(valid_data) < 10:
            return None
        
        y_true = valid_data['is_hallucination'].values.astype(int)
        y_score = valid_data[metric].values
        
        # Invert scores for min_prob, cosine_similarity, aggregate_score (lower = hallucination)
        if metric in ['min_prob', 'cosine_similarity', 'aggregate_score', 'judge_cosine_similarity', 'judge_min_prob']:
            y_score = -y_score
        
        # Compute observed AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        observed_auc = auc(fpr, tpr)
        
        # Bootstrap resampling
        bootstrap_aucs = []
        
        print(f"Running bootstrap for {metric}...")
        n_samples = len(y_true)
        
        for i in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_score_boot = y_score[indices]
            
            # Check if we have both classes
            if len(np.unique(y_true_boot)) < 2:
                continue
            
            try:
                fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_score_boot)
                auc_boot = auc(fpr_boot, tpr_boot)
                bootstrap_aucs.append(auc_boot)
            except:
                continue
        
        bootstrap_aucs = np.array(bootstrap_aucs)
        
        # Compute confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_aucs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2))
        
        return {
            'metric': metric,
            'observed_auc': observed_auc,
            'bootstrap_mean_auc': np.mean(bootstrap_aucs),
            'bootstrap_std_auc': np.std(bootstrap_aucs),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence': confidence,
            'bootstrap_distribution': bootstrap_aucs
        }
    
    def run_all_permutation_tests(self) -> pd.DataFrame:
        """Run permutation tests for all metrics."""
        
        results = []
        
        for metric in self.ci_metrics:
            if metric not in self.df_filtered.columns:
                continue
            
            result = self.permutation_test_correlation(metric)
            if result:
                # Store without distribution (for table)
                results.append({
                    'metric': result['metric'],
                    'observed_r': result['observed_r'],
                    'permutation_p_value': result['permutation_p_value'],
                    'significant': result['significant_permutation']
                })
                
                # Store full result for visualization
                if not hasattr(self, 'permutation_results_full'):
                    self.permutation_results_full = {}
                self.permutation_results_full[metric] = result
        
        self.permutation_results = pd.DataFrame(results)
        return self.permutation_results
    
    def run_all_bootstrap_ci(self) -> pd.DataFrame:
        """Run bootstrap CI for all metrics."""
        
        results = []
        
        for metric in self.ci_metrics:
            if metric not in self.df_filtered.columns:
                continue
            
            result = self.bootstrap_auc_ci(metric)
            if result:
                # Store without distribution (for table)
                results.append({
                    'metric': result['metric'],
                    'observed_auc': result['observed_auc'],
                    'bootstrap_mean_auc': result['bootstrap_mean_auc'],
                    'bootstrap_std_auc': result['bootstrap_std_auc'],
                    'ci_lower': result['ci_lower'],
                    'ci_upper': result['ci_upper'],
                    'confidence': result['confidence']
                })
                
                # Store full result for visualization
                if not hasattr(self, 'bootstrap_results_full'):
                    self.bootstrap_results_full = {}
                self.bootstrap_results_full[metric] = result
        
        self.bootstrap_results = pd.DataFrame(results)
        return self.bootstrap_results
    
    def apply_multiple_comparison_correction(self, p_values: np.ndarray, method: str = 'holm') -> np.ndarray:
        """
        Apply multiple comparison correction.
        
        Args:
            p_values: Array of p-values
            method: 'holm' (Holm-Bonferroni), 'bonferroni', or 'fdr_bh' (Benjamini-Hochberg)
        
        Returns:
            Array of corrected p-values
        """
        
        n = len(p_values)
        
        if method == 'bonferroni':
            return np.minimum(p_values * n, 1.0)
        
        elif method == 'holm':
            # Holm-Bonferroni
            sorted_indices = np.argsort(p_values)
            sorted_pvals = p_values[sorted_indices]
            
            corrected = np.zeros_like(p_values)
            for i, (idx, pval) in enumerate(zip(sorted_indices, sorted_pvals)):
                corrected[idx] = min(pval * (n - i), 1.0)
            
            # Make monotonic
            corrected_sorted = corrected[sorted_indices]
            for i in range(1, n):
                if corrected_sorted[i] < corrected_sorted[i-1]:
                    corrected_sorted[i] = corrected_sorted[i-1]
            
            corrected[sorted_indices] = corrected_sorted
            return corrected
        
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR
            sorted_indices = np.argsort(p_values)
            sorted_pvals = p_values[sorted_indices]
            
            corrected = np.zeros_like(p_values)
            for i in range(n-1, -1, -1):
                idx = sorted_indices[i]
                corrected[idx] = min(sorted_pvals[i] * n / (i + 1), 1.0)
            
            # Make monotonic
            corrected_sorted = corrected[sorted_indices]
            for i in range(n-2, -1, -1):
                if corrected_sorted[i] > corrected_sorted[i+1]:
                    corrected_sorted[i] = corrected_sorted[i+1]
            
            corrected[sorted_indices] = corrected_sorted
            return corrected
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def correct_permutation_pvalues(self, method: str = 'holm') -> pd.DataFrame:
        """Apply multiple comparison correction to permutation test p-values."""
        
        if not hasattr(self, 'permutation_results'):
            raise ValueError("Run permutation tests first")
        
        p_values = self.permutation_results['permutation_p_value'].values
        corrected_p_values = self.apply_multiple_comparison_correction(p_values, method=method)
        
        self.permutation_results['corrected_p_value'] = corrected_p_values
        self.permutation_results['significant_corrected'] = corrected_p_values < 0.05
        self.permutation_results['correction_method'] = method
        
        return self.permutation_results
    
    def visualize_permutation_distributions(self, output_dir: str = "figures"):
        """Visualize permutation test null distributions."""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if not hasattr(self, 'permutation_results_full'):
            print("No permutation results to visualize")
            return
        
        n_metrics = len(self.permutation_results_full)
        fig, axes = plt.subplots(1, min(n_metrics, 3), figsize=(5*min(n_metrics, 3), 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (metric, result) in enumerate(self.permutation_results_full.items()):
            if idx >= 3:
                break
            
            null_dist = result['null_distribution']
            observed = result['observed_r']
            p_value = result['permutation_p_value']
            
            # Plot null distribution
            axes[idx].hist(null_dist, bins=50, alpha=0.7, color='gray', 
                          edgecolor='black', density=True, label='Null distribution')
            
            # Plot observed value
            axes[idx].axvline(observed, color='red', linewidth=2.5, linestyle='--',
                            label=f'Observed r={observed:.3f}')
            
            # Add critical values (5% tails)
            critical_lower = np.percentile(null_dist, 2.5)
            critical_upper = np.percentile(null_dist, 97.5)
            axes[idx].axvline(critical_lower, color='blue', linewidth=1.5, linestyle=':',
                            alpha=0.7, label='5% critical values')
            axes[idx].axvline(critical_upper, color='blue', linewidth=1.5, linestyle=':',
                            alpha=0.7)
            
            axes[idx].set_xlabel('Correlation Coefficient (r)')
            axes[idx].set_ylabel('Density')
            axes[idx].set_title(f'{metric.replace("_", " ").title()}\\np = {p_value:.4f}')
            axes[idx].legend(fontsize=8)
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rq2_permutation_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_dir}/rq2_permutation_distributions.png")
    
    def visualize_bootstrap_distributions(self, output_dir: str = "figures"):
        """Visualize bootstrap AUC distributions with confidence intervals."""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if not hasattr(self, 'bootstrap_results_full'):
            print("No bootstrap results to visualize")
            return
        
        n_metrics = len(self.bootstrap_results_full)
        fig, axes = plt.subplots(1, min(n_metrics, 3), figsize=(5*min(n_metrics, 3), 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (metric, result) in enumerate(self.bootstrap_results_full.items()):
            if idx >= 3:
                break
            
            boot_dist = result['bootstrap_distribution']
            observed = result['observed_auc']
            ci_lower = result['ci_lower']
            ci_upper = result['ci_upper']
            
            # Plot bootstrap distribution
            axes[idx].hist(boot_dist, bins=50, alpha=0.7, color='skyblue', 
                          edgecolor='black', density=True, label='Bootstrap distribution')
            
            # Plot observed value
            axes[idx].axvline(observed, color='red', linewidth=2.5, linestyle='--',
                            label=f'Observed AUC={observed:.3f}')
            
            # Plot confidence interval
            axes[idx].axvline(ci_lower, color='green', linewidth=2, linestyle=':',
                            label=f'95% CI=[{ci_lower:.3f}, {ci_upper:.3f}]')
            axes[idx].axvline(ci_upper, color='green', linewidth=2, linestyle=':')
            
            # Shade CI region
            axes[idx].axvspan(ci_lower, ci_upper, alpha=0.2, color='green')
            
            axes[idx].set_xlabel('AUC Score')
            axes[idx].set_ylabel('Density')
            axes[idx].set_title(f'{metric.replace("_", " ").title()}')
            axes[idx].legend(fontsize=7, loc='upper left')
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rq2_bootstrap_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_dir}/rq2_bootstrap_distributions.png")


def main():
    """Test/demo statistical validation."""
    import sys
    
    # Load data
    data_file = Path(__file__).parent.parent / 'tables' / 'rq2_standalone_data.csv'
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} edges\n")
    
    # Run validation
    validator = RQ2StatisticalValidation(df, n_permutations=10000, n_bootstrap=10000)
    
    print("="*60)
    print("PERMUTATION TESTS (10,000 iterations)")
    print("="*60 + "\n")
    
    # Run permutation tests
    perm_results = validator.run_all_permutation_tests()
    print("\nPermutation Test Results (uncorrected):")
    print(perm_results[['metric', 'observed_r', 'permutation_p_value', 'significant']].to_string(index=False))
    
    # Apply multiple comparison correction
    print("\n" + "="*60)
    print("MULTIPLE COMPARISON CORRECTION (Holm-Bonferroni)")
    print("="*60)
    
    perm_results_corrected = validator.correct_permutation_pvalues(method='holm')
    print("\nCorrected p-values:")
    print(perm_results_corrected[['metric', 'observed_r', 'permutation_p_value', 
                                   'corrected_p_value', 'significant_corrected']].to_string(index=False))
    
    # Run bootstrap CI
    print("\n" + "="*60)
    print("BOOTSTRAP CONFIDENCE INTERVALS (10,000 samples)")
    print("="*60 + "\n")
    
    boot_results = validator.run_all_bootstrap_ci()
    print("\nBootstrap 95% CI for AUC:")
    print(boot_results[['metric', 'observed_auc', 'bootstrap_mean_auc', 
                        'ci_lower', 'ci_upper']].to_string(index=False))
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    output_dir = Path(__file__).parent.parent / 'figures'
    validator.visualize_permutation_distributions(str(output_dir))
    validator.visualize_bootstrap_distributions(str(output_dir))
    
    # Save results
    tables_dir = Path(__file__).parent.parent / 'tables'
    
    perm_results_corrected.to_csv(tables_dir / 'rq2_permutation_tests.csv', index=False)
    boot_results.to_csv(tables_dir / 'rq2_bootstrap_ci.csv', index=False)
    
    print(f"\n✓ Saved: {tables_dir}/rq2_permutation_tests.csv")
    print(f"✓ Saved: {tables_dir}/rq2_bootstrap_ci.csv")
    
    print("\n" + "="*60)
    print("✅ STATISTICAL VALIDATION COMPLETE!")
    print("="*60)
    
    # Summary
    print("\n📊 VALIDATION SUMMARY:")
    print(f"  • {len(perm_results)} metrics tested")
    print(f"  • {perm_results['significant'].sum()} significant before correction")
    print(f"  • {perm_results_corrected['significant_corrected'].sum()} significant after Holm-Bonferroni correction")
    print(f"  • All AUC 95% CIs computed successfully")
    

if __name__ == "__main__":
    main()
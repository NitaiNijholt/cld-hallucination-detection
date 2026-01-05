"""
RQ1 Statistical Validation

Performs statistical tests and computes confidence intervals:
- Bootstrap confidence intervals for classification metrics
- Permutation tests for statistical significance
- Validate that judge performance is significantly better than random
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RQ1StatisticalValidation:
    """
    Statistical validation for RQ1 judge performance.
    
    Ensures that results are statistically significant and not due to chance.
    """
    
    def __init__(self, df: pd.DataFrame, n_bootstrap: int = 10000, n_permutations: int = 10000):
        """
        Initialize statistical validator.
        
        Args:
            df: DataFrame with prepared RQ1 data
            n_bootstrap: Number of bootstrap samples
            n_permutations: Number of permutation samples
        """
        self.df = df
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        
        # Validate required columns
        required_cols = ['is_hallucination', 'judge_predicts_hallucination', 'classification']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info(f"Statistical validator initialized with {len(df)} edges")
        logger.info(f"Bootstrap samples: {n_bootstrap}, Permutation samples: {n_permutations}")
    
    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute all classification metrics for a dataset.
        
        Args:
            df: DataFrame with classifications
            
        Returns:
            Dictionary of metrics
        """
        counts = df['classification'].value_counts().to_dict()
        tp = counts.get('TP', 0)
        fp = counts.get('FP', 0)
        fn = counts.get('FN', 0)
        tn = counts.get('TN', 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        balanced_accuracy = (recall + specificity) / 2
        
        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy
        }
    
    def bootstrap_confidence_intervals(self, group_by: str = None) -> pd.DataFrame:
        """
        Compute bootstrap confidence intervals for metrics.
        
        Args:
            group_by: Optional column to group by
            
        Returns:
            DataFrame with CIs for each metric
        """
        results = []
        
        if group_by is None:
            # Overall CI
            results.append(self._bootstrap_single_group(self.df, 'overall'))
        else:
            # Grouped CI
            for group_name, group_df in self.df.groupby(group_by):
                results.append(self._bootstrap_single_group(group_df, group_name))
        
        return pd.DataFrame(results)
    
    def _bootstrap_single_group(self, df: pd.DataFrame, group_name: str) -> Dict:
        """
        Compute bootstrap CI for a single group.
        
        Args:
            df: DataFrame for this group
            group_name: Name of the group
            
        Returns:
            Dictionary with observed metrics and CIs
        """
        # Observed metrics
        observed = self.compute_metrics(df)
        
        # Bootstrap samples
        bootstrap_metrics = {metric: [] for metric in observed.keys()}
        
        np.random.seed(42)  # For reproducibility
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            sample_df = df.sample(n=len(df), replace=True)
            
            # Recompute classifications
            sample_df = sample_df.copy()
            sample_df['classification'] = [
                'TP' if (h and p) else 'FN' if (h and not p) else 'FP' if (not h and p) else 'TN'
                for h, p in zip(sample_df['is_hallucination'], sample_df['judge_predicts_hallucination'])
            ]
            
            # Compute metrics
            metrics = self.compute_metrics(sample_df)
            for metric, value in metrics.items():
                bootstrap_metrics[metric].append(value)
        
        # Compute percentile CIs (95%)
        result = {'group': group_name}
        
        for metric, values in bootstrap_metrics.items():
            result[f'{metric}_observed'] = observed[metric]
            result[f'{metric}_ci_lower'] = np.percentile(values, 2.5)
            result[f'{metric}_ci_upper'] = np.percentile(values, 97.5)
        
        logger.info(f"Computed bootstrap CI for {group_name}")
        
        return result
    
    def permutation_test(self, group_by: str = None) -> pd.DataFrame:
        """
        Perform permutation test to check if performance is better than random.
        
        H0: Judge predictions are independent of ground truth (random performance)
        H1: Judge predictions are associated with ground truth (better than random)
        
        Args:
            group_by: Optional column to group by
            
        Returns:
            DataFrame with permutation test results
        """
        results = []
        
        if group_by is None:
            results.append(self._permutation_single_group(self.df, 'overall'))
        else:
            for group_name, group_df in self.df.groupby(group_by):
                results.append(self._permutation_single_group(group_df, group_name))
        
        return pd.DataFrame(results)
    
    def _permutation_single_group(self, df: pd.DataFrame, group_name: str) -> Dict:
        """
        Perform permutation test for a single group.
        
        Args:
            df: DataFrame for this group
            group_name: Name of the group
            
        Returns:
            Dictionary with test results
        """
        # Observed performance
        observed = self.compute_metrics(df)
        observed_accuracy = observed['accuracy']
        observed_f1 = observed['f1']
        
        # Permutation distribution
        perm_accuracies = []
        perm_f1s = []
        
        np.random.seed(42)
        
        for _ in range(self.n_permutations):
            # Shuffle predictions (break association with ground truth)
            perm_df = df.copy()
            perm_df['judge_predicts_hallucination'] = np.random.permutation(
                perm_df['judge_predicts_hallucination'].values
            )
            
            # Recompute classifications
            perm_df['classification'] = [
                'TP' if (h and p) else 'FN' if (h and not p) else 'FP' if (not h and p) else 'TN'
                for h, p in zip(perm_df['is_hallucination'], perm_df['judge_predicts_hallucination'])
            ]
            
            # Compute metrics
            metrics = self.compute_metrics(perm_df)
            perm_accuracies.append(metrics['accuracy'])
            perm_f1s.append(metrics['f1'])
        
        # Compute p-values (one-tailed: observed > permuted)
        p_value_accuracy = np.mean([perm >= observed_accuracy for perm in perm_accuracies])
        p_value_f1 = np.mean([perm >= observed_f1 for perm in perm_f1s])
        
        logger.info(f"Permutation test for {group_name}: accuracy p={p_value_accuracy:.4f}, F1 p={p_value_f1:.4f}")
        
        return {
            'group': group_name,
            'observed_accuracy': observed_accuracy,
            'observed_f1': observed_f1,
            'perm_accuracy_mean': np.mean(perm_accuracies),
            'perm_accuracy_std': np.std(perm_accuracies),
            'perm_f1_mean': np.mean(perm_f1s),
            'perm_f1_std': np.std(perm_f1s),
            'p_value_accuracy': p_value_accuracy,
            'p_value_f1': p_value_f1,
            'significant_accuracy': p_value_accuracy < 0.05,
            'significant_f1': p_value_f1 < 0.05
        }
    
    def visualize_bootstrap_distributions(self, output_path: str, group_by: str = None):
        """
        Visualize bootstrap distributions for metrics.
        
        Args:
            output_path: Path to save figure
            group_by: Optional grouping column
        """
        # Compute bootstrap samples for visualization
        if group_by is None:
            groups = [('overall', self.df)]
        else:
            groups = list(self.df.groupby(group_by))
        
        n_groups = len(groups)
        fig, axes = plt.subplots(n_groups, 3, figsize=(15, 5*n_groups))
        
        if n_groups == 1:
            axes = axes.reshape(1, -1)
        
        np.random.seed(42)
        
        for idx, (group_name, group_df) in enumerate(groups):
            # Bootstrap samples
            precisions, recalls, f1s = [], [], []
            
            for _ in range(self.n_bootstrap):
                sample_df = group_df.sample(n=len(group_df), replace=True)
                sample_df = sample_df.copy()
                sample_df['classification'] = [
                    'TP' if (h and p) else 'FN' if (h and not p) else 'FP' if (not h and p) else 'TN'
                    for h, p in zip(sample_df['is_hallucination'], sample_df['judge_predicts_hallucination'])
                ]
                
                metrics = self.compute_metrics(sample_df)
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1s.append(metrics['f1'])
            
            # Observed values
            observed = self.compute_metrics(group_df)
            
            # Plot distributions
            axes[idx, 0].hist(precisions, bins=50, alpha=0.7, edgecolor='black')
            axes[idx, 0].axvline(observed['precision'], color='red', linestyle='--', linewidth=2, label='Observed')
            axes[idx, 0].axvline(np.percentile(precisions, 2.5), color='green', linestyle=':', label='95% CI')
            axes[idx, 0].axvline(np.percentile(precisions, 97.5), color='green', linestyle=':')
            axes[idx, 0].set_xlabel('Precision')
            axes[idx, 0].set_ylabel('Frequency')
            axes[idx, 0].set_title(f'{group_name}: Precision Bootstrap Distribution')
            axes[idx, 0].legend()
            
            axes[idx, 1].hist(recalls, bins=50, alpha=0.7, edgecolor='black')
            axes[idx, 1].axvline(observed['recall'], color='red', linestyle='--', linewidth=2, label='Observed')
            axes[idx, 1].axvline(np.percentile(recalls, 2.5), color='green', linestyle=':', label='95% CI')
            axes[idx, 1].axvline(np.percentile(recalls, 97.5), color='green', linestyle=':')
            axes[idx, 1].set_xlabel('Recall')
            axes[idx, 1].set_ylabel('Frequency')
            axes[idx, 1].set_title(f'{group_name}: Recall Bootstrap Distribution')
            axes[idx, 1].legend()
            
            axes[idx, 2].hist(f1s, bins=50, alpha=0.7, edgecolor='black')
            axes[idx, 2].axvline(observed['f1'], color='red', linestyle='--', linewidth=2, label='Observed')
            axes[idx, 2].axvline(np.percentile(f1s, 2.5), color='green', linestyle=':', label='95% CI')
            axes[idx, 2].axvline(np.percentile(f1s, 97.5), color='green', linestyle=':')
            axes[idx, 2].set_xlabel('F1 Score')
            axes[idx, 2].set_ylabel('Frequency')
            axes[idx, 2].set_title(f'{group_name}: F1 Bootstrap Distribution')
            axes[idx, 2].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved bootstrap distributions to: {output_path}")
    
    def visualize_permutation_distributions(self, output_path: str, group_by: str = None):
        """
        Visualize permutation test null distributions.
        
        Args:
            output_path: Path to save figure
            group_by: Optional grouping column
        """
        # Perform permutation tests and save distributions
        if group_by is None:
            groups = [('overall', self.df)]
        else:
            groups = list(self.df.groupby(group_by))
        
        n_groups = len(groups)
        fig, axes = plt.subplots(n_groups, 2, figsize=(12, 5*n_groups))
        
        if n_groups == 1:
            axes = axes.reshape(1, -1)
        
        np.random.seed(42)
        
        for idx, (group_name, group_df) in enumerate(groups):
            observed = self.compute_metrics(group_df)
            
            # Generate permutation distribution
            perm_accuracies, perm_f1s = [], []
            
            for _ in range(self.n_permutations):
                perm_df = group_df.copy()
                perm_df['judge_predicts_hallucination'] = np.random.permutation(
                    perm_df['judge_predicts_hallucination'].values
                )
                perm_df['classification'] = [
                    'TP' if (h and p) else 'FN' if (h and not p) else 'FP' if (not h and p) else 'TN'
                    for h, p in zip(perm_df['is_hallucination'], perm_df['judge_predicts_hallucination'])
                ]
                
                metrics = self.compute_metrics(perm_df)
                perm_accuracies.append(metrics['accuracy'])
                perm_f1s.append(metrics['f1'])
            
            # Plot accuracy null distribution
            axes[idx, 0].hist(perm_accuracies, bins=50, alpha=0.7, edgecolor='black', label='Null distribution')
            axes[idx, 0].axvline(observed['accuracy'], color='red', linestyle='--', linewidth=2, label='Observed')
            axes[idx, 0].set_xlabel('Accuracy')
            axes[idx, 0].set_ylabel('Frequency')
            axes[idx, 0].set_title(f'{group_name}: Accuracy Permutation Test')
            axes[idx, 0].legend()
            
            # Plot F1 null distribution
            axes[idx, 1].hist(perm_f1s, bins=50, alpha=0.7, edgecolor='black', label='Null distribution')
            axes[idx, 1].axvline(observed['f1'], color='red', linestyle='--', linewidth=2, label='Observed')
            axes[idx, 1].set_xlabel('F1 Score')
            axes[idx, 1].set_ylabel('Frequency')
            axes[idx, 1].set_title(f'{group_name}: F1 Permutation Test')
            axes[idx, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved permutation distributions to: {output_path}")


def main():
    """Example usage of RQ1 statistical validation."""
    
    # Create sample data
    np.random.seed(42)
    n = 100
    
    sample_data = pd.DataFrame({
        'is_hallucination': np.random.choice([True, False], n, p=[0.3, 0.7]),
        'judge_predicts_hallucination': np.random.choice([True, False], n, p=[0.25, 0.75]),
        'corruption_rate': np.random.choice([0.5, 1.0], n)
    })
    
    # Add classifications
    sample_data['classification'] = [
        'TP' if (h and p) else 'FN' if (h and not p) else 'FP' if (not h and p) else 'TN'
        for h, p in zip(sample_data['is_hallucination'], sample_data['judge_predicts_hallucination'])
    ]
    
    # Run analysis
    validator = RQ1StatisticalValidation(sample_data, n_bootstrap=1000, n_permutations=1000)
    
    # Bootstrap CIs
    print("\n" + "="*80)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*80)
    bootstrap_ci = validator.bootstrap_confidence_intervals()
    print(bootstrap_ci.to_string(index=False))
    
    # Permutation tests
    print("\n" + "="*80)
    print("PERMUTATION TESTS")
    print("="*80)
    perm_tests = validator.permutation_test()
    print(perm_tests.to_string(index=False))
    
    # Visualizations
    output_dir = Path("rq1_analyses/test/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validator.visualize_bootstrap_distributions(str(output_dir / "bootstrap_distributions.png"))
    validator.visualize_permutation_distributions(str(output_dir / "permutation_distributions.png"))


if __name__ == "__main__":
    main()

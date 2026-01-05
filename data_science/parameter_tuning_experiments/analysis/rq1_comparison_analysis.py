"""
RQ1 Comparison Analysis

Compares serial vs parallel judging strategies:
- Serial: Single judge evaluation
- Parallel: Multiple judges with majority voting

Analyzes which strategy performs better for hallucination detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RQ1ComparisonAnalysis:
    """
    Compare serial vs parallel judging strategies.
    
    Answers: Does parallel ensemble judging improve hallucination detection?
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize comparison analyzer.
        
        Args:
            df: DataFrame with prepared RQ1 data including judging strategy info
        """
        self.df = df
        
        # Validate required columns
        required_cols = ['is_hallucination', 'judge_predicts_hallucination', 'classification']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info(f"Comparison analyzer initialized with {len(df)} edges")
    
    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare performance metrics across judging strategies.
        
        Returns:
            DataFrame with metrics for each strategy
        """
        if 'judging_strategy' not in self.df.columns:
            logger.warning("No judging_strategy column found")
            # Try to infer from num_judges
            if 'num_judges' in self.df.columns:
                self.df['judging_strategy'] = self.df['num_judges'].apply(
                    lambda x: 'serial' if x == 1 else f'parallel_{x}'
                )
            else:
                logger.error("Cannot determine judging strategy")
                return pd.DataFrame()
        
        results = []
        
        for strategy, group_df in self.df.groupby('judging_strategy'):
            # Compute confusion matrix
            counts = group_df['classification'].value_counts().to_dict()
            tp = counts.get('TP', 0)
            fp = counts.get('FP', 0)
            fn = counts.get('FN', 0)
            tn = counts.get('TN', 0)
            
            # Compute metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            balanced_accuracy = (recall + specificity) / 2
            
            results.append({
                'strategy': strategy,
                'n_edges': len(group_df),
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn
            })
        
        return pd.DataFrame(results)
    
    def statistical_comparison(self) -> pd.DataFrame:
        """
        Perform statistical tests comparing strategies.
        
        Uses McNemar's test for paired binary classifications.
        
        Returns:
            DataFrame with test results
        """
        if 'judging_strategy' not in self.df.columns:
            logger.warning("Cannot perform statistical comparison without judging_strategy")
            return pd.DataFrame()
        
        strategies = self.df['judging_strategy'].unique()
        
        if len(strategies) < 2:
            logger.warning("Need at least 2 strategies for comparison")
            return pd.DataFrame()
        
        results = []
        
        # Pairwise comparisons
        for i, strat1 in enumerate(strategies):
            for strat2 in strategies[i+1:]:
                df1 = self.df[self.df['judging_strategy'] == strat1]
                df2 = self.df[self.df['judging_strategy'] == strat2]
                
                # For McNemar's test, we need same edges judged by both strategies
                # Here we compare overall performance using proportions test
                
                # Compute success rates (correct classifications)
                correct1 = ((df1['classification'] == 'TP') | (df1['classification'] == 'TN')).sum()
                total1 = len(df1)
                correct2 = ((df2['classification'] == 'TP') | (df2['classification'] == 'TN')).sum()
                total2 = len(df2)
                
                # Two-proportion z-test
                p1 = correct1 / total1 if total1 > 0 else 0
                p2 = correct2 / total2 if total2 > 0 else 0
                
                # Pooled proportion
                p_pooled = (correct1 + correct2) / (total1 + total2) if (total1 + total2) > 0 else 0
                
                # Standard error
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/total1 + 1/total2)) if p_pooled > 0 else 1e-10
                
                # Z-statistic
                z = (p1 - p2) / se if se > 0 else 0
                
                # P-value (two-tailed)
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                
                results.append({
                    'strategy_1': strat1,
                    'strategy_2': strat2,
                    'accuracy_1': p1,
                    'accuracy_2': p2,
                    'diff': p1 - p2,
                    'z_statistic': z,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        
        return pd.DataFrame(results)
    
    def visualize_strategy_comparison(self, output_path: str):
        """
        Visualize comparison between judging strategies.
        
        Args:
            output_path: Path to save figure
        """
        comparison = self.compare_strategies()
        
        if comparison.empty:
            logger.warning("No comparison data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        strategies = comparison['strategy'].tolist()
        x = np.arange(len(strategies))
        width = 0.35
        
        # Plot 1: Precision vs Recall
        axes[0, 0].bar(x - width/2, comparison['precision'], width, label='Precision', alpha=0.8)
        axes[0, 0].bar(x + width/2, comparison['recall'], width, label='Recall', alpha=0.8)
        axes[0, 0].set_ylabel('Score', fontsize=11)
        axes[0, 0].set_title('Precision and Recall by Strategy', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(strategies, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].set_ylim([0, 1.05])
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Plot 2: F1 and Accuracy
        axes[0, 1].bar(x - width/2, comparison['f1'], width, label='F1 Score', alpha=0.8, color='purple')
        axes[0, 1].bar(x + width/2, comparison['accuracy'], width, label='Accuracy', alpha=0.8, color='green')
        axes[0, 1].set_ylabel('Score', fontsize=11)
        axes[0, 1].set_title('F1 and Accuracy by Strategy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(strategies, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].set_ylim([0, 1.05])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Confusion matrix comparison (stacked bars)
        tp_vals = comparison['TP'].values
        fp_vals = comparison['FP'].values
        fn_vals = comparison['FN'].values
        tn_vals = comparison['TN'].values
        
        axes[1, 0].bar(x, tp_vals, label='TP', color='#2ecc71', alpha=0.8)
        axes[1, 0].bar(x, fp_vals, bottom=tp_vals, label='FP', color='#e74c3c', alpha=0.8)
        axes[1, 0].bar(x, fn_vals, bottom=tp_vals+fp_vals, label='FN', color='#f39c12', alpha=0.8)
        axes[1, 0].bar(x, tn_vals, bottom=tp_vals+fp_vals+fn_vals, label='TN', color='#3498db', alpha=0.8)
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_title('Classification Breakdown by Strategy', fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(strategies, rotation=45, ha='right')
        axes[1, 0].legend()
        
        # Plot 4: All metrics radar/bar chart
        metrics = ['precision', 'recall', 'specificity', 'f1', 'accuracy', 'balanced_accuracy']
        metric_labels = [m.replace('_', ' ').title() for m in metrics]
        
        bar_width = 0.8 / len(strategies)
        for i, (_, row) in enumerate(comparison.iterrows()):
            values = [row[m] for m in metrics]
            offset = (i - len(strategies)/2) * bar_width + bar_width/2
            axes[1, 1].bar(np.arange(len(metrics)) + offset, values, bar_width, 
                          label=row['strategy'], alpha=0.8)
        
        axes[1, 1].set_ylabel('Score', fontsize=11)
        axes[1, 1].set_title('All Metrics Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(np.arange(len(metrics)))
        axes[1, 1].set_xticklabels(metric_labels, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1.05])
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved strategy comparison to: {output_path}")
    
    def visualize_confusion_matrices_comparison(self, output_path: str):
        """
        Side-by-side confusion matrices for each strategy.
        
        Args:
            output_path: Path to save figure
        """
        if 'judging_strategy' not in self.df.columns:
            logger.warning("No judging_strategy column available")
            return
        
        strategies = sorted(self.df['judging_strategy'].unique())
        n_strategies = len(strategies)
        
        fig, axes = plt.subplots(1, n_strategies, figsize=(6*n_strategies, 5))
        
        if n_strategies == 1:
            axes = [axes]
        
        for idx, strategy in enumerate(strategies):
            strategy_df = self.df[self.df['judging_strategy'] == strategy]
            counts = strategy_df['classification'].value_counts().to_dict()
            
            tp = counts.get('TP', 0)
            fp = counts.get('FP', 0)
            fn = counts.get('FN', 0)
            tn = counts.get('TN', 0)
            
            matrix = np.array([
                [tp, fp],
                [fn, tn]
            ])
            
            sns.heatmap(
                matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Pred Halluc', 'Pred Genuine'],
                yticklabels=['True Halluc', 'True Genuine'],
                ax=axes[idx],
                cbar_kws={'label': 'Count'}
            )
            
            # Add metrics as subtitle
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            axes[idx].set_title(
                f'{strategy}\nP={precision:.2f}, R={recall:.2f}, F1={f1:.2f}',
                fontsize=12,
                fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrices comparison to: {output_path}")


def main():
    """Example usage of RQ1 comparison analysis."""
    
    # Create sample data with different judging strategies
    np.random.seed(42)
    
    # Serial strategy (single judge)
    n_serial = 50
    serial_data = pd.DataFrame({
        'is_hallucination': np.random.choice([True, False], n_serial),
        'judge_predicts_hallucination': np.random.choice([True, False], n_serial),
        'judging_strategy': 'serial',
        'num_judges': 1
    })
    
    # Parallel strategy (3 judges)
    n_parallel = 50
    parallel_data = pd.DataFrame({
        'is_hallucination': np.random.choice([True, False], n_parallel),
        'judge_predicts_hallucination': np.random.choice([True, False], n_parallel, p=[0.3, 0.7]),  # Better performance
        'judging_strategy': 'parallel_3',
        'num_judges': 3
    })
    
    # Combine
    sample_data = pd.concat([serial_data, parallel_data], ignore_index=True)
    
    # Add classifications
    sample_data['classification'] = [
        'TP' if (h and p) else 'FN' if (h and not p) else 'FP' if (not h and p) else 'TN'
        for h, p in zip(sample_data['is_hallucination'], sample_data['judge_predicts_hallucination'])
    ]
    
    # Run analysis
    analyzer = RQ1ComparisonAnalysis(sample_data)
    
    # Compare strategies
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    comparison = analyzer.compare_strategies()
    print(comparison.to_string(index=False))
    
    # Statistical comparison
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)
    stats_comp = analyzer.statistical_comparison()
    print(stats_comp.to_string(index=False))
    
    # Generate visualizations
    output_dir = Path("rq1_analyses/test/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer.visualize_strategy_comparison(str(output_dir / "strategy_comparison.png"))
    analyzer.visualize_confusion_matrices_comparison(str(output_dir / "confusion_matrices_by_strategy.png"))


if __name__ == "__main__":
    main()

"""
RQ1 Classification Analysis

Analyzes LLM-as-a-judge performance in detecting hallucinations:
- Compute confusion matrices
- Calculate precision, recall, F1, accuracy
- Compare performance across different experimental conditions
- Visualize classification performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RQ1ClassificationAnalysis:
    """
    Analyze judge classification performance for RQ1.
    
    Computes standard classification metrics to answer:
    - How well can LLM-as-a-judge detect hallucinations?
    - What is the precision/recall trade-off?
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize classification analyzer.
        
        Args:
            df: DataFrame with prepared RQ1 data (from RQ1DataLoader)
        """
        self.df = df
        
        # Validate required columns
        required_cols = ['is_hallucination', 'judge_predicts_hallucination', 'classification']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info(f"Classification analyzer initialized with {len(df)} edges")
    
    def compute_confusion_matrix(self, group_by: str = None) -> pd.DataFrame:
        """
        Compute confusion matrix (TP/FP/FN/TN counts).
        
        Args:
            group_by: Optional column to group by (e.g., 'judging_strategy', 'combo')
            
        Returns:
            DataFrame with confusion matrix counts
        """
        if group_by is None:
            # Overall confusion matrix
            counts = self.df['classification'].value_counts().to_dict()
            result = pd.DataFrame([{
                'group': 'overall',
                'TP': counts.get('TP', 0),
                'FP': counts.get('FP', 0),
                'FN': counts.get('FN', 0),
                'TN': counts.get('TN', 0)
            }])
        else:
            # Grouped confusion matrix
            results = []
            for group_name, group_df in self.df.groupby(group_by):
                counts = group_df['classification'].value_counts().to_dict()
                results.append({
                    'group': group_name,
                    'TP': counts.get('TP', 0),
                    'FP': counts.get('FP', 0),
                    'FN': counts.get('FN', 0),
                    'TN': counts.get('TN', 0)
                })
            result = pd.DataFrame(results)
        
        # Add total column
        result['total'] = result['TP'] + result['FP'] + result['FN'] + result['TN']
        
        return result
    
    def compute_classification_metrics(self, group_by: str = None) -> pd.DataFrame:
        """
        Compute classification metrics (precision, recall, F1, accuracy).
        
        Args:
            group_by: Optional column to group by
            
        Returns:
            DataFrame with classification metrics
        """
        conf_matrix = self.compute_confusion_matrix(group_by)
        
        metrics = []
        for _, row in conf_matrix.iterrows():
            tp, fp, fn, tn = row['TP'], row['FP'], row['FN'], row['TN']
            
            # Precision: TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Recall (Sensitivity): TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Specificity: TN / (TN + FP)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # F1 score: 2 * (precision * recall) / (precision + recall)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Accuracy: (TP + TN) / (TP + TN + FP + FN)
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            # Balanced accuracy: (recall + specificity) / 2
            balanced_accuracy = (recall + specificity) / 2
            
            metrics.append({
                'group': row['group'],
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
        
        return pd.DataFrame(metrics)
    
    def compute_metrics_by_corruption_rate(self) -> pd.DataFrame:
        """
        Compute metrics stratified by corruption rate.
        
        Returns:
            DataFrame with metrics for each corruption rate
        """
        if 'corruption_rate' not in self.df.columns:
            logger.warning("No corruption_rate column found, skipping stratified analysis")
            return pd.DataFrame()
        
        return self.compute_classification_metrics(group_by='corruption_rate')
    
    def visualize_confusion_matrix(self, output_path: str, group_by: str = None):
        """
        Visualize confusion matrix as heatmap.
        
        Args:
            output_path: Path to save figure
            group_by: Optional grouping column
        """
        conf_matrix = self.compute_confusion_matrix(group_by)
        
        if group_by is None:
            # Single confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create 2x2 matrix for heatmap
            row = conf_matrix.iloc[0]
            matrix = np.array([
                [row['TP'], row['FP']],
                [row['FN'], row['TN']]
            ])
            
            sns.heatmap(
                matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Predicted Hallucination', 'Predicted Genuine'],
                yticklabels=['Actual Hallucination', 'Actual Genuine'],
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            
            ax.set_title('Confusion Matrix: Judge Performance', fontsize=14, fontweight='bold')
            
        else:
            # Multiple confusion matrices
            n_groups = len(conf_matrix)
            fig, axes = plt.subplots(1, n_groups, figsize=(6*n_groups, 5))
            
            if n_groups == 1:
                axes = [axes]
            
            for idx, (_, row) in enumerate(conf_matrix.iterrows()):
                matrix = np.array([
                    [row['TP'], row['FP']],
                    [row['FN'], row['TN']]
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
                
                axes[idx].set_title(f'{row["group"]}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to: {output_path}")
    
    def visualize_metrics_comparison(self, output_path: str, group_by: str = None):
        """
        Visualize comparison of classification metrics.
        
        Args:
            output_path: Path to save figure
            group_by: Optional grouping column
        """
        metrics = self.compute_classification_metrics(group_by)
        
        # Create bar plot comparing metrics
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metric_cols = ['precision', 'recall', 'specificity', 'f1', 'accuracy', 'balanced_accuracy']
        x = np.arange(len(metrics))
        width = 0.12
        
        for i, metric in enumerate(metric_cols):
            offset = (i - len(metric_cols)/2) * width + width/2
            ax.bar(x + offset, metrics[metric], width, label=metric.replace('_', ' ').title())
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Judge Classification Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics['group'], rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metrics comparison to: {output_path}")
    
    def visualize_performance_by_corruption_rate(self, output_path: str):
        """
        Visualize how judge performance varies with corruption rate.
        
        Args:
            output_path: Path to save figure
        """
        if 'corruption_rate' not in self.df.columns:
            logger.warning("No corruption_rate column, skipping corruption rate visualization")
            return
        
        metrics = self.compute_metrics_by_corruption_rate()
        
        if metrics.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Sort by corruption rate for better visualization
        metrics = metrics.sort_values('group')
        
        # Plot 1: Precision and Recall
        axes[0].plot(metrics['group'], metrics['precision'], 'o-', label='Precision', linewidth=2, markersize=8)
        axes[0].plot(metrics['group'], metrics['recall'], 's-', label='Recall', linewidth=2, markersize=8)
        axes[0].set_xlabel('Corruption Rate', fontsize=11)
        axes[0].set_ylabel('Score', fontsize=11)
        axes[0].set_title('Precision and Recall vs Corruption Rate', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim([0, 1.05])
        
        # Plot 2: F1 Score
        axes[1].plot(metrics['group'], metrics['f1'], 'o-', color='purple', linewidth=2, markersize=8)
        axes[1].set_xlabel('Corruption Rate', fontsize=11)
        axes[1].set_ylabel('F1 Score', fontsize=11)
        axes[1].set_title('F1 Score vs Corruption Rate', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        
        # Plot 3: Accuracy
        axes[2].plot(metrics['group'], metrics['accuracy'], 'o-', color='green', linewidth=2, markersize=8)
        axes[2].plot(metrics['group'], metrics['balanced_accuracy'], 's-', color='darkgreen', linewidth=2, markersize=8)
        axes[2].set_xlabel('Corruption Rate', fontsize=11)
        axes[2].set_ylabel('Accuracy', fontsize=11)
        axes[2].set_title('Accuracy vs Corruption Rate', fontsize=12, fontweight='bold')
        axes[2].legend(['Accuracy', 'Balanced Accuracy'])
        axes[2].grid(alpha=0.3)
        axes[2].set_ylim([0, 1.05])
        
        # Plot 4: TP/FP/FN/TN stacked bar
        x = np.arange(len(metrics))
        axes[3].bar(x, metrics['TP'], label='TP', color='#2ecc71')
        axes[3].bar(x, metrics['FP'], bottom=metrics['TP'], label='FP', color='#e74c3c')
        axes[3].bar(x, metrics['FN'], bottom=metrics['TP'] + metrics['FP'], label='FN', color='#f39c12')
        axes[3].bar(x, metrics['TN'], bottom=metrics['TP'] + metrics['FP'] + metrics['FN'], label='TN', color='#3498db')
        axes[3].set_xlabel('Corruption Rate', fontsize=11)
        axes[3].set_ylabel('Count', fontsize=11)
        axes[3].set_title('Classification Breakdown', fontsize=12, fontweight='bold')
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(metrics['group'])
        axes[3].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved corruption rate analysis to: {output_path}")


def main():
    """Example usage of RQ1 classification analysis."""
    
    # This would typically be called after loading data with RQ1DataLoader
    # For demonstration, we create a sample dataset
    
    np.random.seed(42)
    n = 100
    
    sample_data = pd.DataFrame({
        'is_hallucination': np.random.choice([True, False], n),
        'judge_predicts_hallucination': np.random.choice([True, False], n),
        'corruption_rate': np.random.choice([0.0, 0.5, 1.0], n)
    })
    
    # Add classification
    sample_data['classification'] = [
        'TP' if (h and p) else 'FN' if (h and not p) else 'FP' if (not h and p) else 'TN'
        for h, p in zip(sample_data['is_hallucination'], sample_data['judge_predicts_hallucination'])
    ]
    
    # Run analysis
    analyzer = RQ1ClassificationAnalysis(sample_data)
    
    # Compute metrics
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    metrics = analyzer.compute_classification_metrics()
    print(metrics.to_string(index=False))
    
    print("\n" + "="*80)
    print("METRICS BY CORRUPTION RATE")
    print("="*80)
    metrics_by_rate = analyzer.compute_metrics_by_corruption_rate()
    print(metrics_by_rate.to_string(index=False))
    
    # Generate visualizations
    output_dir = Path("rq1_analyses/test/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer.visualize_confusion_matrix(str(output_dir / "confusion_matrix.png"))
    analyzer.visualize_metrics_comparison(str(output_dir / "metrics_comparison.png"))
    analyzer.visualize_performance_by_corruption_rate(str(output_dir / "performance_by_corruption_rate.png"))


if __name__ == "__main__":
    main()

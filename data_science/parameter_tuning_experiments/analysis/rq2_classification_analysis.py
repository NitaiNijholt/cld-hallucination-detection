"""
RQ2 Classification Analysis

Evaluates CI metrics as binary classifiers for hallucination detection.
Computes ROC curves, optimal thresholds, and confusion matrices.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class RQ2ClassificationAnalysis:
    """Classification analysis for RQ2 - treating CI metrics as predictors."""
    
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
        
    def compute_roc_curves(self) -> Dict:
        """Compute ROC curves for each CI metric."""
        
        results = {}
        
        for metric in self.ci_metrics:
            if metric not in self.df_filtered.columns:
                continue
                
            valid_data = self.df_filtered[[metric, 'is_hallucination']].dropna()
            
            if len(valid_data) < 10:
                continue
            
            y_true = valid_data['is_hallucination'].astype(int)
            y_score = valid_data[metric]
            
            # For min_prob, cosine_similarity, aggregate_score: lower = more likely hallucination
            # For perplexity and max_window_entropy: higher = more likely hallucination
            if metric in ['min_prob', 'cosine_similarity', 'aggregate_score', 'judge_cosine_similarity', 'judge_min_prob']:
                y_score = -y_score  # Invert so higher score = higher risk
            
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            results[metric] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc
            }
        
        self.roc_results = results
        return results
    
    def find_optimal_thresholds(self, criterion: str = 'f1') -> pd.DataFrame:
        """Find optimal thresholds for each metric based on criterion."""
        
        results = []
        
        for metric in self.ci_metrics:
            if metric not in self.df_filtered.columns:
                continue
                
            valid_data = self.df_filtered[[metric, 'is_hallucination']].dropna()
            
            if len(valid_data) < 10:
                continue
            
            y_true = valid_data['is_hallucination'].astype(int)
            y_score = valid_data[metric]
            
            # Try different threshold percentiles
            thresholds = np.percentile(y_score, np.arange(10, 91, 5))
            
            best_threshold = None
            best_score = -np.inf
            best_metrics = None
            
            for threshold in thresholds:
                # Classify based on threshold
                if metric in ['min_prob', 'cosine_similarity', 'aggregate_score', 'judge_cosine_similarity', 'judge_min_prob']:
                    y_pred = (y_score < threshold).astype(int)  # Lower = hallucination
                else:
                    y_pred = (y_score > threshold).astype(int)  # Higher = hallucination
                
                # Compute metrics
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                else:
                    continue
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                
                # Select best based on criterion
                if criterion == 'f1':
                    score = f1
                elif criterion == 'accuracy':
                    score = accuracy
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_metrics = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'accuracy': accuracy,
                        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
                    }
            
            if best_metrics:
                results.append({
                    'metric': metric,
                    'optimal_threshold': best_threshold,
                    'criterion': criterion,
                    **best_metrics
                })
        
        self.optimal_thresholds = pd.DataFrame(results)
        return self.optimal_thresholds
    
    def visualize_roc_curves(self, output_dir: str = "figures"):
        """Plot ROC curves for all metrics."""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for metric, results in self.roc_results.items():
            ax.plot(results['fpr'], results['tpr'], 
                   label=f"{metric.replace('_', ' ').title()} (AUC = {results['auc']:.3f})",
                   linewidth=2.5)
        
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', linewidth=1.5)
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves: CI Metrics as Hallucination Classifiers', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rq2_roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_dir}/rq2_roc_curves.png")
    
    def visualize_confusion_matrices(self, output_dir: str = "figures"):
        """Plot confusion matrices at optimal thresholds."""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        n_metrics = len(self.optimal_thresholds)
        if n_metrics == 0:
            print("No optimal thresholds to visualize")
            return
        
        fig, axes = plt.subplots(1, min(n_metrics, 3), figsize=(4*min(n_metrics, 3), 4))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (_, row) in enumerate(self.optimal_thresholds.iterrows()):
            if idx >= 3:  # Max 3 confusion matrices
                break
                
            metric = row['metric']
            threshold = row['optimal_threshold']
            
            valid_data = self.df_filtered[[metric, 'is_hallucination']].dropna()
            y_true = valid_data['is_hallucination'].astype(int)
            y_score = valid_data[metric]
            
            if metric in ['min_prob', 'cosine_similarity', 'aggregate_score', 'judge_cosine_similarity', 'judge_min_prob']:
                y_pred = (y_score < threshold).astype(int)
            else:
                y_pred = (y_score > threshold).astype(int)
            
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Non-Halluc', 'Halluc'],
                       yticklabels=['Non-Halluc', 'Halluc'],
                       cbar_kws={'label': 'Count'})
            
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
            axes[idx].set_title(f'{metric.replace("_", " ").title()}\n' +
                              f'F1={row["f1"]:.3f}, Threshold={threshold:.3f}')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rq2_confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_dir}/rq2_confusion_matrices.png")


def main():
    """Test/demo classification analysis."""
    import sys
    
    # Load data
    data_file = Path(__file__).parent.parent / 'tables' / 'rq2_standalone_data.csv'
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} edges\n")
    
    # Run analysis
    analyzer = RQ2ClassificationAnalysis(df)
    
    print("\n" + "="*60)
    print("ROC CURVE ANALYSIS")
    print("="*60)
    
    # Compute ROC curves
    roc_results = analyzer.compute_roc_curves()
    print("\nROC AUC Scores:")
    for metric, results in roc_results.items():
        print(f"  {metric:20s}: AUC = {results['auc']:.3f}")
    
    # Find optimal thresholds
    print("\n" + "="*60)
    print("OPTIMAL THRESHOLDS (F1 criterion)")
    print("="*60)
    optimal = analyzer.find_optimal_thresholds(criterion='f1')
    print("\n" + optimal[['metric', 'optimal_threshold', 'f1', 'precision', 'recall', 
                          'accuracy']].to_string(index=False))
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    output_dir = Path(__file__).parent.parent / 'figures'
    analyzer.visualize_roc_curves(str(output_dir))
    analyzer.visualize_confusion_matrices(str(output_dir))
    
    # Save results
    tables_dir = Path(__file__).parent.parent / 'tables'
    
    # ROC AUC summary
    roc_summary = pd.DataFrame([
        {'metric': metric, 'roc_auc': results['auc']}
        for metric, results in roc_results.items()
    ])
    roc_summary.to_csv(tables_dir / 'rq2_roc_auc.csv', index=False)
    optimal.to_csv(tables_dir / 'rq2_optimal_thresholds.csv', index=False)
    
    print(f"\n✓ Saved: {tables_dir}/rq2_roc_auc.csv")
    print(f"✓ Saved: {tables_dir}/rq2_optimal_thresholds.csv")
    
    print("\n" + "="*60)
    print("✅ CLASSIFICATION ANALYSIS COMPLETE!")
    print("="*60)
    
    # Show best metric
    best_metric = roc_summary.loc[roc_summary['roc_auc'].idxmax()]
    print(f"\n🎯 Best Metric: {best_metric['metric']} (AUC = {best_metric['roc_auc']:.3f})")
    

if __name__ == "__main__":
    main()
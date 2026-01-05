"""
Compare Spectral vs Synergistic Methods

Statistical comparison of spectral analysis and synergistic information theory
for hallucination detection on TruthfulQA.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonMetrics:
    """Metrics for method comparison."""
    method_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    correlation: float
    p_value: float


class SpectralVsSynergisticComparison:
    """Compare spectral and synergistic hallucination detection methods."""
    
    def __init__(
        self,
        spectral_results_path: str,
        synergistic_results_path: str
    ):
        """
        Initialize comparison.
        
        Args:
            spectral_results_path: Path to spectral experiment results JSON
            synergistic_results_path: Path to synergistic experiment results JSON
        """
        self.spectral_results_path = spectral_results_path
        self.synergistic_results_path = synergistic_results_path
        
        # Load results
        self.spectral_data = self.load_results(spectral_results_path)
        self.synergistic_data = self.load_results(synergistic_results_path)
        
        logger.info("Loaded both result sets for comparison")
    
    def load_results(self, path: str) -> Dict[str, Any]:
        """Load experiment results from JSON."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def extract_spectral_scores(self, results: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract risk scores and ground truth from spectral results.
        
        Args:
            results: List of spectral result dicts
            
        Returns:
            (risk_scores, is_hallucination) arrays
        """
        risk_scores = np.array([r.get('overall_risk', 0.5) for r in results])
        is_truthful = np.array([r.get('is_truthful_actual', True) for r in results])
        is_hallucination = ~is_truthful
        
        return risk_scores, is_hallucination
    
    def extract_synergistic_scores(self, results: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract synergy scores and ground truth from synergistic results.
        
        Args:
            results: List of synergistic result dicts
            
        Returns:
            (synergy_scores, is_hallucination) arrays
        """
        synergy_scores = np.array([r.get('synergy_ratio', 0.5) for r in results])
        is_truthful = np.array([r.get('is_truthful', True) for r in results])
        is_hallucination = ~is_truthful
        
        return synergy_scores, is_hallucination
    
    def compute_method_metrics(
        self,
        scores: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 0.5
    ) -> ComparisonMetrics:
        """
        Compute comprehensive metrics for a method.
        
        Args:
            scores: Detection scores
            ground_truth: True labels (1 = hallucination)
            threshold: Classification threshold
            
        Returns:
            ComparisonMetrics object
        """
        # Binary classification
        predictions = scores > threshold
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
        
        # Metrics
        accuracy = (tp + tn) / len(ground_truth) if len(ground_truth) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # ROC AUC
        if len(np.unique(ground_truth)) > 1:
            fpr, tpr, _ = roc_curve(ground_truth, scores)
            auc_roc = auc(fpr, tpr)
        else:
            auc_roc = 0.5
        
        # Correlation
        if len(scores) > 1:
            correlation, p_value = stats.pearsonr(scores, ground_truth.astype(float))
        else:
            correlation, p_value = 0, 1
        
        return ComparisonMetrics(
            method_name="",  # Set by caller
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            auc_roc=float(auc_roc),
            correlation=float(correlation),
            p_value=float(p_value)
        )
    
    def paired_comparison(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Perform paired comparison for a specific model.
        
        Args:
            model_name: Model to compare
            
        Returns:
            Comparison statistics dictionary
        """
        logger.info(f"\nComparing methods for model: {model_name}")
        
        # Extract results for this model
        spectral_results = self.spectral_data.get('sample_results', [])
        synergistic_results_by_model = self.synergistic_data.get('results_by_model', {})
        
        # Find matching model in synergistic results
        synergistic_results = None
        for key, results in synergistic_results_by_model.items():
            if model_name in key or key in model_name:
                synergistic_results = results
                break
        
        if synergistic_results is None:
            logger.warning(f"No synergistic results found for {model_name}")
            return {}
        
        # Extract scores
        spectral_scores, spectral_labels = self.extract_spectral_scores(spectral_results)
        synergistic_scores, synergistic_labels = self.extract_synergistic_scores(synergistic_results)
        
        # Align samples (use minimum length)
        min_len = min(len(spectral_scores), len(synergistic_scores))
        spectral_scores = spectral_scores[:min_len]
        synergistic_scores = synergistic_scores[:min_len]
        ground_truth = spectral_labels[:min_len]  # Assume same ground truth
        
        logger.info(f"Comparing {min_len} aligned samples")
        
        # Compute metrics for each method
        spectral_metrics = self.compute_method_metrics(spectral_scores, ground_truth)
        spectral_metrics.method_name = "Spectral"
        
        synergistic_metrics = self.compute_method_metrics(synergistic_scores, ground_truth)
        synergistic_metrics.method_name = "Synergistic"
        
        # Paired t-test on F1 scores (via bootstrap)
        # Since we can't directly compute paired F1, we use score differences
        score_diff = synergistic_scores - spectral_scores
        t_stat, t_pvalue = stats.ttest_1samp(score_diff, 0)
        
        # McNemar's test for classification agreement
        spectral_pred = spectral_scores > 0.5
        synergistic_pred = synergistic_scores > 0.5
        
        # Contingency table for McNemar
        both_correct = np.sum((spectral_pred == ground_truth) & (synergistic_pred == ground_truth))
        spectral_only = np.sum((spectral_pred == ground_truth) & (synergistic_pred != ground_truth))
        synergistic_only = np.sum((spectral_pred != ground_truth) & (synergistic_pred == ground_truth))
        both_wrong = np.sum((spectral_pred != ground_truth) & (synergistic_pred != ground_truth))
        
        # McNemar test (b and c are off-diagonal)
        if spectral_only + synergistic_only > 0:
            mcnemar_stat = (abs(spectral_only - synergistic_only) - 1)**2 / (spectral_only + synergistic_only)
            mcnemar_pvalue = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat, mcnemar_pvalue = 0, 1
        
        # Determine winner
        winner = "Synergistic" if synergistic_metrics.f1_score > spectral_metrics.f1_score else "Spectral"
        if abs(synergistic_metrics.f1_score - spectral_metrics.f1_score) < 0.01:
            winner = "Tie"
        
        comparison = {
            'model': model_name,
            'n_samples': int(min_len),
            'spectral_metrics': {
                'accuracy': spectral_metrics.accuracy,
                'precision': spectral_metrics.precision,
                'recall': spectral_metrics.recall,
                'f1_score': spectral_metrics.f1_score,
                'auc_roc': spectral_metrics.auc_roc,
                'correlation': spectral_metrics.correlation
            },
            'synergistic_metrics': {
                'accuracy': synergistic_metrics.accuracy,
                'precision': synergistic_metrics.precision,
                'recall': synergistic_metrics.recall,
                'f1_score': synergistic_metrics.f1_score,
                'auc_roc': synergistic_metrics.auc_roc,
                'correlation': synergistic_metrics.correlation
            },
            'comparison': {
                'winner': winner,
                'f1_improvement': synergistic_metrics.f1_score - spectral_metrics.f1_score,
                'auc_improvement': synergistic_metrics.auc_roc - spectral_metrics.auc_roc
            },
            'statistical_tests': {
                'paired_ttest': {
                    't_statistic': float(t_stat),
                    'p_value': float(t_pvalue),
                    'significant': t_pvalue < 0.05
                },
                'mcnemar': {
                    'statistic': float(mcnemar_stat),
                    'p_value': float(mcnemar_pvalue),
                    'significant': mcnemar_pvalue < 0.05,
                    'agreement_matrix': {
                        'both_correct': int(both_correct),
                        'spectral_only': int(spectral_only),
                        'synergistic_only': int(synergistic_only),
                        'both_wrong': int(both_wrong)
                    }
                }
            }
        }
        
        # Print summary
        logger.info(f"\n{model_name} Comparison:")
        logger.info(f"  Spectral    F1: {spectral_metrics.f1_score:.3f}, AUC: {spectral_metrics.auc_roc:.3f}")
        logger.info(f"  Synergistic F1: {synergistic_metrics.f1_score:.3f}, AUC: {synergistic_metrics.auc_roc:.3f}")
        logger.info(f"  Winner: {winner}")
        logger.info(f"  McNemar p-value: {mcnemar_pvalue:.4f}")
        
        return comparison
    
    def visualize_comparison(
        self,
        comparison: Dict[str, Any],
        output_path: str = "comparison_visualization.png"
    ):
        """
        Create comparison visualizations.
        
        Args:
            comparison: Comparison results
            output_path: Output path
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        model_name = comparison['model']
        
        # Plot 1: Metrics comparison (bar chart)
        ax = axes[0, 0]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
        spectral_values = [
            comparison['spectral_metrics']['accuracy'],
            comparison['spectral_metrics']['precision'],
            comparison['spectral_metrics']['recall'],
            comparison['spectral_metrics']['f1_score'],
            comparison['spectral_metrics']['auc_roc']
        ]
        synergistic_values = [
            comparison['synergistic_metrics']['accuracy'],
            comparison['synergistic_metrics']['precision'],
            comparison['synergistic_metrics']['recall'],
            comparison['synergistic_metrics']['f1_score'],
            comparison['synergistic_metrics']['auc_roc']
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        ax.bar(x - width/2, spectral_values, width, label='Spectral', alpha=0.8)
        ax.bar(x + width/2, synergistic_values, width, label='Synergistic', alpha=0.8)
        ax.set_ylabel('Score')
        ax.set_title(f'{model_name}: Metric Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Agreement matrix (McNemar)
        ax = axes[0, 1]
        agreement = comparison['statistical_tests']['mcnemar']['agreement_matrix']
        matrix = np.array([
            [agreement['both_correct'], agreement['spectral_only']],
            [agreement['synergistic_only'], agreement['both_wrong']]
        ])
        
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Syn Correct', 'Syn Wrong'],
                   yticklabels=['Spec Correct', 'Spec Wrong'])
        ax.set_title(f'{model_name}: Classification Agreement')
        
        # Plot 3: F1 Score comparison
        ax = axes[1, 0]
        methods = ['Spectral', 'Synergistic']
        f1_scores = [
            comparison['spectral_metrics']['f1_score'],
            comparison['synergistic_metrics']['f1_score']
        ]
        colors = ['blue', 'green' if comparison['comparison']['winner'] == 'Synergistic' else 'blue']
        ax.bar(methods, f1_scores, color=colors, alpha=0.7)
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{model_name}: F1 Score\nWinner: {comparison["comparison"]["winner"]}')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 4: Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
{model_name} Comparison Summary

Samples: {comparison['n_samples']}

SPECTRAL METHOD:
  F1 Score: {comparison['spectral_metrics']['f1_score']:.3f}
  AUC-ROC: {comparison['spectral_metrics']['auc_roc']:.3f}
  Correlation: {comparison['spectral_metrics']['correlation']:.3f}

SYNERGISTIC METHOD:
  F1 Score: {comparison['synergistic_metrics']['f1_score']:.3f}
  AUC-ROC: {comparison['synergistic_metrics']['auc_roc']:.3f}
  Correlation: {comparison['synergistic_metrics']['correlation']:.3f}

STATISTICAL TESTS:
  Winner: {comparison['comparison']['winner']}
  F1 Improvement: {comparison['comparison']['f1_improvement']:+.3f}
  
  McNemar Test:
    p-value: {comparison['statistical_tests']['mcnemar']['p_value']:.4f}
    Significant: {comparison['statistical_tests']['mcnemar']['significant']}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
               verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Comparison visualization saved to {output_path}")
        
        try:
            plt.show()
        except:
            pass
    
    def save_comparison_report(
        self,
        all_comparisons: List[Dict[str, Any]],
        output_path: str = "comparison_report.json"
    ):
        """
        Save comprehensive comparison report.
        
        Args:
            all_comparisons: List of comparison results
            output_path: Output path
        """
        report = {
            'comparison_type': 'spectral_vs_synergistic',
            'n_models': len(all_comparisons),
            'comparisons': all_comparisons,
            'summary': self.generate_summary(all_comparisons)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comparison report saved to {output_path}")
    
    def generate_summary(self, all_comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall summary across all models."""
        if not all_comparisons:
            return {}
        
        synergistic_wins = sum(1 for c in all_comparisons if c['comparison']['winner'] == 'Synergistic')
        spectral_wins = sum(1 for c in all_comparisons if c['comparison']['winner'] == 'Spectral')
        ties = sum(1 for c in all_comparisons if c['comparison']['winner'] == 'Tie')
        
        avg_f1_improvement = np.mean([c['comparison']['f1_improvement'] for c in all_comparisons])
        avg_auc_improvement = np.mean([c['comparison']['auc_improvement'] for c in all_comparisons])
        
        significant_improvements = sum(
            1 for c in all_comparisons 
            if c['statistical_tests']['mcnemar']['significant'] and 
               c['comparison']['winner'] == 'Synergistic'
        )
        
        return {
            'total_models': len(all_comparisons),
            'synergistic_wins': int(synergistic_wins),
            'spectral_wins': int(spectral_wins),
            'ties': int(ties),
            'avg_f1_improvement': float(avg_f1_improvement),
            'avg_auc_improvement': float(avg_auc_improvement),
            'significant_improvements': int(significant_improvements),
            'overall_winner': 'Synergistic' if synergistic_wins > spectral_wins else 'Spectral'
        }


def main():
    parser = argparse.ArgumentParser(description="Compare spectral vs synergistic methods")
    parser.add_argument(
        '--spectral-results',
        required=True,
        help='Path to spectral experiment results JSON'
    )
    parser.add_argument(
        '--synergistic-results',
        required=True,
        help='Path to synergistic experiment results JSON'
    )
    parser.add_argument(
        '--output',
        default='comparison_results',
        help='Output file prefix'
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Specific model to compare (default: all)'
    )
    
    args = parser.parse_args()
    
    # Initialize comparison
    comparison_engine = SpectralVsSynergisticComparison(
        spectral_results_path=args.spectral_results,
        synergistic_results_path=args.synergistic_results
    )
    
    logger.info("\n" + "="*80)
    logger.info("Spectral vs Synergistic Comparison")
    logger.info("="*80)
    
    # Determine models to compare
    if args.model:
        models_to_compare = [args.model]
    else:
        # Extract models from synergistic results
        synergistic_data = comparison_engine.synergistic_data
        models_to_compare = list(synergistic_data.get('results_by_model', {}).keys())
    
    # Run comparisons
    all_comparisons = []
    for model_name in models_to_compare:
        comparison = comparison_engine.paired_comparison(model_name)
        if comparison:
            all_comparisons.append(comparison)
            
            # Visualize each model
            comparison_engine.visualize_comparison(
                comparison,
                f"{args.output}_{model_name.replace('/', '_')}.png"
            )
    
    # Save comprehensive report
    comparison_engine.save_comparison_report(
        all_comparisons,
        f"{args.output}.json"
    )
    
    # Print final summary
    if all_comparisons:
        summary = comparison_engine.generate_summary(all_comparisons)
        logger.info("\n" + "="*80)
        logger.info("OVERALL SUMMARY")
        logger.info("="*80)
        logger.info(f"Models compared: {summary['total_models']}")
        logger.info(f"Synergistic wins: {summary['synergistic_wins']}")
        logger.info(f"Spectral wins: {summary['spectral_wins']}")
        logger.info(f"Ties: {summary['ties']}")
        logger.info(f"Average F1 improvement: {summary['avg_f1_improvement']:+.3f}")
        logger.info(f"Significant improvements: {summary['significant_improvements']}")
        logger.info(f"\nOVERALL WINNER: {summary['overall_winner']}")
        logger.info("="*80)


if __name__ == "__main__":
    main()










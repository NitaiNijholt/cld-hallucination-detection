"""
Synergistic TruthfulQA Experiment

Tests synergistic information theory for hallucination detection on TruthfulQA dataset.
Compares synergistic analysis against spectral baseline.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict
import argparse
import logging
from tqdm import tqdm

from synergistic_hidden_state_analysis import (
    SynergisticHiddenStateAnalyzer,
    SynergisticMetrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynergisticTruthfulQAExperiment:
    """Experiment runner for synergistic analysis on TruthfulQA."""
    
    def __init__(
        self,
        extracted_data_dir: str,
        model_names: Optional[List[str]] = None
    ):
        """
        Initialize experiment.
        
        Args:
            extracted_data_dir: Directory containing extracted hidden states
            model_names: List of model names to analyze (None = all subdirs)
        """
        self.extracted_data_dir = Path(extracted_data_dir)
        self.analyzer = SynergisticHiddenStateAnalyzer()
        
        # Discover model directories
        if model_names is None:
            model_dirs = [d for d in self.extracted_data_dir.iterdir() if d.is_dir()]
            self.model_names = [d.name for d in model_dirs]
        else:
            self.model_names = [m.replace('/', '_') for m in model_names]
        
        logger.info(f"Initialized experiment for models: {self.model_names}")
    
    def load_extraction(self, extraction_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load extracted hidden states from disk.
        
        Args:
            extraction_path: Path to extraction (without extension)
            
        Returns:
            Dictionary with hidden states and metadata
        """
        json_path = Path(f"{extraction_path}.json")
        pt_path = Path(f"{extraction_path}.pt")
        
        if not json_path.exists() or not pt_path.exists():
            logger.warning(f"Missing files for {extraction_path}")
            return None
        
        try:
            # Load metadata
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            # Load tensors
            tensors = torch.load(pt_path, map_location='cpu')
            
            return {
                **metadata,
                **tensors
            }
            
        except Exception as e:
            logger.error(f"Failed to load {extraction_path}: {e}")
            return None
    
    def analyze_single_sample(
        self,
        extraction: Dict[str, Any]
    ) -> Optional[SynergisticMetrics]:
        """
        Analyze a single extracted sample.
        
        Args:
            extraction: Extracted hidden states and metadata
            
        Returns:
            SynergisticMetrics or None if analysis fails
        """
        try:
            hidden_states = extraction['hidden_states']
            output_logits = extraction['output_logits']
            
            # Take last token's hidden states and logits for analysis
            # (most relevant for generation)
            hidden_states_last = [h[:, -1, :].squeeze() for h in hidden_states]
            output_logits_last = output_logits[:, -1, :].squeeze()
            
            # Run synergistic analysis
            metrics = self.analyzer.analyze(
                hidden_states_all_layers=hidden_states_last,
                output_logits=output_logits_last
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return None
    
    def run_model_experiment(
        self,
        model_name: str,
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run experiment for a single model.
        
        Args:
            model_name: Model name (directory name)
            max_samples: Maximum samples to analyze (None = all)
            
        Returns:
            List of result dictionaries
        """
        model_dir = self.extracted_data_dir / model_name
        summary_path = model_dir / "extraction_summary.json"
        
        if not summary_path.exists():
            logger.error(f"Summary not found for {model_name}")
            return []
        
        # Load extraction summary
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        samples_metadata = summary['samples']
        if max_samples:
            samples_metadata = samples_metadata[:max_samples]
        
        logger.info(f"\nAnalyzing {len(samples_metadata)} samples for {model_name}")
        
        results = []
        
        for sample_meta in tqdm(samples_metadata, desc=f"Analyzing {model_name}"):
            sample_id = sample_meta['sample_id']
            extraction_path = model_dir / f"sample_{sample_id:04d}"
            
            # Load extraction
            extraction = self.load_extraction(extraction_path)
            if extraction is None:
                continue
            
            # Analyze
            metrics = self.analyze_single_sample(extraction)
            if metrics is None:
                continue
            
            # Store result
            result = {
                'model': model_name,
                'sample_id': sample_id,
                'question': sample_meta.get('question', ''),
                'answer': sample_meta.get('answer', ''),
                'is_truthful': sample_meta.get('is_truthful', False),
                **asdict(metrics)
            }
            results.append(result)
        
        logger.info(f"Completed {model_name}: {len(results)} samples analyzed")
        return results
    
    def run_all_models(
        self,
        max_samples_per_model: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run experiment for all models.
        
        Args:
            max_samples_per_model: Max samples per model
            
        Returns:
            Dictionary mapping model name to results list
        """
        all_results = {}
        
        for model_name in self.model_names:
            results = self.run_model_experiment(model_name, max_samples_per_model)
            all_results[model_name] = results
        
        return all_results
    
    def calculate_statistics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate classification statistics.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {}
        
        # Extract arrays
        is_truthful = np.array([r['is_truthful'] for r in results])
        synergy_ratio = np.array([r['synergy_ratio'] for r in results])
        hallucination_risk = np.array([r['hallucination_risk'] for r in results])
        
        # Use synergy_ratio as detection score
        # Threshold at 0.5 for classification
        threshold = 0.5
        predicted_hallucination = synergy_ratio > threshold
        actual_hallucination = ~is_truthful
        
        # Confusion matrix
        tp = np.sum(actual_hallucination & predicted_hallucination)
        tn = np.sum(~actual_hallucination & ~predicted_hallucination)
        fp = np.sum(~actual_hallucination & predicted_hallucination)
        fn = np.sum(actual_hallucination & ~predicted_hallucination)
        
        # Metrics
        accuracy = (tp + tn) / len(results) if len(results) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Correlation
        if len(synergy_ratio) > 1:
            correlation, p_value = stats.pearsonr(synergy_ratio, actual_hallucination.astype(float))
        else:
            correlation, p_value = 0, 1
        
        # Synergy statistics by truth value
        truthful_synergy = synergy_ratio[is_truthful]
        false_synergy = synergy_ratio[~is_truthful]
        
        statistics = {
            'n_samples': len(results),
            'n_truthful': int(np.sum(is_truthful)),
            'n_false': int(np.sum(~is_truthful)),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': {
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            },
            'synergy_ratio': {
                'mean': float(np.mean(synergy_ratio)),
                'std': float(np.std(synergy_ratio)),
                'min': float(np.min(synergy_ratio)),
                'max': float(np.max(synergy_ratio)),
                'truthful_mean': float(np.mean(truthful_synergy)) if len(truthful_synergy) > 0 else 0,
                'truthful_std': float(np.std(truthful_synergy)) if len(truthful_synergy) > 0 else 0,
                'false_mean': float(np.mean(false_synergy)) if len(false_synergy) > 0 else 0,
                'false_std': float(np.std(false_synergy)) if len(false_synergy) > 0 else 0,
            },
            'correlation': {
                'pearson_r': float(correlation),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        }
        
        # T-test between truthful and false synergy
        if len(truthful_synergy) > 0 and len(false_synergy) > 0:
            t_stat, t_pvalue = stats.ttest_ind(truthful_synergy, false_synergy)
            statistics['ttest'] = {
                't_statistic': float(t_stat),
                'p_value': float(t_pvalue),
                'significant': t_pvalue < 0.05
            }
        
        return statistics
    
    def visualize_results(
        self,
        all_results: Dict[str, List[Dict[str, Any]]],
        output_path: str = "synergistic_experiment_results.png"
    ):
        """
        Visualize experiment results.
        
        Args:
            all_results: Dictionary mapping model to results
            output_path: Output path for figure
        """
        n_models = len(all_results)
        fig, axes = plt.subplots(n_models, 3, figsize=(15, 5 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, results) in enumerate(all_results.items()):
            if not results:
                continue
            
            # Extract data
            is_truthful = np.array([r['is_truthful'] for r in results])
            synergy_ratio = np.array([r['synergy_ratio'] for r in results])
            
            truthful_synergy = synergy_ratio[is_truthful]
            false_synergy = synergy_ratio[~is_truthful]
            
            # Plot 1: Synergy distribution
            ax = axes[idx, 0]
            if len(truthful_synergy) > 0:
                ax.hist(truthful_synergy, bins=20, alpha=0.6, label='Truthful', color='green')
            if len(false_synergy) > 0:
                ax.hist(false_synergy, bins=20, alpha=0.6, label='Hallucinated', color='red')
            ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
            ax.set_xlabel('Synergy Ratio')
            ax.set_ylabel('Count')
            ax.set_title(f'{model_name}: Synergy Distribution')
            ax.legend()
            
            # Plot 2: Box plot
            ax = axes[idx, 1]
            data_to_plot = []
            labels = []
            if len(truthful_synergy) > 0:
                data_to_plot.append(truthful_synergy)
                labels.append('Truthful')
            if len(false_synergy) > 0:
                data_to_plot.append(false_synergy)
                labels.append('Hallucinated')
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                bp['boxes'][0].set_facecolor('green' if labels[0] == 'Truthful' else 'red')
                if len(data_to_plot) > 1:
                    bp['boxes'][1].set_facecolor('red' if labels[1] == 'Hallucinated' else 'green')
            ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            ax.set_ylabel('Synergy Ratio')
            ax.set_title(f'{model_name}: Synergy by Truth Value')
            
            # Plot 3: Scatter
            ax = axes[idx, 2]
            for r in results:
                color = 'green' if r['is_truthful'] else 'red'
                marker = 'o'
                ax.scatter(r['synergy_ratio'], 
                          1 if r['is_truthful'] else 0,
                          color=color, marker=marker, alpha=0.6)
            ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Synergy Ratio')
            ax.set_ylabel('Truth Value (1=Truthful, 0=False)')
            ax.set_title(f'{model_name}: Scatter Plot')
            ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
        
        try:
            plt.show()
        except:
            pass
    
    def save_results(
        self,
        all_results: Dict[str, List[Dict[str, Any]]],
        all_statistics: Dict[str, Dict[str, Any]],
        output_path: str = "synergistic_experiment_results.json"
    ):
        """
        Save results to JSON.
        
        Args:
            all_results: All experiment results
            all_statistics: All statistics
            output_path: Output path
        """
        # Convert numpy types to Python native types
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        output = {
            'experiment': 'synergistic_truthfulqa',
            'models': list(all_results.keys()),
            'results_by_model': convert_to_json_serializable(all_results),
            'statistics_by_model': convert_to_json_serializable(all_statistics)
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run synergistic TruthfulQA experiment")
    parser.add_argument(
        '--extracted-dir',
        default='extracted_synergistic',
        help='Directory with extracted hidden states'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Model names to analyze (default: all in extracted-dir)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per model'
    )
    parser.add_argument(
        '--output',
        default='synergistic_experiment_results',
        help='Output file prefix'
    )
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = SynergisticTruthfulQAExperiment(
        extracted_data_dir=args.extracted_dir,
        model_names=args.models
    )
    
    # Run experiment
    logger.info("\n" + "="*80)
    logger.info("Running Synergistic TruthfulQA Experiment")
    logger.info("="*80)
    
    all_results = experiment.run_all_models(max_samples_per_model=args.max_samples)
    
    # Calculate statistics
    all_statistics = {}
    for model_name, results in all_results.items():
        stats = experiment.calculate_statistics(results)
        all_statistics[model_name] = stats
        
        # Print summary
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Samples: {stats.get('n_samples', 0)}")
        logger.info(f"  Accuracy: {stats.get('accuracy', 0):.2%}")
        logger.info(f"  F1 Score: {stats.get('f1_score', 0):.3f}")
        logger.info(f"  Correlation: {stats.get('correlation', {}).get('pearson_r', 0):.3f} "
                   f"(p={stats.get('correlation', {}).get('p_value', 1):.4f})")
        
        if 'synergy_ratio' in stats:
            sr = stats['synergy_ratio']
            logger.info(f"  Synergy - Truthful: {sr.get('truthful_mean', 0):.3f} ± {sr.get('truthful_std', 0):.3f}")
            logger.info(f"  Synergy - False: {sr.get('false_mean', 0):.3f} ± {sr.get('false_std', 0):.3f}")
    
    # Visualize
    experiment.visualize_results(all_results, f"{args.output}.png")
    
    # Save results
    experiment.save_results(all_results, all_statistics, f"{args.output}.json")
    
    logger.info("\n" + "="*80)
    logger.info("Experiment Complete")
    logger.info("="*80)


if __name__ == "__main__":
    main()










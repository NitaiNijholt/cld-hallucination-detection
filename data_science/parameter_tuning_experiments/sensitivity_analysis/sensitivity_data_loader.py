"""
Sensitivity Analysis Data Loader

Extends RQ2DataLoader to load results from Sobol sensitivity analysis experiments.

Architecture:
- Inherits from RQ2DataLoader
- Maps Sobol samples to experiment results
- Aggregates outcomes for Sobol analysis
- Prepares data for sobol.analyze()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import logging
import sys

# Add parent directory to path to import RQ2DataLoader
sys.path.append(str(Path(__file__).parent.parent / 'analysis'))
from rq2_data_loader import RQ2DataLoader

logger = logging.getLogger(__name__)


class SensitivityDataLoader(RQ2DataLoader):
    """
    Data loader for sensitivity analysis experiments.
    
    Extends RQ2DataLoader with:
    - Loading by experiment ID prefix (e.g., all 'sobol_rq2_*' experiments)
    - Mapping experiments to Sobol samples
    - Aggregating metrics for Sobol analysis
    - Validating sample completeness
    """
    
    def __init__(
        self,
        results_base_dir: str = "results",
        hallucination_mode: str = "judge"
    ):
        """
        Initialize sensitivity data loader.
        
        Args:
            results_base_dir: Base directory containing experiment results
            hallucination_mode: How to define hallucinations ('judge' or 'ground_truth')
        """
        super().__init__(results_base_dir, hallucination_mode)
        logger.info("Initialized SensitivityDataLoader")
    
    def load_sobol_experiments(
        self,
        exp_id_prefix: str,
        samples_csv: str,
        config_manifest_csv: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all experiments matching a prefix and map to Sobol samples.
        
        Args:
            exp_id_prefix: Prefix to filter experiment IDs (e.g., 'sobol_rq2_')
            samples_csv: Path to Sobol samples CSV (with sample_id column)
            config_manifest_csv: Optional path to config manifest for mapping
        
        Returns:
            Tuple of (experiments_df, samples_df):
            - experiments_df: Loaded experiment data with sample_id
            - samples_df: Sobol samples with metadata
        """
        logger.info(f"Loading Sobol experiments with prefix: {exp_id_prefix}")
        
        # Find all experiment IDs matching prefix
        all_exp_ids = self.find_all_experiments()
        matching_exp_ids = [exp_id for exp_id in all_exp_ids if exp_id.startswith(exp_id_prefix)]
        
        if not matching_exp_ids:
            raise ValueError(f"No experiments found with prefix: {exp_id_prefix}")
        
        logger.info(f"Found {len(matching_exp_ids)} experiments matching prefix")
        
        # Load all experiments
        experiments_df = self.load_multiple_experiments(matching_exp_ids)
        
        # Load Sobol samples
        samples_df = pd.read_csv(samples_csv)
        logger.info(f"Loaded {len(samples_df)} Sobol samples")
        
        # Map experiments to samples
        experiments_df = self._map_experiments_to_samples(
            experiments_df,
            samples_df,
            config_manifest_csv
        )
        
        # Validate completeness
        self._validate_completeness(experiments_df, samples_df)
        
        return experiments_df, samples_df
    
    def _map_experiments_to_samples(
        self,
        experiments_df: pd.DataFrame,
        samples_df: pd.DataFrame,
        config_manifest_csv: Optional[str]
    ) -> pd.DataFrame:
        """
        Map experiment results to Sobol samples.
        
        Uses sobol_metadata in experiment configs to link sample_id.
        """
        logger.info("Mapping experiments to Sobol samples...")
        
        # If manifest provided, use it for mapping
        if config_manifest_csv:
            manifest_df = pd.read_csv(config_manifest_csv)
            logger.info(f"Using config manifest: {config_manifest_csv}")
            
            # Extract experiment_id from results and map to sample_id via manifest
            # This is more reliable than trying to parse metadata from CSV
            
            # For now, assume experiment files contain sample_id in their metadata
            # This will be populated by config generator
        
        # Try to extract sample_id from the CSV mapping files
        sample_ids = []
        for _, row in experiments_df.iterrows():
            exp_id = row['experiment_id']
            
            # Read the param_combo_mapping CSV for this experiment
            csv_path = self.results_base_dir / f"param_combo_mapping_{exp_id}.csv"
            
            if csv_path.exists():
                mapping_df = pd.read_csv(csv_path)
                if len(mapping_df) > 0:
                    # Parse parameters JSON to get sobol_metadata
                    params_json = mapping_df.iloc[0]['parameters']
                    params = json.loads(params_json)
                    
                    # Check if config had sobol_metadata (added by config generator)
                    # If not, we'll need to infer from experiment_id or other means
                    sample_id = None
                    
                    # Try to parse sample_id from experiment_id
                    # Expected format: sobol_rq2_sample_0001
                    if 'sample_' in exp_id:
                        try:
                            sample_id = int(exp_id.split('sample_')[-1].split('_')[0])
                        except:
                            pass
                    
                    sample_ids.append(sample_id)
                else:
                    sample_ids.append(None)
            else:
                logger.warning(f"No mapping CSV found for experiment: {exp_id}")
                sample_ids.append(None)
        
        experiments_df['sample_id'] = sample_ids
        
        # Remove rows without sample_id
        n_before = len(experiments_df)
        experiments_df = experiments_df[experiments_df['sample_id'].notna()]
        n_after = len(experiments_df)
        
        if n_after < n_before:
            logger.warning(f"Removed {n_before - n_after} rows without sample_id")
        
        logger.info(f"Mapped {len(experiments_df)} results to {experiments_df['sample_id'].nunique()} unique samples")
        
        return experiments_df
    
    def _validate_completeness(
        self,
        experiments_df: pd.DataFrame,
        samples_df: pd.DataFrame
    ):
        """Validate that all samples have experimental results."""
        
        expected_sample_ids = set(samples_df['sample_id'].unique())
        actual_sample_ids = set(experiments_df['sample_id'].unique())
        
        missing_sample_ids = expected_sample_ids - actual_sample_ids
        extra_sample_ids = actual_sample_ids - expected_sample_ids
        
        if missing_sample_ids:
            logger.warning(f"Missing results for {len(missing_sample_ids)} samples: {sorted(list(missing_sample_ids))[:10]}...")
        
        if extra_sample_ids:
            logger.warning(f"Extra results for {len(extra_sample_ids)} samples not in samples CSV")
        
        coverage = len(actual_sample_ids) / len(expected_sample_ids) * 100
        logger.info(f"Sample coverage: {coverage:.1f}% ({len(actual_sample_ids)}/{len(expected_sample_ids)})")
        
        if coverage < 95:
            logger.error(f"Low sample coverage: {coverage:.1f}%. Need at least 95% for reliable Sobol analysis.")
    
    def aggregate_outcomes_for_sobol(
        self,
        experiments_df: pd.DataFrame,
        samples_df: pd.DataFrame,
        outcome_metrics: List[str] = None
    ) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Aggregate outcomes for Sobol analysis.
        
        For each outcome metric, returns samples aligned with outcome values.
        
        Args:
            experiments_df: Experiment results with sample_id
            samples_df: Sobol samples
            outcome_metrics: List of outcome metrics to aggregate
                           Default: ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        Returns:
            Dictionary mapping metric_name -> (samples_df, outcomes_array)
            Both are aligned by sample_id and ready for sobol.analyze()
        """
        if outcome_metrics is None:
            outcome_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        logger.info(f"Aggregating outcomes for {len(outcome_metrics)} metrics")
        
        results = {}
        
        for metric in outcome_metrics:
            # Compute metric per sample
            if metric == 'accuracy':
                # Accuracy = (TP + TN) / (TP + TN + FP + FN)
                metric_per_sample = experiments_df.groupby('sample_id').apply(
                    lambda g: (
                        len(g[g['classification'] == 'TP']) + len(g[g['classification'] == 'TN'])
                    ) / len(g) if len(g) > 0 else 0
                ).reset_index(name=metric)
            
            elif metric == 'precision':
                # Precision = TP / (TP + FP)
                metric_per_sample = experiments_df.groupby('sample_id').apply(
                    lambda g: (
                        len(g[g['classification'] == 'TP']) /
                        (len(g[g['classification'] == 'TP']) + len(g[g['classification'] == 'FP']))
                        if (len(g[g['classification'] == 'TP']) + len(g[g['classification'] == 'FP'])) > 0 else 0
                    )
                ).reset_index(name=metric)
            
            elif metric == 'recall':
                # Recall = TP / (TP + FN)
                metric_per_sample = experiments_df.groupby('sample_id').apply(
                    lambda g: (
                        len(g[g['classification'] == 'TP']) /
                        (len(g[g['classification'] == 'TP']) + len(g[g['classification'] == 'FN']))
                        if (len(g[g['classification'] == 'TP']) + len(g[g['classification'] == 'FN'])) > 0 else 0
                    )
                ).reset_index(name=metric)
            
            elif metric == 'f1_score':
                # F1 = 2 * (precision * recall) / (precision + recall)
                # Compute per sample
                metric_per_sample = experiments_df.groupby('sample_id').apply(
                    lambda g: self._compute_f1(g)
                ).reset_index(name=metric)
            
            elif metric == 'auc_roc':
                # Would need CI metric values to compute ROC
                # For now, skip or use aggregate_score as proxy
                logger.warning(f"Metric '{metric}' not implemented yet. Skipping.")
                continue
            
            elif metric in experiments_df.columns:
                # Direct metric from data
                metric_per_sample = experiments_df.groupby('sample_id')[metric].mean().reset_index()
            
            else:
                logger.warning(f"Unknown metric: {metric}. Skipping.")
                continue
            
            # Merge with samples to ensure alignment
            merged = samples_df.merge(metric_per_sample, on='sample_id', how='left')
            
            # Check for missing values
            if merged[metric].isna().any():
                n_missing = merged[metric].isna().sum()
                logger.warning(f"Metric '{metric}' has {n_missing} missing values. Filling with 0.")
                merged[metric] = merged[metric].fillna(0)
            
            # Extract outcomes array (must be same order as samples)
            outcomes = merged[metric].values
            
            results[metric] = (merged, outcomes)
            logger.info(f"  {metric}: {len(outcomes)} samples, mean={outcomes.mean():.3f}, std={outcomes.std():.3f}")
        
        return results
    
    def _compute_f1(self, group_df: pd.DataFrame) -> float:
        """Compute F1 score for a group."""
        tp = len(group_df[group_df['classification'] == 'TP'])
        fp = len(group_df[group_df['classification'] == 'FP'])
        fn = len(group_df[group_df['classification'] == 'FN'])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    def find_all_experiments(self) -> List[str]:
        """Find all experiment IDs in results directory."""
        csv_files = list(self.results_base_dir.glob("param_combo_mapping_exp_*.csv"))
        
        exp_ids = []
        for csv_file in csv_files:
            # Extract experiment_id from filename
            # Format: param_combo_mapping_exp_<experiment_id>.csv
            filename = csv_file.stem
            exp_id = filename.replace("param_combo_mapping_", "")
            exp_ids.append(exp_id)
        
        return sorted(exp_ids)


def load_sobol_results(
    exp_id_prefix: str,
    samples_csv: str,
    results_dir: str = "parameter_tuning_experiments/results",
    outcome_metrics: List[str] = None,
    hallucination_mode: str = "judge"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[pd.DataFrame, np.ndarray]]]:
    """
    Convenience function to load Sobol experiment results.
    
    Args:
        exp_id_prefix: Prefix for experiment IDs
        samples_csv: Path to Sobol samples CSV
        results_dir: Results directory
        outcome_metrics: Metrics to aggregate
        hallucination_mode: Hallucination definition mode
    
    Returns:
        Tuple of (experiments_df, samples_df, outcomes_dict)
    """
    loader = SensitivityDataLoader(
        results_base_dir=results_dir,
        hallucination_mode=hallucination_mode
    )
    
    experiments_df, samples_df = loader.load_sobol_experiments(
        exp_id_prefix=exp_id_prefix,
        samples_csv=samples_csv
    )
    
    outcomes_dict = loader.aggregate_outcomes_for_sobol(
        experiments_df=experiments_df,
        samples_df=samples_df,
        outcome_metrics=outcome_metrics
    )
    
    return experiments_df, samples_df, outcomes_dict






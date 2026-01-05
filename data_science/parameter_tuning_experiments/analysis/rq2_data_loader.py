"""
RQ2 Data Loader

Loads and aggregates experiment results from multi-runner output.

Directory structure expected:
    results/
    ├── param_combo_mapping_exp_{experiment_id}.csv
    └── exp_{experiment_id}/
        └── combo_{N}/
            └── {prompt_variant}/
                └── {cld_name}/
                    └── run{N}_{hash}/
                        └── results_*.xlsx
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RQ2DataLoader:
    """
    Load and preprocess experiment results for RQ2 analysis.
    
    This loader uses the param_combo_mapping CSV files as an index to efficiently
    load all experiment results without traversing directory trees.
    """
    
    def __init__(self, results_base_dir: str = "results", hallucination_mode: str = "judge"):
        """
        Initialize the data loader.
        
        Args:
            results_base_dir: Base directory containing experiment results
                             (relative to parameter_tuning_experiments/)
            hallucination_mode: How to define hallucinations:
                - 'judge': Based on judge's aggregate_score < 0.5 (default)
                - 'ground_truth': Based on Classification='FP' (edge not in ground truth CLD)
        """
        self.results_base_dir = Path(results_base_dir)
        if not self.results_base_dir.is_absolute():
            # Make it absolute relative to this file's location
            script_dir = Path(__file__).parent.parent
            self.results_base_dir = script_dir / self.results_base_dir
        
        self.results_base_dir = self.results_base_dir.resolve()
        
        if not self.results_base_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_base_dir}")
        
        if hallucination_mode not in ['judge', 'ground_truth']:
            raise ValueError(f"Invalid hallucination_mode: {hallucination_mode}. Must be 'judge' or 'ground_truth'")
        
        self.hallucination_mode = hallucination_mode
        logger.info(f"Initialized RQ2DataLoader with results_dir: {self.results_base_dir}")
        logger.info(f"Hallucination mode: {hallucination_mode}")
    
    def find_all_experiments(self) -> List[str]:
        """
        Find all experiment IDs by looking for param_combo_mapping CSV files.
        
        Returns:
            List of experiment IDs (e.g., ['exp_20251001_181507_4b7d12de', ...])
        """
        mapping_files = list(self.results_base_dir.glob("param_combo_mapping_exp_*.csv"))
        
        experiment_ids = []
        for f in mapping_files:
            # Extract experiment ID from filename
            # Format: param_combo_mapping_exp_20251001_181507_4b7d12de.csv
            exp_id = f.stem.replace("param_combo_mapping_", "")
            experiment_ids.append(exp_id)
        
        experiment_ids.sort(reverse=True)  # Most recent first
        logger.info(f"Found {len(experiment_ids)} experiments")
        
        return experiment_ids
    
    def load_experiment_metadata(self, experiment_id: str) -> pd.DataFrame:
        """
        Load metadata CSV for a specific experiment.
        
        Args:
            experiment_id: Experiment ID (e.g., 'exp_20251001_181507_4b7d12de')
        
        Returns:
            DataFrame with columns: experiment_id, prompt, cld_prefix, run_idx,
                                   excel_filename, combo_number, parameters
        """
        csv_path = self.results_base_dir / f"param_combo_mapping_{experiment_id}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
        
        metadata = pd.read_csv(csv_path)
        logger.info(f"Loaded metadata for {experiment_id}: {len(metadata)} runs")
        
        return metadata
    
    def load_single_excel_file(self, excel_path: Path, metadata_row: pd.Series) -> pd.DataFrame:
        """
        Load a single Excel result file and add metadata.
        
        Args:
            excel_path: Path to Excel file (may not have timestamp)
            metadata_row: Row from metadata CSV with parameters
        
        Returns:
            DataFrame with edge data + metadata columns
        """
        # The CSV path doesn't include timestamp, but actual file does
        # E.g., CSV: results_..._run1_CLD_test.xlsx
        #       Actual: results_..._run1_CLD_test_20251001_182549.xlsx
        
        if not excel_path.exists():
            # Try to find the file with timestamp suffix
            parent_dir = excel_path.parent
            base_name = excel_path.stem  # Without .xlsx
            
            # Look for files matching pattern: {base_name}_*.xlsx or {base_name}.xlsx
            if parent_dir.exists():
                matching_files = list(parent_dir.glob(f"{base_name}*.xlsx"))
                if matching_files:
                    excel_path = matching_files[0]  # Use first match
                    logger.debug(f"Found file with timestamp: {excel_path.name}")
                else:
                    logger.warning(f"Excel file not found: {excel_path} (searched {parent_dir})")
                    return pd.DataFrame()
            else:
                logger.warning(f"Directory not found: {parent_dir}")
                return pd.DataFrame()
        
        try:
            # Load "All Edges" sheet which has complete data
            df = pd.read_excel(excel_path, sheet_name='All Edges')
            
            # Standardize column names
            df.columns = df.columns.str.strip()
            
            # Add metadata columns
            df['experiment_id'] = metadata_row['experiment_id']
            df['prompt_variant'] = metadata_row['prompt']
            df['cld_name'] = metadata_row['cld_prefix']
            df['run_idx'] = metadata_row['run_idx']
            df['combo_number'] = metadata_row['combo_number']
            df['excel_file'] = str(excel_path.relative_to(self.results_base_dir))
            
            # Parse parameters JSON if available
            if 'parameters' in metadata_row and pd.notna(metadata_row['parameters']):
                try:
                    params = json.loads(metadata_row['parameters'])
                    df['corruption_rate'] = params.get('corruption_rate', np.nan)
                    df['generator_model'] = params.get('generator_config', {}).get('model', 'unknown')
                    df['generator_temperature'] = params.get('generator_temperature', np.nan)
                    df['judge_edges'] = params.get('judge_edges', False)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse parameters JSON for {excel_path.name}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {excel_path}: {e}")
            return pd.DataFrame()
    
    def load_experiment(self, experiment_id: str, max_files: Optional[int] = None) -> pd.DataFrame:
        """
        Load all results from a single experiment.
        
        Args:
            experiment_id: Experiment ID
            max_files: Optional limit on number of files to load (for testing)
        
        Returns:
            DataFrame with all edge data from this experiment
        """
        logger.info(f"Loading experiment: {experiment_id}")
        
        # Load metadata
        metadata = self.load_experiment_metadata(experiment_id)
        
        if max_files:
            metadata = metadata.head(max_files)
            logger.info(f"Limited to {max_files} files for testing")
        
        # Load each Excel file
        all_dfs = []
        for idx, row in metadata.iterrows():
            excel_path = self.results_base_dir / row['excel_filename']
            df = self.load_single_excel_file(excel_path, row)
            
            if not df.empty:
                all_dfs.append(df)
        
        if not all_dfs:
            logger.warning(f"No data loaded for experiment {experiment_id}")
            return pd.DataFrame()
        
        # Concatenate all dataframes
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} edges from {len(all_dfs)} files")
        
        return combined
    
    def load_multiple_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """
        Load results from multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to load
        
        Returns:
            Combined DataFrame with all edges from all experiments
        """
        all_dfs = []
        
        for exp_id in experiment_ids:
            try:
                df = self.load_experiment(exp_id)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to load experiment {exp_id}: {e}")
                continue
        
        if not all_dfs:
            logger.warning("No data loaded from any experiment")
            return pd.DataFrame()
        
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined {len(combined)} edges from {len(all_dfs)} experiments")
        
        return combined
    
    def load_all_experiments(self, max_experiments: Optional[int] = None) -> pd.DataFrame:
        """
        Load all available experiments.
        
        Args:
            max_experiments: Optional limit on number of experiments (for testing)
        
        Returns:
            DataFrame with all edges from all experiments
        """
        experiment_ids = self.find_all_experiments()
        
        if max_experiments:
            experiment_ids = experiment_ids[:max_experiments]
            logger.info(f"Limited to {max_experiments} experiments for testing")
        
        return self.load_multiple_experiments(experiment_ids)
    
    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data quality and clean invalid entries.
        
        This method:
        1. Standardizes column names
        2. Filters to edges with CI metrics (TP/FP only)
        3. Creates binary hallucination labels
        4. Handles missing values appropriately
        
        Args:
            df: Raw loaded DataFrame
        
        Returns:
            Cleaned DataFrame ready for analysis
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to validate_and_clean")
            return df
        
        logger.info(f"Validating and cleaning {len(df)} edges...")
        
        # Standardize column names (handle spaces, capitalization)
        column_mapping = {
            'Source': 'source',
            'Target': 'target',
            'Classification': 'classification',
            'In Session Graph': 'in_session_graph',
            'In Validation Graph': 'in_validation_graph',
            'Judge Verdict': 'judge_verdict',
            'Motivation': 'motivation',
            'Citation': 'citation',
            'Judge Message': 'judge_message',
            'Aggregate Score': 'aggregate_score',
            'Perplexity': 'perplexity',
            'Min Prob': 'min_prob',
            'Max Window Entropy': 'max_window_entropy',
            'Cosine Similarity': 'cosine_similarity',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Check which columns are present
        ci_metrics = ['perplexity', 'min_prob', 'max_window_entropy', 'cosine_similarity']
        missing_cols = [col for col in ci_metrics if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing CI metric columns: {missing_cols}")
        
        # Filter to edges with at least some CI metrics (TP/FP edges only)
        # FN/TN edges don't have CI metrics since they weren't discovered
        available_ci = [col for col in ci_metrics if col in df.columns]
        if available_ci:
            # Keep rows where at least one CI metric is non-null
            df_clean = df.dropna(subset=available_ci, how='all').copy()
            logger.info(f"Filtered to {len(df_clean)} edges with CI metrics (TP/FP only)")
        else:
            logger.warning("No CI metrics found in data!")
            df_clean = df.copy()
        
        # Create binary hallucination label
        df_clean['is_hallucination'] = self._create_hallucination_labels(df_clean)
        
        # Log summary statistics
        if 'is_hallucination' in df_clean.columns:
            n_halluc = df_clean['is_hallucination'].sum()
            n_total = len(df_clean)
            logger.info(f"Hallucinations: {n_halluc}/{n_total} ({n_halluc/n_total*100:.1f}%)")
        
        # Log classification breakdown
        if 'classification' in df_clean.columns:
            class_counts = df_clean['classification'].value_counts()
            logger.info(f"Classification counts:\\n{class_counts}")
        
        return df_clean
    
    def _create_hallucination_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary hallucination labels based on the configured mode.
        
        Mode: 'judge'
            - Uses aggregate_score < 0.5 (judge says edge is not supported)
            - Fallback to judge_verdict text matching
            
        Mode: 'ground_truth'
            - Uses Classification='FP' (edge generated but not in ground truth CLD)
            - These are true hallucinations: edges that should not exist
        
        Args:
            df: DataFrame with judge/classification columns
        
        Returns:
            Boolean Series indicating hallucination (True) or not (False)
        """
        
        if self.hallucination_mode == 'ground_truth':
            # Ground truth mode: FP = hallucination (edge not in ground truth CLD)
            if 'classification' not in df.columns:
                logger.error("Ground truth mode requires 'Classification' column, but it's missing!")
                return pd.Series([False] * len(df), index=df.index)
            
            labels = df['classification'] == 'FP'
            n_halluc = labels.sum()
            n_correct = (df['classification'] == 'TP').sum() if 'classification' in df.columns else 0
            logger.info(f"GROUND TRUTH MODE: FP (hallucination)={n_halluc}, TP (correct)={n_correct}")
            return labels
        
        elif self.hallucination_mode == 'judge':
            # Judge mode: Use aggregate_score or judge_verdict
            
            # Strategy 1: Use aggregate_score (preferred)
            if 'aggregate_score' in df.columns:
                # Edges with score < 0.5 are hallucinations
                # NaN values become False (no score = no hallucination detected)
                labels = df['aggregate_score'].fillna(1.0) < 0.5
                n_halluc = labels.sum()
                n_correct = (~labels).sum()
                logger.info(f"JUDGE MODE (aggregate_score < 0.5): hallucination={n_halluc}, correct={n_correct}")
                return labels
            
            # Strategy 2: Use judge_verdict text
            if 'judge_verdict' in df.columns:
                hallucination_verdicts = [
                    'not supported', 'not_supported', 'contradicted', 
                    'unaddressed', 'partially supported', 'partially_supported'
                ]
                labels = df['judge_verdict'].fillna('').str.lower().isin(hallucination_verdicts)
                logger.info("JUDGE MODE (judge_verdict text matching)")
                return labels
            
            # Fallback
            logger.warning("JUDGE MODE: No aggregate_score or judge_verdict column found!")
            return pd.Series([False] * len(df), index=df.index)
        
        else:
            # Should never reach here due to validation in __init__
            logger.error(f"Invalid hallucination_mode: {self.hallucination_mode}")
            return pd.Series([False] * len(df), index=df.index)
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for loaded data.
        
        Args:
            df: Loaded and cleaned DataFrame
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_edges': len(df),
            'n_experiments': df['experiment_id'].nunique() if 'experiment_id' in df.columns else 0,
            'n_prompts': df['prompt_variant'].nunique() if 'prompt_variant' in df.columns else 0,
            'n_clds': df['cld_name'].nunique() if 'cld_name' in df.columns else 0,
            'n_runs': len(df.groupby(['experiment_id', 'run_idx'])) if 'run_idx' in df.columns else 0,
        }
        
        # CI metrics availability
        ci_metrics = ['perplexity', 'min_prob', 'max_window_entropy', 'cosine_similarity']
        for metric in ci_metrics:
            if metric in df.columns:
                summary[f'{metric}_available'] = int(df[metric].notna().sum())
                summary[f'{metric}_mean'] = float(df[metric].mean())
                summary[f'{metric}_std'] = float(df[metric].std())
        
        # Hallucination statistics
        if 'is_hallucination' in df.columns:
            summary['n_hallucinations'] = int(df['is_hallucination'].sum())
            summary['hallucination_rate'] = float(df['is_hallucination'].mean())
        
        # Classification breakdown
        if 'classification' in df.columns:
            for class_label in ['TP', 'FP', 'FN', 'TN']:
                summary[f'n_{class_label}'] = int((df['classification'] == class_label).sum())
        
        return summary


# Convenience function for quick loading
def load_experiment_quick(experiment_id: str, results_dir: str = "results") -> pd.DataFrame:
    """
    Quick function to load and clean a single experiment.
    
    Args:
        experiment_id: Experiment ID to load
        results_dir: Results directory path
    
    Returns:
        Cleaned DataFrame ready for analysis
    """
    loader = RQ2DataLoader(results_dir)
    df = loader.load_experiment(experiment_id)
    df_clean = loader.validate_and_clean(df)
    return df_clean


# Example usage and testing
if __name__ == "__main__":
    # Example: Load a specific experiment
    loader = RQ2DataLoader()
    
    # Find all experiments
    experiments = loader.find_all_experiments()
    print(f"\\nFound {len(experiments)} experiments")
    print(f"Most recent: {experiments[:3]}")
    
    # Load one experiment
    if experiments:
        exp_id = experiments[0]
        print(f"\\nLoading experiment: {exp_id}")
        
        df = loader.load_experiment(exp_id, max_files=3)  # Limit for testing
        print(f"Loaded {len(df)} edges")
        print(f"Columns: {df.columns.tolist()}")
        
        # Clean and validate
        df_clean = loader.validate_and_clean(df)
        print(f"\\nAfter cleaning: {len(df_clean)} edges")
        
        # Get summary
        summary = loader.get_data_summary(df_clean)
        print(f"\\nSummary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Show sample
        print(f"\\nSample data:")
        print(df_clean[['source', 'target', 'classification', 'perplexity', 
                       'aggregate_score', 'is_hallucination']].head())
"""
RQ1 Data Loader

Loads and prepares experiment data for RQ1 analysis:
- Can LLM-as-a-judge detect hallucinations induced by the corruptor?
- Comparison between serial and parallel judging strategies

This module extracts edge-level judge verdicts and compares them against 
ground truth (is_corrupted flag) to compute TP/FP/FN/TN classifications.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RQ1DataLoader:
    """
    Load and prepare judge performance data for RQ1 analysis.
    
    RQ1 focuses on whether LLM-as-a-judge can detect hallucinations 
    (corrupted edges) introduced by the corruptor agent.
    """
    
    def __init__(self, results_folder: str = None):
        """
        Initialize data loader.
        
        Args:
            results_folder: Path to results folder containing experiment outputs.
                           If None, uses default location.
        """
        if results_folder is None:
            # Default to parameter_tuning_experiments/results
            self.results_folder = Path(__file__).parent.parent / 'results'
        else:
            self.results_folder = Path(results_folder)
        
        logger.info(f"RQ1 Data Loader initialized with results folder: {self.results_folder}")
    
    def load_experiment_data(
        self, 
        experiment_id: str,
        combo_number: Optional[int] = None,
        prompt_set: Optional[str] = None,
        cld_name: Optional[str] = None,
        run_number: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load edge data from a specific experiment.
        
        Args:
            experiment_id: Experiment ID (e.g., "exp_20251008_180446_a0d1b2de")
            combo_number: Optional parameter combo number to filter
            prompt_set: Optional prompt set name to filter
            cld_name: Optional CLD name to filter
            run_number: Optional run number to filter (e.g., "run1", "run2")
            
        Returns:
            DataFrame with edge-level judge performance data
        """
        exp_folder = self.results_folder / experiment_id
        
        if not exp_folder.exists():
            raise FileNotFoundError(f"Experiment folder not found: {exp_folder}")
        
        # Find all Excel files matching the criteria
        excel_files = []
        
        # Navigate folder structure: exp_id/combo_X/prompt_set/cld_name/runX/*.xlsx
        for combo_dir in exp_folder.glob("combo_*"):
            if combo_number is not None and combo_dir.name != f"combo_{combo_number}":
                continue
            
            for prompt_dir in combo_dir.iterdir():
                if not prompt_dir.is_dir():
                    continue
                if prompt_set is not None and prompt_dir.name != prompt_set:
                    continue
                
                for cld_dir in prompt_dir.iterdir():
                    if not cld_dir.is_dir():
                        continue
                    if cld_name is not None and not cld_name in cld_dir.name:
                        continue
                    
                    for run_dir in cld_dir.glob("run*"):
                        if run_number is not None and run_dir.name != run_number:
                            continue
                        
                        # Find Excel file in this run directory
                        for excel_file in run_dir.glob("results_*.xlsx"):
                            excel_files.append({
                                'path': excel_file,
                                'experiment_id': experiment_id,
                                'combo': combo_dir.name,
                                'prompt_set': prompt_dir.name,
                                'cld_name': cld_dir.name,
                                'run': run_dir.name
                            })
        
        if not excel_files:
            raise FileNotFoundError(f"No Excel files found for experiment {experiment_id}")
        
        logger.info(f"Found {len(excel_files)} Excel files for experiment {experiment_id}")
        
        # Load all Excel files and combine
        all_edges = []
        
        for file_info in excel_files:
            try:
                df = pd.read_excel(file_info['path'], sheet_name='Edges')
                
                # Add metadata
                df['experiment_id'] = file_info['experiment_id']
                df['combo'] = file_info['combo']
                df['prompt_set'] = file_info['prompt_set']
                df['cld_name'] = file_info['cld_name']
                df['run'] = file_info['run']
                df['excel_file'] = str(file_info['path'])
                
                all_edges.append(df)
                logger.info(f"Loaded {len(df)} edges from {file_info['run']} ({file_info['combo']}/{file_info['prompt_set']})")
            
            except Exception as e:
                logger.warning(f"Failed to load {file_info['path']}: {e}")
                continue
        
        if not all_edges:
            raise ValueError("No edges loaded successfully")
        
        # Combine all dataframes
        df_combined = pd.concat(all_edges, ignore_index=True)
        logger.info(f"Total edges loaded: {len(df_combined)}")
        
        return df_combined
    
    def prepare_rq1_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for RQ1 analysis.
        
        RQ1 asks: Can LLM-as-a-judge detect hallucinations (corrupted edges)?
        
        Ground truth: is_corrupted column (True = hallucination, False = genuine)
        Judge prediction: aggregate_verdict or aggregate_score
        
        Args:
            df: Raw edge data from experiments
            
        Returns:
            DataFrame with prepared RQ1 analysis columns
        """
        df_prepared = df.copy()
        
        # Ensure required columns exist
        required_cols = ['is_corrupted']
        missing_cols = [col for col in required_cols if col not in df_prepared.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ground truth: is_corrupted indicates hallucination
        df_prepared['is_hallucination'] = df_prepared['is_corrupted'].fillna(False)
        
        # Judge prediction: multiple possible formats
        # 1. aggregate_verdict: "Fully supported" / "Partially supported" / "Not supported"
        # 2. aggregate_score: numeric score (typically 0-1)
        # 3. judge_verdict: individual judge verdict
        
        if 'aggregate_score' in df_prepared.columns:
            # Use aggregate score (higher = more supported = less likely hallucination)
            # Invert score so higher = more likely hallucination
            df_prepared['judge_hallucination_score'] = 1.0 - df_prepared['aggregate_score'].fillna(0.5)
        elif 'aggregate_verdict' in df_prepared.columns:
            # Convert verdict to numeric score
            verdict_map = {
                'Fully supported': 0.0,
                'Partially supported': 0.5,
                'Not supported': 1.0,
                'FULLY_SUPPORTED': 0.0,
                'PARTIALLY_SUPPORTED': 0.5,
                'NOT_SUPPORTED': 1.0,
                'Supported': 0.0,
                'Contradicted': 1.0,
                'Unaddressed': 1.0
            }
            df_prepared['judge_hallucination_score'] = df_prepared['aggregate_verdict'].map(verdict_map).fillna(0.5)
        else:
            logger.warning("No aggregate_score or aggregate_verdict found, using default score")
            df_prepared['judge_hallucination_score'] = 0.5
        
        # Binary judge prediction using threshold (default 0.5)
        threshold = 0.5
        df_prepared['judge_predicts_hallucination'] = df_prepared['judge_hallucination_score'] >= threshold
        
        # Compute TP/FP/FN/TN classification
        df_prepared['classification'] = self._classify_edges(
            df_prepared['is_hallucination'],
            df_prepared['judge_predicts_hallucination']
        )
        
        # Add judging strategy metadata if available
        if 'num_judges' in df_prepared.columns:
            df_prepared['judging_strategy'] = df_prepared['num_judges'].apply(
                lambda x: 'serial' if x == 1 else f'parallel_{x}_judges'
            )
        else:
            df_prepared['judging_strategy'] = 'unknown'
        
        logger.info(f"Prepared {len(df_prepared)} edges for RQ1 analysis")
        logger.info(f"Hallucination rate: {df_prepared['is_hallucination'].mean():.2%}")
        logger.info(f"Judge detection rate: {df_prepared['judge_predicts_hallucination'].mean():.2%}")
        
        return df_prepared
    
    def _classify_edges(self, ground_truth: pd.Series, predictions: pd.Series) -> pd.Series:
        """
        Classify edges into TP/FP/FN/TN.
        
        In RQ1 context:
        - Positive class = Hallucination (corrupted edge)
        - Negative class = Genuine (non-corrupted edge)
        
        Args:
            ground_truth: True labels (is_hallucination)
            predictions: Predicted labels (judge_predicts_hallucination)
            
        Returns:
            Series with classification labels
        """
        classifications = []
        
        for gt, pred in zip(ground_truth, predictions):
            if gt and pred:
                classifications.append('TP')  # Correctly detected hallucination
            elif gt and not pred:
                classifications.append('FN')  # Missed hallucination
            elif not gt and pred:
                classifications.append('FP')  # False alarm
            else:
                classifications.append('TN')  # Correctly accepted genuine edge
        
        return pd.Series(classifications, index=ground_truth.index)
    
    def export_prepared_data(self, df: pd.DataFrame, output_dir: str, experiment_id: str):
        """
        Export prepared data for downstream analysis.
        
        Args:
            df: Prepared DataFrame
            output_dir: Output directory path
            experiment_id: Experiment identifier for filename
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        csv_path = output_path / f'rq1_prepared_data_{experiment_id}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported prepared data to: {csv_path}")
        
        # Export summary statistics
        summary = {
            'total_edges': len(df),
            'n_hallucinations': int(df['is_hallucination'].sum()),
            'n_genuine': int((~df['is_hallucination']).sum()),
            'hallucination_rate': float(df['is_hallucination'].mean()),
            'judge_detection_rate': float(df['judge_predicts_hallucination'].mean()),
            'classification_counts': df['classification'].value_counts().to_dict()
        }
        
        summary_path = output_path / f'rq1_data_summary_{experiment_id}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Exported summary to: {summary_path}")
        
        return csv_path, summary_path


def main():
    """Example usage of RQ1 data loader."""
    
    # Example: Load data from a specific experiment
    loader = RQ1DataLoader()
    
    # You would specify your experiment ID here
    experiment_id = "exp_20251008_180446_a0d1b2de"
    
    try:
        # Load experiment data
        df_raw = loader.load_experiment_data(
            experiment_id=experiment_id,
            # combo_number=1,
            # prompt_set="prompts_Nitai_C",
            # cld_name="Depressive_symptoms"
        )
        
        # Prepare for RQ1 analysis
        df_prepared = loader.prepare_rq1_data(df_raw)
        
        # Print summary
        print("\n" + "="*80)
        print("RQ1 DATA SUMMARY")
        print("="*80)
        print(f"\nTotal edges: {len(df_prepared)}")
        print(f"Hallucinations: {df_prepared['is_hallucination'].sum()} ({df_prepared['is_hallucination'].mean():.1%})")
        print(f"Genuine edges: {(~df_prepared['is_hallucination']).sum()} ({(~df_prepared['is_hallucination']).mean():.1%})")
        print(f"\nClassification breakdown:")
        print(df_prepared['classification'].value_counts())
        
        # Export data
        loader.export_prepared_data(df_prepared, "rq1_analyses/test", experiment_id)
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


if __name__ == "__main__":
    main()

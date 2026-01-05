#!/usr/bin/env python3
"""
Node Ablation Data Loader

Extracts node comparison metrics from experiment result files for ablation analysis.
Similar structure to RQ2 data loader but focused on node generation performance.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
import re


class NodeAblationDataLoader:
    """
    Loads and consolidates node comparison metrics from multiple experiment runs.
    
    Extracts:
    - Hybrid F1, precision, recall (binary + similarity-based)
    - Prompt variant information
    - CLD information
    - Run metadata
    """
    
    def __init__(self, experiment_dir: Path):
        """
        Initialize data loader.
        
        Args:
            experiment_dir: Path to experiment directory (e.g., exp_20251009_163247_a49a01a2/)
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_id = self.experiment_dir.name
        
        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {self.experiment_dir}")
    
    def find_result_files(self) -> List[Path]:
        """Find all Excel result files in the experiment directory."""
        # Pattern: exp_dir/combo_N/prompts_*/CLD_name/runN_hash/results_*.xlsx
        result_files = list(self.experiment_dir.glob("**/results_*.xlsx"))
        return sorted(result_files)
    
    def extract_metadata_from_path(self, file_path: Path) -> Dict:
        """
        Extract metadata from file path.
        
        Args:
            file_path: Path to result Excel file
            
        Returns:
            Dictionary with metadata (prompt, cld, run, combo)
        """
        parts = file_path.parts
        
        # Find combo number
        combo = None
        prompt = None
        cld = None
        run = None
        
        for i, part in enumerate(parts):
            if part.startswith('combo_'):
                combo = int(part.split('_')[1])
            elif part.startswith('prompts_'):
                prompt = part.replace('prompts_', '')
            elif part.startswith('run'):
                run = int(re.search(r'run(\d+)', part).group(1))
                if i > 0:
                    cld = parts[i-1]  # CLD is the parent directory of run
        
        return {
            'combo': combo,
            'prompt': prompt,
            'cld_name': cld,
            'run_number': run,
            'file_path': str(file_path)
        }
    
    def load_node_metrics_from_excel(self, excel_file: Path) -> Optional[Dict]:
        """
        Load node comparison metrics from Excel file.
        
        Args:
            excel_file: Path to results Excel file
            
        Returns:
            Dictionary with node comparison metrics, or None if not found
        """
        try:
            # Check if Node Comparison Summary sheet exists
            xl = pd.ExcelFile(excel_file)
            if 'Node Comparison Summary' not in xl.sheet_names:
                return None
            
            # Read Node Comparison Summary sheet
            df = pd.read_excel(excel_file, sheet_name='Node Comparison Summary')
            
            # Extract metrics
            metrics = {}
            
            for _, row in df.iterrows():
                metric_name = row['Metric']
                value = row['Value']
                
                # Skip section headers (NaN values)
                if pd.isna(value):
                    continue
                
                # Map metric names to our standard names
                metric_map = {
                    'Total Generated Nodes': 'n_generated_nodes',
                    'Total Validation Nodes': 'n_validation_nodes',
                    'Matched Pairs (Binary)': 'n_binary_matches',
                    'Node Precision (Binary)': 'binary_precision',
                    'Node Recall (Binary)': 'binary_recall',
                    'Node F1 Score (Binary)': 'binary_f1',
                    'Cosine Node Precision': 'cosine_precision',
                    'Cosine Node Recall': 'cosine_recall',
                    'Cosine Node F1 Score': 'cosine_f1',
                    'Avg Cosine (Gen→Val)': 'avg_cosine_gen_to_val',
                    'Avg Cosine (Val→Gen)': 'avg_cosine_val_to_gen',
                    'Hybrid Node Precision': 'hybrid_precision',
                    'Hybrid Node Recall': 'hybrid_recall',
                    'Hybrid Node F1 Score': 'hybrid_f1'
                }
                
                if metric_name in metric_map:
                    metrics[metric_map[metric_name]] = float(value)
            
            return metrics if metrics else None
            
        except Exception as e:
            print(f"Warning: Could not load metrics from {excel_file.name}: {e}")
            return None
    
    def load_all_metrics(self) -> pd.DataFrame:
        """
        Load node comparison metrics from all result files.
        
        Returns:
            DataFrame with all metrics and metadata
        """
        result_files = self.find_result_files()
        
        if not result_files:
            raise ValueError(f"No result files found in {self.experiment_dir}")
        
        print(f"Found {len(result_files)} result files in {self.experiment_id}")
        
        all_data = []
        
        for file_path in result_files:
            # Extract metadata
            metadata = self.extract_metadata_from_path(file_path)
            
            # Load metrics
            metrics = self.load_node_metrics_from_excel(file_path)
            
            if metrics is None:
                print(f"  Warning: No node metrics in {file_path.name}")
                continue
            
            # Combine metadata and metrics
            record = {**metadata, **metrics}
            record['experiment_id'] = self.experiment_id
            all_data.append(record)
        
        if not all_data:
            raise ValueError("No valid node metrics found in any result files")
        
        df = pd.DataFrame(all_data)
        
        print(f"\\nLoaded metrics from {len(df)} files:")
        print(f"  • Prompts: {df['prompt'].nunique()} ({', '.join(df['prompt'].unique())})")
        print(f"  • CLDs: {df['cld_name'].nunique()} ({', '.join(df['cld_name'].unique())})")
        print(f"  • Runs per prompt-CLD combo: {df.groupby(['prompt', 'cld_name']).size().values}")
        
        return df
    
    def save_data(self, df: pd.DataFrame, output_dir: Path):
        """
        Save extracted data to CSV.
        
        Args:
            df: DataFrame with all metrics
            output_dir: Directory to save output
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'node_ablation_data_{self.experiment_id}.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\\n✓ Saved data to: {output_file}")
        
        return output_file


def main():
    """CLI for data loader."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load node ablation data from experiment directory"
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to experiment directory (e.g., parameter_tuning_experiments/results/exp_*/)"
    )
    parser.add_argument(
        "--output-dir",
        default="parameter_tuning_experiments/node_ablation_analyses",
        help="Output directory for extracted data"
    )
    
    args = parser.parse_args()
    
    # Load data
    loader = NodeAblationDataLoader(args.experiment_dir)
    df = loader.load_all_metrics()
    
    # Save data
    output_file = loader.save_data(df, args.output_dir)
    
    print(f"\\n{'='*80}")
    print("DATA LOADING COMPLETE")
    print(f"{'='*80}")
    print(f"\\nExtracted {len(df)} records")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()

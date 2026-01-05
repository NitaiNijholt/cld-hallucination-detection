"""
Sensitivity Analysis Config Generator

Generates experiment configuration YAMLs from Sobol samples.
Integrates with existing multirunner infrastructure.

Architecture:
- Takes Sobol samples (from sobol_analyzer.py)
- Generates individual YAML configs for each sample
- Each config has param_grid with single values
- Compatible with existing multirun_parameter_experiments()

This allows Sobol sampling to work seamlessly with the existing
experiment runner without modifying any core infrastructure.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SensitivityConfigGenerator:
    """
    Generate experiment configs from Sobol samples.
    
    Workflow:
    1. Load Sobol samples CSV
    2. Load base experiment config (template)
    3. For each sample, create a config YAML with single-value param_grid
    4. Organize configs by batch for easier management
    """
    
    def __init__(
        self,
        base_config_path: str,
        output_dir: str = "parameter_tuning_experiments/configs/sobol_generated",
        batch_size: int = 100
    ):
        """
        Initialize config generator.
        
        Args:
            base_config_path: Path to base experiment config (template)
            output_dir: Directory to save generated configs
            batch_size: Number of configs per batch directory
        """
        self.base_config_path = Path(base_config_path)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        
        # Load base config
        with open(self.base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        logger.info(f"Initialized config generator")
        logger.info(f"  Base config: {self.base_config_path}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Batch size: {self.batch_size}")
    
    def generate_configs(
        self,
        samples_df: pd.DataFrame,
        rq_name: str = "rq2",
        param_mapping: Optional[Dict[str, str]] = None,
        fixed_params: Optional[Dict[str, Any]] = None
    ) -> List[Path]:
        """
        Generate experiment configs from Sobol samples.
        
        Args:
            samples_df: DataFrame with Sobol samples (from SobolAnalyzer.generate_samples)
            rq_name: Research question name (for organizing outputs)
            param_mapping: Optional mapping from sample param names to config param names
                          Format: {sample_param: config_param}
            fixed_params: Optional dict of fixed parameters to include in all configs
        
        Returns:
            List of paths to generated config files
        """
        logger.info(f"Generating configs for {len(samples_df)} samples")
        
        # Create output directory
        rq_output_dir = self.output_dir / rq_name
        rq_output_dir.mkdir(parents=True, exist_ok=True)
        
        config_paths = []
        
        # Generate configs in batches
        n_samples = len(samples_df)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, n_samples)
            batch_df = samples_df.iloc[batch_start:batch_end]
            
            # Create batch directory
            batch_dir = rq_output_dir / f"batch_{batch_idx + 1:03d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"  Batch {batch_idx + 1}/{n_batches}: samples {batch_start}-{batch_end-1}")
            
            # Generate configs for this batch
            for _, row in batch_df.iterrows():
                sample_id = row['sample_id']
                config_path = self._generate_single_config(
                    row,
                    batch_dir,
                    sample_id,
                    param_mapping,
                    fixed_params
                )
                config_paths.append(config_path)
        
        logger.info(f"Generated {len(config_paths)} config files")
        
        # Create manifest file
        manifest_path = rq_output_dir / "config_manifest.csv"
        self._create_manifest(samples_df, config_paths, manifest_path)
        
        return config_paths
    
    def _generate_single_config(
        self,
        sample_row: pd.Series,
        output_dir: Path,
        sample_id: int,
        param_mapping: Optional[Dict[str, str]],
        fixed_params: Optional[Dict[str, Any]]
    ) -> Path:
        """Generate a single config file for one Sobol sample."""
        
        # Create config from base template
        config = self.base_config.copy()
        
        # Create param_grid with single values from sample
        param_grid = {}
        
        # Add sampled parameters
        for param_name, param_value in sample_row.items():
            if param_name == 'sample_id':
                continue
            
            # Skip categorical index columns
            if param_name.endswith('_index'):
                continue
            
            # Map parameter name if mapping provided
            if param_mapping and param_name in param_mapping:
                config_param_name = param_mapping[param_name]
            else:
                config_param_name = param_name
            
            # Convert numpy types to Python native types
            if hasattr(param_value, 'item'):
                param_value = param_value.item()
            
            # Handle categorical parameters
            if isinstance(param_value, str):
                # Categorical value - keep as string
                param_grid[config_param_name] = [param_value]
            elif isinstance(param_value, (int, np.integer)):
                param_grid[config_param_name] = [int(param_value)]
            elif isinstance(param_value, (float, np.floating)):
                param_grid[config_param_name] = [float(param_value)]
            else:
                param_grid[config_param_name] = [param_value]
        
        # Add fixed parameters
        if fixed_params:
            for param_name, param_value in fixed_params.items():
                if isinstance(param_value, list):
                    param_grid[param_name] = param_value
                else:
                    param_grid[param_name] = [param_value]
        
        # Update config
        config['param_grid'] = param_grid
        config['runs'] = 1  # Single run per Sobol sample
        
        # Add metadata
        config['sobol_metadata'] = {
            'sample_id': int(sample_id),
            'generated_from': str(self.base_config_path),
            'sensitivity_analysis': True
        }
        
        # Save config
        config_filename = f"sobol_config_{sample_id:04d}.yaml"
        config_path = output_dir / config_filename
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return config_path
    
    def _create_manifest(
        self,
        samples_df: pd.DataFrame,
        config_paths: List[Path],
        manifest_path: Path
    ):
        """Create a manifest CSV mapping sample_ids to config files."""
        
        manifest_data = []
        for (_, row), config_path in zip(samples_df.iterrows(), config_paths):
            manifest_data.append({
                'sample_id': row['sample_id'],
                'config_path': str(config_path.relative_to(self.output_dir.parent)),
                **{col: row[col] for col in samples_df.columns if col != 'sample_id'}
            })
        
        manifest_df = pd.DataFrame(manifest_data)
        manifest_df.to_csv(manifest_path, index=False)
        logger.info(f"Created manifest: {manifest_path}")
    
    def generate_run_script(
        self,
        config_paths: List[Path],
        output_script_path: str,
        parallel: int = 1,
        force: bool = True
    ):
        """
        Generate a bash script to run all generated configs.
        
        Args:
            config_paths: List of paths to config files
            output_script_path: Path to save bash script
            parallel: Number of parallel jobs (for GNU parallel)
            force: Whether to use --force flag
        """
        script_lines = [
            "#!/bin/bash",
            "#",
            "# Auto-generated script to run Sobol sensitivity analysis experiments",
            f"# Total configs: {len(config_paths)}",
            "#",
            "",
            "set -e",  # Exit on error
            "",
            "# Change to data_science directory",
            "cd \"$(dirname \"$0\")/../..\"",
            "",
            "echo 'Starting Sobol sensitivity analysis experiments'",
            f"echo 'Total configs: {len(config_paths)}'",
            "",
        ]
        
        if parallel > 1:
            # Use GNU parallel
            script_lines.extend([
                "# Check if GNU parallel is installed",
                "if ! command -v parallel &> /dev/null; then",
                "    echo 'GNU parallel is not installed'",
                "    echo 'Install with: sudo apt-get install parallel'",
                "    exit 1",
                "fi",
                "",
                f"# Run experiments in parallel (j={parallel})",
                "cat << 'EOF' | parallel -j " + str(parallel) + " --eta",
            ])
            
            # Add config paths
            for config_path in config_paths:
                rel_path = config_path.relative_to(Path.cwd())
                force_flag = " --force" if force else ""
                script_lines.append(f"python main_eval_multi_run.py --config {rel_path}{force_flag}")
            
            script_lines.extend([
                "EOF",
                "",
            ])
        else:
            # Sequential execution
            script_lines.append("# Run experiments sequentially")
            for i, config_path in enumerate(config_paths, 1):
                rel_path = config_path.relative_to(Path.cwd())
                force_flag = " --force" if force else ""
                script_lines.extend([
                    f"echo 'Running config {i}/{len(config_paths)}'",
                    f"python main_eval_multi_run.py --config {rel_path}{force_flag}",
                    ""
                ])
        
        script_lines.extend([
            "echo 'All experiments completed!'",
            ""
        ])
        
        # Write script
        script_path = Path(output_script_path)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write('\n'.join(script_lines))
        
        # Make executable
        script_path.chmod(0o755)
        
        logger.info(f"Generated run script: {script_path}")
        logger.info(f"  Parallel jobs: {parallel}")
        logger.info(f"  Execute with: bash {script_path}")


def generate_configs_from_samples(
    samples_csv: str,
    base_config: str,
    output_dir: str,
    rq_name: str = "rq2",
    param_mapping: Optional[Dict[str, str]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    generate_script: bool = True,
    parallel: int = 1
) -> List[Path]:
    """
    Convenience function to generate configs from a samples CSV file.
    
    Args:
        samples_csv: Path to Sobol samples CSV (from SobolAnalyzer.generate_samples)
        base_config: Path to base experiment config
        output_dir: Directory for generated configs
        rq_name: Research question name
        param_mapping: Parameter name mapping
        fixed_params: Fixed parameters for all configs
        generate_script: Whether to generate run script
        parallel: Number of parallel jobs for run script
    
    Returns:
        List of paths to generated config files
    """
    # Load samples
    samples_df = pd.read_csv(samples_csv)
    logger.info(f"Loaded {len(samples_df)} samples from {samples_csv}")
    
    # Generate configs
    generator = SensitivityConfigGenerator(
        base_config_path=base_config,
        output_dir=output_dir
    )
    
    config_paths = generator.generate_configs(
        samples_df=samples_df,
        rq_name=rq_name,
        param_mapping=param_mapping,
        fixed_params=fixed_params
    )
    
    # Generate run script
    if generate_script:
        script_path = Path(output_dir) / rq_name / "run_sobol_experiments.sh"
        generator.generate_run_script(
            config_paths=config_paths,
            output_script_path=str(script_path),
            parallel=parallel,
            force=True
        )
    
    return config_paths


# Importing numpy for type checking
import numpy as np






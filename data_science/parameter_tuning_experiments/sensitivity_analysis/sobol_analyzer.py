"""
Sobol Sensitivity Analysis - Core Analysis Module

Uses SALib to perform global sensitivity analysis with Sobol indices.
This is the MODEL/ANALYSIS layer - handles only analysis logic.

Architecture follows RQ2 pattern:
- sobol_analyzer.py: Core analysis logic (THIS FILE)
- sensitivity_data_loader.py: Data loading and preprocessing
- sensitivity_visualizer.py: Visualization
- sensitivity_config_generator.py: Config generation
- sensitivity_master_pipeline.py: Orchestration

References:
- SALib documentation: https://salib.readthedocs.io/
- Saltelli et al. (2010) "Variance based sensitivity analysis"
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml
import logging
from dataclasses import dataclass

# SALib imports
from SALib.sample import saltelli
from SALib.analyze import sobol

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ParameterDefinition:
    """Definition of a parameter for sensitivity analysis."""
    name: str
    bounds: Tuple[float, float]
    param_type: str  # 'continuous', 'discrete', 'categorical'
    values: Optional[List[Any]] = None  # For discrete/categorical
    description: Optional[str] = None


class SobolAnalyzer:
    """
    Sobol sensitivity analysis using SALib.
    
    Responsibilities (MODEL/ANALYSIS only):
    - Load and validate problem definitions
    - Generate Sobol samples using saltelli sampling
    - Compute Sobol indices (S1, ST, S2)
    - Interpret sensitivity results
    - Compute convergence metrics
    
    Does NOT handle:
    - Visualization (see sensitivity_visualizer.py)
    - Data loading from experiments (see sensitivity_data_loader.py)
    - Config generation (see sensitivity_config_generator.py)
    - Pipeline orchestration (see sensitivity_master_pipeline.py)
    
    Workflow:
    1. Initialize with problem definition (parameters, bounds, types)
    2. Generate Sobol samples: N * (2D + 2) samples
    3. Run experiments with generated samples (external)
    4. Load experimental outcomes
    5. Compute Sobol indices
    6. Interpret and save results
    """
    
    def __init__(
        self,
        problem_definition: Union[str, Dict],
        output_dir: str = "parameter_tuning_experiments/sensitivity_analysis/results",
        calc_second_order: bool = True
    ):
        """
        Initialize Sobol analyzer.
        
        Args:
            problem_definition: Path to problem YAML or dict with problem definition
                Expected format:
                {
                    'problem_name': str,
                    'parameters': [
                        {
                            'name': str,
                            'bounds': [min, max],
                            'type': 'continuous'|'discrete'|'categorical',
                            'values': [...] (optional, for discrete/categorical),
                            'description': str (optional)
                        },
                        ...
                    ]
                }
            output_dir: Directory for analysis outputs
            calc_second_order: Whether to calculate second-order indices (pairwise interactions)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calc_second_order = calc_second_order
        
        # Load problem definition
        if isinstance(problem_definition, str):
            with open(problem_definition, 'r') as f:
                problem_dict = yaml.safe_load(f)
        else:
            problem_dict = problem_definition
        
        self.problem_name = problem_dict.get('problem_name', 'sensitivity_analysis')
        self.problem = self._create_salib_problem(problem_dict)
        self.param_definitions = self._parse_parameter_definitions(problem_dict)
        
        logger.info(f"Initialized Sobol analyzer: {self.problem_name}")
        logger.info(f"Parameters ({self.problem['num_vars']}): {self.problem['names']}")
        logger.info(f"Second-order indices: {'Enabled' if calc_second_order else 'Disabled'}")
    
    def _create_salib_problem(self, problem_dict: Dict) -> Dict:
        """
        Convert problem definition to SALib format.
        
        SALib expects:
        {
            'num_vars': int,
            'names': List[str],
            'bounds': List[Tuple[float, float]]
        }
        """
        parameters = problem_dict['parameters']
        
        problem = {
            'num_vars': len(parameters),
            'names': [p['name'] for p in parameters],
            'bounds': [p['bounds'] for p in parameters]
        }
        
        # Validate
        if problem['num_vars'] == 0:
            raise ValueError("Problem definition must have at least one parameter")
        
        if len(problem['names']) != len(set(problem['names'])):
            raise ValueError("Parameter names must be unique")
        
        return problem
    
    def _parse_parameter_definitions(self, problem_dict: Dict) -> Dict[str, ParameterDefinition]:
        """Parse parameter definitions with type information."""
        definitions = {}
        for param in problem_dict['parameters']:
            definitions[param['name']] = ParameterDefinition(
                name=param['name'],
                bounds=tuple(param['bounds']),
                param_type=param.get('type', 'continuous'),
                values=param.get('values', None),
                description=param.get('description', None)
            )
        return definitions
    
    def generate_samples(
        self,
        n_samples: int = 128,
        seed: Optional[int] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Generate Sobol samples using saltelli sampling.
        
        Saltelli sampling generates N * (2D + 2) samples where:
        - N = base sample size
        - D = number of parameters
        - Total samples = N * (2D + 2)
        
        For example: N=128, D=4 → 128 * 10 = 1,280 samples
        
        Args:
            n_samples: Number of base samples (N)
            seed: Random seed for reproducibility
            save: Whether to save samples to CSV
        
        Returns:
            DataFrame with Sobol samples (one row per sample)
            Columns: ['sample_id', param1, param2, ..., paramN]
        """
        logger.info(f"Generating Sobol samples with N={n_samples}")
        logger.info(f"Expected total samples: {n_samples} * (2*{self.problem['num_vars']} + 2) = "
                   f"{n_samples * (2 * self.problem['num_vars'] + 2)}")
        
        # Generate samples using SALib saltelli sampling
        param_values = saltelli.sample(
            self.problem,
            n_samples,
            calc_second_order=self.calc_second_order,
            seed=seed
        )
        
        total_samples = param_values.shape[0]
        logger.info(f"Generated {total_samples} total samples")
        
        # Convert to DataFrame
        df = pd.DataFrame(param_values, columns=self.problem['names'])
        
        # Process parameters according to their types
        df = self._process_parameter_types(df)
        
        # Add sample index
        df.insert(0, 'sample_id', range(len(df)))
        
        # Save samples
        if save:
            output_file = self.output_dir / f"sobol_samples_n{n_samples}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved samples to {output_file}")
        
        return df
    
    def _process_parameter_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process parameters according to their types.
        
        - Continuous: Keep as-is
        - Discrete: Round to nearest integer, map to valid values
        - Categorical: Map to categorical values
        """
        df = df.copy()
        
        for param_name, param_def in self.param_definitions.items():
            if param_def.param_type == 'discrete':
                # Round to nearest integer
                df[param_name] = df[param_name].round().astype(int)
                
                # Map to valid values if provided
                if param_def.values:
                    valid_values = sorted(param_def.values)
                    df[param_name] = df[param_name].clip(
                        lower=valid_values[0],
                        upper=valid_values[-1]
                    )
                    # Map to nearest valid value
                    df[param_name] = df[param_name].apply(
                        lambda x: min(valid_values, key=lambda v: abs(v - x))
                    )
            
            elif param_def.param_type == 'categorical':
                # Map continuous [0, n-1] to categorical values
                if param_def.values:
                    n_categories = len(param_def.values)
                    # Round and clip to valid indices
                    indices = df[param_name].round().astype(int).clip(0, n_categories - 1)
                    # Store original index for reference
                    df[param_name + '_index'] = indices
                    # Map to actual categorical values
                    df[param_name] = indices.map(lambda i: param_def.values[i])
        
        return df
    
    def analyze(
        self,
        samples_df: pd.DataFrame,
        outcomes: Union[np.ndarray, pd.Series, pd.DataFrame],
        outcome_name: str = "outcome",
        print_results: bool = True,
        save: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute Sobol indices from experimental outcomes.
        
        Args:
            samples_df: DataFrame with Sobol samples (from generate_samples)
                        Must have same number of rows as outcomes
            outcomes: Array/Series/DataFrame of experimental outcomes (Y values)
                     If DataFrame, uses first column
            outcome_name: Name of the outcome variable
            print_results: Whether to print results to console
            save: Whether to save results to CSV
        
        Returns:
            Dictionary with DataFrames:
            - 'S1': First-order indices (direct effects)
            - 'S1_conf': Confidence intervals for S1
            - 'ST': Total-effect indices (direct + interaction effects)
            - 'ST_conf': Confidence intervals for ST
            - 'S2': Second-order indices (pairwise interactions) [if calc_second_order=True]
            - 'S2_conf': Confidence intervals for S2 [if calc_second_order=True]
        """
        logger.info(f"Computing Sobol indices for outcome: {outcome_name}")
        
        # Handle DataFrame input
        if isinstance(outcomes, pd.DataFrame):
            if len(outcomes.columns) > 1:
                logger.warning(f"Multiple outcome columns detected. Using first: {outcomes.columns[0]}")
            outcomes = outcomes.iloc[:, 0]
        
        # Convert to numpy array
        if isinstance(outcomes, pd.Series):
            outcomes = outcomes.values
        
        # Validate
        if len(samples_df) != len(outcomes):
            raise ValueError(
                f"Sample count mismatch: {len(samples_df)} samples "
                f"but {len(outcomes)} outcomes"
            )
        
        # Check for NaN/inf in outcomes
        if np.any(np.isnan(outcomes)):
            n_nan = np.sum(np.isnan(outcomes))
            raise ValueError(f"Outcomes contain {n_nan} NaN values. Cannot compute Sobol indices.")
        
        if np.any(np.isinf(outcomes)):
            n_inf = np.sum(np.isinf(outcomes))
            raise ValueError(f"Outcomes contain {n_inf} infinite values. Cannot compute Sobol indices.")
        
        # Compute Sobol indices using SALib
        logger.info("Running SALib sobol.analyze()...")
        Si = sobol.analyze(
            self.problem,
            outcomes,
            calc_second_order=self.calc_second_order,
            print_to_console=print_results
        )
        
        # Convert to DataFrames for easier handling
        results = self._format_sobol_results(Si)
        
        # Save results
        if save:
            self._save_results(results, outcome_name)
        
        logger.info(f"Sobol analysis complete for {outcome_name}")
        
        return results
    
    def _format_sobol_results(self, Si: Dict) -> Dict[str, pd.DataFrame]:
        """Convert SALib Sobol results to structured DataFrames."""
        results = {}
        
        # First-order indices (direct effect)
        results['S1'] = pd.DataFrame({
            'Parameter': self.problem['names'],
            'S1': Si['S1'],
            'S1_conf': Si['S1_conf']
        }).sort_values('S1', ascending=False)
        
        # Total-effect indices (direct + interaction effects)
        results['ST'] = pd.DataFrame({
            'Parameter': self.problem['names'],
            'ST': Si['ST'],
            'ST_conf': Si['ST_conf']
        }).sort_values('ST', ascending=False)
        
        # Second-order indices (pairwise interactions)
        if self.calc_second_order and 'S2' in Si:
            n_vars = self.problem['num_vars']
            param_names = self.problem['names']
            
            s2_data = []
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    s2_data.append({
                        'Param1': param_names[i],
                        'Param2': param_names[j],
                        'S2': Si['S2'][i, j],
                        'S2_conf': Si['S2_conf'][i, j]
                    })
            
            results['S2'] = pd.DataFrame(s2_data).sort_values('S2', ascending=False)
        
        return results
    
    def _save_results(self, results: Dict[str, pd.DataFrame], outcome_name: str):
        """Save Sobol analysis results to CSV files."""
        for key, df in results.items():
            output_file = self.output_dir / f"sobol_{key}_{outcome_name}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {key} indices to {output_file}")
    
    def interpret_results(
        self,
        results: Dict[str, pd.DataFrame],
        threshold_s1: float = 0.05,
        threshold_st: float = 0.1
    ) -> Dict[str, List[str]]:
        """
        Provide interpretation of Sobol indices.
        
        Interpretation guidelines:
        - High ST (≥ threshold_st): Parameter is influential
        - High S1, low (ST-S1): Direct effect only, no interactions
        - Low S1, high ST: Mainly interacts with other parameters
        - Low ST (< threshold_st): Negligible influence
        
        Args:
            results: Results from analyze()
            threshold_s1: Threshold for considering S1 significant
            threshold_st: Threshold for considering ST significant
        
        Returns:
            Dictionary with parameter classifications:
            - 'influential': High ST (important overall)
            - 'direct_only': High S1, low (ST - S1)
            - 'interaction': Low S1, high ST (mainly interactions)
            - 'negligible': Low ST (not important)
        """
        interpretation = {
            'influential': [],
            'direct_only': [],
            'interaction': [],
            'negligible': []
        }
        
        s1_df = results['S1'].set_index('Parameter')
        st_df = results['ST'].set_index('Parameter')
        
        for param in self.problem['names']:
            s1 = s1_df.loc[param, 'S1']
            st = st_df.loc[param, 'ST']
            interaction = st - s1
            
            if st >= threshold_st:
                interpretation['influential'].append(param)
                
                if interaction < threshold_s1:
                    interpretation['direct_only'].append(param)
                else:
                    interpretation['interaction'].append(param)
            else:
                interpretation['negligible'].append(param)
        
        # Log interpretation
        self._log_interpretation(interpretation, s1_df, st_df, threshold_st)
        
        return interpretation
    
    def _log_interpretation(
        self,
        interpretation: Dict,
        s1_df: pd.DataFrame,
        st_df: pd.DataFrame,
        threshold_st: float
    ):
        """Log sensitivity interpretation to console."""
        logger.info("\n" + "="*70)
        logger.info("SOBOL SENSITIVITY INTERPRETATION")
        logger.info("="*70)
        
        logger.info(f"\n📊 Influential Parameters (ST ≥ {threshold_st}):")
        if interpretation['influential']:
            for param in interpretation['influential']:
                s1 = s1_df.loc[param, 'S1']
                st = st_df.loc[param, 'ST']
                logger.info(f"  • {param:30s}  S1={s1:6.3f}  ST={st:6.3f}")
        else:
            logger.info("  (none)")
        
        logger.info(f"\n🎯 Direct Effects Only (high S1, low interaction):")
        if interpretation['direct_only']:
            for param in interpretation['direct_only']:
                logger.info(f"  • {param}")
        else:
            logger.info("  (none)")
        
        logger.info(f"\n🔗 Interaction Effects (low S1, high ST):")
        if interpretation['interaction']:
            for param in interpretation['interaction']:
                logger.info(f"  • {param}")
        else:
            logger.info("  (none)")
        
        logger.info(f"\n❌ Negligible Parameters (ST < {threshold_st}):")
        if interpretation['negligible']:
            for param in interpretation['negligible']:
                st = st_df.loc[param, 'ST']
                logger.info(f"  • {param:30s}  ST={st:6.3f}")
        else:
            logger.info("  (none)")
        
        logger.info("\n" + "="*70)
    
    def compute_convergence_metrics(
        self,
        samples_df: pd.DataFrame,
        outcomes: Union[np.ndarray, pd.Series],
        n_points: int = 10
    ) -> pd.DataFrame:
        """
        Compute Sobol indices for increasing sample sizes to check convergence.
        
        Helps determine if N is large enough by checking if indices stabilize.
        
        Args:
            samples_df: DataFrame with Sobol samples
            outcomes: Array of experimental outcomes
            n_points: Number of sample sizes to test
        
        Returns:
            DataFrame with columns:
            ['n', 'n_total_samples', 'parameter', 'S1', 'S1_conf', 'ST', 'ST_conf']
        """
        logger.info("Computing convergence metrics...")
        
        if isinstance(outcomes, pd.Series):
            outcomes = outcomes.values
        
        # Sample sizes to test
        total_samples = len(samples_df)
        n_vars = self.problem['num_vars']
        samples_per_n = 2 * n_vars + 2
        
        max_n = total_samples // samples_per_n
        
        if max_n < 4:
            raise ValueError(
                f"Not enough samples for convergence analysis. "
                f"Need at least {4 * samples_per_n} samples, have {total_samples}"
            )
        
        n_values = np.linspace(max_n // 4, max_n, n_points, dtype=int)
        
        # Store results
        convergence_data = []
        
        for n in n_values:
            # Use first n sets of samples
            n_samples = n * samples_per_n
            
            logger.info(f"  Computing for n={n} ({n_samples} samples)...")
            
            try:
                Si = sobol.analyze(
                    self.problem,
                    outcomes[:n_samples],
                    calc_second_order=False,  # Skip S2 for convergence
                    print_to_console=False
                )
                
                for i, param in enumerate(self.problem['names']):
                    convergence_data.append({
                        'n': n,
                        'n_total_samples': n_samples,
                        'parameter': param,
                        'S1': Si['S1'][i],
                        'S1_conf': Si['S1_conf'][i],
                        'ST': Si['ST'][i],
                        'ST_conf': Si['ST_conf'][i]
                    })
            except Exception as e:
                logger.warning(f"Failed to compute indices for n={n}: {e}")
                continue
        
        convergence_df = pd.DataFrame(convergence_data)
        
        # Save convergence results
        output_file = self.output_dir / "sobol_convergence_metrics.csv"
        convergence_df.to_csv(output_file, index=False)
        logger.info(f"Saved convergence metrics to {output_file}")
        
        return convergence_df
    
    def get_summary_statistics(self, results: Dict[str, pd.DataFrame]) -> Dict:
        """
        Compute summary statistics for Sobol analysis.
        
        Returns:
            Dictionary with summary stats
        """
        summary = {
            'n_parameters': self.problem['num_vars'],
            'parameter_names': self.problem['names'],
            'calc_second_order': self.calc_second_order
        }
        
        # S1 statistics
        s1_df = results['S1']
        summary['s1_max'] = s1_df['S1'].max()
        summary['s1_min'] = s1_df['S1'].min()
        summary['s1_mean'] = s1_df['S1'].mean()
        summary['s1_sum'] = s1_df['S1'].sum()
        summary['s1_top_param'] = s1_df.iloc[0]['Parameter']
        
        # ST statistics
        st_df = results['ST']
        summary['st_max'] = st_df['ST'].max()
        summary['st_min'] = st_df['ST'].min()
        summary['st_mean'] = st_df['ST'].mean()
        summary['st_sum'] = st_df['ST'].sum()
        summary['st_top_param'] = st_df.iloc[0]['Parameter']
        
        # Interaction strength (mean of ST - S1)
        merged = pd.merge(s1_df, st_df, on='Parameter')
        merged['interaction'] = merged['ST'] - merged['S1']
        summary['interaction_mean'] = merged['interaction'].mean()
        summary['interaction_max'] = merged['interaction'].max()
        
        return summary


def create_problem_definition_from_config(
    config_path: str,
    output_path: str,
    parameter_ranges: Optional[Dict[str, Dict]] = None,
    problem_name: Optional[str] = None
):
    """
    Helper function to create a problem definition YAML from an experiment config.
    
    Args:
        config_path: Path to experiment config YAML
        output_path: Path to save problem definition YAML
        parameter_ranges: Optional dict specifying ranges/types for each parameter
                         Format: {param_name: {'bounds': [min, max], 'type': '...'}}
        problem_name: Optional name for the problem
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    param_grid = config.get('param_grid', {})
    
    if not param_grid:
        raise ValueError(f"No param_grid found in {config_path}")
    
    # Create problem definition
    parameters = []
    for param_name, values in param_grid.items():
        if parameter_ranges and param_name in parameter_ranges:
            # Use provided ranges
            param_def = parameter_ranges[param_name]
        else:
            # Infer from values
            if isinstance(values, list) and len(values) > 0:
                min_val = min(values)
                max_val = max(values)
                param_def = {
                    'bounds': [min_val, max_val],
                    'type': 'continuous'
                }
        
        parameters.append({
            'name': param_name,
            'bounds': param_def['bounds'],
            'type': param_def.get('type', 'continuous'),
            'values': values if param_def.get('type') in ['discrete', 'categorical'] else None,
            'description': param_def.get('description', f"Parameter: {param_name}")
        })
    
    if problem_name is None:
        problem_name = f"Sensitivity Analysis - {Path(config_path).stem}"
    
    problem_def = {
        'problem_name': problem_name,
        'parameters': parameters
    }
    
    # Save
    with open(output_path, 'w') as f:
        yaml.dump(problem_def, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Created problem definition: {output_path}")
    logger.info(f"  Parameters: {len(parameters)}")
    logger.info(f"  Names: {[p['name'] for p in parameters]}")
    
    return problem_def






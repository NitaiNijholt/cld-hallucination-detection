"""
Sensitivity Analysis Master Pipeline

Orchestrates the complete Sobol sensitivity analysis workflow.
This is the CONTROLLER layer - coordinates all modules.

Architecture (matches RQ2 pattern):
- Imports all analysis modules
- Orchestrates workflow from sample generation to final report
- Generates comprehensive summary with tables and figures
"""

import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Import sensitivity analysis modules
from sobol_analyzer import SobolAnalyzer, create_problem_definition_from_config
from sensitivity_config_generator import SensitivityConfigGenerator, generate_configs_from_samples
from sensitivity_data_loader import SensitivityDataLoader, load_sobol_results
from sensitivity_visualizer import SensitivityVisualizer

logger = logging.getLogger(__name__)


class SensitivityMasterPipeline:
    """
    Master pipeline for Sobol sensitivity analysis.
    
    Orchestrates:
    1. Problem definition and sample generation
    2. Experiment config generation
    3. Data loading and preprocessing (after experiments run)
    4. Sobol analysis (S1, ST, S2 computation)
    5. Visualization
    6. Summary report generation
    """
    
    def __init__(
        self,
        rq_name: str = "rq2",
        output_base_dir: str = "parameter_tuning_experiments/sensitivity_analysis"
    ):
        """
        Initialize master pipeline.
        
        Args:
            rq_name: Research question name (rq1, rq2, or rq3)
            output_base_dir: Base directory for all outputs
        """
        self.rq_name = rq_name
        self.output_base_dir = Path(output_base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.results_dir = self.output_base_dir / "results" / rq_name
        self.configs_dir = self.output_base_dir / "configs"
        self.figures_dir = self.output_base_dir / "visualizations" / rq_name
        self.tables_dir = self.output_base_dir / "tables" / rq_name
        
        for dir in [self.results_dir, self.configs_dir, self.figures_dir, self.tables_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Sensitivity Master Pipeline for {rq_name}")
        logger.info(f"  Results: {self.results_dir}")
        logger.info(f"  Figures: {self.figures_dir}")
        logger.info(f"  Tables: {self.tables_dir}")
    
    def run_sample_generation(
        self,
        problem_definition: str,
        n_samples: int = 128,
        seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """
        Step 1: Generate Sobol samples.
        
        Args:
            problem_definition: Path to problem definition YAML
            n_samples: Base sample size
            seed: Random seed
        
        Returns:
            DataFrame with Sobol samples
        """
        logger.info("="*70)
        logger.info("STEP 1: GENERATING SOBOL SAMPLES")
        logger.info("="*70)
        
        # Initialize analyzer
        analyzer = SobolAnalyzer(
            problem_definition=problem_definition,
            output_dir=str(self.results_dir),
            calc_second_order=True
        )
        
        # Generate samples
        samples_df = analyzer.generate_samples(
            n_samples=n_samples,
            seed=seed,
            save=True
        )
        
        logger.info(f"Generated {len(samples_df)} samples")
        logger.info(f"Samples saved to: {self.results_dir}")
        
        return samples_df
    
    def run_config_generation(
        self,
        samples_csv: str,
        base_config: str,
        param_mapping: Optional[Dict[str, str]] = None,
        fixed_params: Optional[Dict] = None,
        generate_script: bool = True,
        parallel: int = 4
    ) -> List[Path]:
        """
        Step 2: Generate experiment configs from Sobol samples.
        
        Args:
            samples_csv: Path to Sobol samples CSV
            base_config: Path to base experiment config
            param_mapping: Parameter name mapping
            fixed_params: Fixed parameters
            generate_script: Whether to generate run script
            parallel: Number of parallel jobs
        
        Returns:
            List of config file paths
        """
        logger.info("="*70)
        logger.info("STEP 2: GENERATING EXPERIMENT CONFIGS")
        logger.info("="*70)
        
        config_paths = generate_configs_from_samples(
            samples_csv=samples_csv,
            base_config=base_config,
            output_dir=str(self.configs_dir.parent / "configs" / "sobol_generated"),
            rq_name=self.rq_name,
            param_mapping=param_mapping,
            fixed_params=fixed_params,
            generate_script=generate_script,
            parallel=parallel
        )
        
        logger.info(f"Generated {len(config_paths)} config files")
        
        if generate_script:
            script_path = self.configs_dir.parent / "configs" / "sobol_generated" / self.rq_name / "run_sobol_experiments.sh"
            logger.info(f"\n📜 Run script generated: {script_path}")
            logger.info(f"   Execute with: bash {script_path}")
        
        return config_paths
    
    def run_analysis(
        self,
        problem_definition: str,
        samples_csv: str,
        exp_id_prefix: str,
        outcome_metrics: Optional[List[str]] = None,
        print_results: bool = True
    ) -> Dict[str, Dict]:
        """
        Step 3: Load experimental results and run Sobol analysis.
        
        Args:
            problem_definition: Path to problem definition YAML
            samples_csv: Path to Sobol samples CSV
            exp_id_prefix: Prefix for experiment IDs
            outcome_metrics: Metrics to analyze
            print_results: Whether to print results
        
        Returns:
            Dictionary mapping outcome_name -> sobol_results
        """
        logger.info("="*70)
        logger.info("STEP 3: LOADING RESULTS AND RUNNING SOBOL ANALYSIS")
        logger.info("="*70)
        
        # Initialize analyzer
        analyzer = SobolAnalyzer(
            problem_definition=problem_definition,
            output_dir=str(self.results_dir),
            calc_second_order=True
        )
        
        # Load experimental results
        experiments_df, samples_df, outcomes_dict = load_sobol_results(
            exp_id_prefix=exp_id_prefix,
            samples_csv=samples_csv,
            results_dir="../results",  # Relative to parameter_tuning_experiments
            outcome_metrics=outcome_metrics,
            hallucination_mode="judge"
        )
        
        logger.info(f"Loaded {len(experiments_df)} experiment results")
        logger.info(f"Matched to {len(samples_df)} Sobol samples")
        
        # Run Sobol analysis for each outcome
        all_results = {}
        
        for outcome_name, (merged_df, outcomes) in outcomes_dict.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing outcome: {outcome_name}")
            logger.info(f"{'='*60}")
            
            # Run analysis
            results = analyzer.analyze(
                samples_df=merged_df,
                outcomes=outcomes,
                outcome_name=outcome_name,
                print_results=print_results,
                save=True
            )
            
            # Interpret results
            interpretation = analyzer.interpret_results(
                results=results,
                threshold_s1=0.05,
                threshold_st=0.1
            )
            
            all_results[outcome_name] = {
                'results': results,
                'interpretation': interpretation,
                'summary': analyzer.get_summary_statistics(results)
            }
        
        logger.info(f"\nCompleted Sobol analysis for {len(all_results)} outcomes")
        
        return all_results
    
    def run_visualization(
        self,
        all_results: Dict[str, Dict],
        create_convergence_plots: bool = False,
        convergence_data: Optional[pd.DataFrame] = None
    ):
        """
        Step 4: Create visualizations.
        
        Args:
            all_results: Results dict from run_analysis()
            create_convergence_plots: Whether to create convergence plots
            convergence_data: Convergence data (if available)
        """
        logger.info("="*70)
        logger.info("STEP 4: CREATING VISUALIZATIONS")
        logger.info("="*70)
        
        # Initialize visualizer
        visualizer = SensitivityVisualizer(
            output_dir=str(self.figures_dir)
        )
        
        # Create plots for each outcome
        for outcome_name, outcome_data in all_results.items():
            logger.info(f"\nCreating plots for {outcome_name}...")
            
            results = outcome_data['results']
            
            # Main Sobol indices plot
            visualizer.plot_sobol_indices(
                results=results,
                outcome_name=outcome_name,
                figsize=(14, 10),
                show_confidence=True
            )
            
            # Parameter ranking plot
            visualizer.plot_parameter_ranking(
                results=results,
                outcome_name=outcome_name,
                figsize=(10, 6)
            )
        
        # Multi-outcome comparison (if multiple outcomes)
        if len(all_results) > 1:
            logger.info("\nCreating multi-outcome comparison...")
            results_for_comparison = {
                name: data['results'] for name, data in all_results.items()
            }
            visualizer.plot_multi_outcome_comparison(
                results_dict=results_for_comparison,
                figsize=(16, 10)
            )
        
        # Convergence plots (if data provided)
        if create_convergence_plots and convergence_data is not None:
            logger.info("\nCreating convergence plots...")
            for outcome_name in all_results.keys():
                visualizer.plot_convergence(
                    convergence_df=convergence_data,
                    outcome_name=outcome_name
                )
        
        logger.info(f"\nVisualizations saved to: {self.figures_dir}")
    
    def generate_summary_report(
        self,
        all_results: Dict[str, Dict],
        problem_definition: str,
        n_samples: int
    ):
        """
        Step 5: Generate comprehensive summary report.
        
        Args:
            all_results: Results from run_analysis()
            problem_definition: Path to problem definition
            n_samples: Number of base samples
        """
        logger.info("="*70)
        logger.info("STEP 5: GENERATING SUMMARY REPORT")
        logger.info("="*70)
        
        # Load problem definition
        with open(problem_definition, 'r') as f:
            problem_def = yaml.safe_load(f)
        
        report_lines = []
        report_lines.append("# SOBOL SENSITIVITY ANALYSIS SUMMARY REPORT")
        report_lines.append("="*70)
        report_lines.append(f"Research Question: {self.rq_name.upper()}")
        report_lines.append(f"Problem: {problem_def['problem_name']}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*70)
        report_lines.append("")
        
        # Experimental setup
        report_lines.append("## EXPERIMENTAL SETUP")
        report_lines.append(f"Base samples (N): {n_samples}")
        n_params = len(problem_def['parameters'])
        total_samples = n_samples * (2 * n_params + 2)
        report_lines.append(f"Total samples: {total_samples} ({n_samples} * (2*{n_params} + 2))")
        report_lines.append(f"Parameters analyzed: {n_params}")
        report_lines.append("")
        
        # Parameters
        report_lines.append("### Parameters:")
        for param in problem_def['parameters']:
            report_lines.append(f"  - {param['name']}: {param['bounds']} ({param['type']})")
        report_lines.append("")
        
        # Results for each outcome
        for outcome_name, outcome_data in all_results.items():
            report_lines.append(f"## RESULTS: {outcome_name.upper()}")
            report_lines.append("-"*70)
            
            results = outcome_data['results']
            interpretation = outcome_data['interpretation']
            summary = outcome_data['summary']
            
            # Summary statistics
            report_lines.append(f"### Summary Statistics:")
            report_lines.append(f"  S1 sum: {summary['s1_sum']:.3f}")
            report_lines.append(f"  ST sum: {summary['st_sum']:.3f}")
            report_lines.append(f"  Mean interaction strength: {summary['interaction_mean']:.3f}")
            report_lines.append("")
            
            # Top parameters
            report_lines.append(f"### Most Influential Parameters (by ST):")
            st_df = results['ST'].head(5)
            for _, row in st_df.iterrows():
                report_lines.append(f"  {row['Parameter']:30s}  ST={row['ST']:.3f} ± {row['ST_conf']:.3f}")
            report_lines.append("")
            
            # Interpretation
            report_lines.append(f"### Interpretation:")
            report_lines.append(f"  Influential: {len(interpretation['influential'])} parameters")
            report_lines.append(f"  Direct effects only: {len(interpretation['direct_only'])} parameters")
            report_lines.append(f"  Interaction effects: {len(interpretation['interaction'])} parameters")
            report_lines.append(f"  Negligible: {len(interpretation['negligible'])} parameters")
            report_lines.append("")
            
            if interpretation['influential']:
                report_lines.append(f"  Influential parameters:")
                for param in interpretation['influential']:
                    report_lines.append(f"    - {param}")
            report_lines.append("")
        
        # Save report
        report_path = self.tables_dir / f"sensitivity_summary_{self.rq_name}_{self.timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report saved to: {report_path}")
        
        # Print to console
        print("\n" + "\n".join(report_lines))
    
    def run_full_pipeline(
        self,
        problem_definition: str,
        base_config: str,
        n_samples: int = 128,
        param_mapping: Optional[Dict[str, str]] = None,
        fixed_params: Optional[Dict] = None,
        skip_config_generation: bool = False,
        skip_analysis: bool = True,  # Default True because experiments need to run first
        exp_id_prefix: Optional[str] = None,
        outcome_metrics: Optional[List[str]] = None
    ):
        """
        Run the complete sensitivity analysis pipeline.
        
        Note: This pipeline has 3 phases:
        Phase 1: Sample + config generation (this function)
        Phase 2: Run experiments (manual - use generated script)
        Phase 3: Analysis + visualization (this function with skip_analysis=False)
        
        Args:
            problem_definition: Path to problem YAML
            base_config: Path to base experiment config
            n_samples: Base sample size
            param_mapping: Parameter name mapping
            fixed_params: Fixed parameters
            skip_config_generation: Skip config generation (if already done)
            skip_analysis: Skip analysis (set False after experiments complete)
            exp_id_prefix: Prefix for loading experiments
            outcome_metrics: Metrics to analyze
        """
        logger.info("\n" + "="*70)
        logger.info("SOBOL SENSITIVITY ANALYSIS - FULL PIPELINE")
        logger.info("="*70)
        
        # Phase 1: Sample and config generation
        if not skip_config_generation:
            # Step 1: Generate samples
            samples_df = self.run_sample_generation(
                problem_definition=problem_definition,
                n_samples=n_samples,
                seed=42
            )
            
            samples_csv = self.results_dir / f"sobol_samples_n{n_samples}.csv"
            
            # Step 2: Generate configs
            self.run_config_generation(
                samples_csv=str(samples_csv),
                base_config=base_config,
                param_mapping=param_mapping,
                fixed_params=fixed_params,
                generate_script=True,
                parallel=4
            )
            
            logger.info("\n" + "="*70)
            logger.info("PHASE 1 COMPLETE: Sample and config generation")
            logger.info("="*70)
            logger.info("\nNext steps:")
            logger.info("1. Run the generated experiment script")
            logger.info("2. Wait for all experiments to complete")
            logger.info("3. Re-run this pipeline with skip_config_generation=True and skip_analysis=False")
        
        # Phase 3: Analysis and visualization (after experiments complete)
        if not skip_analysis:
            if exp_id_prefix is None:
                exp_id_prefix = f"sobol_{self.rq_name}_"
            
            samples_csv = self.results_dir / f"sobol_samples_n{n_samples}.csv"
            
            # Step 3: Analysis
            all_results = self.run_analysis(
                problem_definition=problem_definition,
                samples_csv=str(samples_csv),
                exp_id_prefix=exp_id_prefix,
                outcome_metrics=outcome_metrics
            )
            
            # Step 4: Visualization
            self.run_visualization(all_results=all_results)
            
            # Step 5: Summary report
            self.generate_summary_report(
                all_results=all_results,
                problem_definition=problem_definition,
                n_samples=n_samples
            )
            
            logger.info("\n" + "="*70)
            logger.info("PHASE 3 COMPLETE: Analysis and visualization")
            logger.info("="*70)
            logger.info(f"\nAll results saved to:")
            logger.info(f"  Tables: {self.tables_dir}")
            logger.info(f"  Figures: {self.figures_dir}")
            logger.info(f"  Results: {self.results_dir}")






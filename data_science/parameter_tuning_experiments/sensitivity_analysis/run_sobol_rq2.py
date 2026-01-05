"""
Run Sobol Sensitivity Analysis for RQ2

Example usage:
    python run_sobol_rq2.py --phase 1  # Generate samples and configs
    # ... run experiments using generated script ...
    python run_sobol_rq2.py --phase 3  # Run analysis and visualization
"""

import argparse
import logging
from pathlib import Path

from sensitivity_master_pipeline import SensitivityMasterPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run Sobol sensitivity analysis for RQ2')
    parser.add_argument('--phase', type=int, required=True, choices=[1, 3],
                       help='Pipeline phase: 1=generate samples/configs, 3=analyze results')
    parser.add_argument('--n-samples', type=int, default=128,
                       help='Base sample size (default: 128)')
    parser.add_argument('--exp-prefix', type=str, default='sobol_rq2_',
                       help='Experiment ID prefix for loading results')
    
    args = parser.parse_args()
    
    # Paths (relative to this file)
    base_dir = Path(__file__).parent
    problem_def = base_dir / "configs" / "problem_rq2.yaml"
    base_config = base_dir.parent / "configs" / "experiment_rq2_social_norms.yaml"
    
    # Initialize pipeline
    pipeline = SensitivityMasterPipeline(
        rq_name="rq2",
        output_base_dir=str(base_dir)
    )
    
    # Run appropriate phase
    if args.phase == 1:
        logger.info("Running Phase 1: Sample and config generation")
        
        pipeline.run_full_pipeline(
            problem_definition=str(problem_def),
            base_config=str(base_config),
            n_samples=args.n_samples,
            skip_config_generation=False,
            skip_analysis=True
        )
        
        logger.info("\n" + "="*70)
        logger.info("PHASE 1 COMPLETE")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("1. Navigate to the generated config directory")
        logger.info("2. Run: bash run_sobol_experiments.sh")
        logger.info("3. Wait for all experiments to complete")
        logger.info("4. Run: python run_sobol_rq2.py --phase 3")
    
    elif args.phase == 3:
        logger.info("Running Phase 3: Analysis and visualization")
        
        pipeline.run_full_pipeline(
            problem_definition=str(problem_def),
            base_config=str(base_config),
            n_samples=args.n_samples,
            skip_config_generation=True,
            skip_analysis=False,
            exp_id_prefix=args.exp_prefix,
            outcome_metrics=['accuracy', 'precision', 'recall', 'f1_score']
        )
        
        logger.info("\n" + "="*70)
        logger.info("PHASE 3 COMPLETE")
        logger.info("="*70)
        logger.info("\nResults available in:")
        logger.info(f"  - {base_dir}/tables/rq2/")
        logger.info(f"  - {base_dir}/visualizations/rq2/")
        logger.info(f"  - {base_dir}/results/rq2/")


if __name__ == "__main__":
    main()






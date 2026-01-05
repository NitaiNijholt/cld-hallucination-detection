#!/usr/bin/env python3
"""
RQ1b Fresh Run - Complete experiment with correct target variable
Generates a brand new base CLD with target_variable='Obesity Prevalence'
"""
import sys
sys.path.insert(0, '.')
from modules import run_discovery_experiment
import logging
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

logger.info('='*80)
logger.info('RQ1b FRESH EXPERIMENT: Complete Run with Correct Target Variable')
logger.info('='*80)
logger.info(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
logger.info(f'Target Variable: Obesity Prevalence (CORRECTED)')
logger.info(f'Temporal Scale: Years')
logger.info(f'Spatial Scale: Population')
logger.info('='*80)

# Configuration
generator_config = {'provider': 'openai', 'model': 'gpt-4.1', 'temperature': 0.7}
corruptor_config = {'provider': 'openai', 'model': 'gpt-4.1', 'temperature': 0.7}
judge_config = {'provider': 'openai', 'model': 'gpt-4.1', 'temperature': 0.7}
corrector_models = ['gpt-4.1']

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'parameter_tuning_experiments/results/rq1b_fresh_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

logger.info(f'\nOutput directory: {output_dir}')
logger.info(f'\nLLM Configuration:')
logger.info(f'  Generator: {generator_config["model"]}')
logger.info(f'  Corruptor: {corruptor_config["model"]}')
logger.info(f'  Judge: {judge_config["model"]}')
logger.info(f'  Corrector: {corrector_models[0]}')

# Ground truth file
excel_file = 'parameter_tuning_experiments/ground_truth_clds_for_experiments/Social_norms_and_obesity_prevalence.xlsx'

logger.info(f'\nGround truth: {excel_file}')
logger.info('\n' + '='*80)
logger.info('STARTING EXPERIMENT')
logger.info('='*80)

# Run the complete experiment
# NOTE: Not providing retrieved_session_id, so it will generate a NEW base CLD
result = run_discovery_experiment(
    retrieved_session_id='False',  # Generate new base CLD
    excel_path=excel_file,
    overide_target_variable='Obesity Prevalence',  # OVERRIDE target variable
    generator_config=generator_config,
    corruptor_config=corruptor_config,
    judge_config=judge_config,
    corruption_rate=0.3,
    judge_models=[judge_config['model']],
    num_judges=1,
    judge_approach='correctness',
    corrector_models=corrector_models,
    run_correction=True,  # Enable correction
    output_dir=output_dir,
    # Parallelization settings
    generation_parallel=True,
    generation_max_workers=5,
    corruption_parallel=False,
    corruption_max_workers=10,
    judge_parallel=True,
    judge_max_workers=10,
    rejudge_parallel=True,
    rejudge_max_workers=10,
    rejudge_approach='correctness',
    # Enable judging
    judge_edges=True,
    # Output settings
    export_json=True,
    plot_graph=True,
    plot_session_graph=True,
    plot_validation_graph=True
)

logger.info('\n' + '='*80)
logger.info('EXPERIMENT COMPLETED')
logger.info('='*80)

# Print session IDs
if result:
    logger.info(f"\nSession IDs:")
    logger.info(f"  Base: {result.get('base_session_id', 'N/A')}")
    logger.info(f"  Corrupted: {result.get('corrupted_session_id', 'N/A')}")
    logger.info(f"  Judged (no correction): {result.get('judged_no_correction_session_id', 'N/A')}")
    logger.info(f"  Corrected: {result.get('corrected_session_id', 'N/A')}")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("\nTo analyze results, run:")
    logger.info(f"  python analyze_corruption_experiment.py {output_dir}")

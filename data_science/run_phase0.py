#!/usr/bin/env python3
"""
RQ1 Phase 0: Generate Clean Base CLDs
"""

from modules import multirun_parameter_experiments

print('='*80)
print('RQ1 PHASE 0: Generating Clean Base CLDs')
print('='*80)
print()
print('Configuration:')
print('  - 2 CLDs (Depressive symptoms + Social norms)')
print('  - 3 runs per CLD = 6 total sessions')
print('  - 0% corruption (clean base CLDs)')
print('  - Expected time: 30-45 minutes')
print('  - Expected cost: ~$15-20')
print()
print('Starting experiment...')
print()

multirun_parameter_experiments(
    PROMPTS_FOLDER='parameter_tuning_experiments/alternative_prompts',
    CONFIG_FILE='parameter_tuning_experiments/configs/experiment_rq1_phase0_base.yaml',
    CLD_FOLDER='parameter_tuning_experiments/ground_truth_clds_for_experiments',
    RESULTS_FOLDER='parameter_tuning_experiments/results'
)

print()
print('='*80)
print('PHASE 0 COMPLETE!')
print('='*80)
print()
print('Next step: Get the experiment ID by running:')
print('  cd parameter_tuning_experiments/results')
print('  ls -t | grep exp_ | head -1')

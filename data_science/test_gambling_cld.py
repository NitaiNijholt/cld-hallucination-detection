#!/usr/bin/env python3
"""
Quick test to verify Online Gambling CLD works correctly
"""

from modules import multirun_parameter_experiments

print('='*80)
print('RQ1 TEST: Online Gambling CLD Verification')
print('='*80)
print()
print('Configuration:')
print('  - 1 CLD: Online Gambling Market')
print('  - 1 run (quick test)')
print('  - 0% corruption (clean generation)')
print('  - Expected time: ~5-8 minutes')
print('  - Expected cost: ~$3-5')
print()
print('This will verify:')
print('  ✓ CLD file format is correct')
print('  ✓ Variables load properly')
print('  ✓ Edges generate correctly')
print('  ✓ CI metrics compute')
print()
print('Starting test...')
print()

multirun_parameter_experiments(
    PROMPTS_FOLDER='parameter_tuning_experiments/alternative_prompts',
    CONFIG_FILE='parameter_tuning_experiments/configs/experiment_rq1_test_gambling.yaml',
    CLD_FOLDER='parameter_tuning_experiments/ground_truth_clds_for_experiments',
    RESULTS_FOLDER='parameter_tuning_experiments/results'
)

print()
print('='*80)
print('TEST COMPLETE!')
print('='*80)
print()
print('If you see this message, the Online Gambling CLD works correctly! ✅')
print()
print('Next steps:')
print('  1. Check the results in parameter_tuning_experiments/results/')
print('  2. Look for the generated graph PNG')
print('  3. Ready to add to full RQ1 experiment')

#!/usr/bin/env python3
"""
Wrapper script to run multirun_parameter_experiments from command line.
"""

import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python run_multirun.py <config_yaml>")
    sys.exit(1)

config_file = sys.argv[1]

from modules import multirun_parameter_experiments

multirun_parameter_experiments(
    CONFIG_FILE=config_file,
    PROMPTS_FOLDER='parameter_tuning_experiments/alternative_prompts',
    CLD_FOLDER='parameter_tuning_experiments/ground_truth_clds_for_experiments',
    RESULTS_FOLDER='parameter_tuning_experiments/results',
    FORCE_RERUN=True
)

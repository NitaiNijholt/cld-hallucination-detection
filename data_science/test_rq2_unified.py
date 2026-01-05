#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, '/home/nitai/code/causalix.ai/data_science')

# Import the functions
import importlib.util
spec = importlib.util.spec_from_file_location("run_rq2", "run_rq2.py")
run_rq2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_rq2)

# Test file finding
exp_dir = Path("parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de")
excel_files = run_rq2.find_excel_files(exp_dir)

print(f"Found {len(excel_files)} Excel files:")
for f in excel_files:
    print(f"  - {f}")

"""
Centralized path resolution for RQ2 scripts.

These scripts are invoked by `final_runs/RQ2_unified_analysis.py`.
To keep the unified runner flexible, we allow overriding output locations via env vars:

- RQ2_ANALYSES_DIR: where intermediate analysis artifacts are written/read
- RQ2_OUTPUT_DIR: where final thesis-facing artifacts (tables/figures/json) are written

If the env vars are not set, defaults match the historical project layout:
- analyses:  <repo_root>/parameter_tuning_experiments/rq2_analyses
- outputs:   <repo_root>/final_runs/RQ2_uq_hallucination_detection
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple


def repo_root() -> Path:
    # This file lives under final_runs/RQ2_uq_hallucination_detection/scripts/
    return Path(__file__).resolve().parents[3]


def rq2_dirs() -> Tuple[Path, Path]:
    root = repo_root()
    default_analyses = root / "parameter_tuning_experiments" / "rq2_analyses"
    default_output = root / "final_runs" / "RQ2_uq_hallucination_detection"

    analyses_dir = Path(os.environ.get("RQ2_ANALYSES_DIR", str(default_analyses))).expanduser()
    output_dir = Path(os.environ.get("RQ2_OUTPUT_DIR", str(default_output))).expanduser()

    # Do not resolve() unconditionally (could be a not-yet-existing path)
    return analyses_dir, output_dir



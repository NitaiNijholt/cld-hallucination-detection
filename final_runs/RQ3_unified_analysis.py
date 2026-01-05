#!/usr/bin/env python3
"""
RQ3 Unified Analysis Script Wrapper
Wrapper for the existing comprehensive RQ3 analysis script.

Usage:
    python3 final_runs/RQ3_unified_analysis.py
    python3 final_runs/RQ3_unified_analysis.py --run-dir /tmp/rq3_run_20260103
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse

def main():
    # Points to the existing unified script
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    script_path = PROJECT_ROOT / "final_runs/RQ3_deep_research_validation/analysis_scripts/rq3_unified_analysis.py"
    
    parser = argparse.ArgumentParser(description="RQ3 unified analysis wrapper")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="If set, write RQ3 outputs under this folder (inputs remain under final_runs/RQ3_deep_research_validation by default).",
    )
    parsed, passthrough = parser.parse_known_args(sys.argv[1:])

    env = os.environ.copy()
    env["RQ3_INPUT_DIR"] = str(PROJECT_ROOT / "final_runs/RQ3_deep_research_validation")
    if parsed.run_dir:
        run_dir = Path(parsed.run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        env["RQ3_OUTPUT_DIR"] = str(run_dir)

    # Pass remaining arguments through to the original script
    args = " ".join(passthrough)
    cmd = f"python3 {script_path} {args}".strip()
    
    print(f"Running RQ3 Unified Analysis: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

    # Create 'latest' symlink for reproducibility
    if parsed.run_dir:
        run_dir = Path(parsed.run_dir).expanduser().resolve()
        latest_link = run_dir.parent / "RQ3_latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir, target_is_directory=True)
        print(f"\n✓ Created symlink: RQ3_latest → {run_dir.name}")

if __name__ == "__main__":
    main()




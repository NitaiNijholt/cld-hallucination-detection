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
import shutil
from pathlib import Path
import argparse

def run_command(cmd, description, *, env=None, cwd=None):
    """Run a shell command with logging."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}\n")
    try:
        subprocess.run(cmd, shell=True, check=True, env=env, cwd=cwd)
        print(f"\n✅ Successfully completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Error running: {description}")
        print(f"Exit code: {e.returncode}")
        return False

def main():
    # Points to the existing unified script
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    RQ3_DIR = PROJECT_ROOT / "final_runs/RQ3_deep_research_validation"
    script_path = RQ3_DIR / "analysis_scripts/rq3_unified_analysis.py"
    
    parser = argparse.ArgumentParser(description="RQ3 unified analysis wrapper")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="If set, write RQ3 outputs under this folder (inputs remain under final_runs/RQ3_deep_research_validation by default).",
    )
    parsed, passthrough = parser.parse_known_args(sys.argv[1:])

    env = os.environ.copy()
    env["RQ3_INPUT_DIR"] = str(RQ3_DIR)
    if parsed.run_dir:
        run_dir = Path(parsed.run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        env["RQ3_OUTPUT_DIR"] = str(run_dir)
    else:
        run_dir = None

    # Pass remaining arguments through to the original script
    args = " ".join(passthrough)
    cmd = f"python3 {script_path} {args}".strip()
    
    print(f"Running RQ3 Unified Analysis: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

    # Run additional scripts for generated/ tables
    scripts_dir = RQ3_DIR / "analysis_scripts"
    
    # Generate validation distribution tables (rq3_verdict_distribution_table.tex)
    run_command(
        f"python3 {scripts_dir / 'generate_validation_distribution_tables.py'}",
        "RQ3: Generate Validation Distribution Tables",
        env=env, cwd=str(RQ3_DIR)
    )
    
    # Estimate smart trigger DR enrichment (rq3_smart_trigger_*.tex)
    run_command(
        f"python3 {scripts_dir / 'estimate_smart_trigger_dr_enrichment.py'}",
        "RQ3: Estimate Smart Trigger DR Enrichment",
        env=env, cwd=str(RQ3_DIR)
    )

    # Copy additional thesis tables to output directory
    if run_dir:
        tables_dir = run_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        thesis_tables = [
            "dr_cost_table.tex",
            "rq3_enriched_gt_table.tex",
            "rq3_main_table.tex",
            "rq3_per_cld_table.tex",
        ]
        for table in thesis_tables:
            src = RQ3_DIR / table
            if src.exists():
                dst = tables_dir / table
                shutil.copy(src, dst)
                print(f"  ✓ Copied: {table}")
        
        # Copy validation figures
        validation_dir = run_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        val_assets = [
            "all_edges_threshold_tradeoff.png",
            "dr_cost_efficiency_comparison.png",
        ]
        for asset in val_assets:
            src = RQ3_DIR / "validation" / asset
            if src.exists():
                dst = validation_dir / asset
                shutil.copy(src, dst)
                print(f"  ✓ Copied: validation/{asset}")
        
        # Create 'latest' symlink for reproducibility
        latest_link = run_dir.parent / "RQ3_latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir, target_is_directory=True)
        print(f"\n✓ Created symlink: RQ3_latest → {run_dir.name}")

if __name__ == "__main__":
    main()




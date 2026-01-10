#!/usr/bin/env python3
"""
Preliminary Unified Analysis Script
Single entry point for all preliminary analyses:
1. Generator Model Comparison
2. Random Baseline Generator Comparison
3. Temperature Sensitivity Analysis

Usage:
    python3 final_runs/PRELIM_unified_analysis.py --run-dir /path/to/output

Outputs:
    - generator_model_comparison.tex
    - random_baseline_comparison.png
    - random_baseline_table.tex
    - temperature_sensitivity_table.tex
"""

import sys
import os
import subprocess
import argparse
import shutil
from datetime import datetime
from pathlib import Path


def run_command(cmd, description, *, env=None, cwd=None):
    """Run a shell command with logging."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    if cwd:
        print(f"CWD: {cwd}")
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
    parser = argparse.ArgumentParser(description="Preliminary unified analysis runner")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional output run directory."
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    FINAL_RUNS = PROJECT_ROOT / "final_runs"
    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {PROJECT_ROOT}")

    print("\n" + "#"*80)
    print("PRELIMINARY: Random Baseline and Temperature Sensitivity Analyses")
    print("#"*80)

    env = os.environ.copy()
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = FINAL_RUNS / f"prelim_unified_output_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPreliminary output directory: {run_dir}")

    generated_outputs = []

    # =========================================================================
    # 1. GENERATOR MODEL COMPARISON
    # =========================================================================
    print("\n" + "-"*80)
    print("1. Generator Model Comparison")
    print("-"*80)

    generator_model_script = FINAL_RUNS / "prelim_generator_model_comparison/analysis_scripts/analyze_generator_models.py"
    
    if generator_model_script.exists():
        cmd = f"python3 {generator_model_script}"
        success = run_command(cmd, "Generator Model Comparison", env=env,
                             cwd=str(FINAL_RUNS / "prelim_generator_model_comparison"))
    else:
        print(f"  ⚠️ Script not found: {generator_model_script}")
        success = False
    
    # Copy outputs (either generated or pre-computed)
    fallback_dirs = [
        FINAL_RUNS / "prelim_generator_model_comparison/Output",
        FINAL_RUNS / "prelim_generator_model_comparison/analysis_scripts",
        FINAL_RUNS / "prelim_generator_model_comparison",
    ]
    
    for src_dir in fallback_dirs:
        src_table = src_dir / "generator_model_comparison.tex"
        if src_table.exists():
            dst = run_dir / "generator_model_comparison.tex"
            shutil.copy(src_table, dst)
            generated_outputs.append(dst)
            print(f"  ✓ Copied: generator_model_comparison.tex")
            break
    
    # Also copy PNG if available
    for src_dir in fallback_dirs:
        src_fig = src_dir / "generator_model_comparison.png"
        if src_fig.exists():
            dst = run_dir / "generator_model_comparison.png"
            shutil.copy(src_fig, dst)
            generated_outputs.append(dst)
            print(f"  ✓ Copied: generator_model_comparison.png")
            break

    # =========================================================================
    # 2. RANDOM BASELINE GENERATOR COMPARISON
    # =========================================================================
    print("\n" + "-"*80)
    print("2. Random Baseline Generator Comparison")
    print("-"*80)

    random_baseline_script = FINAL_RUNS / "prelim_random_baseline_generator/analysis_scripts/compute_random_baseline.py"
    random_baseline_output = run_dir / "random_baseline"
    random_baseline_output.mkdir(parents=True, exist_ok=True)

    if random_baseline_script.exists():
        env_rb = env.copy()
        env_rb["RANDOM_BASELINE_OUTPUT_DIR"] = str(random_baseline_output)
        cmd = f"python3 {random_baseline_script}"
        success = run_command(cmd, "Random Baseline Computation", env=env_rb, 
                             cwd=str(FINAL_RUNS / "prelim_random_baseline_generator"))
    else:
        print(f"  ⚠️ Script not found: {random_baseline_script}")
        success = False
    
    # Copy outputs (either generated or pre-computed)
    scripts_dir = FINAL_RUNS / "prelim_random_baseline_generator/analysis_scripts"
    fallback_dirs = [
        scripts_dir,
        FINAL_RUNS / "prelim_random_baseline_generator/Output",
        random_baseline_output,
    ]
    
    # Copy figure
    for src_dir in fallback_dirs:
        src_fig = src_dir / "random_baseline_comparison.png"
        if src_fig.exists():
            dst = run_dir / "random_baseline_comparison.png"
            shutil.copy(src_fig, dst)
            generated_outputs.append(dst)
            print(f"  ✓ Copied: random_baseline_comparison.png")
            break
    
    # Copy table if exists
    for src_dir in fallback_dirs:
        src_table = src_dir / "random_baseline_table.tex"
        if src_table.exists():
            dst = run_dir / "random_baseline_table.tex"
            shutil.copy(src_table, dst)
            generated_outputs.append(dst)
            print(f"  ✓ Copied: random_baseline_table.tex")
            break

    # =========================================================================
    # 3. TEMPERATURE SENSITIVITY ANALYSIS
    # =========================================================================
    print("\n" + "-"*80)
    print("3. Temperature Sensitivity Analysis")
    print("-"*80)

    # Use local copy in analysis_lib (self-contained, no data_science dependency)
    temperature_script = PROJECT_ROOT / "final_runs/analysis_lib/analyze_temperature_sensitivity.py"
    temperature_output = run_dir / "temperature_sensitivity"
    temperature_output.mkdir(parents=True, exist_ok=True)

    if temperature_script.exists():
        env_temp = env.copy()
        env_temp["TEMPERATURE_OUTPUT_DIR"] = str(temperature_output)
        cmd = f"python3 {temperature_script}"
        success = run_command(cmd, "Temperature Sensitivity Analysis", env=env_temp)
        
        if success:
            # Script outputs to hardcoded path - check multiple locations
            source_dirs = [
                FINAL_RUNS / "temperature_sensitivity_analysis",
                FINAL_RUNS / "prelim_temperature_sensitivity/Output",
                temperature_output,
            ]
            for src_dir in source_dirs:
                if src_dir.exists():
                    for f in src_dir.glob("temperature_sensitivity*"):
                        dst = run_dir / f.name
                        shutil.copy(f, dst)
                        print(f"  ✓ Copied: {f.name}")
                        generated_outputs.append(dst)
                    break
    else:
        print(f"  ⚠️ Script not found: {temperature_script}")

    # =========================================================================
    # Create 'latest' symlink
    # =========================================================================
    latest_link = run_dir.parent / "PRELIM_latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)
    print(f"\n✓ Created symlink: PRELIM_latest → {run_dir.name}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("PRELIMINARY ANALYSES SUMMARY")
    print("="*80)
    print(f"\nOutput directory: {run_dir}")
    print(f"\nGenerated outputs ({len(generated_outputs)}):")
    for out in generated_outputs:
        print(f"  • {out.name}")

    print("\n" + "="*80)
    print("✅ Preliminary analyses completed!")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())









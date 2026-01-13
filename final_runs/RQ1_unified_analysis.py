#!/usr/bin/env python3
"""
RQ1 Unified Analysis Script
Single entry point for all RQ1-related analyses:
1. RQ1: Judge Verification (TruthfulQA)
2. RQ1a: LLM-as-a-Judge (Hallucination Detection)
3. RQ1b: LLM-as-a-Corrector

Usage:
    python3 final_runs/RQ1_unified_analysis.py
    python3 final_runs/RQ1_unified_analysis.py --run-dir /tmp/rq1_run_20260103
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path
import glob
import argparse
from datetime import datetime

def run_command(cmd: str, description: str, env: dict | None = None):
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}\n")
    try:
        subprocess.run(cmd, shell=True, check=True, env=env)
        print(f"\n✅ Successfully completed: {description}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running: {description}")
        print(f"Exit code: {e.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="RQ1 unified analysis runner (reproducible)")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="If set, write all analysis outputs under this folder (inputs remain under final_runs/* unless overridden in scripts).",
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {PROJECT_ROOT}")

    # Shared env passed to all subprocesses
    env_vars = os.environ.copy()

    # Optional run folder for co-locating outputs
    run_dir: Path | None = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run output directory: {run_dir}")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- RQ1: TruthfulQA Verification ---
    print("\n" + "#"*80)
    print("RQ1: Judge Capability Verification (TruthfulQA)")
    print("#"*80)
    
    # Find results file (check new structure first, then old)
    truthqa_dir = PROJECT_ROOT / "final_runs/validation_RQ1_truthfulqa_judge/Data"
    if not truthqa_dir.exists():
        # Fall back to old structure
        truthqa_dir = PROJECT_ROOT / "final_runs/validation_RQ1_truthfulqa_judge/Data_runs"
    
    results_files = glob.glob(str(truthqa_dir / "**/*results*.json"), recursive=True)
    if not results_files:
        print("⚠️  No TruthfulQA results found. Skipping analysis.")
    else:
        # Use latest
        latest_result = max(results_files, key=os.path.getmtime)
        print(f"Found latest results: {latest_result}")
        
        script_path = PROJECT_ROOT / "final_runs/validation_RQ1_truthfulqa_judge/analysis_scripts/recalculate_metrics.py"
        if not script_path.exists():
            # Fall back to old structure
            script_path = PROJECT_ROOT / "final_runs/validation_RQ1_truthfulqa_judge/analysis_script/recalculate_metrics.py"
        run_command(f"python3 {script_path} {latest_result}", "RQ1 TruthfulQA Metrics Analysis", env=env_vars)

    # --- RQ1a: LLM-as-a-Judge ---
    print("\n" + "#"*80)
    print("RQ1a: LLM-as-a-Judge (Hallucination Detection)")
    print("#"*80)

    # Use local copies in analysis_lib (self-contained, no data_science dependency)
    script_path_agg = PROJECT_ROOT / "final_runs/analysis_lib/analyze_rq1a_aggregate_enhanced.py"
    script_path_gt = PROJECT_ROOT / "final_runs/analysis_lib/analyze_rq1a_ground_truth_enhanced.py"

    rq1a_runs = [
        {
            "name": "GT Synth (Correctness judge)",
            "base_dir": PROJECT_ROOT / "final_runs/RQ1a_gt_synth_correctness",
            "script": script_path_agg,
            "extra_args": "",
            "out_subdir": "RQ1a_gt_synth_correctness",
        },
        {
            "name": "GT Synth (Citation judge)",
            "base_dir": PROJECT_ROOT / "final_runs/RQ1a_gt_synth_citation",
            "script": script_path_agg,
            "extra_args": "",
            "out_subdir": "RQ1a_gt_synth_citation",
        },
        {
            "name": "GT Lit (Correctness judge)",
            "base_dir": PROJECT_ROOT / "final_runs/RQ1a_gt_lit_correctness",
            "script": script_path_gt,
            "extra_args": "--judge-type correctness",
            "out_subdir": "RQ1a_gt_lit_correctness",
        },
        {
            "name": "GT Lit (Citation judge)",
            "base_dir": PROJECT_ROOT / "final_runs/RQ1a_gt_lit_citation",
            "script": script_path_gt,
            "extra_args": "--judge-type citation",
            "out_subdir": "RQ1a_gt_lit_citation",
        },
    ]

    for item in rq1a_runs:
        base_dir = item["base_dir"]
        if not base_dir.exists():
            print(f"⚠️  Directory not found, skipping: {base_dir}")
            continue

        if run_dir:
            out_dir = run_dir / item["out_subdir"] / f"enhanced_analysis_{run_ts}"
        else:
            out_dir = base_dir / f"enhanced_analysis_{run_ts}"

        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = f"python3 {item['script']} --base_dir {base_dir} --output-dir {out_dir} {item['extra_args']}".strip()
        run_command(cmd, f"RQ1a: {item['name']}", env=env_vars)

    # --- RQ1b: LLM-as-a-Corrector ---
    print("\n" + "#"*80)
    print("RQ1b: LLM-as-a-Corrector")
    print("#"*80)
    
    # Check new structure first, then fall back to old
    script_path_corrector = PROJECT_ROOT / "final_runs/RQ1b_corrector_ablation/analysis_scripts/analyze_corrector_ablation.py"
    if not script_path_corrector.exists():
        script_path_corrector = PROJECT_ROOT / "final_runs/RQ1b_corrector_ablation/scripts/analyze_corrector_ablation.py"
    
    if not script_path_corrector.exists():
        print(f"⚠️  Missing script: {script_path_corrector} (skipping RQ1b analysis)")
    else:
        if run_dir:
            rq1b_out_dir = run_dir / "RQ1b_corrector_ablation"
        else:
            rq1b_out_dir = PROJECT_ROOT / "final_runs/RQ1b_corrector_ablation"

        cmd = (
            f"python3 {script_path_corrector} "
            f"--type both "
            f"--base-path {PROJECT_ROOT / 'final_runs'} "
            f"--output-dir {rq1b_out_dir}"
        )
        run_command(cmd, "RQ1b Corrector Ablation Analysis", env=env_vars)

    # Create 'latest' symlink for reproducibility
    if run_dir:
        latest_link = run_dir.parent / "RQ1_latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir, target_is_directory=True)
        print(f"\n✓ Created symlink: RQ1_latest → {run_dir.name}")

    print("\n" + "="*80)
    print("✅ All RQ1 analyses completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()




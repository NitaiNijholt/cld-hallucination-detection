#!/usr/bin/env python3
"""
RQ1a_ground_truth: Multi-Run Controller for Clean CLD Judging

Orchestrates multiple runs of judging clean (non-corrupted) CLDs to assess
judge ability to distinguish True Positives from False Positives.

Key differences from corruption multirunner:
- NO corruption step (judges base sessions directly)
- CI metrics ENABLED (for RQ2 bonus analysis)
- Output directory: RQ1a_gt_lit_correctness

Usage:
    python run_rq1a_ground_truth_multirun.py --runs 3 --clds depressive social_norms emergency_department --yes
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional, Dict

# Ground truth Excel files
EXCEL_FILES = {
    "depressive": "ground_truth_clds_for_experiments/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx",
    "social_norms": "ground_truth_clds_for_experiments/Social_norms_and_obesity_prevalence.xlsx",
    "emergency_department": "ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx"
}

def load_base_sessions() -> Dict:
    """Load base session registry from JSON file.
    
    Checks multiple locations:
    1. Local: data_science/parameter_tuning_experiments/rq1a_ground_truth_base_sessions.json
    2. Final runs: final_runs/RQ1a_gt_lit_correctness/rq1a_ground_truth_base_sessions.json
    """
    # Try local location first
    local_path = Path(__file__).parent / "rq1a_ground_truth_base_sessions.json"
    
    # Try final runs directory as fallback
    project_root = Path(__file__).parent.parent.parent
    final_runs_path = project_root / "final_runs" / "RQ1a_gt_lit_correctness" / "rq1a_ground_truth_base_sessions.json"
    
    registry_path = None
    if local_path.exists():
        registry_path = local_path
        print(f"Loading base sessions from: {local_path}")
    elif final_runs_path.exists():
        registry_path = final_runs_path
        print(f"Loading base sessions from: {final_runs_path}")
    else:
        raise FileNotFoundError(
            f"Base session registry not found in either location:\n"
            f"  1. {local_path}\n"
            f"  2. {final_runs_path}\n"
            f"Please ensure the base sessions have been generated."
        )
    
    with open(registry_path, 'r') as f:
        return json.load(f)

def check_run_complete(cld: str, run_number: int, base_dir: Path) -> bool:
    """Check if a run is complete (has judged files and metadata)."""
    run_dir = base_dir / cld / f"run_{run_number}"
    if not run_dir.exists():
        return False
    
    # Check for judged files and metadata
    judged_files = list(run_dir.glob("judged_*.xlsx"))
    metadata_files = list(run_dir.glob("judging_metadata_*.json"))
    
    return len(judged_files) >= 3 and len(metadata_files) >= 1

def run_judging(
    cld: str, 
    run_number: int, 
    base_session_id: str, 
    seed: int, 
    excel_file: str, 
    output_dir: Path, 
    judge_type: str = "correctness",
    local_llm: bool = False,
    model: str = None
) -> bool:
    """Run the judging script for a single CLD/run."""
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent
    excel_path = script_dir / excel_file
    
    cmd = [
        "python3",
        str(script_dir / "run_rq1a_ground_truth_judge.py"),
        base_session_id,
        str(excel_path),
        str(output_dir),
        "--seed", str(seed),
        "--judge-type", judge_type
    ]
    
    # Add local LLM flags if specified
    if local_llm:
        cmd.append("--local-llm")
    if model:
        cmd.extend(["--model", model])
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join([c if i < 3 else f'{c[:20]}...' for i, c in enumerate(cmd)])}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, cwd=script_dir)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="RQ1a_ground_truth Multi-Run Controller")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per CLD (default: 3)")
    parser.add_argument(
        "--clds",
        nargs="+",
        default=["depressive", "social_norms", "emergency_department"],
        choices=["depressive", "social_norms", "emergency_department"],
        help="CLDs to process (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if results exist"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--judge-type",
        type=str,
        default="correctness",
        choices=["correctness", "citation"],
        help="Type of judging: correctness or citation (default: correctness)"
    )
    parser.add_argument(
        "--local-llm",
        action="store_true",
        help="Use local LLM via Ollama instead of OpenAI API"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name (e.g., qwen2.5:32b-instruct for local, gpt-4.1 for OpenAI)"
    )
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    dir_suffix = "correctness" if args.judge_type == "correctness" else "citation"
    # Use separate directory for local LLM runs
    if args.local_llm:
        dir_suffix += "_local"
    base_dir = project_root / "final_runs" / f"RQ1a_ground_truth_{dir_suffix}"
    
    # Load base sessions from registry
    try:
        base_sessions = load_base_sessions()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    
    print(f"\n{'='*80}")
    print("RQ1a_ground_truth: CLEAN CLD JUDGING - MULTI-RUN CONTROLLER")
    print(f"{'='*80}")
    print(f"Judge type: {args.judge_type}")
    print(f"Runs per CLD: {args.runs}")
    print(f"CLDs to process: {', '.join(args.clds)}")
    print(f"Force re-run: {args.force}")
    print(f"Output directory: {base_dir}")
    print(f"Web search: {'ENABLED' if args.judge_type == 'citation' else 'DISABLED'}")
    print(f"CI metrics: ENABLED (for RQ2 analysis)")
    print(f"Session strategy: Different base sessions per run (seeds 10, 20, 30)")
    print(f"Local LLM mode: {'ENABLED' if args.local_llm else 'DISABLED'}")
    if args.model:
        print(f"Model override: {args.model}")
    print()
    
    # Check what needs to be run
    to_run = []
    to_skip = []
    
    for cld in args.clds:
        if cld not in base_sessions:
            print(f"❌ ERROR: No base sessions found for {cld} in registry")
            continue
        if cld not in EXCEL_FILES:
            print(f"❌ ERROR: No Excel file configured for {cld}")
            continue
            
        for run_num in range(1, args.runs + 1):
            # Map run_num to seed (run_1 → seed_1, run_2 → seed_2, etc.)
            seed_key = f"seed_{run_num}"
            if seed_key not in base_sessions[cld]:
                print(f"❌ ERROR: No session found for {cld} {seed_key}")
                continue
            
            if "error" in base_sessions[cld][seed_key]:
                print(f"⚠️  WARNING: {cld} {seed_key} had generation error, skipping")
                continue
            
            if not args.force and check_run_complete(cld, run_num, base_dir):
                to_skip.append((cld, run_num))
            else:
                to_run.append((cld, run_num))
    
    print("Status:")
    for cld, run_num in to_skip:
        print(f"  ✓ {cld}/run_{run_num}: EXISTS (skipping)")
    for cld, run_num in to_run:
        print(f"  - {cld}/run_{run_num}: MISSING (will run)")
    
    print()
    print(f"Total to run: {len(to_run)}")
    print(f"Total to skip: {len(to_skip)}")
    
    if not to_run:
        print("\nAll runs complete. Nothing to do.")
        return 0
    
    # Estimate time (judging only, ~15-20 min per run with CI metrics)
    est_minutes = len(to_run) * 20
    print(f"Estimated time: ~{est_minutes} minutes ({len(to_run)} x 20 min/run)")
    print("Note: CI metrics add ~5 min per run (but enable RQ2 analysis!)")
    
    # Confirm
    if not args.yes:
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1
    else:
        print("\nAuto-confirmed with --yes flag. Starting experiments...")
    
    # Run experiments
    successes = 0
    failures = 0
    
    for cld, run_num in to_run:
        print(f"\n{'='*80}")
        print(f"[{successes + failures + 1}/{len(to_run)}] {cld.upper()} - RUN {run_num}")
        print(f"{'='*80}")
        
        # Get base session info from registry (map run_num to seed_key)
        # run_1 → seed_1 (actual seed 10), run_2 → seed_2 (actual seed 20), etc.
        seed_key = f"seed_{run_num}"
        session_info = base_sessions[cld][seed_key]
        base_session_id = session_info["session_id"]
        actual_seed = session_info["seed"]  # Get the actual seed (10, 20, or 30)
        excel_file = EXCEL_FILES[cld]
        
        print(f">> Base session: {base_session_id[:8]}... (seed={actual_seed}, clean CLD)")
        print(f">> Excel file: {excel_file}")
        print(f">> Run number: {run_num}")
        
        # Create output directory
        output_dir = base_dir / cld / f"run_{run_num}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save base session info
        session_info_file = output_dir / "base_session_info.txt"
        experiment_type = f"RQ1a_ground_truth_{args.judge_type}"
        with open(session_info_file, 'w') as f:
            f.write(f"base_session_id={base_session_id}\n")
            f.write(f"base_session_seed={actual_seed}\n")
            f.write(f"cld_name={cld}\n")
            f.write(f"run_number={run_num}\n")
            f.write(f"excel_file={excel_file}\n")
            f.write(f"experiment_type={experiment_type}\n")
            f.write(f"judge_type={args.judge_type}\n")
            f.write(f"corruption_applied=False\n")
            f.write(f"ci_metrics_enabled=True\n")
            f.write(f"local_llm_mode={args.local_llm}\n")
            if args.model:
                f.write(f"model_override={args.model}\n")
            f.write(f"generation_timestamp={session_info.get('timestamp', 'unknown')}\n")
        
        # Run the judging
        success = run_judging(
            cld, run_num, base_session_id, actual_seed, excel_file, output_dir, 
            judge_type=args.judge_type,
            local_llm=args.local_llm,
            model=args.model
        )
        
        if success:
            successes += 1
            print(f"\n✅ SUCCESS: {cld}/run_{run_num} COMPLETE")
        else:
            failures += 1
            print(f"\n❌ FAILED: {cld}/run_{run_num}")
            if not args.yes:
                response = input("\nContinue with remaining runs? (y/n): ")
                if response.lower() != 'y':
                    print("Aborted.")
                    break
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total runs attempted: {successes + failures}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"\nResults directory: {base_dir}")
    
    if successes > 0:
        print(f"\n🎯 Next steps:")
        print(f"   1. Analyze individual runs: analyze_rq1a_ground_truth_tp_fp.py")
        print(f"   2. Aggregate analysis: run_rq1a_ground_truth_aggregate.py")
        print(f"   3. BONUS: RQ2 analysis using CI metrics in output files!")
    
    return 0 if failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

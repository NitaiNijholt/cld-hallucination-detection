#!/usr/bin/env python3
"""
RQ1a Corruption Detection - Citation Judge Multi-Run Controller

Uses existing base sessions to run multiple corruption+citation judging experiments.
Reuses base sessions across runs to save time and ensure consistency.

This script runs the same corruption experiments but judges using CITATION SUPPORT
instead of correctness, testing 3 citation judge prompt variants:
- Baseline: Simple citation support evaluation
- Mechanistic: Mechanistic evidence in citations
- SCE: Structured criteria evaluation of citations

Usage:
    python run_rq1a_corruption_citation_multirun.py --runs 3 --clds depressive social_norms emergency_department
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Base session IDs from rq1_base_generation_original_20251014_004654
# These are clean base CLDs generated from ground truth Excel files WITH CITATIONS
BASE_SESSIONS = {
    "depressive": "49d62c01-e85c-4392-bb6d-eb89f53136a8",
    "social_norms": "278d89e0-7980-4cd5-baf3-d5ed4ee27a13",
    "emergency_department": "a7d55436-d8c9-409b-a591-78f837a44951"
}

def check_run_complete(cld: str, run_number: int, base_dir: Path) -> bool:
    """Check if a run is complete (has analysis folder and README)."""
    run_dir = base_dir / cld / f"run_{run_number}"
    if not run_dir.exists():
        return False
    # Check for both analysis folder and README to confirm completion
    return (run_dir / "analysis").exists() and (run_dir / "README.md").exists()

def run_pipeline(cld: str, run_number: int, base_session_id: Optional[str], model: Optional[str] = None, output_suffix: Optional[str] = None) -> bool:
    """Run the corruption detection pipeline for a single CLD/run with CITATION judging."""
    cmd = [
        str(Path(__file__).parent / "run_corruption_detection_pipeline.sh"),
        cld,
        str(run_number)
    ]
    
    # Add base session ID if provided
    if base_session_id:
        cmd.append(base_session_id)
    
    # Add judge type flag for citation judging
    cmd.append("--judge-type")
    cmd.append("citation")
    
    # Add model override if specified
    if model:
        cmd.append("--model")
        cmd.append(model)
    
    # Add output suffix if specified
    if output_suffix:
        cmd.append("--output-suffix")
        cmd.append(output_suffix)
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0

def load_base_session_from_run(cld: str, run_number: int, base_dir: Path) -> Optional[str]:
    """Load base session ID from a previous run's base_session_info.txt."""
    run_dir = base_dir / cld / f"run_{run_number}"
    session_info_file = run_dir / "base_session_info.txt"
    
    if session_info_file.exists():
        with open(session_info_file) as f:
            for line in f:
                if line.startswith("base_session_id="):
                    return line.strip().split("=")[1]
    return None

def main():
    parser = argparse.ArgumentParser(description="RQ1a Corruption Detection - Citation Judge Multi-Run Controller")
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
        "--model",
        type=str,
        default=None,
        help="Override judge model (e.g., gpt-5-mini, gpt-4.1)"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Add suffix to output directory (e.g., 'gpt5mini' -> RQ1a_gt_synth_citation)"
    )
    args = parser.parse_args()
    
    # Setup paths - DIFFERENT OUTPUT DIRECTORY for citation judging
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    output_dir_name = "RQ1a_corruption_detection_citation"
    if args.output_suffix:
        output_dir_name = f"{output_dir_name}_{args.output_suffix}"
    base_dir = project_root / "final_runs" / output_dir_name
    
    print(f"\n{'='*60}")
    print("RQ1a: CORRUPTION DETECTION - CITATION JUDGE CONTROLLER")
    print(f"{'='*60}")
    print(f"Runs per CLD: {args.runs}")
    print(f"CLDs to process: {', '.join(args.clds)}")
    print(f"Force re-run: {args.force}")
    print(f"Judge type: CITATION (3 variants: Baseline, Mechanistic, SCE)")
    if args.model:
        print(f"Judge model: {args.model}")
    print(f"Output directory: {base_dir}")
    print()
    
    # Check what needs to be run
    to_run = []
    to_skip = []
    
    for cld in args.clds:
        for run_num in range(1, args.runs + 1):
            if not args.force and check_run_complete(cld, run_num, base_dir):
                to_skip.append((cld, run_num))
            else:
                to_run.append((cld, run_num))
    
    print("Status:")
    for cld, run_num in to_skip:
        print(f"  * {cld}/run_{run_num}: EXISTS (skipping)")
    for cld, run_num in to_run:
        print(f"  - {cld}/run_{run_num}: MISSING (will run)")
    
    print()
    print(f"Total to run: {len(to_run)}")
    print(f"Total to skip: {len(to_skip)}")
    
    if not to_run:
        print("\nAll runs complete. Nothing to do.")
        return 0
    
    # Estimate time
    est_minutes = len(to_run) * 25
    print(f"Estimated time: ~{est_minutes} minutes ({len(to_run)} x 25 min/run)")
    
    # Confirm
    if not args.yes:
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1
    else:
        print("\nAuto-confirmed with --yes flag. Starting experiments...")
    
    # Run experiments
    base_session_tracker = BASE_SESSIONS.copy()
    successes = 0
    failures = 0
    
    for cld, run_num in to_run:
        print(f"\n{'='*80}")
        print(f"[{successes + failures + 1}/{len(to_run)}] {cld.upper()} - RUN {run_num}")
        print(f"{'='*80}")
        
        # Get base session ID (reuse if available)
        base_session_id = base_session_tracker.get(cld)
        
        # If no base session yet, check if we created one in a previous run
        if not base_session_id and run_num > 1:
            base_session_id = load_base_session_from_run(cld, run_num - 1, base_dir)
            if base_session_id:
                base_session_tracker[cld] = base_session_id
                print(f">> Reusing base session from run_{run_num - 1}: {base_session_id[:8]}...")
        
        if base_session_id:
            print(f">> Using existing base session: {base_session_id[:8]}...")
        else:
            print(">> Will generate new base session")
        
        # Run the pipeline with CITATION judging
        success = run_pipeline(cld, run_num, base_session_id, model=args.model, output_suffix=args.output_suffix)
        
        if success:
            successes += 1
            print(f"\nSUCCESS: {cld}/run_{run_num} COMPLETE")
            
            # Load the base session ID from this run for future reuse
            if not base_session_tracker.get(cld):
                new_base_session = load_base_session_from_run(cld, run_num, base_dir)
                if new_base_session:
                    base_session_tracker[cld] = new_base_session
                    print(f">> Saved base session for future runs: {new_base_session[:8]}...")
        else:
            failures += 1
            print(f"\nFAILED: {cld}/run_{run_num}")
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
    print(f"\nNext step: Run aggregate analysis script on {base_dir}")
    
    return 0 if failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

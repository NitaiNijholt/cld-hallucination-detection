#!/usr/bin/env python3
"""
RQ1b Correction - Multi-Run Controller

Orchestrates correction experiments across multiple CLDs and runs,
testing all corrector prompt variants.

Similar structure to run_rq1a_ground_truth_multirun.py

Usage:
    python run_rq1b_correction_multirun.py --runs 3 --clds depressive social_norms emergency_department --yes
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Excel files (ground truth CLDs)
EXCEL_FILES = {
    "depressive": "ground_truth_clds_for_experiments/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx",
    "social_norms": "ground_truth_clds_for_experiments/Social_norms_and_obesity_prevalence.xlsx",
    "emergency_department": "ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx"
}

def load_session_info(rq1a_dir: Path) -> tuple:
    """Load corrupted and base session IDs from RQ1a run directory."""
    session_info_file = rq1a_dir / "session_info.txt"
    if not session_info_file.exists():
        return None, None
    
    info = {}
    with open(session_info_file, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                info[k] = v
    
    return info.get('corrupted_session_id'), info.get('base_session_id')

def check_run_complete(cld: str, run_number: int, base_dir: Path) -> bool:
    """Check if a correction run is complete (has corrected files and metadata)."""
    run_dir = base_dir / cld / f"run_{run_number}"
    if not run_dir.exists():
        return False
    
    # Check for corrected files (at least 4 variants)
    corrected_files = list(run_dir.glob("corrected_*.xlsx"))
    metadata_files = list(run_dir.glob("correction_metadata_*.json"))
    
    return len(corrected_files) >= 4 and len(metadata_files) >= 1

def run_correction(
    corrupted_session_id: str,
    base_session_id: str,
    excel_file: str,
    output_dir: Path
) -> bool:
    """Run the correction script for a single CLD/run."""
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent
    excel_path = script_dir / excel_file
    
    cmd = [
        "python3",
        str(script_dir / "run_rq1b_correction_single.py"),
        corrupted_session_id,
        base_session_id,
        str(excel_path),
        str(output_dir)
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join([c if i < 3 else f'{c[:20]}...' for i, c in enumerate(cmd)])}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, cwd=script_dir)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="RQ1b Correction Multi-Run Controller")
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
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    rq1a_base_dir = project_root / "final_runs" / "RQ1a_gt_synth_correctness"
    rq1b_base_dir = project_root / "final_runs" / "RQ1b_correction_experiment_final"
    
    print(f"\n{'='*80}")
    print("RQ1b: CORRECTION EXPERIMENT - MULTI-RUN CONTROLLER")
    print(f"{'='*80}")
    print(f"Runs per CLD: {args.runs}")
    print(f"CLDs to process: {', '.join(args.clds)}")
    print(f"Force re-run: {args.force}")
    print(f"Input (RQ1a corrupted): {rq1a_base_dir}")
    print(f"Output (RQ1b corrected): {rq1b_base_dir}")
    print(f"Corrector variants: {len(CORRECTOR_VARIANTS)}")
    print()
    
    # Check what needs to be run
    to_run = []
    to_skip = []
    
    for cld in args.clds:
        if cld not in EXCEL_FILES:
            print(f"❌ ERROR: No Excel file configured for {cld}")
            continue
            
        for run_num in range(1, args.runs + 1):
            # Load session IDs from RQ1a
            rq1a_run_dir = rq1a_base_dir / cld / f"run_{run_num}"
            if not rq1a_run_dir.exists():
                print(f"⚠️  WARNING: RQ1a run not found: {rq1a_run_dir}")
                continue
            
            corrupted_id, base_id = load_session_info(rq1a_run_dir)
            if not corrupted_id or not base_id:
                print(f"⚠️  WARNING: Session IDs not found in {rq1a_run_dir / 'session_info.txt'}")
                continue
            
            if not args.force and check_run_complete(cld, run_num, rq1b_base_dir):
                to_skip.append((cld, run_num))
            else:
                to_run.append((cld, run_num, corrupted_id, base_id))
    
    print("Status:")
    for cld, run_num in to_skip:
        print(f"  ✓ {cld}/run_{run_num}: EXISTS (skipping)")
    for cld, run_num, _, _ in to_run:
        print(f"  - {cld}/run_{run_num}: MISSING (will run)")
    
    print()
    print(f"Total to run: {len(to_run)}")
    print(f"Total to skip: {len(to_skip)}")
    
    if not to_run:
        print("\nAll runs complete. Nothing to do.")
        return 0
    
    # Estimate time (4 variants × ~5 min per variant = ~20 min per run)
    est_minutes = len(to_run) * 20
    print(f"Estimated time: ~{est_minutes} minutes ({len(to_run)} × 20 min/run)")
    print(f"Note: Each run tests {len(CORRECTOR_VARIANTS)} corrector variants")
    
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
    
    for cld, run_num, corrupted_id, base_id in to_run:
        print(f"\n{'='*80}")
        print(f"[{successes + failures + 1}/{len(to_run)}] {cld.upper()} - RUN {run_num}")
        print(f"{'='*80}")
        
        excel_file = EXCEL_FILES[cld]
        
        print(f">> Corrupted session: {corrupted_id[:8]}...")
        print(f">> Base session (GT): {base_id[:8]}...")
        print(f">> Excel file: {excel_file}")
        
        # Create output directory
        output_dir = rq1b_base_dir / cld / f"run_{run_num}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save session info
        session_info_file = output_dir / "session_info.txt"
        with open(session_info_file, 'w') as f:
            f.write(f"corrupted_session_id={corrupted_id}\n")
            f.write(f"base_session_id={base_id}\n")
            f.write(f"excel_file={excel_file}\n")
            f.write(f"cld_name={cld}\n")
            f.write(f"run_number={run_num}\n")
        
        # Run the correction
        success = run_correction(corrupted_id, base_id, excel_file, output_dir)
        
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
    print(f"\nResults directory: {rq1b_base_dir}")
    
    if successes > 0:
        print(f"\n🎯 Next steps:")
        print(f"   1. Analyze results: python analyze_rq1b_correction_variants.py <output_dir>")
    
    return 0 if failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())





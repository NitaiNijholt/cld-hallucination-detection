#!/usr/bin/env python3
"""
Run mech_lit citation judge for all existing RQ1a corruption detection runs.

This script runs the mechanistic_lit prompt variant on all 9 existing runs
(3 CLDs x 3 runs) without re-running other prompts.

Usage:
    python run_rq1a_mech_lit_all_runs.py [--dry-run]
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BASE_DIR = PROJECT_ROOT / "final_runs" / "RQ1a_gt_synth_citation"
GT_CLD_DIR = SCRIPT_DIR / "ground_truth_clds_for_experiments"

# CLD configurations
CLDS = {
    "depressive": {
        "excel": "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx",
        "sessions": {
            1: "6aac24e5-7b53-40fc-9688-6cf87462b02f",
            2: "aaf0c142-c72b-4e38-8b25-2754d426018b",
            3: "00f0553c-397a-4306-abda-8c7c4b448985"
        }
    },
    "social_norms": {
        "excel": "Social_norms_and_obesity_prevalence.xlsx",
        "sessions": {
            1: "a0e16f57-b126-4bd9-8d22-c803c2a03466",
            2: "a6754751-7b00-47f8-add1-326b729a8ba8",
            3: "f735f5ea-a01c-4479-aa69-439dc7fa4227"
        }
    },
    "emergency_department": {
        "excel": "older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx",
        "sessions": {
            1: "3caefe91-ff94-4e3b-a69a-7c7dabdba5f5",
            2: "2695f15b-2820-497e-aa95-896afd627124",
            3: "81dbcd6a-47ed-4b21-baff-1097c0828998"
        }
    }
}

PROMPT = "mechanistic_lit"
MODEL = "gpt-5-mini"
JUDGE_TYPE = "citation"


def check_already_complete(cld: str, run_num: int) -> bool:
    """Check if mech_lit judging already exists for this run."""
    run_dir = BASE_DIR / cld / f"run_{run_num}"
    # Look for judged_citation_*_mechanistic_lit_*.xlsx
    existing = list(run_dir.glob(f"judged_citation_*_{PROMPT}_*.xlsx"))
    return len(existing) > 0


def run_single_prompt(cld: str, run_num: int, session_id: str, excel_file: str, dry_run: bool = False) -> bool:
    """Run the single prompt judging for one run."""
    output_dir = BASE_DIR / cld / f"run_{run_num}"
    excel_path = GT_CLD_DIR / excel_file
    
    cmd = [
        "uv", "run", "python3", "run_rq1a_judge_single_prompt.py",
        "--session-id", session_id,
        "--prompt", PROMPT,
        "--excel", str(excel_path),
        "--output-dir", str(output_dir),
        "--model", MODEL,
        "--judge-type", JUDGE_TYPE
    ]
    
    logging.info(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        logging.info("(dry run - not executing)")
        return True
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run mech_lit for all RQ1a corruption runs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    
    logging.info("=" * 80)
    logging.info("RQ1a MECH_LIT RUNNER - ALL RUNS")
    logging.info("=" * 80)
    logging.info(f"Prompt: {PROMPT}")
    logging.info(f"Model: {MODEL}")
    logging.info(f"Judge type: {JUDGE_TYPE}")
    logging.info(f"Dry run: {args.dry_run}")
    logging.info("")
    
    # Count what needs to run
    to_run = []
    to_skip = []
    
    for cld, config in CLDS.items():
        for run_num, session_id in config["sessions"].items():
            if not args.force and check_already_complete(cld, run_num):
                to_skip.append((cld, run_num))
            else:
                to_run.append((cld, run_num, session_id, config["excel"]))
    
    logging.info("Status:")
    for cld, run_num in to_skip:
        logging.info(f"  ✓ {cld}/run_{run_num}: EXISTS (skipping)")
    for cld, run_num, _, _ in to_run:
        logging.info(f"  - {cld}/run_{run_num}: WILL RUN")
    
    logging.info(f"\nTotal to run: {len(to_run)}")
    logging.info(f"Total to skip: {len(to_skip)}")
    
    if not to_run:
        logging.info("\nAll runs complete. Nothing to do.")
        return 0
    
    # Estimate time
    est_minutes = len(to_run) * 25
    logging.info(f"Estimated time: ~{est_minutes} minutes")
    
    if not args.dry_run and not args.yes:
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            logging.info("Aborted.")
            return 1
    
    # Run experiments
    successes = 0
    failures = 0
    
    for i, (cld, run_num, session_id, excel_file) in enumerate(to_run, 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"[{i}/{len(to_run)}] {cld.upper()} - RUN {run_num}")
        logging.info(f"{'='*80}")
        logging.info(f"Session ID: {session_id}")
        
        success = run_single_prompt(cld, run_num, session_id, excel_file, dry_run=args.dry_run)
        
        if success:
            successes += 1
            logging.info(f"✅ {cld}/run_{run_num} COMPLETE")
        else:
            failures += 1
            logging.error(f"❌ {cld}/run_{run_num} FAILED")
            if not args.dry_run and not args.yes:
                response = input("Continue with remaining runs? (y/n): ")
                if response.lower() != 'y':
                    break
    
    # Summary
    logging.info(f"\n{'='*80}")
    logging.info("SUMMARY")
    logging.info(f"{'='*80}")
    logging.info(f"Total runs attempted: {successes + failures}")
    logging.info(f"Successes: {successes}")
    logging.info(f"Failures: {failures}")
    logging.info(f"\nResults directory: {BASE_DIR}")
    
    if not args.dry_run and successes > 0:
        logging.info(f"\nNext step: Re-run aggregate analysis:")
        logging.info(f"  python analyze_rq1a_corruption_detection_enhanced.py {BASE_DIR}")
    
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())







#!/usr/bin/env python3
"""
Add CI metrics with local embeddings to both emergency_department sessions.
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))

from add_ci_metrics_to_excel import compute_ci_metrics_for_session, update_excel_with_ci_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SESSIONS = [
    {
        "name": "run_2_mechanistic",
        "session_id": "071b547b-5882-4dd9-a99f-d02ef3319eb5",
        "excel_file": Path("../final_runs/RQ1a_gt_lit_citation/emergency_department/run_2/judged_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_mechanistic_20251125_032605_20251125_040815_20251125_041829.xlsx")
    },
    {
        "name": "run_3_cot",
        "session_id": "77928a30-4375-4a1c-8b17-373968a4b6f4",
        "excel_file": None  # Will export from Neo4j
    }
]

def process_session(session_info):
    """Process a single session."""
    logger.info("=" * 80)
    logger.info(f"PROCESSING: {session_info['name']}")
    logger.info("=" * 80)
    logger.info(f"Session ID: {session_info['session_id']}")
    
    try:
        # Step 1: Compute CI metrics with local embeddings
        logger.info("Computing CI metrics with local GPU embeddings...")
        ci_metrics = compute_ci_metrics_for_session(
            session_id=session_info['session_id'],
            compute_embeddings=True,
            embedding_provider="local",
            fill_citations=False,  # Citations already exist
            embedding_device="cuda"
        )
        logger.info(f"✓ Computed CI metrics for {len(ci_metrics)} edges")
        
        # Step 2: Update Excel file if it exists
        if session_info['excel_file'] and session_info['excel_file'].exists():
            logger.info(f"Updating Excel file: {session_info['excel_file'].name}")
            update_excel_with_ci_metrics(session_info['excel_file'], ci_metrics, backup=True)
            logger.info(f"✅ {session_info['name']} completed successfully!")
        else:
            # For run_3, need to export from Neo4j
            logger.info("Excel file doesn't exist, will need to export from Neo4j...")
            logger.info(f"Session {session_info['name']} has {len(ci_metrics)} edges with CI metrics computed")
            logger.info("Use export script to create Excel file from Neo4j")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to process {session_info['name']}: {e}", exc_info=True)
        return False

def main():
    logger.info("\n" + "=" * 80)
    logger.info("ADD CI METRICS TO EMERGENCY DEPARTMENT SESSIONS")
    logger.info("=" * 80)
    logger.info("Using: local embeddings (all-mpnet-base-v2)")
    logger.info("Device: cuda (GPU)")
    logger.info("Skip citations: Yes (already filled during judging)")
    logger.info("")
    
    results = []
    for session_info in SESSIONS:
        success = process_session(session_info)
        results.append((session_info['name'], success))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{name}: {status}")

if __name__ == "__main__":
    main()



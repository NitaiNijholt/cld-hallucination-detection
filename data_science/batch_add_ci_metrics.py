#!/usr/bin/env python3
"""
Batch Add CI Metrics to Multiple Excel Files

This script processes multiple Excel files in a directory, computing CI metrics
and embeddings for each one.

Usage:
    python batch_add_ci_metrics.py <directory_path> [--no-embeddings]
    
Example:
    python batch_add_ci_metrics.py parameter_tuning_experiments/results/rq1_base_judging_correctness_20251011_031233/
    python batch_add_ci_metrics.py parameter_tuning_experiments/results/rq1_base_judging_correctness_20251011_031233/ --no-embeddings
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from add_ci_metrics_to_excel import (
    read_session_id_from_excel,
    compute_ci_metrics_for_session,
    update_excel_with_ci_metrics
)


def find_excel_files(directory: Path) -> list:
    """Find all judged Excel files in a directory (excluding backups)."""
    excel_files = []
    for pattern in ['judged_*.xlsx', '*_judged.xlsx', '*base*.xlsx']:
        for file in directory.rglob(pattern):
            # Skip backup files
            if '.backup' not in file.name and file.name.startswith('judged_'):
                excel_files.append(file)
    return excel_files


def process_excel_files(excel_files: list, compute_embeddings: bool = True):
    """Process multiple Excel files, adding CI metrics to each."""
    total = len(excel_files)
    success = 0
    failed = 0
    
    logger.info(f"Found {total} Excel files to process")
    print()
    
    for idx, excel_file in enumerate(excel_files, 1):
        print("=" * 80)
        print(f"Processing {idx}/{total}: {excel_file.name}")
        print("=" * 80)
        
        try:
            # Step 1: Read session_id
            session_id = read_session_id_from_excel(excel_file)
            logger.info(f"✓ Session ID: {session_id}")
            
            # Step 2: Compute CI metrics
            logger.info("Computing CI metrics...")
            ci_metrics = compute_ci_metrics_for_session(session_id, compute_embeddings=compute_embeddings)
            logger.info(f"✓ Computed metrics for {len(ci_metrics)} edges")
            
            # Step 3: Update Excel file
            logger.info("Updating Excel file...")
            update_excel_with_ci_metrics(excel_file, ci_metrics, backup=True)
            logger.info(f"✓ Updated successfully")
            
            success += 1
            
        except Exception as e:
            logger.error(f"✗ Failed: {e}")
            failed += 1
        
        print()
    
    return success, failed


def main():
    """Main execution."""
    if len(sys.argv) < 2:
        print("Usage: python batch_add_ci_metrics.py <directory_path> [--no-embeddings]")
        print("\nOptions:")
        print("  --no-embeddings    Skip expensive embedding computations (faster)")
        print("\nExample:")
        print("  python batch_add_ci_metrics.py parameter_tuning_experiments/results/rq1_base_judging_correctness_20251011_031233/")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    compute_embeddings = '--no-embeddings' not in sys.argv
    
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    
    print("=" * 80)
    print("BATCH ADD CI METRICS TO EXCEL FILES")
    print("=" * 80)
    print(f"Directory: {directory}")
    print(f"Compute embeddings: {compute_embeddings}")
    print()
    
    # Find all Excel files
    logger.info("Searching for Excel files...")
    excel_files = find_excel_files(directory)
    
    if not excel_files:
        logger.error("No judged Excel files found in directory")
        sys.exit(1)
    
    # Process all files
    start_time = datetime.now()
    success, failed = process_excel_files(excel_files, compute_embeddings=compute_embeddings)
    elapsed = datetime.now() - start_time
    
    # Summary
    print("=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total files: {len(excel_files)}")
    print(f"✓ Success: {success}")
    print(f"✗ Failed: {failed}")
    print(f"⏱ Time elapsed: {elapsed}")
    print()


if __name__ == "__main__":
    main()

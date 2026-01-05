#!/usr/bin/env python3
"""
Batch process all RQ1a_gt_lit_correctness Excel files to add CI metrics with local embeddings.

Usage:
    python batch_add_ci_metrics_all_rq1a.py
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = Path(__file__).parent / "logs" / f"batch_ci_all_rq1a_{timestamp}.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Process all judged Excel files in RQ1a_gt_lit_correctness."""
    
    # Find all judged Excel files
    rq1a_dir = Path(__file__).parent.parent / "final_runs" / "RQ1a_gt_lit_correctness"
    excel_files = sorted(rq1a_dir.glob("*/*/judged_*.xlsx"))
    
    # Exclude backup files
    excel_files = [f for f in excel_files if not f.name.endswith('.backup.xlsx')]
    
    logger.info("=" * 80)
    logger.info("BATCH ADD CI METRICS TO ALL RQ1a FILES")
    logger.info("=" * 80)
    logger.info(f"Found {len(excel_files)} Excel files to process")
    logger.info(f"Using: all-mpnet-base-v2 (local embeddings)")
    logger.info(f"Device: CPU (to avoid GPU contention)")
    logger.info(f"Parallel processing: 10 workers")
    logger.info("")
    
    # Process each file
    start_time = datetime.now()
    successful = 0
    failed = 0
    failed_files = []
    
    for i, excel_file in enumerate(excel_files, 1):
        rel_path = excel_file.relative_to(rq1a_dir)
        logger.info(f"[{i}/{len(excel_files)}] Processing: {rel_path}")
        
        # Create individual log file for this worker process
        worker_log = Path(__file__).parent / "logs" / f"ci_worker_{i}_{excel_file.stem}.log"
        logger.info(f"  📝 Worker log: {worker_log.name}")
        
        try:
            # Run add_ci_metrics_to_excel.py for this file with streaming output
            with open(worker_log, 'w') as log_file:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).parent / "add_ci_metrics_to_excel.py"),
                        str(excel_file),
                        "--provider", "local",
                        "--device", "cpu"  # Use CPU to avoid GPU contention with other processes
                    ],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout for single log
                    text=True,
                    timeout=18000  # 5 hour timeout per file (citations + embeddings take time)
                )
            
            if result.returncode == 0:
                logger.info(f"  ✅ SUCCESS")
                successful += 1
            else:
                logger.error(f"  ❌ FAILED (exit code {result.returncode})")
                logger.error(f"  Check log: {worker_log}")
                failed += 1
                failed_files.append(str(rel_path))
                
        except subprocess.TimeoutExpired:
            logger.error(f"  ❌ TIMEOUT (exceeded 5 hours)")
            failed += 1
            failed_files.append(str(rel_path))
        except Exception as e:
            logger.error(f"  ❌ EXCEPTION: {e}")
            failed += 1
            failed_files.append(str(rel_path))
        
        logger.info("")
    
    # Summary
    duration = datetime.now() - start_time
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files: {len(excel_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Duration: {duration.total_seconds():.1f}s ({duration.total_seconds()/60:.1f} minutes)")
    logger.info("")
    
    if failed_files:
        logger.info("Failed files:")
        for f in failed_files:
            logger.info(f"  - {f}")
        logger.info("")
    
    logger.info(f"📁 Log file: {log_file}")
    logger.info("")
    logger.info("🎉 Batch processing complete!")


if __name__ == "__main__":
    main()

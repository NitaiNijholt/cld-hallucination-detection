#!/usr/bin/env python3
"""
Batch process all RQ1a_ground_truth Excel files to add citations and CI metrics.

This script:
1. Finds all judged Excel files in final_runs/RQ1a_gt_lit_correctness/
2. For each file:
   - Fills missing citations via Brave search
   - Computes CI metrics (including cosine similarity) using LOCAL 1536-dim embeddings
   - Updates the Excel file in-place (creates .backup.xlsx)
3. Runs in parallel for speed

Usage:
    python batch_add_ci_metrics_rq1a_ground_truth.py [--max-workers N]
    
Example:
    python batch_add_ci_metrics_rq1a_ground_truth.py
    python batch_add_ci_metrics_rq1a_ground_truth.py --max-workers 3
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Setup logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"batch_ci_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from add_ci_metrics_to_excel import (
    read_session_id_from_excel,
    compute_ci_metrics_for_session,
    update_excel_with_ci_metrics
)


def process_single_file(excel_path: Path, embedding_device: str = "cuda") -> dict:
    """Process a single Excel file: fill citations, compute CI metrics, update file."""
    try:
        logger.info(f"Processing: {excel_path.name}")
        
        # Step 1: Read session_id
        session_id = read_session_id_from_excel(excel_path)
        logger.info(f"  Session ID: {session_id[:12]}...")
        
        # Step 2: Compute CI metrics with LOCAL embeddings (SKIP citation filling - already done!)
        ci_metrics = compute_ci_metrics_for_session(
            session_id,
            compute_embeddings=True,
            embedding_provider="local",
            fill_citations=False,  # Citations already filled, just recompute embeddings!
            embedding_device=embedding_device
        )
        logger.info(f"  CI metrics computed for {len(ci_metrics)} edges")
        
        # Step 3: Update Excel file
        num_updated = update_excel_with_ci_metrics(excel_path, ci_metrics, backup=True)
        logger.info(f"  ✅ Updated {num_updated} edges in {excel_path.name}")
        
        return {
            'file': str(excel_path),
            'status': 'success',
            'edges_updated': num_updated,
            'session_id': session_id
        }
        
    except Exception as e:
        logger.error(f"  ❌ Failed to process {excel_path.name}: {e}", exc_info=True)
        return {
            'file': str(excel_path),
            'status': 'failed',
            'error': str(e)
        }


def find_all_excel_files(base_dir: Path) -> list:
    """Find all judged Excel files in RQ1a_gt_lit_correctness directory."""
    excel_files = []
    
    # Pattern: judged_*.xlsx (but NOT .backup.xlsx)
    for excel_file in base_dir.rglob("judged_*.xlsx"):
        if ".backup" not in excel_file.name:
            excel_files.append(excel_file)
    
    return sorted(excel_files)


def main():
    parser = argparse.ArgumentParser(
        description='Batch add citations and CI metrics to RQ1a_ground_truth Excel files'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=3,
        help='Maximum number of parallel workers (default: 3)'
    )
    parser.add_argument(
        '--test-single',
        action='store_true',
        help='Test on a single file first'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for embeddings: cpu, cuda (default), cuda:0, etc.'
    )
    args = parser.parse_args()
    
    # Find base directory
    base_dir = Path(__file__).parent.parent / "final_runs" / "RQ1a_gt_lit_correctness"
    
    if not base_dir.exists():
        logger.error(f"Directory not found: {base_dir}")
        sys.exit(1)
    
    # Find all Excel files
    excel_files = find_all_excel_files(base_dir)
    
    if not excel_files:
        logger.error(f"No Excel files found in {base_dir}")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("BATCH ADD CITATIONS + CI METRICS TO RQ1A_GROUND_TRUTH FILES")
    logger.info("=" * 80)
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Found {len(excel_files)} Excel files to process")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Embedding device: {args.device}")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    
    # Test mode: process single file
    if args.test_single:
        logger.info("TEST MODE: Processing single file...")
        test_file = excel_files[0]
        logger.info(f"Test file: {test_file.name}")
        result = process_single_file(test_file, embedding_device=args.device)
        logger.info(f"Test result: {result}")
        logger.info("✅ Test complete!")
        return
    
    # Process all files in parallel
    results = []
    failed = []
    
    start_time = datetime.now()
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, excel_file, args.device): excel_file
            for excel_file in excel_files
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_file), 1):
            excel_file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    logger.info(f"[{i}/{len(excel_files)}] ✅ {Path(result['file']).name}")
                else:
                    failed.append(result)
                    logger.warning(f"[{i}/{len(excel_files)}] ❌ {Path(result['file']).name}")
                    
            except Exception as e:
                logger.error(f"[{i}/{len(excel_files)}] ❌ {excel_file.name}: {e}")
                failed.append({
                    'file': str(excel_file),
                    'status': 'failed',
                    'error': str(e)
                })
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total files: {len(excel_files)}")
    logger.info(f"Successful: {len(results) - len(failed)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    logger.info("")
    
    if failed:
        logger.warning("⚠️  Failed files:")
        for result in failed:
            logger.warning(f"  - {Path(result['file']).name}: {result.get('error', 'Unknown error')}")
        logger.info("")
    
    total_edges = sum(r.get('edges_updated', 0) for r in results if r['status'] == 'success')
    logger.info(f"📊 Total edges updated: {total_edges}")
    logger.info(f"📁 Log file: {log_file}")
    logger.info("")
    logger.info("🎉 All done! Citations filled + CI metrics computed with LOCAL embeddings (FREE!)")


if __name__ == "__main__":
    main()

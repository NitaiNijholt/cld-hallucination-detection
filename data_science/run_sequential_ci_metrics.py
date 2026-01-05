#!/usr/bin/env python3
"""
Sequential processing of CI metrics for all RQ1a_ground_truth files.
Avoids multiprocessing issues.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def main():
    # Find all Excel files
    base_dir = Path("../final_runs/RQ1a_gt_lit_correctness")
    excel_files = sorted([
        f for f in base_dir.rglob("judged_*.xlsx")
        if ".backup" not in f.name and ".old_backup" not in f.name
    ])
    
    total = len(excel_files)
    success = 0
    failed = 0
    
    print("=" * 80)
    print("SEQUENTIAL CI METRICS PROCESSING")
    print("=" * 80)
    print(f"Total files: {total}")
    print(f"Started: {datetime.now()}")
    print()
    
    for i, excel_file in enumerate(excel_files, 1):
        print(f"[{i}/{total}] Processing: {excel_file.name}")
        
        try:
            result = subprocess.run(
                [sys.executable, "add_ci_metrics_to_excel.py", str(excel_file), "--provider", "local"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes per file
            )
            
            if result.returncode == 0:
                print(f"  ✅ SUCCESS")
                success += 1
            else:
                print(f"  ❌ FAILED: {result.stderr[:200]}")
                failed += 1
        except subprocess.TimeoutExpired:
            print(f"  ❌ TIMEOUT (>5 minutes)")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1
        
        print()
    
    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total: {total}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Completed: {datetime.now()}")

if __name__ == "__main__":
    main()

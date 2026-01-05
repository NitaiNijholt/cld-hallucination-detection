#!/usr/bin/env python3
"""
Monitors the social norms experiment and automatically runs RQ2 analysis when complete.
"""
import subprocess
import time
import sys
import os
from pathlib import Path

LOG_FILE = "/home/nitai/code/causalix.ai/data_science/rq2_social_norms_preload.log"
CHECK_INTERVAL = 60  # Check every minute

print("🔍 Monitoring social norms CLD experiment...")
print(f"Log file: {LOG_FILE}")
print("")

def is_experiment_running():
    """Check if experiment process is still running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "main_eval_multi_run.py.*experiment_rq2_clean_test"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error checking process: {e}")
        return False

def get_log_size():
    """Get size of log file."""
    try:
        return os.path.getsize(LOG_FILE)
    except:
        return 0

# Monitor experiment
check_count = 0
while is_experiment_running():
    check_count += 1
    log_size = get_log_size()
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] Check #{check_count}: Experiment running (log: {log_size:,} bytes)")
    time.sleep(CHECK_INTERVAL)

print("")
print("✅ EXPERIMENT COMPLETED!")
print("")

# Find output directory
time.sleep(5)  # Wait for log to flush
try:
    with open(LOG_FILE, 'r', errors='ignore') as f:
        log_content = f.read()
        
    # Find output directory
    for line in log_content.split('\n'):
        if 'OUTPUT DIRECTORY (END)' in line:
            output_dir = line.split(': ')[-1].strip()
            print(f"Output directory: {output_dir}")
            
            # Find Excel file
            excel_files = list(Path(output_dir).glob("*.xlsx"))
            if excel_files:
                excel_file = str(excel_files[0])
                print(f"Excel file: {excel_file}")
                print("")
                
                # Save path for analysis
                with open("/home/nitai/code/causalix.ai/data_science/latest_rq2_excel.txt", "w") as f:
                    f.write(excel_file)
                print("✅ Excel path saved to latest_rq2_excel.txt")
                break
    
except Exception as e:
    print(f"❌ Error finding output: {e}")
    sys.exit(1)

print("")
print("="*60)
print("EXPERIMENT MONITORING COMPLETE")
print("Ready for analysis pipeline!")
print("="*60)

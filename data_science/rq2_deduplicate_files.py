#!/usr/bin/env python3
"""
Deduplicate RQ2 file inventory by keeping only the latest timestamp
for each (experiment_type, CLD, run, prompt_type) combination.
"""

import json
from pathlib import Path
from datetime import datetime
import re

def extract_timestamp_from_filepath(filepath):
    """Extract timestamp from filename (format: YYYYMMDD_HHMMSS)."""
    # Look for pattern: YYYYMMDD_HHMMSS
    pattern = r'(\d{8}_\d{6})'
    matches = re.findall(pattern, filepath)
    
    if matches:
        # Take the first timestamp (generation time)
        timestamp_str = matches[0]
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except:
            return None
    return None

def deduplicate_files(files):
    """Keep only the latest file for each (exp_type, cld, run, prompt) combination."""
    
    # Group by key
    groups = {}
    
    for file in files:
        key = (
            file['experiment_type'],
            file['cld'],
            file['run'],
            file['prompt_type']
        )
        
        timestamp = extract_timestamp_from_filepath(file['filepath'])
        
        if key not in groups:
            groups[key] = {'file': file, 'timestamp': timestamp}
        else:
            # Keep the one with the latest timestamp
            existing_ts = groups[key]['timestamp']
            
            if timestamp and existing_ts:
                if timestamp > existing_ts:
                    groups[key] = {'file': file, 'timestamp': timestamp}
            elif timestamp and not existing_ts:
                groups[key] = {'file': file, 'timestamp': timestamp}
    
    # Extract deduplicated files
    deduplicated = [item['file'] for item in groups.values()]
    
    return deduplicated

def main():
    print("="*80)
    print("RQ2 FILE DEDUPLICATION")
    print("="*80)
    
    # Load the valid files inventory
    inventory_dir = Path("parameter_tuning_experiments/rq2_analyses")
    inventory_files = list(inventory_dir.glob("rq2_valid_files_*.json"))
    
    if not inventory_files:
        print("❌ No valid files inventory found!")
        return 1
    
    # Use the most recent inventory
    inventory_file = max(inventory_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📋 Loading: {inventory_file.name}")
    
    with open(inventory_file, 'r') as f:
        files = json.load(f)
    
    print(f"   Original files: {len(files)}")
    
    # Show duplicates before deduplication
    from collections import Counter
    keys = [(f['experiment_type'], f['cld'], f['run'], f['prompt_type']) for f in files]
    duplicates = {k: v for k, v in Counter(keys).items() if v > 1}
    
    if duplicates:
        print(f"\n🔍 Found {len(duplicates)} keys with duplicates:")
        for key, count in sorted(duplicates.items()):
            exp, cld, run, prompt = key
            print(f"   {exp}/{cld}/{run}/{prompt}: {count} files")
            
            # Show the files
            matching = [f for f in files if 
                       f['experiment_type'] == exp and 
                       f['cld'] == cld and 
                       f['run'] == run and 
                       f['prompt_type'] == prompt]
            
            for f in matching:
                ts = extract_timestamp_from_filepath(f['filepath'])
                filename = Path(f['filepath']).name[:60]
                print(f"      - {ts.strftime('%Y-%m-%d %H:%M:%S') if ts else 'no timestamp'}: {filename}")
    
    # Deduplicate
    deduplicated = deduplicate_files(files)
    
    print(f"\n✅ After deduplication: {len(deduplicated)} files")
    print(f"   Removed: {len(files) - len(deduplicated)} duplicates")
    
    # Show breakdown
    from collections import defaultdict
    by_exp = defaultdict(int)
    for f in deduplicated:
        by_exp[f['experiment_type']] += 1
    
    print(f"\nBy experiment:")
    for exp, count in sorted(by_exp.items()):
        print(f"   {exp:15s}: {count} files")
    
    # Save deduplicated list
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = inventory_dir / f"rq2_valid_files_deduplicated_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(deduplicated, f, indent=2)
    
    print(f"\n📄 Saved deduplicated list:")
    print(f"   {output_file}")
    
    # Expected counts
    print(f"\n📊 Expected vs. Actual:")
    print(f"   Citation:    Expected 36 (3 CLDs × 3 runs × 4 prompts), Got {by_exp.get('citation', 0)}")
    print(f"   Correctness: Expected 27 (3 CLDs × 3 runs × 3 prompts), Got {by_exp.get('correctness', 0)}")
    print(f"   Total:       Expected 63, Got {len(deduplicated)}")
    
    if len(deduplicated) == 63:
        print(f"\n✅ Perfect! Counts match expectations.")
    else:
        print(f"\n⚠️  Warning: Counts don't match. Investigate further.")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())



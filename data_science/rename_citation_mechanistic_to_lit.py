#!/usr/bin/env python3
"""
Rename citation experiment files from 'mechanistic' to 'mechanistic_lit'.
Only affects files in RQ1a_gt_lit_citation folder.
Excludes deprecated folders and backup files.
"""

import os
from pathlib import Path

def rename_mechanistic_to_lit():
    """Rename citation mechanistic files to mechanistic_lit."""
    
    base_path = Path("final_runs/RQ1a_gt_lit_citation")
    
    # Find all mechanistic files (excluding mechanistic_original and backups)
    all_files = []
    for root, dirs, files in os.walk(base_path):
        # Skip deprecated folders
        if 'deprecated' in root:
            continue
        
        for file in files:
            if file.endswith('.xlsx') and '_mechanistic_' in file:
                # Exclude mechanistic_original and backup files
                if 'mechanistic_original' not in file and '.backup' not in file:
                    full_path = Path(root) / file
                    all_files.append(full_path)
    
    print("="*80)
    print("RENAMING CITATION MECHANISTIC → MECHANISTIC_LIT")
    print("="*80)
    print(f"\nFound {len(all_files)} files to rename\n")
    
    renamed_count = 0
    for old_path in sorted(all_files):
        # Create new filename by replacing _mechanistic_ with _mechanistic_lit_
        new_filename = old_path.name.replace('_mechanistic_', '_mechanistic_lit_')
        new_path = old_path.parent / new_filename
        
        print(f"Renaming:")
        print(f"  OLD: {old_path.name}")
        print(f"  NEW: {new_filename}")
        
        # Perform rename
        old_path.rename(new_path)
        renamed_count += 1
        print(f"  ✓ Done\n")
    
    print("="*80)
    print(f"✅ RENAMED {renamed_count} FILES")
    print("="*80)
    
    return renamed_count

if __name__ == "__main__":
    renamed = rename_mechanistic_to_lit()
    print(f"\nTotal files renamed: {renamed}")


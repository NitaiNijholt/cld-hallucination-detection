#!/usr/bin/env python3
"""
RQ2 File Inventory Script
Scans three RQ1a experimental directories and collects valid Excel files for analysis.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

# Target directories
# NOTE: Corruption experiments are EXCLUDED because corruption is inserted POST-generation.
# Generator CI metrics (logits) reflect the original text, not the corrupted edges.
TARGET_DIRS = {
    'citation': 'final_runs/RQ1a_gt_lit_citation',
    'correctness': 'final_runs/RQ1a_gt_lit_correctness',
    # 'corruption': 'final_runs/RQ1a_gt_synth_correctness'  # EXCLUDED - see note above
}

# Hallucination definitions per experiment type
HALLUCINATION_DEFS = {
    'citation': 'FP_OR_FN',  # Classification == 'FP' OR Classification == 'FN'
    'correctness': 'FP_OR_FN',  # Classification == 'FP' OR Classification == 'FN'
    # 'corruption': 'IS_CORRUPTED'  # EXCLUDED - corruption inserted post-generation
}

# Expected CLDs
CLDS = ['depressive', 'social_norms', 'emergency_department']

# Expected runs
RUNS = ['run_1', 'run_2', 'run_3']

# Prompt types
PROMPT_TYPES = ['mechanistic_lit', 'mechanistic', 'mechanistic_original', 'cot', 'baseline']


def is_valid_excel_file(filepath: Path) -> bool:
    """Check if file should be included in analysis."""
    filename = filepath.name.lower()
    
    # Exclude backup files
    if '.backup.' in filename or filename.endswith('.backup.xlsx'):
        return False
    
    # Exclude deprecated folders
    if 'deprecated' in str(filepath):
        return False
    
    # Exclude analysis outputs
    excluded_patterns = [
        'prompt_variants_analysis',
        'enhanced_results',
        'enhanced_analysis',
        'rq1a_ground_truth_enhanced',
        'rq1a_enhanced'
    ]
    
    for pattern in excluded_patterns:
        if pattern in filename:
            return False
    
    # Must start with "judged_" or "CORRUPTED_" (for corruption files)
    if not (filename.startswith('judged_') or filename.startswith('corrupted_')):
        return False
    
    return True


def extract_metadata_from_path(filepath: Path, experiment_type: str) -> Dict:
    """Extract metadata from file path."""
    parts = filepath.parts
    filename = filepath.name
    
    metadata = {
        'filepath': str(filepath),
        'filename': filename,
        'experiment_type': experiment_type,
        'hallucination_def': HALLUCINATION_DEFS[experiment_type]
    }
    
    # Extract CLD
    for cld in CLDS:
        if cld in str(filepath):
            metadata['cld'] = cld
            break
    else:
        metadata['cld'] = 'unknown'
    
    # Extract run
    for run in RUNS:
        if run in str(filepath):
            metadata['run'] = run
            break
    else:
        metadata['run'] = 'unknown'
    
    # Extract prompt type from filename
    filename_lower = filename.lower()
    prompt_type = None
    
    # Check for more specific variants first (mechanistic_lit, mechanistic_original before mechanistic)
    if 'mechanistic_original' in filename_lower:
        prompt_type = 'mechanistic_original'
    elif 'mechanistic_lit' in filename_lower:
        prompt_type = 'mechanistic_lit'
    elif 'mechanistic' in filename_lower:
        prompt_type = 'mechanistic'
    elif 'cot' in filename_lower:
        prompt_type = 'cot'
    elif 'baseline' in filename_lower:
        prompt_type = 'baseline'
    elif 'corrupted' in filename_lower:
        # For corruption files without prompt in name, need to check
        prompt_type = 'unknown'
    
    metadata['prompt_type'] = prompt_type if prompt_type else 'unknown'
    
    return metadata


def check_excel_columns(filepath: Path) -> Tuple[bool, List[str], str]:
    """
    Check if Excel file has required columns for RQ2 analysis.
    Returns: (is_valid, missing_columns, error_message)
    """
    required_ci_metrics = [
        'Gen Perplexity',
        'Gen Min Prob',
        'Gen Max Window Entropy',
        'Gen Cosine Similarity'
    ]
    
    required_classification_cols = ['Classification']
    
    try:
        # Try to read the Excel file
        df = pd.read_excel(filepath, sheet_name='All Edges', nrows=5)
        
        missing = []
        
        # Check CI metrics
        for col in required_ci_metrics:
            if col not in df.columns:
                missing.append(col)
        
        # Check classification column
        if 'Classification' not in df.columns:
            missing.append('Classification')
        
        if missing:
            return False, missing, f"Missing columns: {', '.join(missing)}"
        
        return True, [], "OK"
        
    except Exception as e:
        return False, [], f"Error reading file: {str(e)}"


def scan_directory(base_path: Path, experiment_type: str) -> List[Dict]:
    """Scan a directory for valid Excel files."""
    print(f"\n{'='*80}")
    print(f"Scanning: {experiment_type.upper()}")
    print(f"Path: {base_path}")
    print(f"{'='*80}")
    
    files = []
    excel_files = list(base_path.glob("**/*.xlsx"))
    
    print(f"Found {len(excel_files)} total .xlsx files")
    
    valid_count = 0
    invalid_count = 0
    
    for filepath in excel_files:
        if is_valid_excel_file(filepath):
            metadata = extract_metadata_from_path(filepath, experiment_type)
            
            # Check if file has required columns
            is_valid, missing_cols, error_msg = check_excel_columns(filepath)
            
            metadata['has_required_columns'] = is_valid
            metadata['column_check_status'] = error_msg
            metadata['missing_columns'] = missing_cols
            
            files.append(metadata)
            
            if is_valid:
                valid_count += 1
                status_icon = "✓"
            else:
                invalid_count += 1
                status_icon = "✗"
            
            print(f"  {status_icon} {metadata['cld']:20s} | {metadata['run']:8s} | {metadata['prompt_type']:20s} | {filepath.name[:50]}")
            if not is_valid:
                print(f"      └─ {error_msg}")
    
    print(f"\nSummary: {valid_count} valid, {invalid_count} invalid")
    
    return files


def generate_summary_stats(inventory: Dict) -> Dict:
    """Generate summary statistics from inventory."""
    stats = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'by_experiment': {},
        'by_cld': {},
        'by_run': {},
        'by_prompt_type': {},
        'by_hallucination_def': {}
    }
    
    for exp_type, files in inventory.items():
        stats['total_files'] += len(files)
        
        exp_stats = {
            'total': len(files),
            'valid': sum(1 for f in files if f['has_required_columns']),
            'invalid': sum(1 for f in files if not f['has_required_columns']),
            'by_cld': {},
            'by_prompt': {}
        }
        
        for file in files:
            if file['has_required_columns']:
                stats['valid_files'] += 1
            else:
                stats['invalid_files'] += 1
            
            # Count by CLD
            cld = file['cld']
            if cld not in exp_stats['by_cld']:
                exp_stats['by_cld'][cld] = 0
            exp_stats['by_cld'][cld] += 1
            
            if cld not in stats['by_cld']:
                stats['by_cld'][cld] = 0
            stats['by_cld'][cld] += 1
            
            # Count by prompt type
            prompt = file['prompt_type']
            if prompt not in exp_stats['by_prompt']:
                exp_stats['by_prompt'][prompt] = 0
            exp_stats['by_prompt'][prompt] += 1
            
            if prompt not in stats['by_prompt_type']:
                stats['by_prompt_type'][prompt] = 0
            stats['by_prompt_type'][prompt] += 1
            
            # Count by run
            run = file['run']
            if run not in stats['by_run']:
                stats['by_run'][run] = 0
            stats['by_run'][run] += 1
            
            # Count by hallucination def
            halluc_def = file['hallucination_def']
            if halluc_def not in stats['by_hallucination_def']:
                stats['by_hallucination_def'][halluc_def] = 0
            stats['by_hallucination_def'][halluc_def] += 1
        
        stats['by_experiment'][exp_type] = exp_stats
    
    return stats


def main():
    """Main execution."""
    print("="*80)
    print("RQ2 FILE INVENTORY GENERATION")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Scan all directories
    inventory = {}
    
    for exp_type, rel_path in TARGET_DIRS.items():
        base_path = Path(rel_path)
        if not base_path.exists():
            print(f"\n⚠️  WARNING: Directory not found: {base_path}")
            inventory[exp_type] = []
            continue
        
        files = scan_directory(base_path, exp_type)
        inventory[exp_type] = files
    
    # Generate summary statistics
    stats = generate_summary_stats(inventory)
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"\nTotal files scanned: {stats['total_files']}")
    print(f"  ✓ Valid files:     {stats['valid_files']}")
    print(f"  ✗ Invalid files:   {stats['invalid_files']}")
    
    print(f"\nBy experiment type:")
    for exp_type, exp_stats in stats['by_experiment'].items():
        print(f"  {exp_type:15s}: {exp_stats['valid']:2d} valid / {exp_stats['total']:2d} total")
    
    print(f"\nBy CLD:")
    for cld, count in stats['by_cld'].items():
        print(f"  {cld:25s}: {count:2d} files")
    
    print(f"\nBy prompt type:")
    for prompt, count in stats['by_prompt_type'].items():
        print(f"  {prompt:25s}: {count:2d} files")
    
    print(f"\nBy hallucination definition:")
    for halluc_def, count in stats['by_hallucination_def'].items():
        print(f"  {halluc_def:25s}: {count:2d} files")
    
    # Save inventory to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("parameter_tuning_experiments/rq2_analyses")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"rq2_file_inventory_{timestamp}.json"
    
    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'target_directories': TARGET_DIRS,
            'hallucination_definitions': HALLUCINATION_DEFS
        },
        'statistics': stats,
        'inventory': inventory
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ INVENTORY SAVED")
    print(f"{'='*80}")
    print(f"Location: {output_file}")
    print(f"\nValid files ready for RQ2 analysis: {stats['valid_files']}")
    
    # Also save a simple list of valid files
    valid_files_list = []
    for exp_type, files in inventory.items():
        for file in files:
            if file['has_required_columns']:
                valid_files_list.append({
                    'filepath': file['filepath'],
                    'experiment_type': exp_type,
                    'hallucination_def': file['hallucination_def'],
                    'cld': file['cld'],
                    'run': file['run'],
                    'prompt_type': file['prompt_type']
                })
    
    valid_files_path = output_dir / f"rq2_valid_files_{timestamp}.json"
    with open(valid_files_path, 'w') as f:
        json.dump(valid_files_list, f, indent=2)
    
    print(f"Valid files list: {valid_files_path}")
    
    return output_file, valid_files_list


if __name__ == "__main__":
    main()


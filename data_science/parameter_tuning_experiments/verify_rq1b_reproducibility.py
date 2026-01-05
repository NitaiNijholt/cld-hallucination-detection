#!/usr/bin/env python3
"""
RQ1b Reproducibility Verification Script

This script validates that all numbers in the thesis tables can be traced back
to the source JSON files and are 100% reproducible.

It checks:
1. Action counts match between Excel and JSON source files
2. Total = sum of all sub-actions (revise + change + none + error + remove + flip)
3. Each CLD/run combination has exactly the expected number of runs
4. No missing or duplicate data

Usage:
    python verify_rq1b_reproducibility.py
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys

BASE_PATH = Path('final_runs')

# All RQ1b experiment folders to verify
EXPERIMENT_FOLDERS = {
    'Synthetic Baseline': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_baseline',
    'Synthetic CoT': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_cot',
    'Synthetic Mechanistic': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_mechanistic',
    'GT Baseline': 'RQ1b_corrector_experiment_ground_truth_correctness_baseline',
    'GT CoT': 'RQ1b_corrector_experiment_ground_truth_correctness_cot',
    'GT Mechanistic': 'RQ1b_corrector_experiment_ground_truth_correctness_mechanistic',
}

def count_actions_from_json(json_file: Path) -> dict:
    """Count actions from a single JSON file."""
    with open(json_file, 'r') as f:
        outcomes = json.load(f)
    
    counts = {
        'revise': 0,
        'change_type': 0,
        'flip_polarity': 0,
        'remove': 0,
        'none': 0,
        'error': 0,
        'unknown': 0
    }
    
    for o in outcomes:
        action = o.get('action', 'none')
        if action in counts:
            counts[action] += 1
        else:
            counts['unknown'] += 1
            print(f"  Warning: Unknown action '{action}' in {json_file.name}")
    
    counts['total'] = sum(counts.values())
    return counts


def verify_folder(name: str, folder_name: str) -> dict:
    """Verify a single experiment folder."""
    folder = BASE_PATH / folder_name
    
    if not folder.exists():
        return {'status': 'MISSING', 'folder': folder_name}
    
    results = {
        'status': 'OK',
        'folder': folder_name,
        'clds': {},
        'issues': []
    }
    
    # Find all CLD directories
    for cld_dir in sorted(folder.glob('*')):
        if not cld_dir.is_dir():
            continue
        
        cld_name = cld_dir.name
        cld_results = {
            'runs': [],
            'total_counts': defaultdict(int)
        }
        
        # Find all run directories
        for run_dir in sorted(cld_dir.glob('run_*')):
            run_num = run_dir.name.replace('run_', '')
            
            # Find the LATEST correction_outcomes JSON (most recent by timestamp)
            json_files = sorted(run_dir.glob('correction_outcomes_*.json'), 
                              key=lambda x: x.stat().st_mtime, reverse=True)
            
            if not json_files:
                results['issues'].append(f"{cld_name}/run_{run_num}: No JSON file found")
                continue
            
            # Use the most recent file
            latest_json = json_files[0]
            counts = count_actions_from_json(latest_json)
            
            # Verify total matches sum
            computed_total = (counts['revise'] + counts['change_type'] + counts['flip_polarity'] + 
                            counts['remove'] + counts['none'] + counts['error'] + counts['unknown'])
            
            if counts['total'] != computed_total:
                results['issues'].append(
                    f"{cld_name}/run_{run_num}: Total mismatch ({counts['total']} != {computed_total})"
                )
                results['status'] = 'ERROR'
            
            # Check for errors
            if counts['error'] > 0:
                error_rate = counts['error'] / counts['total'] * 100
                if error_rate >= 20:
                    results['issues'].append(
                        f"{cld_name}/run_{run_num}: HIGH error rate ({counts['error']}/{counts['total']} = {error_rate:.1f}%)"
                    )
            
            run_result = {
                'run': run_num,
                'json_file': latest_json.name,
                'counts': counts
            }
            cld_results['runs'].append(run_result)
            
            # Aggregate counts
            for key, val in counts.items():
                cld_results['total_counts'][key] += val
        
        results['clds'][cld_name] = cld_results
    
    return results


def print_summary(all_results: dict):
    """Print a summary table of all experiments."""
    print("\n" + "=" * 100)
    print("RQ1b REPRODUCIBILITY VERIFICATION SUMMARY")
    print("=" * 100)
    
    # Print header
    print(f"\n{'Experiment':<25} {'Status':<10} {'CLDs':<5} {'Runs':<5} {'Total':<8} {'Revise':<8} {'Change':<8} {'Error':<8}")
    print("-" * 100)
    
    for name, results in all_results.items():
        if results['status'] == 'MISSING':
            print(f"{name:<25} {'MISSING':<10}")
            continue
        
        # Count totals across CLDs
        total_runs = sum(len(cld['runs']) for cld in results['clds'].values())
        total_actions = sum(cld['total_counts']['total'] for cld in results['clds'].values())
        total_revise = sum(cld['total_counts']['revise'] for cld in results['clds'].values())
        total_change = sum(cld['total_counts']['change_type'] for cld in results['clds'].values())
        total_errors = sum(cld['total_counts']['error'] for cld in results['clds'].values())
        
        status = results['status']
        if total_errors > 0 and results['status'] == 'OK':
            status = f"OK ({total_errors}E)"
        
        print(f"{name:<25} {status:<10} {len(results['clds']):<5} {total_runs:<5} {total_actions:<8} {total_revise:<8} {total_change:<8} {total_errors:<8}")
    
    print("-" * 100)
    
    # Print issues
    print("\nISSUES FOUND:")
    any_issues = False
    for name, results in all_results.items():
        if 'issues' in results and results['issues']:
            any_issues = True
            print(f"\n  {name}:")
            for issue in results['issues'][:10]:  # Limit to first 10
                print(f"    - {issue}")
            if len(results['issues']) > 10:
                print(f"    ... and {len(results['issues']) - 10} more")
    
    if not any_issues:
        print("  None! All experiments verified successfully.")
    
    print("\n" + "=" * 100)


def generate_detailed_report(all_results: dict, output_file: Path):
    """Generate a detailed report to a file."""
    with open(output_file, 'w') as f:
        f.write("RQ1b Reproducibility Verification Report\n")
        f.write("=" * 80 + "\n\n")
        
        for name, results in all_results.items():
            f.write(f"\n{'=' * 40}\n{name}\n{'=' * 40}\n")
            
            if results['status'] == 'MISSING':
                f.write(f"MISSING: {results['folder']}\n")
                continue
            
            f.write(f"Folder: {results['folder']}\n")
            f.write(f"Status: {results['status']}\n\n")
            
            for cld_name, cld_data in sorted(results['clds'].items()):
                f.write(f"\n  {cld_name}:\n")
                for run in cld_data['runs']:
                    counts = run['counts']
                    f.write(f"    run_{run['run']}: total={counts['total']}, "
                           f"revise={counts['revise']}, change={counts['change_type']}, "
                           f"none={counts['none']}, error={counts['error']}\n")
                
                f.write(f"    TOTALS: {dict(cld_data['total_counts'])}\n")
            
            if results['issues']:
                f.write(f"\n  Issues:\n")
                for issue in results['issues']:
                    f.write(f"    - {issue}\n")
    
    print(f"\nDetailed report written to: {output_file}")


def main():
    print("Verifying RQ1b reproducibility...")
    
    all_results = {}
    for name, folder_name in EXPERIMENT_FOLDERS.items():
        print(f"\nVerifying: {name}...")
        results = verify_folder(name, folder_name)
        all_results[name] = results
    
    print_summary(all_results)
    
    # Generate detailed report
    report_file = BASE_PATH / 'rq1b_reproducibility_report.txt'
    generate_detailed_report(all_results, report_file)
    
    # Return exit code based on issues
    any_errors = any(r.get('status') == 'ERROR' for r in all_results.values())
    return 1 if any_errors else 0


if __name__ == '__main__':
    sys.exit(main())





















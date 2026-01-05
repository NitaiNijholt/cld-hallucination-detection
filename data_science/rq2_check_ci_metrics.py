#!/usr/bin/env python3
"""
Check which files have all required CI metrics.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Required CI metrics
REQUIRED_METRICS = [
    'Gen Perplexity',
    'Gen Min Prob',
    'Gen Max Window Entropy',
    'Gen Cosine Similarity'
]

def check_file_metrics(filepath):
    """Check which CI metrics are present in a file."""
    try:
        df = pd.read_excel(filepath, sheet_name='All Edges', nrows=5)
        
        present = []
        missing = []
        
        for metric in REQUIRED_METRICS:
            if metric in df.columns:
                present.append(metric)
            else:
                missing.append(metric)
        
        return {
            'has_all': len(missing) == 0,
            'present': present,
            'missing': missing,
            'count_present': len(present)
        }
    except Exception as e:
        return {
            'has_all': False,
            'present': [],
            'missing': REQUIRED_METRICS,
            'count_present': 0,
            'error': str(e)
        }

def main():
    print("="*80)
    print("RQ2 CI METRICS AVAILABILITY CHECK")
    print("="*80)
    
    # Load deduplicated files
    inventory_dir = Path("parameter_tuning_experiments/rq2_analyses")
    deduplicated_files = list(inventory_dir.glob("rq2_valid_files_deduplicated_*.json"))
    
    if not deduplicated_files:
        print("❌ No deduplicated files found. Run rq2_deduplicate_files.py first.")
        return 1
    
    inventory_file = max(deduplicated_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📋 Loading: {inventory_file.name}")
    
    with open(inventory_file, 'r') as f:
        files = json.load(f)
    
    print(f"   Files to check: {len(files)}")
    
    # Check each file
    print(f"\n🔍 Checking CI metric availability...")
    
    results = []
    for i, file_info in enumerate(files, 1):
        filepath = file_info['filepath']
        
        metrics_status = check_file_metrics(filepath)
        
        result = {
            **file_info,
            **metrics_status
        }
        results.append(result)
        
        if not metrics_status['has_all']:
            filename = Path(filepath).name[:60]
            missing_str = ', '.join(metrics_status['missing'])
            print(f"   [{i}/{len(files)}] ✗ {file_info['experiment_type']}/{file_info['cld']}/{file_info['run']}/{file_info['prompt_type']}")
            print(f"              Missing: {missing_str}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    has_all = [r for r in results if r['has_all']]
    missing_some = [r for r in results if not r['has_all']]
    
    print(f"\n✅ Files with ALL 4 CI metrics: {len(has_all)}/{len(files)}")
    print(f"✗  Files missing metrics: {len(missing_some)}/{len(files)}")
    
    if missing_some:
        print(f"\n⚠️  MISSING METRICS BREAKDOWN:")
        
        # Group by what's missing
        by_missing = defaultdict(list)
        for r in missing_some:
            missing_key = tuple(sorted(r['missing']))
            by_missing[missing_key].append(r)
        
        for missing_combo, file_list in sorted(by_missing.items()):
            print(f"\n   Missing {missing_combo}:")
            for r in file_list:
                print(f"      - {r['experiment_type']}/{r['cld']}/{r['run']}/{r['prompt_type']}")
    
    # By experiment
    print(f"\n{'='*80}")
    print("BY EXPERIMENT TYPE")
    print(f"{'='*80}")
    
    by_exp = defaultdict(lambda: {'total': 0, 'complete': 0})
    for r in results:
        exp = r['experiment_type']
        by_exp[exp]['total'] += 1
        if r['has_all']:
            by_exp[exp]['complete'] += 1
    
    for exp, counts in sorted(by_exp.items()):
        complete_pct = counts['complete'] / counts['total'] * 100 if counts['total'] > 0 else 0
        print(f"   {exp:15s}: {counts['complete']}/{counts['total']} ({complete_pct:.1f}%) with all metrics")
    
    # By CLD
    print(f"\n{'='*80}")
    print("BY CLD")
    print(f"{'='*80}")
    
    by_cld = defaultdict(lambda: {'total': 0, 'complete': 0})
    for r in results:
        cld = r['cld']
        by_cld[cld]['total'] += 1
        if r['has_all']:
            by_cld[cld]['complete'] += 1
    
    for cld, counts in sorted(by_cld.items()):
        complete_pct = counts['complete'] / counts['total'] * 100 if counts['total'] > 0 else 0
        print(f"   {cld:30s}: {counts['complete']}/{counts['total']} ({complete_pct:.1f}%) with all metrics")
    
    # By prompt type
    print(f"\n{'='*80}")
    print("BY PROMPT TYPE")
    print(f"{'='*80}")
    
    by_prompt = defaultdict(lambda: {'total': 0, 'complete': 0})
    for r in results:
        prompt = r['prompt_type']
        by_prompt[prompt]['total'] += 1
        if r['has_all']:
            by_prompt[prompt]['complete'] += 1
    
    for prompt, counts in sorted(by_prompt.items()):
        complete_pct = counts['complete'] / counts['total'] * 100 if counts['total'] > 0 else 0
        print(f"   {prompt:25s}: {counts['complete']}/{counts['total']} ({complete_pct:.1f}%) with all metrics")
    
    # Save detailed results
    output_file = inventory_dir / "rq2_ci_metrics_availability.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_files': len(files),
                'files_with_all_metrics': len(has_all),
                'files_missing_metrics': len(missing_some)
            },
            'by_experiment': dict(by_exp),
            'by_cld': dict(by_cld),
            'by_prompt': dict(by_prompt),
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {output_file.name}")
    
    # Save filtered list (only files with all metrics)
    if len(has_all) < len(files):
        filtered_file = inventory_dir / "rq2_valid_files_complete_metrics.json"
        with open(filtered_file, 'w') as f:
            json.dump(has_all, f, indent=2)
        
        print(f"📄 Filtered list (complete metrics only): {filtered_file.name}")
        print(f"   Use this list for analysis: {len(has_all)} files")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())



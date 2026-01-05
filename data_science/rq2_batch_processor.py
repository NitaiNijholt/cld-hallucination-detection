#!/usr/bin/env python3
"""
RQ2 Batch Processor
Runs run_rq2_single_file_v2.py on all valid Excel files from the inventory.
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
import sys

def load_inventory(inventory_file: str):
    """Load the file inventory JSON."""
    with open(inventory_file, 'r') as f:
        data = json.load(f)
    return data

def run_rq2_analysis(excel_file: str, metric_source='generator'):
    """
    Run RQ2 analysis on a single file.
    
    Args:
        excel_file: Path to Excel file
        metric_source: 'generator' or 'judge'
    
    Returns:
        (success: bool, output: str, error: str)
    """
    cmd = [
        'python3',
        'data_science/run_rq2_single_file_v2.py',
        excel_file,
        '--metric-source', metric_source,
        '--n-permutations', '5000',  # Reduced for speed
        '--n-bootstrap', '5000'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per file
        )
        
        if result.returncode == 0:
            return True, result.stdout, result.stderr
        else:
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "", "Timeout expired (10 minutes)"
    except Exception as e:
        return False, "", str(e)


def main():
    print("="*80)
    print("RQ2 BATCH PROCESSOR")
    print("="*80)
    
    # Find the most recent inventory file
    inventory_dir = Path("parameter_tuning_experiments/rq2_analyses")
    inventory_files = list(inventory_dir.glob("rq2_file_inventory_*.json"))
    
    if not inventory_files:
        print("❌ No inventory files found!")
        print(f"   Run rq2_file_inventory.py first")
        sys.exit(1)
    
    # Use the most recent inventory
    inventory_file = max(inventory_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📋 Loading inventory: {inventory_file.name}")
    
    data = load_inventory(inventory_file)
    inventory = data['inventory']
    stats = data['statistics']
    
    print(f"   Total valid files: {stats['valid_files']}")
    print(f"   Total invalid files: {stats['invalid_files']}")
    
    # Collect all valid files
    valid_files = []
    for exp_type, files in inventory.items():
        for file_info in files:
            if file_info['has_required_columns']:
                valid_files.append({
                    'filepath': file_info['filepath'],
                    'experiment_type': exp_type,
                    'cld': file_info['cld'],
                    'run': file_info['run'],
                    'prompt_type': file_info['prompt_type']
                })
    
    print(f"\n✅ Found {len(valid_files)} valid files to process")
    
    # Create batch output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = inventory_dir / f"rq2_batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Batch output directory: {batch_dir}")
    
    # Process each file
    results = []
    successful = 0
    failed = 0
    
    for i, file_info in enumerate(valid_files, 1):
        filepath = file_info['filepath']
        exp_type = file_info['experiment_type']
        cld = file_info['cld']
        run = file_info['run']
        prompt = file_info['prompt_type']
        
        print(f"\n{'─'*80}")
        print(f"[{i}/{len(valid_files)}] Processing: {exp_type}/{cld}/{run}/{prompt}")
        print(f"{'─'*80}")
        print(f"   File: {Path(filepath).name}")
        
        # Run analysis
        success, stdout, stderr = run_rq2_analysis(filepath, metric_source='generator')
        
        result_info = {
            'index': i,
            'filepath': filepath,
            'experiment_type': exp_type,
            'cld': cld,
            'run': run,
            'prompt_type': prompt,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            print(f"   ✅ SUCCESS")
            successful += 1
            result_info['status'] = 'success'
        else:
            print(f"   ❌ FAILED")
            print(f"   Error: {stderr[:200]}")
            failed += 1
            result_info['status'] = 'failed'
            result_info['error'] = stderr[:500]
        
        results.append(result_info)
        
        # Save progress after each file
        progress_file = batch_dir / "batch_progress.json"
        with open(progress_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_files': len(valid_files),
                'processed': i,
                'successful': successful,
                'failed': failed,
                'results': results
            }, f, indent=2)
    
    # Final summary
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"\nTotal files:  {len(valid_files)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed:     {failed}")
    print(f"Success rate: {successful/len(valid_files)*100:.1f}%")
    
    # Save final results
    final_results_file = batch_dir / "batch_results.json"
    with open(final_results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_files': len(valid_files),
            'successful': successful,
            'failed': failed,
            'success_rate': successful/len(valid_files),
            'results': results
        }, f, indent=2)
    
    print(f"\n📄 Results saved to: {final_results_file}")
    
    # Print failed files if any
    if failed > 0:
        print(f"\n⚠️  Failed files:")
        for result in results:
            if not result['success']:
                print(f"   - {result['filepath']}")
                if 'error' in result:
                    print(f"     Error: {result['error'][:100]}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())



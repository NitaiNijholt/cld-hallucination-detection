#!/usr/bin/env python3
"""
RQ1a Unified Analysis Entry Point

This script provides a unified interface for running all RQ1a analyses:
- Corruption detection (correctness and citation-based)
- Ground truth validation (correctness and citation-based)

Usage:
    python analyze_rq1a.py --type all
    python analyze_rq1a.py --type corruption_correctness
    python analyze_rq1a.py --type corruption_citation
    python analyze_rq1a.py --type groundtruth_correctness
    python analyze_rq1a.py --type groundtruth_citation
"""

import argparse
import subprocess
import sys
import uuid
from pathlib import Path
from datetime import datetime


def generate_run_id() -> str:
    """Generate a short unique run ID for tracing analysis runs."""
    # Format: 8 character hex string from UUID
    return uuid.uuid4().hex[:8]


# Configuration for each experiment type
EXPERIMENTS = {
    'corruption_correctness': {
        'script': 'analyze_rq1a_aggregate_enhanced.py',
        'base_dir': 'final_runs/RQ1a_gt_synth_correctness',
        'description': 'Corruption Detection - Correctness Judge'
    },
    'corruption_citation': {
        'script': 'analyze_rq1a_aggregate_enhanced.py',
        'base_dir': 'final_runs/RQ1a_gt_synth_citation',
        'description': 'Corruption Detection - Citation Judge'
    },
    'groundtruth_correctness': {
        'script': 'analyze_rq1a_ground_truth_enhanced.py',
        'base_dir': 'final_runs/RQ1a_gt_lit_correctness',
        'description': 'Ground Truth Validation - Correctness Judge'
    },
    'groundtruth_citation': {
        'script': 'analyze_rq1a_ground_truth_enhanced.py',
        'base_dir': 'final_runs/RQ1a_gt_lit_citation',
        'description': 'Ground Truth Validation - Citation Judge'
    }
}


def run_analysis(experiment_type: str, workspace_root: Path, run_id: str, timestamp: str) -> dict:
    """
    Run analysis for a specific experiment type.
    
    Args:
        experiment_type: One of 'corruption_correctness', 'corruption_citation', 
                        'groundtruth_correctness', 'groundtruth_citation'
        workspace_root: Path to workspace root
        run_id: Unique identifier for this unified run
        timestamp: Timestamp string for folder naming
    
    Returns:
        Dict with 'success' bool and 'output_dir' Path
    """
    config = EXPERIMENTS.get(experiment_type)
    if not config:
        print(f"Unknown experiment type: {experiment_type}")
        return {'success': False, 'output_dir': None}
    
    script_path = workspace_root / 'data_science' / 'parameter_tuning_experiments' / config['script']
    base_dir = workspace_root / config['base_dir']
    
    # Create output directory with run_id for traceability
    output_dir_name = f"enhanced_analysis_{timestamp}_{run_id}"
    output_dir = base_dir / output_dir_name
    
    print(f"\n{'='*80}")
    print(f"Running: {config['description']}")
    print(f"{'='*80}")
    print(f"Script: {script_path}")
    print(f"Data: {base_dir}")
    print(f"Output: {output_dir}")
    print()
    
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return {'success': False, 'output_dir': None}
    
    if not base_dir.exists():
        print(f"ERROR: Data directory not found: {base_dir}")
        return {'success': False, 'output_dir': None}
    
    # Run the analysis script with explicit output directory
    cmd = [
        sys.executable,
        str(script_path),
        '--base_dir', str(base_dir),
        '--output-dir', str(output_dir)
    ]
    
    try:
        result = subprocess.run(cmd, cwd=str(workspace_root), check=True)
        success = result.returncode == 0
        return {'success': success, 'output_dir': output_dir if success else None}
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Analysis failed with exit code {e.returncode}")
        return {'success': False, 'output_dir': None}
    except Exception as e:
        print(f"ERROR: {e}")
        return {'success': False, 'output_dir': None}


def main():
    parser = argparse.ArgumentParser(
        description='RQ1a Unified Analysis - Run corruption detection and ground truth validation analyses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_rq1a.py --type all                    # Run all 4 analyses
  python analyze_rq1a.py --type corruption_correctness # Run only corruption detection (correctness)
  python analyze_rq1a.py --type groundtruth_citation   # Run only ground truth (citation)

Experiment Types:
  corruption_correctness  - Synthetic corruption detection using correctness judge
  corruption_citation     - Synthetic corruption detection using citation judge  
  groundtruth_correctness - Ground truth validation using correctness judge
  groundtruth_citation    - Ground truth validation using citation judge
  all                     - Run all experiments
        """
    )
    
    parser.add_argument(
        '--type', '-t',
        required=True,
        choices=['all'] + list(EXPERIMENTS.keys()),
        help='Type of analysis to run'
    )
    
    parser.add_argument(
        '--workspace', '-w',
        type=Path,
        default=None,
        help='Workspace root directory (auto-detected if not specified)'
    )
    
    args = parser.parse_args()
    
    # Determine workspace root
    if args.workspace:
        workspace_root = args.workspace
    else:
        # Auto-detect from script location
        workspace_root = Path(__file__).resolve().parents[2]
    
    # Generate unique run ID and timestamp for this unified run
    run_id = generate_run_id()
    run_timestamp = datetime.now()
    timestamp_str = run_timestamp.strftime('%Y%m%d_%H%M%S')
    
    print("=" * 80)
    print("RQ1a UNIFIED ANALYSIS")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Timestamp: {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workspace: {workspace_root}")
    
    # Determine which experiments to run
    if args.type == 'all':
        experiment_types = list(EXPERIMENTS.keys())
    else:
        experiment_types = [args.type]
    
    print(f"Experiments: {', '.join(experiment_types)}")
    
    # Track results
    results = {}
    
    for exp_type in experiment_types:
        result = run_analysis(exp_type, workspace_root, run_id, timestamp_str)
        results[exp_type] = result
    
    # Print summary
    end_timestamp = datetime.now()
    duration = (end_timestamp - run_timestamp).total_seconds()
    
    print("\n" + "=" * 80)
    print("UNIFIED RUN COMPLETE")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Started: {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ended: {end_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.1f} seconds")
    print()
    
    print("=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    
    all_success = True
    for exp_type, result in results.items():
        success = result['success']
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"\n{exp_type}: {status}")
        if result['output_dir']:
            print(f"  Output: {result['output_dir']}")
        if not success:
            all_success = False
    
    print()
    print("=" * 80)
    print("OUTPUT PATHS")
    print("=" * 80)
    
    for exp_type, result in results.items():
        if result['output_dir']:
            print(f"\n📁 {EXPERIMENTS[exp_type]['description']}:")
            print(f"   {result['output_dir']}")
    
    print()
    print("=" * 80)
    
    if all_success:
        print(f"✅ All analyses completed successfully!")
        print(f"   Run ID: {run_id}")
        print(f"   Use this ID to locate all outputs from this unified run.")
        return 0
    else:
        print("❌ Some analyses failed. Check output above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())


#!/usr/bin/env python3
"""
RQ1 Preliminaries Unified Analysis Script

This script orchestrates all preliminary analyses for RQ1:
1. Generator Model Comparison (GPT-4.1 vs Sonar-Pro vs Claude-Sonnet-4)
2. Random Baseline Comparison (LLM vs Monte Carlo)
3. Temperature Sensitivity Analysis (justifies T=0.7)
4. TruthfulQA Judge Verification (validates correctness judge)

Each analysis calls existing scripts and copies outputs to a unified directory
with a unique run ID for traceability.

Usage:
    python analyze_rq1_preliminaries.py --type all
    python analyze_rq1_preliminaries.py --type generator_model
    python analyze_rq1_preliminaries.py --type random_baseline
    python analyze_rq1_preliminaries.py --type temperature_sensitivity
    python analyze_rq1_preliminaries.py --type truthfulqa_verification

Output:
    final_runs/RQ1_preliminaries/unified_analysis_{timestamp}_{run_id}/
"""

import argparse
import subprocess
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
import uuid


# Configuration for each preliminary analysis
ANALYSES = {
    'generator_model': {
        'description': 'Generator Model Comparison (GPT-4.1 vs Sonar-Pro vs Claude-Sonnet-4)',
        'script': 'final_runs/generator_model_comparison_analysis/analyze_generator_models.py',
        'output_files': [
            'generator_model_comparison.pdf',
            'generator_model_comparison.png',
            'generator_model_comparison.tex',
            'generator_model_stats.csv',
            'generator_model_stats.json',
            'generator_model_stats.xlsx',
            'generator_precision_recall.pdf',
            'generator_precision_recall.png'
        ],
        'source_dir': 'final_runs/generator_model_comparison_analysis'
    },
    'random_baseline': {
        'description': 'Random Baseline Comparison (LLM vs Monte Carlo)',
        'script': 'final_runs/random_baseline_generator/compute_random_baseline.py',
        'output_files': [
            'random_baseline_results.json',
            'random_baseline_summary.txt',
            'llm_f1_variability_results.json'
        ],
        'source_dir': 'final_runs/random_baseline_generator',
        # Also need the figure from thesis/figures
        'extra_files': [
            ('thesis/final_thesis/figures/random_baseline_comparison.pdf', 'random_baseline_comparison.pdf'),
            ('thesis/final_thesis/figures/random_baseline_comparison.png', 'random_baseline_comparison.png')
        ]
    },
    'temperature_sensitivity': {
        'description': 'Temperature Sensitivity Analysis (justifies T=0.7)',
        'script': 'data_science/parameter_tuning_experiments/analyze_temperature_sensitivity.py',
        'output_files': [
            'temperature_sensitivity_table.tex',
            'temperature_sensitivity_stats.json',
            'temperature_sensitivity_anova.json',
            'temperature_sensitivity_raw.csv'
        ],
        'source_dir': 'final_runs/temperature_sensitivity_analysis',
        'extra_files': []
    },
    'truthfulqa_verification': {
        'description': 'TruthfulQA Judge Verification (validates correctness judge)',
        'script': 'final_runs/RQ1_verification_Truth_QA/analysis_script/recalculate_metrics.py',
        'script_args': ['final_runs/RQ1_verification_Truth_QA/Data_runs/truthqa_judge_verification_baseline_seed42_temp0/truthqa_judge_baseline_results_20251115_204928.json'],
        'output_files': [],  # Output is to stdout, we'll capture it
        'source_dir': 'final_runs/RQ1_verification_Truth_QA',
        'capture_stdout': True
    }
}


def run_analysis(analysis_type: str, workspace_root: Path, output_dir: Path) -> dict:
    """
    Run a single analysis and copy outputs to unified directory.
    
    Args:
        analysis_type: One of the ANALYSES keys
        workspace_root: Path to workspace root
        output_dir: Unified output directory for this run
    
    Returns:
        dict with status, files copied, and any errors
    """
    config = ANALYSES.get(analysis_type)
    if not config:
        return {'status': 'error', 'message': f'Unknown analysis type: {analysis_type}'}
    
    result = {
        'status': 'success',
        'analysis': analysis_type,
        'description': config['description'],
        'files_copied': [],
        'script_output': None,
        'errors': []
    }
    
    print(f"\n{'='*80}")
    print(f"Running: {config['description']}")
    print(f"{'='*80}")
    
    # Create subdirectory for this analysis
    analysis_output_dir = output_dir / analysis_type
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the script if it exists
    script_path = workspace_root / config['script']
    if script_path.exists():
        print(f"  Script: {script_path}")
        
        cmd = [sys.executable, str(script_path)]
        if 'script_args' in config:
            cmd.extend([str(workspace_root / arg) for arg in config['script_args']])
        
        try:
            if config.get('capture_stdout'):
                proc_result = subprocess.run(
                    cmd, 
                    cwd=str(workspace_root), 
                    capture_output=True, 
                    text=True,
                    timeout=300
                )
                result['script_output'] = proc_result.stdout
                
                # Save stdout to file
                stdout_file = analysis_output_dir / 'analysis_output.txt'
                with open(stdout_file, 'w') as f:
                    f.write(proc_result.stdout)
                    if proc_result.stderr:
                        f.write("\n\n=== STDERR ===\n")
                        f.write(proc_result.stderr)
                result['files_copied'].append(str(stdout_file.name))
                print(f"  ✓ Script output saved: {stdout_file.name}")
            else:
                subprocess.run(cmd, cwd=str(workspace_root), check=True, timeout=300)
                print(f"  ✓ Script completed successfully")
        except subprocess.TimeoutExpired:
            result['errors'].append('Script timed out (300s)')
            print(f"  ⚠ Script timed out")
        except subprocess.CalledProcessError as e:
            result['errors'].append(f'Script failed: {e}')
            print(f"  ⚠ Script failed: {e}")
    else:
        print(f"  ⚠ Script not found: {script_path}")
        result['errors'].append(f'Script not found: {script_path}')
    
    # Copy output files from source directory
    source_dir = workspace_root / config['source_dir']
    for filename in config.get('output_files', []):
        src = source_dir / filename
        if src.exists():
            dst = analysis_output_dir / filename
            shutil.copy2(src, dst)
            result['files_copied'].append(filename)
            print(f"  ✓ Copied: {filename}")
        else:
            result['errors'].append(f'File not found: {src}')
            print(f"  ⚠ File not found: {filename}")
    
    # Copy extra files (from different locations)
    for src_rel, dst_name in config.get('extra_files', []):
        src = workspace_root / src_rel
        if src.exists():
            dst = analysis_output_dir / dst_name
            shutil.copy2(src, dst)
            result['files_copied'].append(dst_name)
            print(f"  ✓ Copied: {dst_name} (from {src_rel})")
        else:
            result['errors'].append(f'Extra file not found: {src}')
            print(f"  ⚠ Extra file not found: {src_rel}")
    
    if result['errors']:
        result['status'] = 'partial'
    
    return result


def generate_summary_report(results: dict, output_dir: Path, run_id: str):
    """Generate a summary report of all analyses."""
    report_path = output_dir / 'analysis_summary.json'
    
    summary = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'output_directory': str(output_dir),
        'analyses': results
    }
    
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary report saved: {report_path}")
    return report_path


def verify_temperature_table(output_dir: Path, workspace_root: Path):
    """
    Temperature sensitivity table is now generated by analyze_temperature_sensitivity.py
    which reads raw Excel files and computes statistics reproducibly.
    This function just verifies the table was generated.
    """
    table_path = output_dir / 'temperature_sensitivity' / 'temperature_sensitivity_table.tex'
    if table_path.exists():
        print(f"  ✓ Temperature sensitivity table generated: {table_path.name}")
    else:
        print(f"  ⚠ Temperature sensitivity table not found at: {table_path}")


def main():
    parser = argparse.ArgumentParser(
        description='RQ1 Preliminaries Unified Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_rq1_preliminaries.py --type all                    # Run all 4 analyses
  python analyze_rq1_preliminaries.py --type generator_model        # Run only generator model comparison
  python analyze_rq1_preliminaries.py --type temperature_sensitivity # Run only temperature sensitivity

Analysis Types:
  generator_model         - Generator Model Comparison (GPT-4.1 vs Sonar-Pro vs Claude-Sonnet-4)
  random_baseline         - Random Baseline Comparison (LLM vs Monte Carlo)  
  temperature_sensitivity - Temperature Sensitivity Analysis (justifies T=0.7)
  truthfulqa_verification - TruthfulQA Judge Verification (validates correctness judge)
  all                     - Run all analyses
        """
    )
    
    parser.add_argument(
        '--type', '-t',
        required=True,
        choices=['all'] + list(ANALYSES.keys()),
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
        workspace_root = Path(__file__).resolve().parents[2]
    
    # Generate unique run ID
    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory
    output_dir = workspace_root / 'final_runs' / 'RQ1_preliminaries' / f'unified_analysis_{timestamp}_{run_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("RQ1 PRELIMINARIES UNIFIED ANALYSIS")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workspace: {workspace_root}")
    print(f"Run ID: {run_id}")
    print(f"Output: {output_dir}")
    
    # Determine which analyses to run
    if args.type == 'all':
        analysis_types = list(ANALYSES.keys())
    else:
        analysis_types = [args.type]
    
    print(f"Analyses: {', '.join(analysis_types)}")
    
    # Run analyses
    results = {}
    for analysis_type in analysis_types:
        results[analysis_type] = run_analysis(analysis_type, workspace_root, output_dir)
    
    # Generate temperature sensitivity LaTeX table if applicable
    if 'temperature_sensitivity' in analysis_types:
        verify_temperature_table(output_dir, workspace_root)
    
    # Generate summary report
    generate_summary_report(results, output_dir, run_id)
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    
    all_success = True
    for analysis_type, result in results.items():
        status_icon = "✓" if result['status'] == 'success' else "⚠" if result['status'] == 'partial' else "✗"
        print(f"\n{status_icon} {result['description']}")
        print(f"   Files: {len(result['files_copied'])}")
        if result['errors']:
            all_success = False
            for err in result['errors']:
                print(f"   ⚠ {err}")
    
    print("\n" + "=" * 80)
    print("OUTPUT PATHS")
    print("=" * 80)
    for analysis_type in analysis_types:
        print(f"\n📁 {ANALYSES[analysis_type]['description']}:")
        print(f"   {output_dir / analysis_type}")
    
    print("\n" + "=" * 80)
    if all_success:
        print(f"✅ All analyses completed successfully!")
    else:
        print(f"⚠️ Some analyses had warnings - check output above")
    print(f"   Run ID: {run_id}")
    print(f"   Output: {output_dir}")
    
    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())


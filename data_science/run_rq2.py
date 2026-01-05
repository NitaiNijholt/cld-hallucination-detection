#!/usr/bin/env python3
"""
Unified RQ2 Analysis Tool

Modes:
  1. Single file:  python run_rq2.py path/to/results.xlsx
  2. Experiment:   python run_rq2.py path/to/exp_YYYYMMDD_HHMMSS_hash/

When given an experiment directory, automatically:
  - Finds all Excel result files
  - Runs RQ2 analysis on each file
  - Generates cross-CLD aggregate report
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def is_excel_file(path: Path) -> bool:
    """Check if path is an Excel file."""
    return path.is_file() and path.suffix in ['.xlsx', '.xls']

def is_experiment_dir(path: Path) -> bool:
    """Check if path looks like an experiment directory."""
    return path.is_dir() and path.name.startswith('exp_')

def find_excel_files(exp_dir: Path) -> list:
    """Find all Excel result files in an experiment directory."""
    excel_files = []
    
    # Pattern: exp_dir/combo_N/prompts_*/CLD_name/runN_hash/results_*.xlsx
    # Also support: exp_dir/CLD_name/runN_hash/results_*.xlsx (legacy)
    
    # Try to find combo directories first
    combo_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith('combo_')]
    
    if combo_dirs:
        # New structure with combo directories
        for combo_dir in combo_dirs:
            for prompt_dir in combo_dir.iterdir():
                if not prompt_dir.is_dir():
                    continue
                
                for cld_dir in prompt_dir.iterdir():
                    if not cld_dir.is_dir():
                        continue
                    
                    for run_dir in cld_dir.iterdir():
                        if not run_dir.is_dir() or not run_dir.name.startswith('run'):
                            continue
                        
                        for file in run_dir.iterdir():
                            if file.name.startswith('results_') and file.suffix == '.xlsx':
                                excel_files.append(file)
    else:
        # Legacy structure: exp_dir/CLD_name/runN_hash/results_*.xlsx
        for cld_dir in exp_dir.iterdir():
            if not cld_dir.is_dir():
                continue
            
            for run_dir in cld_dir.iterdir():
                if not run_dir.is_dir() or not run_dir.name.startswith('run'):
                    continue
                
                for file in run_dir.iterdir():
                    if file.name.startswith('results_') and file.suffix == '.xlsx':
                        excel_files.append(file)
    
    return sorted(excel_files)

def run_single_analysis(excel_file: Path, metric_source: str, hallucination_def: str) -> tuple:
    """Run RQ2 analysis on a single Excel file."""
    cmd = [
        sys.executable,
        "run_rq2_single_file_v2.py",
        "--metric-source", metric_source,
        "--hallucination-def", hallucination_def,
        str(excel_file)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return (True, None)
    except subprocess.CalledProcessError as e:
        return (False, f"Return code {e.returncode}")
    except Exception as e:
        return (False, str(e))

def run_aggregate_analysis(experiment_id: str, metric_source: str, hallucination_def: str) -> tuple:
    """Run aggregate meta-analysis across all RQ2 analyses."""
    cmd = [
        sys.executable,
        "run_rq2_aggregate.py",
        experiment_id,
        "--metric-source", metric_source,
        "--hallucination-def", hallucination_def
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output directly
            text=True,
            check=True
        )
        return (True, None)
    except subprocess.CalledProcessError as e:
        return (False, f"Return code {e.returncode}")
    except Exception as e:
        return (False, str(e))

def main():
    parser = argparse.ArgumentParser(
        description="RQ2 Analysis Tool - Single file or full experiment mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file analysis
  python run_rq2.py results_exp_20251008_180446_a0d1b2de_..._run1_....xlsx
  
  # Full experiment analysis (all files + aggregate)
  python run_rq2.py parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/
  
  # With custom options
  python run_rq2.py exp_dir/ --metric-source judge --hallucination-def judge_aggregate_low
        """
    )
    parser.add_argument(
        "path",
        help="Path to Excel file OR experiment directory"
    )
    parser.add_argument(
        "--metric-source",
        choices=["generator", "judge"],
        default="generator",
        help="CI metric source (default: generator)"
    )
    parser.add_argument(
        "--hallucination-def",
        choices=["ground_truth_FP", "judge_aggregate_low"],
        default="ground_truth_FP",
        help="Hallucination definition (default: ground_truth_FP)"
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip aggregate analysis (only run individual files)"
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"❌ Path does not exist: {path}")
        return 1
    
    # MODE 1: Single Excel file
    if is_excel_file(path):
        print("=" * 100)
        print("RQ2 ANALYSIS - SINGLE FILE MODE")
        print("=" * 100)
        print()
        print(f"📄 File: {path.name}")
        print(f"🔧 Metric source: {args.metric_source}")
        print(f"🎯 Hallucination def: {args.hallucination_def}")
        print()
        
        success, error = run_single_analysis(path, args.metric_source, args.hallucination_def)
        
        if success:
            print()
            print("✅ Analysis complete!")
            return 0
        else:
            print()
            print(f"❌ Analysis failed: {error}")
            return 1
    
    # MODE 2: Experiment directory
    elif is_experiment_dir(path):
        print("=" * 100)
        print("RQ2 ANALYSIS - EXPERIMENT DIRECTORY MODE")
        print("=" * 100)
        print()
        print(f"📁 Experiment: {path.name}")
        print(f"🔧 Metric source: {args.metric_source}")
        print(f"🎯 Hallucination def: {args.hallucination_def}")
        print()
        
        # Find all Excel files
        excel_files = find_excel_files(path)
        
        if not excel_files:
            print(f"❌ No Excel result files found in {path}")
            return 1
        
        print(f"📊 Found {len(excel_files)} result files to analyze:")
        for i, f in enumerate(excel_files, 1):
            # Extract CLD and run from path
            run_dir = f.parent.name
            cld_dir = f.parent.parent.name
            print(f"   {i}. {cld_dir[:50]:<50s} ({run_dir})")
        print()
        
        # Run individual analyses
        print("=" * 100)
        print("STEP 1/2: INDIVIDUAL FILE ANALYSES")
        print("=" * 100)
        print()
        
        results = []
        for i, excel_file in enumerate(excel_files, 1):
            print(f"[{i}/{len(excel_files)}] Analyzing: {excel_file.name}")
            print("-" * 100)
            
            success, error = run_single_analysis(excel_file, args.metric_source, args.hallucination_def)
            results.append((excel_file, success, error))
            
            if success:
                print("✅ SUCCESS")
            else:
                print(f"❌ FAILED: {error}")
            print()
        
        # Summary of individual analyses
        successful = sum(1 for _, success, _ in results if success)
        print("=" * 100)
        print(f"INDIVIDUAL ANALYSES COMPLETE: {successful}/{len(results)} successful")
        print("=" * 100)
        print()
        
        if successful == 0:
            print("❌ All individual analyses failed. Skipping aggregate analysis.")
            return 1
        
        # Run aggregate analysis
        if not args.skip_aggregate:
            print("=" * 100)
            print("STEP 2/2: CROSS-CLD AGGREGATE META-ANALYSIS")
            print("=" * 100)
            print()
            
            experiment_id = path.name
            success, error = run_aggregate_analysis(experiment_id, args.metric_source, args.hallucination_def)
            
            if success:
                print()
                print("=" * 100)
                print("✅ COMPLETE RQ2 ANALYSIS PIPELINE FINISHED!")
                print("=" * 100)
                print()
                print(f"📊 Individual analyses: {successful}/{len(results)}")
                print(f"📈 Aggregate report: Generated")
                print()
                print("📁 Outputs:")
                print(f"   • Individual analyses: parameter_tuning_experiments/rq2_analyses/rq2_analysis_*/")
                print(f"   • Aggregate report: parameter_tuning_experiments/rq2_analyses/rq2_aggregate_*/")
                print()
                return 0
            else:
                print()
                print(f"⚠️  Individual analyses completed, but aggregate failed: {error}")
                print(f"   You can manually run: python run_rq2_aggregate.py {experiment_id}")
                return 1
        else:
            print("=" * 100)
            print("✅ INDIVIDUAL ANALYSES COMPLETE (aggregate skipped)")
            print("=" * 100)
            return 0
    
    else:
        print(f"❌ Path is neither an Excel file nor an experiment directory: {path}")
        print()
        print("Expected:")
        print("  • Excel file: results_exp_*.xlsx")
        print("  • Experiment directory: exp_YYYYMMDD_HHMMSS_hash/")
        return 1

if __name__ == "__main__":
    sys.exit(main())

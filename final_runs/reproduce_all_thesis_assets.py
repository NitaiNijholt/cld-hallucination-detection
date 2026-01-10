#!/usr/bin/env python3
"""
Master Thesis Reproducibility Script
=====================================

Single-command entry point to reproduce ALL thesis figures and tables from
the raw experimental data in final_runs/.

This script:
1. Creates a clean timestamped output directory
2. Runs all 4 unified analysis scripts (RQ1, RQ2, RQ3, SUPP)
3. Creates thesis-compatible symlinks/copies
4. Generates a verification report comparing outputs to thesis requirements

Usage:
    # Reproduce to default timestamped directory
    uv run python final_runs/reproduce_all_thesis_assets.py

    # Reproduce to specific directory
    uv run python final_runs/reproduce_all_thesis_assets.py --output-dir /tmp/thesis_repro

    # Generate verification report only (after reproduction)
    uv run python final_runs/reproduce_all_thesis_assets.py --verify-only --output-dir /path/to/existing

Expected outputs (21 assets):
    RQ1:  12 figures (LLM-as-a-Judge performance metrics)
    RQ2:   4 figures, 1 table (UQ hallucination detection)
    RQ3:   3 figures, 3 tables (Deep Research validation)
    SUPP:  5 figures (time/cost, parallelization, sensitivity, ablation)
"""

import sys
import os
import subprocess
import argparse
import shutil
import json
from datetime import datetime
from pathlib import Path


def run_script(script_path: Path, args: str, description: str, env: dict) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'#'*80}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"Args: {args}")
    print(f"{'#'*80}\n")
    
    cmd = f"python3 {script_path} {args}".strip()
    try:
        subprocess.run(cmd, shell=True, check=True, env=env)
        print(f"\n✅ Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Failed: {description} (exit code: {e.returncode})")
        return False


def count_assets(directory: Path, patterns: list[str]) -> dict:
    """Count files matching patterns in a directory."""
    counts = {}
    for pattern in patterns:
        files = list(directory.rglob(pattern))
        counts[pattern] = len(files)
    return counts


def generate_verification_report(output_dir: Path, project_root: Path) -> dict:
    """Generate a verification report comparing outputs to thesis requirements."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "output_directory": str(output_dir),
        "categories": {},
        "summary": {"total_expected": 0, "total_found": 0, "coverage": 0.0}
    }
    
    # Define expected assets for each category
    expected = {
        "RQ1": {
            "description": "LLM-as-a-Judge Performance Figures",
            "patterns": ["*.png"],
            "expected_count": 12,
            "subdirs": ["RQ1a_gt_synth_correctness", "RQ1a_gt_synth_citation", 
                       "RQ1a_gt_lit_correctness", "RQ1a_gt_lit_citation", "RQ1b_corrector_ablation"],
        },
        "RQ2": {
            "description": "UQ Hallucination Detection Figures & Tables",
            "patterns": ["*.png", "*.tex"],
            "expected_count": 5,
            "subdirs": ["artifacts", "analyses"],
        },
        "RQ3": {
            "description": "Deep Research Validation Figures & Tables",
            "patterns": ["*.png", "*.tex"],
            "expected_count": 6,
            "subdirs": ["figures", "validation", "tables"],
        },
        "SUPP": {
            "description": "Supplementary Analysis Figures",
            "patterns": ["*.png"],
            "expected_count": 5,
            "subdirs": ["."],
        },
        "PRELIM": {
            "description": "Preliminary Analysis Figures & Tables",
            "patterns": ["*.png", "*.tex"],
            "expected_count": 4,
            "subdirs": ["."],
        },
        "VALIDATION": {
            "description": "Validation Tables (TruthfulQA, Human, Uleman's, Physics)",
            "patterns": ["*.tex"],
            "expected_count": 6,
            "subdirs": ["."],
        },
    }
    
    total_expected = 0
    total_found = 0
    
    for category, spec in expected.items():
        category_dir = output_dir / category
        found_files = []
        
        if category_dir.exists():
            for subdir in spec["subdirs"]:
                search_dir = category_dir / subdir if subdir != "." else category_dir
                if search_dir.exists():
                    for pattern in spec["patterns"]:
                        found_files.extend(search_dir.rglob(pattern))
        
        found_count = len(set(found_files))  # Deduplicate
        expected_count = spec["expected_count"]
        
        report["categories"][category] = {
            "description": spec["description"],
            "expected": expected_count,
            "found": found_count,
            "coverage": round(min(found_count / expected_count, 1.0) * 100, 1) if expected_count > 0 else 100.0,
            "files": [str(f.relative_to(output_dir)) for f in sorted(set(found_files))]
        }
        
        total_expected += expected_count
        total_found += min(found_count, expected_count)
    
    report["summary"]["total_expected"] = total_expected
    report["summary"]["total_found"] = total_found
    report["summary"]["coverage"] = round(total_found / total_expected * 100, 1) if total_expected > 0 else 100.0
    
    return report


def print_verification_report(report: dict):
    """Print a human-readable verification report."""
    print("\n" + "="*80)
    print("REPRODUCIBILITY VERIFICATION REPORT")
    print("="*80)
    print(f"Generated: {report['generated_at']}")
    print(f"Output Directory: {report['output_directory']}")
    
    print("\n" + "-"*80)
    print(f"{'Category':<15} {'Description':<40} {'Expected':>10} {'Found':>10} {'Coverage':>10}")
    print("-"*80)
    
    for category, data in report["categories"].items():
        coverage_str = f"{data['coverage']:.1f}%"
        if data['coverage'] >= 100:
            coverage_str = "✓ 100%"
        elif data['coverage'] >= 80:
            coverage_str = f"~ {data['coverage']:.1f}%"
        else:
            coverage_str = f"✗ {data['coverage']:.1f}%"
        
        print(f"{category:<15} {data['description']:<40} {data['expected']:>10} {data['found']:>10} {coverage_str:>10}")
    
    print("-"*80)
    summary = report["summary"]
    total_coverage = f"{summary['coverage']:.1f}%"
    print(f"{'TOTAL':<15} {'All thesis assets':<40} {summary['total_expected']:>10} {summary['total_found']:>10} {total_coverage:>10}")
    print("="*80)
    
    if summary['coverage'] >= 100:
        print("\n✅ ALL THESIS ASSETS SUCCESSFULLY REPRODUCED!")
    elif summary['coverage'] >= 80:
        print(f"\n⚠️ PARTIAL REPRODUCTION: {summary['coverage']:.1f}% of assets generated")
    else:
        print(f"\n❌ LOW REPRODUCTION RATE: {summary['coverage']:.1f}% of assets generated")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all thesis figures and tables from final_runs data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for reproduced assets. Default: timestamped directory in /tmp/",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification on existing output directory (skip reproduction)",
    )
    parser.add_argument(
        "--skip-supp",
        action="store_true",
        help="Skip supplementary analyses (faster reproduction)",
    )
    args = parser.parse_args()

    # Resolve paths
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    FINAL_RUNS = PROJECT_ROOT / "final_runs"
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"/tmp/thesis_reproducibility_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("THESIS REPRODUCIBILITY RUNNER")
    print("="*80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Final Runs:   {FINAL_RUNS}")
    print(f"Output Dir:   {output_dir}")
    print(f"Mode:         {'Verify Only' if args.verify_only else 'Full Reproduction'}")
    print("="*80)

    env = os.environ.copy()
    os.chdir(PROJECT_ROOT)

    if not args.verify_only:
        # Track success/failure for each RQ
        results = {}
        
        # =====================================================================
        # RQ1: LLM-as-a-Judge
        # =====================================================================
        rq1_script = FINAL_RUNS / "RQ1_unified_analysis.py"
        rq1_output = output_dir / "RQ1"
        results["RQ1"] = run_script(
            rq1_script,
            f"--run-dir {rq1_output}",
            "RQ1: LLM-as-a-Judge Analysis",
            env
        )
        
        # =====================================================================
        # RQ2: UQ Hallucination Detection
        # =====================================================================
        rq2_script = FINAL_RUNS / "RQ2_unified_analysis.py"
        rq2_output = output_dir / "RQ2"
        results["RQ2"] = run_script(
            rq2_script,
            f"--run-dir {rq2_output}",
            "RQ2: UQ Hallucination Detection",
            env
        )
        
        # =====================================================================
        # RQ3: Deep Research Validation
        # =====================================================================
        rq3_script = FINAL_RUNS / "RQ3_unified_analysis.py"
        rq3_output = output_dir / "RQ3"
        results["RQ3"] = run_script(
            rq3_script,
            f"--run-dir {rq3_output}",
            "RQ3: Deep Research Validation",
            env
        )
        
        # =====================================================================
        # SUPP: Supplementary Analyses
        # =====================================================================
        if not args.skip_supp:
            supp_script = FINAL_RUNS / "SUPP_unified_analysis.py"
            supp_output = output_dir / "SUPP"
            results["SUPP"] = run_script(
                supp_script,
                f"--run-dir {supp_output}",
                "SUPP: Supplementary Analyses",
                env
            )
        else:
            print("\n⏭️ Skipping supplementary analyses (--skip-supp)")
            results["SUPP"] = None
        
        # =====================================================================
        # PRELIM: Preliminary Analyses (Random Baseline + Temperature Sensitivity)
        # =====================================================================
        prelim_script = FINAL_RUNS / "PRELIM_unified_analysis.py"
        prelim_output = output_dir / "PRELIM"
        results["PRELIM"] = run_script(
            prelim_script,
            f"--run-dir {prelim_output}",
            "PRELIM: Preliminary Analyses",
            env
        )
        
        # =====================================================================
        # VALIDATION: Validation Studies (TruthfulQA, Human, Uleman's, Physics)
        # =====================================================================
        validation_script = FINAL_RUNS / "VALIDATION_unified_analysis.py"
        validation_output = output_dir / "VALIDATION"
        results["VALIDATION"] = run_script(
            validation_script,
            f"--run-dir {validation_output}",
            "VALIDATION: Validation Studies",
            env
        )
        
        # =====================================================================
        # Effect Size Summary Table (generated/)
        # =====================================================================
        effect_size_script = FINAL_RUNS / "retrieve_effect_sizes.py"
        if effect_size_script.exists():
            print("\n" + "#"*80)
            print("Running: Effect Size Summary Table")
            print("#"*80)
            try:
                import subprocess
                subprocess.run(f"python3 {effect_size_script}", shell=True, check=True, env=env)
                results["EFFECT_SIZES"] = True
                print("✅ Completed: Effect Size Summary Table")
            except:
                results["EFFECT_SIZES"] = False
                print("❌ Failed: Effect Size Summary Table")
        
        # Print run summary
        print("\n" + "="*80)
        print("ANALYSIS RUNS SUMMARY")
        print("="*80)
        for rq, success in results.items():
            if success is None:
                status = "⏭️ Skipped"
            elif success:
                status = "✅ Success"
            else:
                status = "❌ Failed"
            print(f"  {rq}: {status}")

    # =========================================================================
    # Generate Verification Report
    # =========================================================================
    report = generate_verification_report(output_dir, PROJECT_ROOT)
    print_verification_report(report)
    
    # Save report to JSON
    report_path = output_dir / "reproducibility_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Report saved to: {report_path}")
    
    # Create master 'latest' symlink
    latest_link = output_dir.parent / "thesis_latest"
    if latest_link.exists() or latest_link.is_symlink():
        try:
            latest_link.unlink()
        except PermissionError:
            pass
    try:
        latest_link.symlink_to(output_dir, target_is_directory=True)
        print(f"🔗 Created symlink: {latest_link} → {output_dir.name}")
    except (PermissionError, OSError):
        pass

    # =========================================================================
    # Auto-map outputs to thesis figure structure
    # =========================================================================
    if not args.verify_only:
        thesis_figures_dir = PROJECT_ROOT / "thesis" / "reproducible_version" / "Figures" / "final_runs"
        create_structure_script = FINAL_RUNS / "create_thesis_figure_structure.py"
        
        if create_structure_script.exists() and thesis_figures_dir.parent.exists():
            print("\n" + "="*80)
            print("MAPPING OUTPUTS TO THESIS STRUCTURE")
            print("="*80)
            try:
                cmd = f"python3 {create_structure_script} --source {output_dir} --target {thesis_figures_dir}"
                subprocess.run(cmd, shell=True, check=True, env=env)
                print(f"\n✅ Outputs mapped to: {thesis_figures_dir}")
            except subprocess.CalledProcessError as e:
                print(f"\n⚠️ Failed to map outputs (exit code: {e.returncode})")
        else:
            print(f"\n⚠️ Skipping thesis mapping (script or thesis dir not found)")

    print("\n" + "="*80)
    print("✅ REPRODUCIBILITY RUN COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Thesis figures:   {PROJECT_ROOT / 'thesis' / 'reproducible_version' / 'Figures' / 'final_runs'}")
    print(f"\nNext step: Compile the thesis")
    print(f"  cd thesis/reproducible_version")
    print(f"  sed -i 's/oneside, draft/oneside/' main.tex  # Switch to final mode")
    print(f"  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex")


if __name__ == "__main__":
    main()







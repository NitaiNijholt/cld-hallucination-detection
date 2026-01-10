#!/usr/bin/env python3
"""
Validation Unified Analysis Script
Single entry point for all validation analyses and tables:
1. TruthfulQA Judge Verification
2. Human-LLM Agreement Validation
3. Uleman's External CLD Validation
4. Physics CLDs Validation

Usage:
    python3 final_runs/VALIDATION_unified_analysis.py --run-dir /path/to/output

Outputs:
    - truthqa_verification_table.tex
    - human_validation_table.tex
    - human_validation_confusion_table.tex
    - search_provider_comparison_table.tex
    - physics_comparison_table.tex
    - retrieval_comparison_table.tex
"""

import sys
import os
import subprocess
import argparse
import shutil
from datetime import datetime
from pathlib import Path


def run_command(cmd, description, *, env=None, cwd=None):
    """Run a shell command with logging."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    if cwd:
        print(f"CWD: {cwd}")
    print(f"{'='*80}\n")
    try:
        subprocess.run(cmd, shell=True, check=True, env=env, cwd=cwd)
        print(f"\n✅ Successfully completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Error running: {description}")
        print(f"Exit code: {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validation unified analysis runner")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional output run directory."
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    FINAL_RUNS = PROJECT_ROOT / "final_runs"
    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {PROJECT_ROOT}")

    print("\n" + "#"*80)
    print("VALIDATION: Judge Verification and External Validation Studies")
    print("#"*80)

    env = os.environ.copy()
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = FINAL_RUNS / f"validation_unified_output_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nValidation output directory: {run_dir}")

    generated_tables = []

    # =========================================================================
    # 1. TRUTHFULQA JUDGE VERIFICATION
    # =========================================================================
    print("\n" + "-"*80)
    print("1. TruthfulQA Judge Verification")
    print("-"*80)

    truthqa_script = FINAL_RUNS / "validation_RQ1_truthfulqa_judge/analysis_scripts/recalculate_metrics.py"
    if truthqa_script.exists():
        success = run_command(f"python3 {truthqa_script}", "TruthfulQA Verification", env=env,
                             cwd=str(FINAL_RUNS / "validation_RQ1_truthfulqa_judge"))
    
    # Copy table (whether generated or pre-existing)
    src = FINAL_RUNS / "validation_RQ1_truthfulqa_judge/truthqa_verification_table.tex"
    if src.exists():
        dst = run_dir / "truthqa_verification_table.tex"
        shutil.copy(src, dst)
        generated_tables.append(dst)
        print(f"  ✓ Copied: truthqa_verification_table.tex")

    # =========================================================================
    # 2. HUMAN-LLM AGREEMENT VALIDATION
    # =========================================================================
    print("\n" + "-"*80)
    print("2. Human-LLM Agreement Validation")
    print("-"*80)

    human_script = FINAL_RUNS / "validation_RQ1a_human_agreement/analysis_scripts/analyze_human_validation.py"
    if human_script.exists():
        success = run_command(f"python3 {human_script}", "Human Validation Analysis", env=env,
                             cwd=str(FINAL_RUNS / "validation_RQ1a_human_agreement"))
    
    # Copy tables
    tables = ["human_validation_table.tex", "human_validation_confusion_table.tex"]
    for table in tables:
        src = FINAL_RUNS / f"validation_RQ1a_human_agreement/{table}"
        if src.exists():
            dst = run_dir / table
            shutil.copy(src, dst)
            generated_tables.append(dst)
            print(f"  ✓ Copied: {table}")

    # =========================================================================
    # 3. ULEMAN'S EXTERNAL CLD VALIDATION
    # =========================================================================
    print("\n" + "-"*80)
    print("3. Uleman's External CLD Validation")
    print("-"*80)

    ulemans_script = FINAL_RUNS / "validation_RQ1a_ulemans_external/analysis_scripts/compare_jina_vs_scraper.py"
    if ulemans_script.exists():
        success = run_command(f"python3 {ulemans_script}", "Uleman's Search Provider Comparison", env=env,
                             cwd=str(FINAL_RUNS / "validation_RQ1a_ulemans_external"))
    
    # Copy table
    src = FINAL_RUNS / "validation_RQ1a_ulemans_external/search_provider_comparison_table.tex"
    if src.exists():
        dst = run_dir / "search_provider_comparison_table.tex"
        shutil.copy(src, dst)
        generated_tables.append(dst)
        print(f"  ✓ Copied: search_provider_comparison_table.tex")

    # =========================================================================
    # 4. PHYSICS CLDS VALIDATION
    # =========================================================================
    print("\n" + "-"*80)
    print("4. Physics CLDs Validation")
    print("-"*80)

    physics_script = FINAL_RUNS / "validation_RQ1a_physics_clds/analysis_scripts/generate_comparison_table.py"
    if physics_script.exists():
        success = run_command(f"python3 {physics_script}", "Physics Comparison Table", env=env,
                             cwd=str(FINAL_RUNS / "validation_RQ1a_physics_clds"))
    
    retrieval_script = FINAL_RUNS / "validation_RQ1a_physics_clds/analysis_scripts/generate_retrieval_comparison_table.py"
    if retrieval_script.exists():
        success = run_command(f"python3 {retrieval_script}", "Retrieval Comparison Table", env=env,
                             cwd=str(FINAL_RUNS / "validation_RQ1a_physics_clds"))
    
    # Copy tables
    tables = ["physics_comparison_table.tex", "retrieval_comparison_table.tex"]
    for table in tables:
        src = FINAL_RUNS / f"validation_RQ1a_physics_clds/{table}"
        if src.exists():
            dst = run_dir / table
            shutil.copy(src, dst)
            generated_tables.append(dst)
            print(f"  ✓ Copied: {table}")

    # =========================================================================
    # Create 'latest' symlink
    # =========================================================================
    latest_link = run_dir.parent / "VALIDATION_latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)
    print(f"\n✓ Created symlink: VALIDATION_latest → {run_dir.name}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("VALIDATION ANALYSES SUMMARY")
    print("="*80)
    print(f"\nOutput directory: {run_dir}")
    print(f"\nGenerated tables ({len(generated_tables)}):")
    for table in generated_tables:
        print(f"  • {table.name}")

    print("\n" + "="*80)
    print("✅ Validation analyses completed!")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())


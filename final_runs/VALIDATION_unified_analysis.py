#!/usr/bin/env python3
"""
Validation Unified Analysis Script
Single entry point for all validation analyses and tables:
1. TruthfulQA Judge Verification
2. Human-LLM Agreement Validation
3. Uleman's External CLD Validation
4. Physics CLDs Validation
5. Effect Size Summary Table (cross-RQ aggregate)

Usage:
    python3 final_runs/VALIDATION_unified_analysis.py --run-dir /path/to/output

Outputs:
    - truthqa_verification_table.tex
    - human_validation_table.tex
    - human_validation_confusion_table.tex
    - search_provider_comparison_table.tex
    - physics_comparison_table.tex
    - retrieval_comparison_table.tex
    - effect_size_summary_table.tex
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

    truthqa_base = FINAL_RUNS / "validation_RQ1_truthfulqa_judge"
    truthqa_output = truthqa_base / "truthqa_verification_table.tex"
    truthqa_script = truthqa_base / "analysis_scripts/recalculate_metrics.py"
    
    if truthqa_script.exists():
        # Find the most recent results file
        data_files = list((truthqa_base / "Data").glob("truthqa_judge_baseline_results_*.json"))
        if data_files:
            latest_data = sorted(data_files)[-1]
            success = run_command(
                f"python3 {truthqa_script} {latest_data} --output {truthqa_output}", 
                "TruthfulQA Verification", env=env, cwd=str(truthqa_base)
            )
    
    # Copy generated table to run_dir
    if truthqa_output.exists():
        dst = run_dir / "truthqa_verification_table.tex"
        shutil.copy(truthqa_output, dst)
        generated_tables.append(dst)
        print(f"  ✓ Generated: truthqa_verification_table.tex")

    # =========================================================================
    # 2. HUMAN-LLM AGREEMENT VALIDATION
    # =========================================================================
    print("\n" + "-"*80)
    print("2. Human-LLM Agreement Validation")
    print("-"*80)

    human_base = FINAL_RUNS / "validation_RQ1a_human_agreement"
    human_script = human_base / "analysis_scripts/analyze_human_validation.py"
    human_output = human_base / "human_validation_table.tex"
    human_confusion_output = human_base / "human_validation_confusion_table.tex"
    
    if human_script.exists():
        success = run_command(
            f"python3 {human_script} --output {human_output} --confusion-output {human_confusion_output}",
            "Human-LLM Agreement Analysis", env=env, cwd=str(human_base)
        )
    
    # Copy generated tables to run_dir
    for src, name in [(human_output, "human_validation_table.tex"),
                      (human_confusion_output, "human_validation_confusion_table.tex")]:
        if src.exists():
            content = src.read_text()
            if "Placeholder" not in content and "[Table:" not in content:
                dst = run_dir / name
                shutil.copy(src, dst)
                generated_tables.append(dst)
                print(f"  ✓ Generated: {name}")

    # =========================================================================
    # 3. ULEMAN'S EXTERNAL CLD VALIDATION
    # =========================================================================
    print("\n" + "-"*80)
    print("3. Uleman's External CLD Validation")
    print("-"*80)

    ulemans_base = FINAL_RUNS / "validation_RQ1a_ulemans_external"
    ulemans_output = ulemans_base / "search_provider_comparison_table.tex"
    
    # Generate search provider comparison table
    provider_script = ulemans_base / "analysis_scripts/generate_search_provider_table.py"
    if provider_script.exists():
        success = run_command(
            f"python3 {provider_script} --output {ulemans_output}",
            "Search Provider Comparison Table", env=env, cwd=str(ulemans_base)
        )
    
    # Also run Jina vs Scraper comparison (generates xlsx)
    jina_script = ulemans_base / "analysis_scripts/compare_jina_vs_scraper.py"
    if jina_script.exists():
        success = run_command(f"python3 {jina_script}", "Jina vs ContentScraper Comparison", env=env,
                             cwd=str(ulemans_base))
    
    # Copy table to run_dir
    if ulemans_output.exists():
        dst = run_dir / "search_provider_comparison_table.tex"
        shutil.copy(ulemans_output, dst)
        generated_tables.append(dst)
        print(f"  ✓ Generated: search_provider_comparison_table.tex")

    # =========================================================================
    # 4. PHYSICS CLDS VALIDATION
    # =========================================================================
    print("\n" + "-"*80)
    print("4. Physics CLDs Validation")
    print("-"*80)

    physics_base = FINAL_RUNS / "validation_RQ1a_physics_clds"
    physics_output = physics_base / "physics_comparison_table.tex"
    retrieval_output = physics_base / "retrieval_comparison_table.tex"
    
    physics_script = physics_base / "analysis_scripts/generate_comparison_table.py"
    if physics_script.exists():
        success = run_command(f"python3 {physics_script} --output {physics_output}", 
                             "Physics Comparison Table", env=env, cwd=str(physics_base))
    
    retrieval_script = physics_base / "analysis_scripts/generate_retrieval_comparison_table.py"
    if retrieval_script.exists():
        success = run_command(f"python3 {retrieval_script} --output {retrieval_output}", 
                             "Retrieval Comparison Table", env=env, cwd=str(physics_base))
    
    # Copy generated tables to run_dir
    for src, name in [(physics_output, "physics_comparison_table.tex"), 
                      (retrieval_output, "retrieval_comparison_table.tex")]:
        if src.exists():
            dst = run_dir / name
            shutil.copy(src, dst)
            generated_tables.append(dst)
            print(f"  ✓ Generated: {name}")

    # =========================================================================
    # 5. EFFECT SIZE SUMMARY TABLE
    # =========================================================================
    print("\n" + "-"*80)
    print("5. Effect Size Summary Table")
    print("-"*80)

    effect_size_script = FINAL_RUNS / "retrieve_effect_sizes.py"
    if effect_size_script.exists():
        success = run_command(f"python3 {effect_size_script} --run-dir {run_dir}", 
                             "Effect Size Summary Table", env=env)
        if success:
            if (run_dir / "effect_size_summary_table.tex").exists():
                generated_tables.append(run_dir / "effect_size_summary_table.tex")
                print(f"  ✓ Generated: effect_size_summary_table.tex")
    else:
        print(f"  ⚠️ Script not found: {effect_size_script}")
    
    # Copy RQ3 enrichment rows (static fragment included in effect size table)
    enrichment_rows_src = PROJECT_ROOT / "thesis/reproducible_version/generated/effect_size_summary_rq3_enrichment_rows.tex"
    if enrichment_rows_src.exists():
        shutil.copy(enrichment_rows_src, run_dir / "effect_size_summary_rq3_enrichment_rows.tex")
        print(f"  ✓ Copied: effect_size_summary_rq3_enrichment_rows.tex")

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


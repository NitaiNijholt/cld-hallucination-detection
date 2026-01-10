#!/usr/bin/env python3
"""
Supplementary Unified Analysis Script
Single entry point for all supplementary analyses:
1. Time/Cost Scaling Analysis (Figure 3: Generation vs Judging costs)
2. Parallelization Benchmark Analysis (Parallelization thesis figure)
3. Prompt Sensitivity Analysis (Judge and Corrector sensitivity figures)

Usage:
    python3 final_runs/SUPP_unified_analysis.py --run-dir /path/to/output

Outputs:
    - figure3_generation_vs_judging.png (time/cost scaling)
    - parallelization_thesis_figure.png (parallelization benchmark)
    - prompt_sensitivity_figure.png (judge sensitivity)
    - corrector_sensitivity_figure.png (corrector sensitivity)
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
    parser = argparse.ArgumentParser(description="Supplementary unified analysis runner")
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Optional output run directory. When set, all supplementary scripts will write "
            "their outputs to this directory."
        ),
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    FINAL_RUNS = PROJECT_ROOT / "final_runs"
    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {PROJECT_ROOT}")

    print("\n" + "#"*80)
    print("SUPPLEMENTARY: Time/Cost, Parallelization, and Sensitivity Analyses")
    print("#"*80)

    env = os.environ.copy()
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = FINAL_RUNS / f"supp_unified_output_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSupplementary output directory: {run_dir}")

    # Track generated figures for final summary
    generated_figures = []

    # =========================================================================
    # 1. TIME/COST SCALING ANALYSIS
    # =========================================================================
    print("\n" + "-"*80)
    print("1. Time/Cost Scaling Analysis")
    print("-"*80)

    time_cost_script = FINAL_RUNS / "supp_time_cost_scaling/analysis_scripts/aggregate_time_cost_scaling.py"
    time_cost_output = run_dir / "time_cost_scaling"
    time_cost_output.mkdir(parents=True, exist_ok=True)

    if time_cost_script.exists():
        cmd = f"python3 {time_cost_script} --base_dir {FINAL_RUNS} --output_dir {time_cost_output}"
        success = run_command(cmd, "Time/Cost Scaling Aggregation", env=env)
        
        # Find and copy the main thesis figure.
        # IMPORTANT: be specific here; otherwise we can accidentally copy figure1_scaling_analysis.png
        # and overwrite the intended combined scaling plot.
        if success:
            dst = run_dir / "figure3_generation_vs_judging.png"

            # Preferred exact path (this is where aggregate_time_cost_scaling.py writes it)
            preferred = time_cost_output / "figures" / "figure3_generation_vs_judging.png"
            if preferred.exists():
                shutil.copy(preferred, dst)
                generated_figures.append(dst)
                print(f"  ✓ Copied: {preferred.name} → figure3_generation_vs_judging.png")
            else:
                # Fallback: exact filename anywhere under the output directory
                candidates = list(time_cost_output.rglob("figure3_generation_vs_judging.png"))
                if candidates:
                    src = candidates[0]
                    shutil.copy(src, dst)
                    generated_figures.append(dst)
                    print(f"  ✓ Copied: {src.name} → figure3_generation_vs_judging.png")
                else:
                    # Final fallback: try pattern match, but avoid copying figure1/scaling_analysis
                    pattern_candidates = [
                        p for p in time_cost_output.rglob("*generation*judging*.png")
                        if "figure1" not in p.name.lower() and "scaling_analysis" not in p.name.lower()
                    ]
                    if pattern_candidates:
                        src = pattern_candidates[0]
                        shutil.copy(src, dst)
                        generated_figures.append(dst)
                        print(f"  ✓ Copied: {src.name} → figure3_generation_vs_judging.png")
                    else:
                        print("  ⚠️ Could not find figure3_generation_vs_judging.png in time/cost scaling outputs.")
    else:
        print(f"  ⚠️ Script not found: {time_cost_script}")

    # Generate Table 21: Scaling Summary (for thesis)
    table21_script = FINAL_RUNS / "supp_cost_analysis/analysis_scripts/generate_table21_scaling_summary.py"
    if table21_script.exists():
        cmd = f"python3 {table21_script}"
        success = run_command(cmd, "Table 21: Scaling Summary", env=env)
        if success:
            # Copy the generated table to output
            src_table = FINAL_RUNS / "latex/table21_scaling_summary.tex"
            if src_table.exists():
                latex_dir = run_dir / "latex"
                latex_dir.mkdir(parents=True, exist_ok=True)
                dst = latex_dir / "table21_scaling_summary.tex"
                shutil.copy(src_table, dst)
                print(f"  ✓ Copied: table21_scaling_summary.tex")

    # =========================================================================
    # 2. PARALLELIZATION BENCHMARK ANALYSIS
    # =========================================================================
    print("\n" + "-"*80)
    print("2. Parallelization Benchmark Analysis")
    print("-"*80)

    parallel_script = FINAL_RUNS / "supp_parallelization_benchmark/analysis_scripts/analyze_parallelization_results.py"
    parallel_output = run_dir / "parallelization"
    parallel_output.mkdir(parents=True, exist_ok=True)

    if parallel_script.exists():
        # Find the latest results file
        parallel_data = FINAL_RUNS / "supp_parallelization_benchmark/Data"
        results_files = list(parallel_data.glob("*.xlsx")) if parallel_data.exists() else []
        
        if results_files:
            latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
            cmd = f"python3 {parallel_script} {latest_results}"
            # Run in the output directory to capture figures
            success = run_command(cmd, "Parallelization Analysis", env=env, cwd=str(parallel_output))
            
            if success:
                # Find generated figures - check both output dir and source dir
                found = False
                search_dirs = [
                    parallel_output,
                    FINAL_RUNS / "supp_parallelization_benchmark/figures",
                    FINAL_RUNS / "supp_parallelization_benchmark",
                ]
                for search_dir in search_dirs:
                    if not search_dir.exists():
                        continue
                    for fig in search_dir.glob("*thesis*.png"):
                        dst = run_dir / "parallelization_thesis_figure.png"
                        shutil.copy(fig, dst)
                        generated_figures.append(dst)
                        print(f"  ✓ Copied: {fig.name} → parallelization_thesis_figure.png")
                        found = True
                        break
                    if found:
                        break
        else:
            print(f"  ⚠️ No results files found in {parallel_data}")
    else:
        print(f"  ⚠️ Script not found: {parallel_script}")

    # =========================================================================
    # 3. PROMPT SENSITIVITY ANALYSIS (JUDGE)
    # =========================================================================
    print("\n" + "-"*80)
    print("3. Prompt Sensitivity Analysis (Judge)")
    print("-"*80)

    sensitivity_script = FINAL_RUNS / "supp_prompt_sensitivity/analysis_scripts/sensitivity_analysis_simple.py"
    sensitivity_output = run_dir / "prompt_sensitivity"
    sensitivity_output.mkdir(parents=True, exist_ok=True)

    # Check for existing pre-computed figures first (these are expensive to regenerate)
    precomputed_judge = FINAL_RUNS / "supp_prompt_sensitivity/prompt_sensitivity_figure.png"
    precomputed_corrector = FINAL_RUNS / "supp_prompt_sensitivity/corrector_sensitivity_figure.png"

    if precomputed_judge.exists():
        dst = run_dir / "prompt_sensitivity_figure.png"
        shutil.copy(precomputed_judge, dst)
        generated_figures.append(dst)
        print(f"  ✓ Copied pre-computed: prompt_sensitivity_figure.png")
    elif sensitivity_script.exists():
        # Try running the script (requires embeddings, may be slow)
        env_sens = env.copy()
        env_sens["SUPP_OUTPUT_DIR"] = str(sensitivity_output)
        cmd = f"python3 {sensitivity_script}"
        success = run_command(cmd, "Judge Sensitivity Analysis", env=env_sens, cwd=str(FINAL_RUNS / "supp_prompt_sensitivity"))
        
        if success:
            # Script outputs to hardcoded path - check there first
            hardcoded_fig = FINAL_RUNS / "Sensitivity_analysis_simple/prompt_sensitivity_figure.png"
            if hardcoded_fig.exists():
                dst = run_dir / "prompt_sensitivity_figure.png"
                shutil.copy(hardcoded_fig, dst)
                generated_figures.append(dst)
                print(f"  ✓ Copied: prompt_sensitivity_figure.png")
            else:
                for fig in sensitivity_output.glob("*sensitivity*.png"):
                    if "corrector" not in fig.name.lower():
                        dst = run_dir / "prompt_sensitivity_figure.png"
                        shutil.copy(fig, dst)
                        generated_figures.append(dst)
                        print(f"  ✓ Copied: {fig.name} → prompt_sensitivity_figure.png")
                        break

    # =========================================================================
    # 4. CORRECTOR SENSITIVITY ANALYSIS
    # =========================================================================
    print("\n" + "-"*80)
    print("4. Corrector Sensitivity Analysis")
    print("-"*80)

    corrector_script = FINAL_RUNS / "supp_prompt_sensitivity/analysis_scripts/sensitivity_analysis_corrector.py"

    if precomputed_corrector.exists():
        dst = run_dir / "corrector_sensitivity_figure.png"
        shutil.copy(precomputed_corrector, dst)
        generated_figures.append(dst)
        print(f"  ✓ Copied pre-computed: corrector_sensitivity_figure.png")
    elif corrector_script.exists():
        env_corr = env.copy()
        env_corr["SUPP_OUTPUT_DIR"] = str(sensitivity_output)
        cmd = f"python3 {corrector_script}"
        success = run_command(cmd, "Corrector Sensitivity Analysis", env=env_corr, cwd=str(FINAL_RUNS / "supp_prompt_sensitivity"))
        
        if success:
            # Script outputs to hardcoded path - check there first
            hardcoded_fig = FINAL_RUNS / "Sensitivity_analysis_simple/corrector_sensitivity_figure.png"
            if hardcoded_fig.exists():
                dst = run_dir / "corrector_sensitivity_figure.png"
                shutil.copy(hardcoded_fig, dst)
                generated_figures.append(dst)
                print(f"  ✓ Copied: corrector_sensitivity_figure.png")
            else:
                for fig in sensitivity_output.glob("*corrector*sensitivity*.png"):
                    dst = run_dir / "corrector_sensitivity_figure.png"
                    shutil.copy(fig, dst)
                    generated_figures.append(dst)
                    print(f"  ✓ Copied: {fig.name} → corrector_sensitivity_figure.png")
                    break

    # =========================================================================
    # 5. COPY STATIC ASSETS (edge ablation figure - not generated, just copied)
    # =========================================================================
    print("\n" + "-"*80)
    print("5. Static Assets (Edge Ablation)")
    print("-"*80)

    edge_ablation_src = PROJECT_ROOT / "Figures/edge_ablation_f1_ordered.png"
    if edge_ablation_src.exists():
        dst = run_dir / "edge_ablation_f1_ordered.png"
        shutil.copy(edge_ablation_src, dst)
        generated_figures.append(dst)
        print(f"  ✓ Copied static asset: edge_ablation_f1_ordered.png")
    else:
        # Try alternative location
        alt_src = PROJECT_ROOT / "thesis/Figures/edge_ablation_f1_ordered.png"
        if alt_src.exists():
            dst = run_dir / "edge_ablation_f1_ordered.png"
            shutil.copy(alt_src, dst)
            generated_figures.append(dst)
            print(f"  ✓ Copied static asset: edge_ablation_f1_ordered.png")
        else:
            print(f"  ⚠️ Static asset not found: edge_ablation_f1_ordered.png")

    # =========================================================================
    # Create 'latest' symlink
    # =========================================================================
    latest_link = run_dir.parent / "SUPP_latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir, target_is_directory=True)
    print(f"\n✓ Created symlink: SUPP_latest → {run_dir.name}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUPPLEMENTARY ANALYSES SUMMARY")
    print("="*80)
    print(f"\nOutput directory: {run_dir}")
    print(f"\nGenerated figures ({len(generated_figures)}):")
    for fig in generated_figures:
        print(f"  • {fig.name}")
    
    expected_figures = [
        "figure3_generation_vs_judging.png",
        "parallelization_thesis_figure.png",
        "prompt_sensitivity_figure.png",
        "corrector_sensitivity_figure.png",
        "edge_ablation_f1_ordered.png",
    ]
    
    missing = [f for f in expected_figures if not (run_dir / f).exists()]
    if missing:
        print(f"\n⚠️ Missing figures ({len(missing)}):")
        for f in missing:
            print(f"  • {f}")
    else:
        print(f"\n✅ All expected supplementary figures generated!")

    print("\n" + "="*80)
    print("✅ Supplementary analyses completed!")
    print("="*80)


if __name__ == "__main__":
    main()



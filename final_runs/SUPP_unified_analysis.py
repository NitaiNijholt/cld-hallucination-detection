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

    figure3_generated = False
    if time_cost_script.exists():
        # Include generator timing data from final_runs folder
        gen_dir = FINAL_RUNS / "supp_time_cost_scaling" / "Data" / "generation_timing"
        gen_dir_arg = f"--gen-dir {gen_dir}" if gen_dir.exists() else ""
        cmd = f"python3 {time_cost_script} --base_dir {FINAL_RUNS} --output_dir {time_cost_output} {gen_dir_arg}"
        success = run_command(cmd, "Time/Cost Scaling Aggregation", env=env)
        
        # Find and copy the main thesis figure (script outputs to figures/ subdirectory)
        if success:
            figures_dir = time_cost_output / "figures"
            thesis_fig = figures_dir / "figure3_generation_vs_judging.png"
            if thesis_fig.exists():
                # Copy to run_dir for reproducibility tracking
                dst = run_dir / "figure3_generation_vs_judging.png"
                shutil.copy(thesis_fig, dst)
                generated_figures.append(dst)
                # Also copy to thesis-expected location
                thesis_dst = FINAL_RUNS / "supp_time_cost_scaling/Output/figure3_generation_vs_judging.png"
                thesis_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(thesis_fig, thesis_dst)
                print(f"  ✓ Generated: figure3_generation_vs_judging.png")
                figure3_generated = True
    else:
        print(f"  ⚠️ Script not found: {time_cost_script}")
    
    # Fallback: copy pre-existing figure3 if not generated (requires private generator data)
    if not figure3_generated:
        static_fig3 = FINAL_RUNS / "supp_time_cost_scaling/Output/figure3_generation_vs_judging.png"
        if static_fig3.exists():
            dst = run_dir / "figure3_generation_vs_judging.png"
            shutil.copy(static_fig3, dst)
            generated_figures.append(dst)
            print(f"  ✓ Copied static asset: figure3_generation_vs_judging.png (requires private data to regenerate)")

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
        # Find the combined results file (has tokens_per_edge column)
        parallel_data = FINAL_RUNS / "supp_parallelization_benchmark/Data"
        combined_file = parallel_data / "benchmark_results_combined_20251218_011948.xlsx"
        
        if combined_file.exists():
            latest_results = combined_file
        else:
            results_files = list(parallel_data.glob("*combined*.xlsx")) if parallel_data.exists() else []
            latest_results = results_files[0] if results_files else None
        
        if latest_results and latest_results.exists():
            cmd = f"python3 {parallel_script} {latest_results}"
            success = run_command(cmd, "Parallelization Analysis", env=env)
            
            if success:
                # Script outputs to its own figures/ directory
                script_figures_dir = FINAL_RUNS / "supp_parallelization_benchmark/figures"
                thesis_fig = script_figures_dir / "parallelization_thesis_figure.png"
                if thesis_fig.exists():
                    dst = run_dir / "parallelization_thesis_figure.png"
                    shutil.copy(thesis_fig, dst)
                    generated_figures.append(dst)
                    print(f"  ✓ Generated: parallelization_thesis_figure.png")
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
    # Note: Script outputs to FINAL_RUNS / "Sensitivity_analysis_simple" (hardcoded)
    sensitivity_actual_output = FINAL_RUNS / "Sensitivity_analysis_simple"

    # ALWAYS run the script to generate fresh outputs (no precomputed fallbacks)
    judge_generated = False
    if sensitivity_script.exists():
        cmd = f"python3 {sensitivity_script}"
        success = run_command(cmd, "Judge Sensitivity Analysis", cwd=str(FINAL_RUNS / "supp_prompt_sensitivity"))
        
        if success:
            # Look in actual output location (hardcoded in script)
            for fig in sensitivity_actual_output.glob("*sensitivity*.png"):
                if "corrector" not in fig.name.lower():
                    dst = run_dir / "prompt_sensitivity_figure.png"
                    shutil.copy(fig, dst)
                    generated_figures.append(dst)
                    print(f"  ✓ Generated: {fig.name} → prompt_sensitivity_figure.png")
                    judge_generated = True
                    break
    
    if not judge_generated:
        print(f"  ⚠️ Failed to generate prompt_sensitivity_figure.png")
    
    # Copy prompt sensitivity table to thesis-expected location
    prompt_table_src = sensitivity_actual_output / "prompt_sensitivity_table.tex"
    if prompt_table_src.exists():
        dst = FINAL_RUNS / "supp_prompt_sensitivity/prompt_sensitivity_table.tex"
        shutil.copy(prompt_table_src, dst)
        print(f"  ✓ Copied: prompt_sensitivity_table.tex")

    # =========================================================================
    # 4. CORRECTOR SENSITIVITY ANALYSIS
    # =========================================================================
    print("\n" + "-"*80)
    print("4. Corrector Sensitivity Analysis")
    print("-"*80)

    corrector_script = FINAL_RUNS / "supp_prompt_sensitivity/analysis_scripts/sensitivity_analysis_corrector.py"
    # Note: Script outputs to FINAL_RUNS / "Sensitivity_analysis_simple" (hardcoded)

    # ALWAYS run the script to generate fresh outputs (no precomputed fallbacks)
    corrector_generated = False
    if corrector_script.exists():
        cmd = f"python3 {corrector_script}"
        success = run_command(cmd, "Corrector Sensitivity Analysis", cwd=str(FINAL_RUNS / "supp_prompt_sensitivity"))
        
        if success:
            # Look in actual output location (hardcoded in script)
            for fig in sensitivity_actual_output.glob("*corrector*sensitivity*.png"):
                dst = run_dir / "corrector_sensitivity_figure.png"
                shutil.copy(fig, dst)
                generated_figures.append(dst)
                print(f"  ✓ Generated: {fig.name} → corrector_sensitivity_figure.png")
                corrector_generated = True
                break
    
    if not corrector_generated:
        print(f"  ⚠️ Failed to generate corrector_sensitivity_figure.png")
    
    # Copy corrector sensitivity table to thesis-expected location
    corrector_table_src = sensitivity_actual_output / "corrector_sensitivity_table.tex"
    if corrector_table_src.exists():
        dst = FINAL_RUNS / "supp_prompt_sensitivity/corrector_sensitivity_table.tex"
        shutil.copy(corrector_table_src, dst)
        print(f"  ✓ Copied: corrector_sensitivity_table.tex")

    # =========================================================================
    # 4b. SENSITIVITY RM-ANOVA ASSUMPTIONS TABLES
    # =========================================================================
    print("\n" + "-"*80)
    print("4b. Sensitivity RM-ANOVA Assumptions Tables")
    print("-"*80)

    # Prompt sensitivity RM-ANOVA
    rm_anova_prompt = FINAL_RUNS / "supp_prompt_sensitivity/analysis_scripts/sensitivity_analysis_simple_rm_anova.py"
    if rm_anova_prompt.exists():
        run_command(f"python3 {rm_anova_prompt}", "Prompt Sensitivity RM-ANOVA", cwd=str(FINAL_RUNS / "supp_prompt_sensitivity"))
        # Copy output to expected location
        src = sensitivity_actual_output / "prompt_sensitivity_assumptions_table.tex"
        if src.exists():
            dst = FINAL_RUNS / "supp_prompt_sensitivity/prompt_sensitivity_assumptions_table.tex"
            shutil.copy(src, dst)
            print(f"  ✓ Generated: prompt_sensitivity_assumptions_table.tex")

    # Corrector sensitivity RM-ANOVA
    rm_anova_corrector = FINAL_RUNS / "supp_prompt_sensitivity/analysis_scripts/sensitivity_analysis_corrector_rm_anova.py"
    if rm_anova_corrector.exists():
        run_command(f"python3 {rm_anova_corrector}", "Corrector Sensitivity RM-ANOVA", cwd=str(FINAL_RUNS / "supp_prompt_sensitivity"))
        # Copy output to expected location
        src = sensitivity_actual_output / "corrector_sensitivity_assumptions_table.tex"
        if src.exists():
            dst = FINAL_RUNS / "supp_prompt_sensitivity/corrector_sensitivity_assumptions_table.tex"
            shutil.copy(src, dst)
            print(f"  ✓ Generated: corrector_sensitivity_assumptions_table.tex")

    # =========================================================================
    # 4c. TABLE21 SCALING SUMMARY
    # =========================================================================
    print("\n" + "-"*80)
    print("4c. Token Cost Scaling Summary (Table 21)")
    print("-"*80)

    table21_script = FINAL_RUNS / "supp_cost_analysis/analysis_scripts/generate_table21_scaling_summary.py"
    if table21_script.exists():
        run_command(
            f"python3 {table21_script} --repo-root {PROJECT_ROOT}",
            "Generate Table 21 (Scaling Summary)",
            cwd=str(FINAL_RUNS / "supp_cost_analysis")
        )
        table21_out = FINAL_RUNS / "latex/table21_scaling_summary.tex"
        if table21_out.exists():
            print(f"  ✓ Generated: table21_scaling_summary.tex")

    # =========================================================================
    # 5. COPY STATIC ASSETS (edge ablation figure - not generated, just copied)
    # =========================================================================
    print("\n" + "-"*80)
    print("5. Static Assets (Edge Ablation)")
    print("-"*80)

    # Try multiple locations for edge ablation figure
    edge_ablation_locations = [
        FINAL_RUNS / "supp_edge_ablation/edge_ablation_f1_ordered.png",
        PROJECT_ROOT / "Figures/edge_ablation_f1_ordered.png",
        PROJECT_ROOT / "thesis/Figures/edge_ablation_f1_ordered.png",
    ]
    
    edge_copied = False
    for edge_ablation_src in edge_ablation_locations:
        if edge_ablation_src.exists():
            dst = run_dir / "edge_ablation_f1_ordered.png"
            shutil.copy(edge_ablation_src, dst)
            generated_figures.append(dst)
            print(f"  ✓ Copied static asset: edge_ablation_f1_ordered.png")
            edge_copied = True
            break
    
    if not edge_copied:
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


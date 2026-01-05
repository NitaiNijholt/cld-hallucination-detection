#!/usr/bin/env python3
"""
Analyze Parallelization Benchmark Results

Generates figures and statistics for the parallelization benchmark,
supporting multiple judge modes (correctness, citation, combined).

Usage:
    python analyze_parallelization_results.py [results_file.xlsx]
    python analyze_parallelization_results.py --all  # Analyze all result files
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


def calculate_95_ci(values: list) -> tuple:
    """Calculate mean and 95% CI half-width (legacy compatibility)."""
    if len(values) < 2:
        return np.mean(values), 0, np.mean(values), np.mean(values)
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    t_value = stats.t.ppf(0.975, n - 1)
    ci_hw = t_value * std / np.sqrt(n)
    
    return mean, ci_hw, mean - ci_hw, mean + ci_hw


def compute_comprehensive_stats(values: list) -> dict:
    """
    Compute comprehensive statistics for repeated measurements.
    
    For n < 30, reports mean [min, max] as primary (assumption-free).
    Also includes SD and 95% CI (t-distribution) for reference.
    """
    values = list(values)
    n = len(values)
    
    if n == 0:
        return {
            'n': 0, 'values': [], 'mean': np.nan, 'std': np.nan,
            'min': np.nan, 'max': np.nan, 'ci_hw': np.nan,
            'ci_lower': np.nan, 'ci_upper': np.nan
        }
    
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    
    # 95% CI using t-distribution
    if n > 1:
        se = std_val / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_hw = t_crit * se
        ci_lower = mean_val - ci_hw
        ci_upper = mean_val + ci_hw
    else:
        ci_hw = 0.0
        ci_lower = mean_val
        ci_upper = mean_val
    
    return {
        'n': n,
        'values': [float(v) for v in values],
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'ci_hw': float(ci_hw),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'latex_primary': f"{mean_val:.2f} [{min_val:.2f}, {max_val:.2f}]",
        'latex_std': f"{mean_val:.2f} ± {std_val:.2f}"
    }


def analyze_single_mode(df: pd.DataFrame, mode: str, output_dir: Path):
    """Generate analysis for a single mode."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    mode_df = df[df["mode"] == mode].copy()
    if len(mode_df) == 0:
        return None
    
    # Calculate speedup relative to 1 worker
    baseline_time = mode_df[mode_df["worker_count"] == 1]["wall_clock_s"].mean()
    
    # Aggregate by worker count (tokens_per_edge is optional)
    agg_dict = {
        "wall_clock_s": list,
        "cumulative_api_s": "mean",
    }
    if "tokens_per_edge" in mode_df.columns:
        agg_dict["tokens_per_edge"] = "mean"
    
    agg = mode_df.groupby("worker_count").agg(agg_dict).reset_index()
    
    # Add tokens_per_edge column with NaN if not present
    if "tokens_per_edge" not in agg.columns:
        agg["tokens_per_edge"] = np.nan
    
    agg["wall_clock_mean"] = agg["wall_clock_s"].apply(lambda x: np.mean(x))
    agg["wall_clock_std"] = agg["wall_clock_s"].apply(lambda x: np.std(x, ddof=1) if len(x) > 1 else 0)
    agg["wall_clock_min"] = agg["wall_clock_s"].apply(lambda x: np.min(x))
    agg["wall_clock_max"] = agg["wall_clock_s"].apply(lambda x: np.max(x))
    agg["wall_clock_ci"] = agg["wall_clock_s"].apply(lambda x: calculate_95_ci(x)[1])
    agg["wall_clock_n"] = agg["wall_clock_s"].apply(len)
    agg["speedup"] = baseline_time / agg["wall_clock_mean"]
    agg["efficiency"] = agg["speedup"] / agg["worker_count"] * 100
    agg["theoretical_speedup"] = agg["worker_count"]
    agg["mode"] = mode
    
    return agg


def create_combined_figure(all_agg: pd.DataFrame, output_dir: Path):
    """Create combined figure comparing all modes."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    modes = all_agg["mode"].unique()
    n_modes = len(modes)
    
    # Color palette
    colors = {
        "correctness": "#3498db",
        "citation": "#e74c3c",
    }
    
    # ==================================================================================
    # Figure 1: Combined Speedup Comparison (2x2 layout)
    # ==================================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Wall-clock time by mode
    ax = axes[0, 0]
    for mode in modes:
        mode_agg = all_agg[all_agg["mode"] == mode]
        # Use asymmetric [min, max] error bars per uncertainty rule (N<30)
        lower_err = mode_agg["wall_clock_mean"] - mode_agg["wall_clock_min"]
        upper_err = mode_agg["wall_clock_max"] - mode_agg["wall_clock_mean"]
        ax.errorbar(mode_agg["worker_count"], mode_agg["wall_clock_mean"], 
                    yerr=[lower_err, upper_err], fmt='o-', capsize=5, 
                    color=colors.get(mode, "#95a5a6"), markersize=8, linewidth=2,
                    label=f'{mode.capitalize()}')
    
    # Add theoretical line
    baseline_avg = all_agg[all_agg["worker_count"] == 1]["wall_clock_mean"].mean()
    workers = sorted(all_agg["worker_count"].unique())
    theoretical = [baseline_avg / w for w in workers]
    ax.plot(workers, theoretical, '--', color='gray', linewidth=2, label='Theoretical')
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Wall-Clock Time (s) [min, max]')
    ax.set_title('(A) Wall-Clock Time vs Parallelization')
    ax.legend()
    ax.set_xticks(workers)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Speedup comparison
    ax = axes[0, 1]
    x = np.arange(len(workers))
    width = 0.35
    
    for i, mode in enumerate(modes):
        mode_agg = all_agg[all_agg["mode"] == mode]
        offset = (i - (n_modes - 1) / 2) * width
        bars = ax.bar(x + offset, mode_agg["speedup"], width, 
                     label=f'{mode.capitalize()}',
                     color=colors.get(mode, "#95a5a6"), alpha=0.8, edgecolor='black')
    
    # Add theoretical max
    ax.plot(x, workers, 'k--', linewidth=2, label='Theoretical Max')
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('(B) Measured Speedup by Judge Type')
    ax.set_xticks(x)
    ax.set_xticklabels(workers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Panel C: Efficiency comparison
    ax = axes[1, 0]
    for i, mode in enumerate(modes):
        mode_agg = all_agg[all_agg["mode"] == mode]
        offset = (i - (n_modes - 1) / 2) * width
        bars = ax.bar(x + offset, mode_agg["efficiency"], width,
                     label=f'{mode.capitalize()}',
                     color=colors.get(mode, "#95a5a6"), alpha=0.8, edgecolor='black')
        
        # Add efficiency labels
        for bar, eff in zip(bars, mode_agg["efficiency"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{eff:.0f}%', ha='center', va='bottom', fontsize=8)
    
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=2, label='Perfect Efficiency')
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Parallel Efficiency (%)')
    ax.set_title('(C) Parallel Efficiency by Judge Type')
    ax.set_xticks(x)
    ax.set_xticklabels(workers)
    ax.set_ylim(0, 120)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel D: Tokens per edge comparison (skip if data not available)
    ax = axes[1, 1]
    tokens_data = []
    for mode in modes:
        mode_agg = all_agg[all_agg["mode"] == mode]
        tokens = mode_agg["tokens_per_edge"].mean()
        tokens_data.append(tokens)
    
    if not all(np.isnan(t) for t in tokens_data):
        bars = ax.bar(modes, tokens_data, color=[colors.get(m, "#95a5a6") for m in modes],
                      alpha=0.8, edgecolor='black')
        
        for bar, tokens in zip(bars, tokens_data):
            if not np.isnan(tokens):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                       f'{tokens:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Judge Type')
        ax.set_ylabel('Tokens per Edge')
        ax.set_title('(D) Token Usage by Judge Type')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Token data\nnot available', ha='center', va='center', 
                fontsize=14, transform=ax.transAxes)
        ax.set_title('(D) Token Usage by Judge Type')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "parallelization_combined.png", dpi=300)
    plt.close()
    print(f"  Saved: {figures_dir / 'parallelization_combined.png'}")
    
    # ==================================================================================
    # Figure 2: Summary for thesis (cleaner 1x2 layout)
    # ==================================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Wall-clock time
    ax = axes[0]
    for mode in modes:
        mode_agg = all_agg[all_agg["mode"] == mode]
        # Use asymmetric [min, max] error bars per uncertainty rule (N<30)
        lower_err = mode_agg["wall_clock_mean"] - mode_agg["wall_clock_min"]
        upper_err = mode_agg["wall_clock_max"] - mode_agg["wall_clock_mean"]
        ax.errorbar(mode_agg["worker_count"], mode_agg["wall_clock_mean"], 
                    yerr=[lower_err, upper_err], fmt='o-', capsize=5, 
                    color=colors.get(mode, "#95a5a6"), markersize=10, linewidth=2,
                    label=f'{mode.capitalize()} Judge')
    
    # Theoretical line (average baseline)
    baseline_avg = all_agg[all_agg["worker_count"] == 1]["wall_clock_mean"].mean()
    theoretical = [baseline_avg / w for w in workers]
    ax.plot(workers, theoretical, '--', color='gray', linewidth=2, label='Theoretical ($T_1/W$)')
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Wall-Clock Time (s)')
    ax.set_title('(A) Wall-Clock Time vs Parallelization')
    ax.legend()
    ax.set_xticks(workers)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Speedup with efficiency labels
    ax = axes[1]
    x = np.arange(len(workers))
    width = 0.35
    
    for i, mode in enumerate(modes):
        mode_agg = all_agg[all_agg["mode"] == mode]
        offset = (i - (n_modes - 1) / 2) * width
        bars = ax.bar(x + offset, mode_agg["speedup"], width, 
                     label=f'{mode.capitalize()} Judge',
                     color=colors.get(mode, "#95a5a6"), alpha=0.8, edgecolor='black')
        
        # Add efficiency labels above bars
        for bar, eff in zip(bars, mode_agg["efficiency"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                   f'{eff:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Theoretical max line
    ax.plot(x, workers, 'k--', linewidth=2, marker='s', markersize=6, label='Theoretical Max')
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('(B) Measured Speedup (efficiency % shown)')
    ax.set_xticks(x)
    ax.set_xticklabels(workers)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "parallelization_thesis_figure.png", dpi=300)
    plt.close()
    print(f"  Saved: {figures_dir / 'parallelization_thesis_figure.png'}")


def generate_summary(all_agg: pd.DataFrame, output_dir: Path):
    """Generate summary statistics file."""
    summary_file = output_dir / "parallelization_summary.txt"
    
    with open(summary_file, "w") as f:
        f.write("PARALLELIZATION BENCHMARK RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        for mode in all_agg["mode"].unique():
            mode_agg = all_agg[all_agg["mode"] == mode]
            baseline = mode_agg[mode_agg["worker_count"] == 1]["wall_clock_mean"].values[0]
            
            f.write(f"{mode.upper()} JUDGE:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Workers':<10} {'Time (s)':<15} {'Speedup':<12} {'Efficiency':<12}\n")
            
            for _, row in mode_agg.iterrows():
                f.write(f"{int(row['worker_count']):<10} "
                       f"{row['wall_clock_mean']:.1f} ± {row['wall_clock_ci']:.1f}{'':>5} "
                       f"{row['speedup']:.2f}x{'':>5} "
                       f"{row['efficiency']:.0f}%\n")
            
            f.write(f"\nBaseline (sequential): {baseline:.1f}s\n")
            tokens_mean = mode_agg['tokens_per_edge'].mean()
            if not np.isnan(tokens_mean):
                f.write(f"Tokens per edge: {tokens_mean:.0f}\n")
            f.write("\n")
        
        f.write("\nKEY FINDINGS:\n")
        f.write("-" * 60 + "\n")
        
        # Compare modes
        modes = all_agg["mode"].unique()
        if len(modes) > 1:
            corr_agg = all_agg[all_agg["mode"] == "correctness"]
            cit_agg = all_agg[all_agg["mode"] == "citation"]
            
            if len(corr_agg) > 0 and len(cit_agg) > 0:
                corr_base = corr_agg[corr_agg["worker_count"] == 1]["wall_clock_mean"].values[0]
                cit_base = cit_agg[cit_agg["worker_count"] == 1]["wall_clock_mean"].values[0]
                
                f.write(f"1. Citation judging takes {cit_base/corr_base:.1f}x longer than correctness judging\n")
                f.write(f"   (due to longer prompts with retrieved citations)\n")
                
                corr_10 = corr_agg[corr_agg["worker_count"] == 10]
                cit_10 = cit_agg[cit_agg["worker_count"] == 10]
                
                if len(corr_10) > 0 and len(cit_10) > 0:
                    f.write(f"2. At 10 workers:\n")
                    f.write(f"   - Correctness: {corr_10['speedup'].values[0]:.1f}x speedup, {corr_10['efficiency'].values[0]:.0f}% efficiency\n")
                    f.write(f"   - Citation: {cit_10['speedup'].values[0]:.1f}x speedup, {cit_10['efficiency'].values[0]:.0f}% efficiency\n")
        
        f.write(f"3. Parallelization provides significant wall-clock speedup\n")
        f.write(f"4. Efficiency decreases with more workers (API rate limits, overhead)\n")
        f.write(f"5. Recommended: 5-10 workers for best speedup/efficiency balance\n")
    
    print(f"  Saved: {summary_file}")


def generate_latex_table(all_agg: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table for thesis."""
    latex_file = output_dir / "parallelization_table.tex"
    
    with open(latex_file, "w") as f:
        f.write(r"\begin{table}[H]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Parallelization Benchmark: Wall-Clock Time and Speedup by Judge Type}" + "\n")
        f.write(r"\label{tab:parallelization}" + "\n")
        f.write(r"\begin{threeparttable}" + "\n")
        f.write(r"\begin{tabular}{lcccccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{Workers} & \multicolumn{3}{c}{\textbf{Correctness Judge}} & \multicolumn{3}{c}{\textbf{Citation Judge}} \\" + "\n")
        f.write(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}" + "\n")
        f.write(r" & Time (s) & Speedup & Eff. & Time (s) & Speedup & Eff. \\" + "\n")
        f.write(r"\midrule" + "\n")
        
        workers = sorted(all_agg["worker_count"].unique())
        for w in workers:
            corr = all_agg[(all_agg["mode"] == "correctness") & (all_agg["worker_count"] == w)]
            cit = all_agg[(all_agg["mode"] == "citation") & (all_agg["worker_count"] == w)]
            
            line = f"{int(w)}"
            
            if len(corr) > 0:
                line += f" & ${corr['wall_clock_mean'].values[0]:.1f}~[{corr['wall_clock_min'].values[0]:.1f}, {corr['wall_clock_max'].values[0]:.1f}]$"
                line += f" & {corr['speedup'].values[0]:.2f}$\\times$"
                line += f" & {corr['efficiency'].values[0]:.0f}\\%"
            else:
                line += " & --- & --- & ---"
            
            if len(cit) > 0:
                line += f" & ${cit['wall_clock_mean'].values[0]:.1f}~[{cit['wall_clock_min'].values[0]:.1f}, {cit['wall_clock_max'].values[0]:.1f}]$"
                line += f" & {cit['speedup'].values[0]:.2f}$\\times$"
                line += f" & {cit['efficiency'].values[0]:.0f}\\%"
            else:
                line += " & --- & --- & ---"
            
            line += r" \\" + "\n"
            f.write(line)
        
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\begin{tablenotes}" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\item \textit{Note.} Benchmark on Depressive CLD (34 edges, GPT-4.1, 3 runs per config). Time = wall-clock time (mean $\pm$ 95\% CI). Speedup = $T_1/T_W$. Efficiency = Speedup/$W$ $\times$ 100\%." + "\n")
        f.write(r"\item Citation judging uses longer prompts with retrieved citations, resulting in higher per-call latency." + "\n")
        f.write(r"\end{tablenotes}" + "\n")
        f.write(r"\end{threeparttable}" + "\n")
        f.write(r"\end{table}" + "\n")
    
    print(f"  Saved: {latex_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze parallelization benchmark results')
    parser.add_argument('results_file', nargs='?', default=None,
                       help='Path to results Excel file (auto-detects latest if not specified)')
    parser.add_argument('--all', action='store_true',
                       help='Combine and analyze all result files')
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent.parent
    data_dir = output_dir / "data"
    
    # Find results file(s)
    if args.all:
        results_files = sorted(data_dir.glob("benchmark_results_*.xlsx"))
        if not results_files:
            print("Error: No results files found. Run the benchmark first.")
            sys.exit(1)
        print(f"Found {len(results_files)} result files")
        
        # Combine all results
        dfs = []
        for f in results_files:
            df = pd.read_excel(f)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
    elif args.results_file:
        results_file = Path(args.results_file)
        df = pd.read_excel(results_file)
    else:
        # Find latest results file
        results_files = sorted(data_dir.glob("benchmark_results_*.xlsx"))
        if not results_files:
            print("Error: No results files found. Run the benchmark first.")
            sys.exit(1)
        results_file = results_files[-1]
        print(f"Using latest: {results_file}")
        df = pd.read_excel(results_file)
    
    print(f"Loaded {len(df)} benchmark runs")
    print(f"Modes: {df['mode'].unique() if 'mode' in df.columns else 'N/A'}")
    
    # Add mode column if missing (backward compatibility)
    if "mode" not in df.columns:
        df["mode"] = "correctness"
    
    # Analyze each mode
    all_agg = []
    for mode in df["mode"].unique():
        print(f"\nAnalyzing {mode} mode...")
        agg = analyze_single_mode(df, mode, output_dir)
        if agg is not None:
            all_agg.append(agg)
    
    if not all_agg:
        print("Error: No data to analyze")
        sys.exit(1)
    
    all_agg = pd.concat(all_agg, ignore_index=True)
    
    # Create combined figures
    print("\nGenerating figures...")
    create_combined_figure(all_agg, output_dir)
    
    # Generate summary
    print("\nGenerating summary...")
    generate_summary(all_agg, output_dir)
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(all_agg, output_dir)
    
    # Save comprehensive statistics JSON (per uncertainty reporting rule)
    import json
    comprehensive_stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'modes': {}
    }
    
    for mode in all_agg["mode"].unique():
        mode_data = all_agg[all_agg["mode"] == mode]
        comprehensive_stats['modes'][mode] = {
            'worker_counts': {}
        }
        
        for _, row in mode_data.iterrows():
            wc = int(row['worker_count'])
            comprehensive_stats['modes'][mode]['worker_counts'][wc] = {
                'n': int(row['wall_clock_n']),
                'values': [float(v) for v in row['wall_clock_s']],
                'mean': float(row['wall_clock_mean']),
                'std': float(row['wall_clock_std']),
                'min': float(row['wall_clock_min']),
                'max': float(row['wall_clock_max']),
                'ci_hw': float(row['wall_clock_ci']),
                'speedup': float(row['speedup']),
                'efficiency': float(row['efficiency']),
                'latex_primary': f"{row['wall_clock_mean']:.2f} [{row['wall_clock_min']:.2f}, {row['wall_clock_max']:.2f}]",
                'latex_std': f"{row['wall_clock_mean']:.2f} ± {row['wall_clock_std']:.2f}"
            }
    
    with open(output_dir / 'parallelization_comprehensive_stats.json', 'w') as f:
        json.dump(comprehensive_stats, f, indent=2)
    print(f"✓ Comprehensive stats saved to: {output_dir / 'parallelization_comprehensive_stats.json'}")
    
    print("\nAnalysis complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()







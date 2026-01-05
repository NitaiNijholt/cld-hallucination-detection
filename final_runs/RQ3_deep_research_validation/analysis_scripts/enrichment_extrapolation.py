#!/usr/bin/env python3
"""
Enrichment Extrapolation for Deep Research Results

This script:
1. Calculates validation metrics from human-annotated sample
2. Selects an applicability threshold using a simple elbow/gradient decision rule
3. Performs sensitivity analysis across thresholds
4. Extrapolates enrichment estimates to the full dataset
5. Generates summary tables for thesis

THRESHOLD SELECTION RULE (Elbow / Gradient, Option B):
    Compute the discrete tradeoff curve across thresholds:
        x(t) = coverage = pass-rate among edges
        y(t) = correctness = proportion of passing edges labeled "causal"

    For each adjacent step from stricter→looser thresholds (e.g., 0.8→0.7),
    compute a "gradient" score:
        score = Δx / |Δy|

    Choose the threshold as the END of the best segment (the lower threshold),
    i.e., choose to_t for the segment with maximum score.
    Intuition: pick the point where you gain the most coverage for the smallest
    loss in correctness (the elbow / diminishing-returns point).

Author: Generated for RQ3 Deep Research validation
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = Path(os.environ.get("RQ3_INPUT_DIR", str(Path(__file__).parent.parent))).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("RQ3_OUTPUT_DIR", str(INPUT_DIR))).expanduser().resolve()

VALIDATION_FILE = INPUT_DIR / "validation" / "deep_research_edge_validation_sample_20251226_055509_nitai_validated_cleaned.xlsx"
DATA_DIR = INPUT_DIR / "Data"
OUTPUT_DIR = OUTPUT_ROOT / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds for sensitivity analysis
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

# Plotting
PLOTS_DIR = OUTPUT_DIR / "plots"
SINGLE_PLOT_ONLY = True  # If True, generate only a single FP tradeoff plot
TRADEOFF_SCOPE = "ALL"   # "FP" (default) or "ALL" (ignore TP/FP/FN and pool)

# Thesis layout tuning
# This plot is placed next to a small table in LaTeX (minipages). The defaults
# (7x5 with a title + long axis labels) render visually larger/taller than the
# table. These settings bias towards a more compact, table-matching figure.
THESIS_TRADEOFF_FIGURE_MODE = True
THESIS_TRADEOFF_FIGSIZE = (6.4, 3.15)  # inches; slightly taller to avoid label clipping
THESIS_TRADEOFF_DPI = 300
THESIS_TRADEOFF_PAD_INCHES = 0.04

# LaTeX output (for thesis reproducibility)
# OUTPUT_DIR = .../final_runs/RQ3_deep_research_validation/validation
# repo root is 3 levels up: validation -> RQ3_deep_research -> final_runs -> causalix.ai
REPO_ROOT = INPUT_DIR.parent.parent
THESIS_GENERATED_DIR = REPO_ROOT / "thesis" / "final_thesis" / "generated"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple:
    """Calculate Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    
    p = successes / n
    z = stats.norm.ppf(1 - alpha / 2)
    
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    
    return (max(0, center - spread), min(1, center + spread))


def load_validation_data(filepath: Path) -> pd.DataFrame:
    """Load and clean the validation data."""
    df = pd.read_excel(filepath)
    
    # Clean human_verdict (strip whitespace)
    df['human_verdict'] = df['human_verdict'].astype(str).str.strip()
    df.loc[df['human_verdict'] == 'nan', 'human_verdict'] = np.nan
    
    # Create standardized categories
    df['evidence_type'] = df['human_verdict'].apply(lambda x:
        'correct_causal' if x == 'causal' else
        'related_causal' if x == 'different vars, causal' else
        'invalid' if pd.notna(x) else 'missing')
    
    df['is_correct_causal'] = df['evidence_type'] == 'correct_causal'
    df['is_any_causal'] = df['evidence_type'].isin(['correct_causal', 'related_causal'])
    
    return df


def load_full_dr_results(data_dir: Path) -> pd.DataFrame:
    """Load all edges from the full DR results."""
    all_edges = []
    
    for json_file in data_dir.glob("deep_research_results_*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for edge in data.get("results", []):
            edge["_source_file"] = json_file.name
            all_edges.append(edge)
    
    return pd.DataFrame(all_edges)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def calculate_validation_metrics(df: pd.DataFrame) -> dict:
    """Calculate validation metrics by classification."""
    metrics = {}
    
    for cls in ['TP', 'FP', 'FN']:
        subset = df[df['classification'] == cls]
        n = len(subset)
        
        if n == 0:
            continue
        
        # Correct causal rate
        correct = subset['is_correct_causal'].sum()
        correct_rate = correct / n
        correct_ci = wilson_ci(correct, n)
        
        # Any causal rate
        any_causal = subset['is_any_causal'].sum()
        any_rate = any_causal / n
        any_ci = wilson_ci(any_causal, n)
        
        # Applicability stats
        app_mean = subset['Fully applicable'].mean()
        app_std = subset['Fully applicable'].std()
        
        metrics[cls] = {
            'n': n,
            'correct_causal_n': correct,
            'correct_causal_rate': correct_rate,
            'correct_causal_ci': correct_ci,
            'any_causal_n': any_causal,
            'any_causal_rate': any_rate,
            'any_causal_ci': any_ci,
            'applicability_mean': app_mean,
            'applicability_std': app_std,
        }
    
    return metrics


def sensitivity_analysis(df: pd.DataFrame, thresholds: list) -> pd.DataFrame:
    """Analyze correct causal rate at different applicability thresholds."""
    results = []
    
    for cls in ['TP', 'FP', 'FN']:
        subset = df[df['classification'] == cls]
        
        for threshold in thresholds:
            above = subset[subset['Fully applicable'] >= threshold]
            n = len(above)
            
            if n == 0:
                continue
            
            correct = above['is_correct_causal'].sum()
            rate = correct / n
            ci = wilson_ci(correct, n)
            
            # Proportion of class that passes threshold
            pass_rate = n / len(subset) if len(subset) > 0 else 0
            
            results.append({
                'classification': cls,
                'threshold': threshold,
                'n_above_threshold': n,
                'n_total': len(subset),
                'pass_rate': pass_rate,
                'correct_causal_n': correct,
                'correct_causal_rate': rate,
                'ci_low': ci[0],
                'ci_high': ci[1],
            })
    
    return pd.DataFrame(results)


def build_threshold_tradeoff_table(
    df: pd.DataFrame,
    thresholds: list,
    target_class: str = "FP",
) -> pd.DataFrame:
    """
    Build a table of threshold tradeoffs for a given class.

    Columns include:
    - pass_rate: fraction of edges in class with applicability >= threshold
    - mean_app_above: mean applicability among edges above threshold
    - causal_rate_above: fraction of edges above threshold labeled correct_causal
    - deltas vs previous threshold (coverage drop, mean applicability change, causal-rate change)
    - marginal cost ratios to reveal an elbow/knee:
        coverage_drop_per_1pt_causal_gain
        coverage_drop_per_0.01_mean_app_gain
    """
    if target_class == "ALL":
        subset = df.copy()
    else:
        subset = df[df["classification"] == target_class].copy()
    n_total = len(subset)
    if n_total == 0:
        return pd.DataFrame()

    rows = []
    thresholds_sorted = sorted(thresholds)
    for t in thresholds_sorted:
        above = subset[subset["Fully applicable"] >= t]
        n_above = len(above)
        pass_rate = n_above / n_total
        mean_app = float(above["Fully applicable"].mean()) if n_above > 0 else np.nan
        causal_rate = float(above["is_correct_causal"].mean()) if n_above > 0 else np.nan
        ci_low, ci_high = wilson_ci(int(above["is_correct_causal"].sum()), n_above) if n_above > 0 else (np.nan, np.nan)

        rows.append(
            {
                "classification": target_class,
                "threshold": float(t),
                "n_total": int(n_total),
                "n_above": int(n_above),
                "pass_rate": float(pass_rate),
                "mean_app_above": mean_app,
                "causal_rate_above": causal_rate,
                "ci_low": float(ci_low) if not np.isnan(ci_low) else np.nan,
                "ci_high": float(ci_high) if not np.isnan(ci_high) else np.nan,
            }
        )

    trade = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    # Compute deltas (t_k - t_{k-1})
    trade["delta_pass_rate"] = trade["pass_rate"].diff()
    trade["delta_mean_app"] = trade["mean_app_above"].diff()
    trade["delta_causal_rate"] = trade["causal_rate_above"].diff()

    # Define "coverage drop" as negative delta pass_rate when threshold increases
    trade["coverage_drop"] = (-trade["delta_pass_rate"]).clip(lower=0)
    trade["causal_gain"] = trade["delta_causal_rate"].clip(lower=0)
    trade["mean_app_gain"] = trade["delta_mean_app"].clip(lower=0)

    # Marginal cost ratios; add eps to avoid division by 0
    eps = 1e-9
    trade["coverage_drop_per_1pt_causal_gain"] = trade["coverage_drop"] / (trade["causal_gain"] + eps) / 0.01
    trade["coverage_drop_per_0.01_mean_app_gain"] = trade["coverage_drop"] / (trade["mean_app_gain"] + eps) / 0.01

    return trade


def plot_threshold_tradeoffs(trade: pd.DataFrame, plots_dir: Path, title_prefix: str = ""):
    """
    Save plots that visualize the elbow / diminishing returns:
    1) Pass rate vs threshold
    2) Causal rate vs threshold (with 95% CI)
    3) Mean applicability above threshold vs threshold
    4) Marginal tradeoff: coverage_drop per 1pt causal gain (lower is better)
    """
    if trade.empty:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    cls = trade["classification"].iloc[0]
    x = trade["threshold"].values

    # 1) Coverage (pass rate)
    plt.figure(figsize=(7, 4))
    plt.plot(x, trade["pass_rate"].values, marker="o")
    plt.title(f"{title_prefix}{cls}: Coverage (pass rate) vs threshold")
    plt.xlabel("Applicability threshold")
    plt.ylabel("Pass rate (coverage)")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"tradeoff_{cls}_coverage.png", dpi=200)
    plt.close()

    # 2) Causal rate with CI
    plt.figure(figsize=(7, 4))
    y = trade["causal_rate_above"].values
    yerr_low = y - trade["ci_low"].values
    yerr_high = trade["ci_high"].values - y
    plt.errorbar(x, y, yerr=[yerr_low, yerr_high], marker="o", capsize=4)
    plt.title(f"{title_prefix}{cls}: Causal rate vs threshold (95% CI)")
    plt.xlabel("Applicability threshold")
    plt.ylabel("Correct-causal rate among passing edges")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"tradeoff_{cls}_causal_rate_ci.png", dpi=200)
    plt.close()

    # 3) Mean applicability among passing edges
    plt.figure(figsize=(7, 4))
    plt.plot(x, trade["mean_app_above"].values, marker="o")
    plt.title(f"{title_prefix}{cls}: Mean applicability (passing edges) vs threshold")
    plt.xlabel("Applicability threshold")
    plt.ylabel("Mean applicability among passing edges")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"tradeoff_{cls}_mean_app.png", dpi=200)
    plt.close()

    # 4) Marginal tradeoff ratios (start from 2nd point where delta exists)
    plt.figure(figsize=(7, 4))
    y_cost = trade["coverage_drop_per_1pt_causal_gain"].values
    plt.plot(x, y_cost, marker="o")
    plt.title(f"{title_prefix}{cls}: Marginal cost = coverage drop per +1pp causal gain")
    plt.xlabel("Applicability threshold")
    plt.ylabel("Coverage drop per +1 percentage point causal gain")
    plt.ylim(0, np.nanmax(y_cost[~np.isinf(y_cost)]) * 1.1 if np.any(~np.isinf(y_cost)) else 10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"tradeoff_{cls}_marginal_cost.png", dpi=200)
    plt.close()


def plot_single_fp_tradeoff(trade: pd.DataFrame, output_path: Path):
    """
    Single plot for decision-making (FP only):
      x-axis: coverage (pass_rate)
      y-axis: correct-causal rate among passing edges (with 95% CI)
      points labeled by threshold

    This creates a Pareto-style curve where the knee/elbow is visually apparent.
    """
    if trade.empty:
        return
    # Allow plotting for pooled ("ALL") or FP-only
    scope = trade["classification"].iloc[0]
    if scope not in ("FP", "ALL"):
        return

    x = trade["pass_rate"].values
    y = trade["causal_rate_above"].values
    yerr_low = y - trade["ci_low"].values
    yerr_high = trade["ci_high"].values - y
    labels = trade["threshold"].values

    thesis_mode = THESIS_TRADEOFF_FIGURE_MODE
    rc = {}
    figsize = (7, 5)
    dpi = 250
    pad_inches = 0.1
    if thesis_mode:
        figsize = THESIS_TRADEOFF_FIGSIZE
        dpi = THESIS_TRADEOFF_DPI
        pad_inches = THESIS_TRADEOFF_PAD_INCHES
        # Thesis mode: smaller physical figure but larger typography for readability.
        rc = {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 1.6,
            "lines.markersize": 6,
        }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o-", capsize=3)

        title_scope = "FP" if scope == "FP" else "All edges"
        # Title is redundant with the LaTeX caption; omit in thesis mode for a tighter figure.
        if not thesis_mode:
            ax.set_title(f"{title_scope} threshold tradeoff: coverage vs correct-causal rate (95% CI)")

        ax.set_xlabel("Coverage (pass rate)")
        ax.set_ylabel("Correct-causal rate")
        if thesis_mode:
            # Thesis mode: zoom in to the relevant region (points + CIs) to avoid wasted whitespace.
            x_min = max(0.0, float(np.min(x)) - 0.06)
            x_max = min(1.02, float(np.max(x)) + 0.06)
            # Add extra headroom so point labels don't get clipped by tight bounding boxes.
            y_min = max(0.0, float(np.min(y - yerr_low)) - 0.06)
            y_max = min(1.10, float(np.max(y + yerr_high)) + 0.10)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)

        # Annotate thresholds next to points
        # Stagger labels so they don't overlap in the high-coverage cluster.
        label_offsets = {
            0.5: (6, 6),
            0.6: (6, 16),
            0.7: (6, 6),
            0.8: (6, 16),
            0.9: (6, 10),
        }
        for xi, yi, t in zip(x, y, labels):
            dx, dy = label_offsets.get(float(t), (6, 6))
            ax.annotate(
                f"{t:.1f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=12,
                annotation_clip=False,
            )

        fig.tight_layout(pad=0.15)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
        plt.close(fig)


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def write_human_validation_latex(
    *,
    trade: pd.DataFrame,
    selected_threshold: float,
    out_dir: Path,
    scope: str,
):
    """Write LaTeX macros + a small table summarizing the pooled tradeoff curve."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grab row at selected threshold
    row = trade[trade["threshold"] == float(selected_threshold)]
    if row.empty:
        raise ValueError(f"Selected threshold {selected_threshold} not found in trade table")
    r = row.iloc[0]

    n_total = int(r["n_total"])
    coverage = float(r["pass_rate"])
    correctness = float(r["causal_rate_above"])

    # Macros
    numbers_path = out_dir / "rq3_human_validation_numbers.tex"
    numbers_tex = "\n".join(
        [
            "% Auto-generated by enrichment_extrapolation.py",
            # Use a distinct prefix to avoid clashing with enrichment macros
            f"\\newcommand{{\\rqThreeHVN}}{{{n_total}}}",
            f"\\newcommand{{\\rqThreeHVScope}}{{{scope}}}",
            f"\\newcommand{{\\rqThreeHVCoverage}}{{{fmt3(coverage)}}}",
            f"\\newcommand{{\\rqThreeHVCorrect}}{{{fmt3(correctness)}}}",
            "",
        ]
    )
    numbers_path.write_text(numbers_tex)

    # Table macro: pooled tradeoff curve (used inside a minipage in the thesis).
    # NOTE: final_results.tex expects this file to DEFINE a macro named
    # \rqThreeHumanValidationTradeoffTableBody (not to emit a standalone table float).
    table_path = out_dir / "rq3_human_validation_table.tex"
    lines = ["% Auto-generated by enrichment_extrapolation.py"]
    lines.append("\\newcommand{\\rqThreeHumanValidationTradeoffTableBody}{%")
    lines.append("\\begin{threeparttable}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Threshold} & \\textbf{Coverage} & \\textbf{Correct-causal rate} & \\textbf{n pass} \\\\")
    lines.append("\\midrule")

    for _, rr in trade.sort_values("threshold").iterrows():
        t = float(rr["threshold"])
        cov = float(rr["pass_rate"])
        cor = float(rr["causal_rate_above"])
        n_pass = int(rr["n_above"])
        star = "\\textbf{" if abs(t - float(selected_threshold)) < 1e-9 else ""
        end = "}" if star else ""
        lines.append(f"{star}{t:.1f}{end} & {star}{fmt3(cov)}{end} & {star}{fmt3(cor)}{end} & {star}{n_pass}{end} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item Coverage = fraction of validated edges with applicability $\\ge t$. Correct-causal rate = fraction of passing edges labeled \\texttt{causal}.")
    lines.append("\\item Bold row indicates selected threshold $t=\\rqThreeValThresh$.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("}%")
    lines.append("")
    table_path.write_text("\n".join(lines))

    return numbers_path, table_path

def select_optimal_threshold(
    sens_df: pd.DataFrame,
    target_class: str = 'FP',
    min_ci_lower: float = 0.80,
    min_sample: int = 5,
) -> tuple:
    """
    Select optimal threshold using decision rule.
    
    DECISION RULE:
        Select the LOWEST threshold where:
        1. The lower bound of 95% CI for causal rate >= min_ci_lower (default 80%)
        2. Sample size above threshold >= min_sample (default 5)
    
    Returns:
        (selected_threshold, selection_details)
    """
    subset = sens_df[sens_df['classification'] == target_class].copy()
    subset = subset.sort_values('threshold')
    
    # Filter by minimum sample size
    subset = subset[subset['n_above_threshold'] >= min_sample]
    
    # Find lowest threshold meeting CI criterion
    meets_criterion = subset[subset['ci_low'] >= min_ci_lower]
    
    if len(meets_criterion) > 0:
        selected = meets_criterion.iloc[0]
        selection_reason = f"Lowest threshold with 95% CI lower bound ≥ {min_ci_lower*100:.0f}%"
    else:
        # Fallback: select threshold with highest CI lower bound
        selected = subset.loc[subset['ci_low'].idxmax()]
        selection_reason = f"Highest CI lower bound (none met {min_ci_lower*100:.0f}% criterion)"
    
    details = {
        'threshold': selected['threshold'],
        'causal_rate': selected['correct_causal_rate'],
        'ci_low': selected['ci_low'],
        'ci_high': selected['ci_high'],
        'n_above': selected['n_above_threshold'],
        'pass_rate': selected['pass_rate'],
        'reason': selection_reason,
        'criterion_met': len(meets_criterion) > 0,
    }
    
    return selected['threshold'], details


def select_threshold_by_gradient(trade: pd.DataFrame) -> tuple[float, dict]:
    """
    Elbow/gradient selection (Option B):
      - Build segments moving from stricter -> looser thresholds (descending t)
      - score = Δcoverage / |Δcorrectness|
      - choose threshold = to_t for the segment with maximum score
    """
    if trade.empty:
        return (np.nan, {"reason": "empty tradeoff table"})

    trade_desc = trade.sort_values("threshold", ascending=False).reset_index(drop=True)

    segments = []
    for i in range(len(trade_desc) - 1):
        t_from = float(trade_desc.loc[i, "threshold"])
        t_to = float(trade_desc.loc[i + 1, "threshold"])
        x_from = float(trade_desc.loc[i, "pass_rate"])
        x_to = float(trade_desc.loc[i + 1, "pass_rate"])
        y_from = float(trade_desc.loc[i, "causal_rate_above"])
        y_to = float(trade_desc.loc[i + 1, "causal_rate_above"])

        dx = x_to - x_from
        dy = y_to - y_from  # typically negative when going looser
        y_drop = -dy
        score = dx / y_drop if y_drop > 0 else np.inf

        segments.append(
            {
                "from_t": t_from,
                "to_t": t_to,
                "dx": dx,
                "dy": dy,
                "y_drop": y_drop,
                "score": score,
            }
        )

    seg = pd.DataFrame(segments)
    seg_finite = seg.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
    if seg_finite.empty:
        # If no drops in y anywhere, pick lowest threshold (max coverage)
        chosen = float(trade["threshold"].min())
        return chosen, {"reason": "no y-drop across thresholds; chose minimum threshold", "segments": seg.to_dict("records")}

    best = seg_finite.loc[seg_finite["score"].idxmax()]
    chosen = float(best["to_t"])  # Option B: choose END of best segment

    details = {
        "reason": "max Δcoverage / |Δcorrectness| (Option B: choose to_t)",
        "chosen_threshold": chosen,
        "best_segment": best.to_dict(),
        "segments": seg.to_dict("records"),
    }
    return chosen, details


def estimate_enrichment(
    validation_df: pd.DataFrame,
    full_df: pd.DataFrame,
    threshold: float
) -> dict:
    """
    Estimate enrichment for the full dataset based on validation sample.
    
    Returns estimates for:
    - FP edges to promote (enrichment)
    - FN edges to add (discovery)
    """
    results = {}
    
    for cls in ['FP', 'FN']:
        # Validation sample stats
        val_subset = validation_df[validation_df['classification'] == cls]
        val_above = val_subset[val_subset['Fully applicable'] >= threshold]
        
        if len(val_above) == 0:
            results[cls] = {
                'error': f'No {cls} edges above threshold {threshold}'
            }
            continue
        
        # Sample statistics
        sample_n = len(val_subset)
        sample_above_n = len(val_above)
        sample_pass_rate = sample_above_n / sample_n
        
        correct_n = val_above['is_correct_causal'].sum()
        causal_rate = correct_n / sample_above_n
        causal_ci = wilson_ci(correct_n, sample_above_n)
        
        # Full dataset: count edges with direct_causal_found=True
        full_subset = full_df[full_df['classification'] == cls]
        full_with_direct = full_subset[full_subset['direct_causal_found'] == True]
        full_n = len(full_with_direct)
        
        # Extrapolate
        est_above_threshold = int(full_n * sample_pass_rate)
        est_truly_causal = int(est_above_threshold * causal_rate)
        est_conservative = int(est_above_threshold * causal_ci[0])
        
        results[cls] = {
            # Validation sample
            'sample_n': sample_n,
            'sample_above_threshold': sample_above_n,
            'sample_pass_rate': sample_pass_rate,
            'sample_causal_n': correct_n,
            'sample_causal_rate': causal_rate,
            'sample_causal_ci': causal_ci,
            
            # Full dataset
            'full_direct_causal_n': full_n,
            
            # Estimates
            'est_above_threshold': est_above_threshold,
            'est_truly_causal': est_truly_causal,
            'est_conservative': est_conservative,
            'enrichment_rate': est_truly_causal / full_n if full_n > 0 else 0,
        }
    
    return results


def enrichment_sensitivity(
    validation_df: pd.DataFrame,
    full_df: pd.DataFrame,
    thresholds: list
) -> pd.DataFrame:
    """Run enrichment estimation across multiple thresholds."""
    results = []
    
    for threshold in thresholds:
        enrichment = estimate_enrichment(validation_df, full_df, threshold)
        
        for cls in ['FP', 'FN']:
            if 'error' in enrichment.get(cls, {}):
                continue
            
            e = enrichment[cls]
            results.append({
                'classification': cls,
                'threshold': threshold,
                'sample_pass_rate': e['sample_pass_rate'],
                'sample_causal_rate': e['sample_causal_rate'],
                'sample_causal_ci_low': e['sample_causal_ci'][0],
                'sample_causal_ci_high': e['sample_causal_ci'][1],
                'full_eligible': e['full_direct_causal_n'],
                'est_promotions': e['est_truly_causal'],
                'est_conservative': e['est_conservative'],
                'enrichment_rate': e['enrichment_rate'],
            })
    
    return pd.DataFrame(results)


# =============================================================================
# REPORTING FUNCTIONS
# =============================================================================

def print_validation_metrics(metrics: dict):
    """Print validation metrics table."""
    print("\n" + "=" * 80)
    print("VALIDATION METRICS BY CLASSIFICATION")
    print("=" * 80)
    
    print(f"\n{'Metric':<40} {'TP':>12} {'FP':>12} {'FN':>12}")
    print("-" * 80)
    
    # Sample size
    print(f"{'Sample size':<40}", end="")
    for cls in ['TP', 'FP', 'FN']:
        print(f"{metrics[cls]['n']:>12}", end="")
    print()
    
    # Correct causal rate
    print(f"{'Correct causal evidence (%)':<40}", end="")
    for cls in ['TP', 'FP', 'FN']:
        rate = metrics[cls]['correct_causal_rate'] * 100
        print(f"{rate:>11.1f}%", end="")
    print()
    
    # 95% CI for correct causal
    print(f"{'  95% CI':<40}", end="")
    for cls in ['TP', 'FP', 'FN']:
        ci = metrics[cls]['correct_causal_ci']
        print(f"  [{ci[0]*100:.0f}-{ci[1]*100:.0f}%]", end="")
    print()
    
    # Any causal rate
    print(f"{'Any causal evidence (%)':<40}", end="")
    for cls in ['TP', 'FP', 'FN']:
        rate = metrics[cls]['any_causal_rate'] * 100
        print(f"{rate:>11.1f}%", end="")
    print()
    
    # Applicability mean
    print(f"{'Mean applicability (0-1)':<40}", end="")
    for cls in ['TP', 'FP', 'FN']:
        mean = metrics[cls]['applicability_mean']
        print(f"{mean:>12.2f}", end="")
    print()
    
    # Applicability SD
    print(f"{'SD applicability':<40}", end="")
    for cls in ['TP', 'FP', 'FN']:
        sd = metrics[cls]['applicability_std']
        print(f"{sd:>12.2f}", end="")
    print()
    
    print("-" * 80)


def print_sensitivity_analysis(sens_df: pd.DataFrame):
    """Print sensitivity analysis table."""
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS: Correct Causal Rate by Threshold")
    print("=" * 80)
    
    for cls in ['TP', 'FP', 'FN']:
        print(f"\n{cls}:")
        subset = sens_df[sens_df['classification'] == cls]
        print(f"  {'Threshold':<12} {'Pass Rate':<12} {'n':<8} {'Causal %':<12} {'95% CI':<15}")
        print("  " + "-" * 60)
        for _, row in subset.iterrows():
            ci_str = f"[{row['ci_low']*100:.0f}-{row['ci_high']*100:.0f}%]"
            print(f"  {row['threshold']:<12.1f} {row['pass_rate']*100:>6.1f}%{'':<5} "
                  f"{row['n_above_threshold']:>4}/{row['n_total']:<3} "
                  f"{row['correct_causal_rate']*100:>6.1f}%{'':<5} {ci_str:<15}")


def print_enrichment_results(enrich_df: pd.DataFrame, primary_threshold: float):
    """Print enrichment estimation results."""
    print("\n" + "=" * 80)
    print("ENRICHMENT ESTIMATION")
    print("=" * 80)
    
    print(f"\nPrimary threshold: {primary_threshold}")
    
    primary = enrich_df[enrich_df['threshold'] == primary_threshold]
    
    for cls in ['FP', 'FN']:
        row = primary[primary['classification'] == cls]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        
        action = "Promote to TP" if cls == 'FP' else "Add to system output"
        print(f"\n{cls} edges ({action}):")
        print(f"  Full dataset eligible: {row['full_eligible']}")
        print(f"  Sample pass rate (≥{primary_threshold}): {row['sample_pass_rate']*100:.1f}%")
        print(f"  Sample causal rate: {row['sample_causal_rate']*100:.1f}% "
              f"[95% CI: {row['sample_causal_ci_low']*100:.0f}-{row['sample_causal_ci_high']*100:.0f}%]")
        print(f"  Estimated promotions: {row['est_promotions']}")
        print(f"  Conservative estimate (lower CI): {row['est_conservative']}")
        print(f"  Enrichment rate: {row['enrichment_rate']*100:.1f}%")
    
    print("\n" + "-" * 80)
    print("SENSITIVITY: Enrichment by Threshold")
    print("-" * 80)
    
    for cls in ['FP', 'FN']:
        print(f"\n{cls}:")
        subset = enrich_df[enrich_df['classification'] == cls]
        print(f"  {'Threshold':<12} {'Pass %':<10} {'Causal %':<12} {'Est. Promotions':<18} {'Conservative':<12}")
        print("  " + "-" * 65)
        for _, row in subset.iterrows():
            print(f"  {row['threshold']:<12.1f} {row['sample_pass_rate']*100:>5.1f}%{'':<4} "
                  f"{row['sample_causal_rate']*100:>6.1f}%{'':<5} "
                  f"{row['est_promotions']:>8}{'':<10} {row['est_conservative']:>8}")


def print_statistical_tests(df: pd.DataFrame):
    """Print statistical tests for TP vs FP discrimination."""
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)
    
    tp = df[df['classification'] == 'TP']
    fp = df[df['classification'] == 'FP']
    
    # Fisher's exact for correct causal
    tp_correct = tp['is_correct_causal'].sum()
    fp_correct = fp['is_correct_causal'].sum()
    
    contingency = [
        [tp_correct, len(tp) - tp_correct],
        [fp_correct, len(fp) - fp_correct]
    ]
    odds_ratio, p_fisher = stats.fisher_exact(contingency)
    
    print(f"\nFisher's Exact Test (TP vs FP - Correct Causal):")
    print(f"  TP: {tp_correct}/{len(tp)} = {tp_correct/len(tp)*100:.1f}%")
    print(f"  FP: {fp_correct}/{len(fp)} = {fp_correct/len(fp)*100:.1f}%")
    print(f"  Odds Ratio: {odds_ratio:.2f}" if odds_ratio != np.inf else "  Odds Ratio: ∞")
    print(f"  p-value: {p_fisher:.4f}")
    
    # Mann-Whitney for applicability
    stat, p_mw = stats.mannwhitneyu(
        tp['Fully applicable'].dropna(),
        fp['Fully applicable'].dropna(),
        alternative='two-sided'
    )
    print(f"\nMann-Whitney U Test (TP vs FP - Applicability):")
    print(f"  TP mean: {tp['Fully applicable'].mean():.2f}")
    print(f"  FP mean: {fp['Fully applicable'].mean():.2f}")
    print(f"  U statistic: {stat:.1f}")
    print(f"  p-value: {p_mw:.4f}")


def save_results(
    metrics: dict,
    sens_df: pd.DataFrame,
    enrich_df: pd.DataFrame,
    output_dir: Path
):
    """Save results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save sensitivity analysis
    sens_path = output_dir / f"sensitivity_analysis_{timestamp}.csv"
    sens_df.to_csv(sens_path, index=False)
    print(f"\nSaved: {sens_path}")
    
    # Save enrichment results
    enrich_path = output_dir / f"enrichment_estimates_{timestamp}.csv"
    enrich_df.to_csv(enrich_path, index=False)
    print(f"Saved: {enrich_path}")
    
    # Save summary metrics as JSON
    metrics_path = output_dir / f"validation_metrics_{timestamp}.json"
    # Convert numpy types for JSON serialization
    metrics_json = {}
    for cls, m in metrics.items():
        metrics_json[cls] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else 
               [float(x) for x in v] if isinstance(v, tuple) else v
            for k, v in m.items()
        }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Saved: {metrics_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("ENRICHMENT EXTRAPOLATION FOR DEEP RESEARCH")
    print("=" * 80)
    print(f"Validation file: {VALIDATION_FILE}")
    print(f"Data directory: {DATA_DIR}")
    
    # Load data
    print("\nLoading validation data...")
    val_df = load_validation_data(VALIDATION_FILE)
    print(f"Loaded {len(val_df)} validated edges")
    
    print("\nLoading full DR results...")
    full_df = load_full_dr_results(DATA_DIR)
    print(f"Loaded {len(full_df)} total edges")
    
    # Calculate metrics
    print("\nCalculating validation metrics...")
    metrics = calculate_validation_metrics(val_df)
    print_validation_metrics(metrics)
    
    # Statistical tests
    print_statistical_tests(val_df)
    
    # Sensitivity analysis
    print("\nRunning sensitivity analysis...")
    sens_df = sensitivity_analysis(val_df, THRESHOLDS)
    print_sensitivity_analysis(sens_df)

    # ==========================================================================
    # PLOTTING: show elbows / diminishing returns
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PLOTTING THRESHOLD TRADEOFFS")
    print("=" * 80)
    if SINGLE_PLOT_ONLY:
        trade = build_threshold_tradeoff_table(val_df, THRESHOLDS, target_class=TRADEOFF_SCOPE)
        out_name = "fp_threshold_tradeoff.png" if TRADEOFF_SCOPE == "FP" else "all_edges_threshold_tradeoff.png"
        out_path = OUTPUT_DIR / out_name
        plot_single_fp_tradeoff(trade, out_path)
        print(f"Saved: {out_path}")
    else:
        print(f"Saving plots to: {PLOTS_DIR}")
        for cls in ["FP", "FN", "TP"]:
            trade = build_threshold_tradeoff_table(val_df, THRESHOLDS, target_class=cls)
            if trade.empty:
                continue
            plot_threshold_tradeoffs(trade, PLOTS_DIR, title_prefix="Validation sample - ")
            # Also save the tradeoff table as CSV for inspection
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            trade_path = PLOTS_DIR / f"tradeoff_table_{cls}_{ts}.csv"
            trade.to_csv(trade_path, index=False)
            print(f"Saved: {trade_path}")
    
    # ==========================================================================
    # THRESHOLD SELECTION
    # ==========================================================================
    print("\n" + "=" * 80)
    print("THRESHOLD SELECTION")
    print("=" * 80)

    print("""
DECISION RULE (Elbow / Gradient, Option B):
    For adjacent thresholds (stricter→looser), compute:
        score = Δcoverage / |Δcorrectness|
    Select the threshold as the END of the segment with maximum score (to_t).
""")

    trade_for_selection = build_threshold_tradeoff_table(val_df, THRESHOLDS, target_class=TRADEOFF_SCOPE)
    selected_threshold, selection_details = select_threshold_by_gradient(trade_for_selection)

    print(f"SELECTED THRESHOLD: {selected_threshold}")
    best = selection_details.get("best_segment", {})
    if best:
        print(f"  Best segment: {best.get('from_t')} → {best.get('to_t')}")
        print(f"    Δcoverage: {best.get('dx'):.3f}")
        print(f"    Δcorrectness: {best.get('dy'):.3f}")
        print(f"    score (Δx/|Δy|): {best.get('score'):.3f}")

    # Write LaTeX artifacts for thesis reproducibility (human validation results)
    numbers_path, table_path = write_human_validation_latex(
        trade=trade_for_selection,
        selected_threshold=selected_threshold,
        out_dir=THESIS_GENERATED_DIR,
        scope=TRADEOFF_SCOPE,
    )
    print(f"Saved LaTeX numbers: {numbers_path}")
    print(f"Saved LaTeX table:   {table_path}")
    
    # Enrichment estimation with selected threshold
    print("\n" + "=" * 80)
    print("ENRICHMENT ESTIMATION")
    print("=" * 80)
    
    enrich_df = enrichment_sensitivity(val_df, full_df, THRESHOLDS)
    print_enrichment_results(enrich_df, selected_threshold)
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Add selected threshold to output (convert numpy types)
    def to_python(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v) if isinstance(v, np.floating) else int(v)
        return v
    
    selection_info = {
        'selected_threshold': to_python(selected_threshold),
        'decision_rule': "Elbow/gradient Option B: choose to_t of max Δcoverage/|Δcorrectness| segment",
        **{f'selection_{k}': to_python(v) for k, v in selection_details.items()}
    }
    
    save_results(metrics, sens_df, enrich_df, OUTPUT_DIR)
    
    # Save selection info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selection_path = OUTPUT_DIR / f"threshold_selection_{timestamp}.json"
    with open(selection_path, 'w') as f:
        json.dump(selection_info, f, indent=2)
    print(f"Saved: {selection_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nSelected threshold: {selected_threshold}")
    print("Decision rule: Elbow/gradient Option B (choose to_t of max Δcoverage/|Δcorrectness|)")


if __name__ == "__main__":
    main()


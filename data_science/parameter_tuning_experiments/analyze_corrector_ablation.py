#!/usr/bin/env python3
"""
Analyze Corrector Ablation Study Results

Compares performance across prompt variants:
- F1 Delta (precision, recall)
- Judge Score Delta
- Action distribution (revise vs change_type)
- Per-CLD breakdown
- Macro-averaged aggregate statistics

Supports both synthetic corruptions and ground truth validation.

Usage:
    python analyze_corrector_ablation.py --type synthetic
    python analyze_corrector_ablation.py --type groundtruth
    python analyze_corrector_ablation.py --type both
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from typing import Any, Dict, Optional, Tuple

VARIANTS = ['baseline', 'cot', 'mechanistic']
BASE_PATH = Path('final_runs')

# Folder patterns for each experiment type
# {variant} is replaced by baseline/cot/mechanistic
# {judge_suffix} is replaced by '' for baseline judge or '_mechanistic_judge' for mechanistic judge
FOLDER_PATTERNS = {
    'synthetic': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_{variant}{judge_suffix}',
    'groundtruth': 'RQ1b_corrector_experiment_ground_truth_correctness_{variant}{judge_suffix}'
}


def bonferroni_correction(p_values: dict, alpha: float = 0.05) -> dict:
    """
    Apply Bonferroni correction to a family of p-values.
    """
    m = len(p_values)
    if m == 0:
        return {'method': 'Bonferroni', 'n_tests': 0, 'results': {}}
    
    alpha_adj = alpha / m
    
    results = {}
    for name, p in p_values.items():
        if p is None or np.isnan(p):
            results[name] = {
                'p_original': p,
                'p_adjusted': np.nan,
                'significant_original': False,
                'significant_adjusted': False,
                'changed': False
            }
        else:
            p_adj = min(float(p) * m, 1.0)
            results[name] = {
                'p_original': float(p),
                'p_adjusted': float(p_adj),
                'significant_original': bool(p < alpha),
                'significant_adjusted': bool(p < alpha_adj),
                'changed': bool((p < alpha) != (p < alpha_adj))
            }
    
    return {
        'method': 'Bonferroni',
        'n_tests': m,
        'alpha_original': alpha,
        'alpha_adjusted': alpha_adj,
        'results': results,
        'n_significant_original': sum(1 for r in results.values() if r.get('significant_original', False)),
        'n_significant_adjusted': sum(1 for r in results.values() if r.get('significant_adjusted', False))
    }


def load_results(variant: str, experiment_type: str, judge_suffix: str = '') -> pd.DataFrame:
    """Load results for a specific variant and experiment type."""
    folder_pattern = FOLDER_PATTERNS[experiment_type]
    folder = BASE_PATH / folder_pattern.format(variant=variant, judge_suffix=judge_suffix)
    results_file = folder / 'rq1b_all_results.xlsx'
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")
    
    # Try different sheet names (Excel structure varies)
    xl = pd.ExcelFile(results_file)
    sheet_names = xl.sheet_names
    
    if 'All_Results' in sheet_names:
        df = pd.read_excel(results_file, sheet_name='All_Results')
    elif 'Sheet1' in sheet_names:
        df = pd.read_excel(results_file, sheet_name='Sheet1')
    else:
        df = pd.read_excel(results_file, sheet_name=0)  # First sheet
    
    df['Variant'] = variant
    return df


def _detect_run_column(df: pd.DataFrame) -> str:
    """Detect a run identifier column for blocking."""
    for col in ["Run", "run", "Seed", "seed"]:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find a run/seed column for blocking. Available columns: {list(df.columns)}")


def _format_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _p_stars(p: Optional[float]) -> str:
    """
    Conventional significance markers for display (two-sided).
    """
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _latex_superscript(stars: str) -> str:
    return f"\\textsuperscript{{{stars}}}" if stars else ""


def _rank_biserial_from_diffs(diffs: np.ndarray) -> float:
    """
    Rank-biserial correlation r_rb for (paired/one-sample) Wilcoxon signed-rank.
    Computed from signed differences after dropping zeros (wilcox convention).
    """
    from scipy.stats import rankdata

    x = np.asarray(diffs, dtype=float)
    x = x[~np.isnan(x)]
    x = x[x != 0.0]
    n = int(x.size)
    if n == 0:
        return float("nan")

    ranks = rankdata(np.abs(x), method="average")
    w_plus = float(ranks[x > 0].sum())
    w_minus = float(ranks[x < 0].sum())
    denom = n * (n + 1) / 2.0
    return (w_plus - w_minus) / denom


def compute_blocked_prompt_tests(combined: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute blocked prompt-effect and post-hoc tests on F1_Delta.

    Blocked design:
      blocks = (CLD, run)
      treatments = Variant in {baseline, cot, mechanistic}

    Returns dict with:
      - friedman_p, chi2, kendalls_w, n_blocks
      - posthoc pairs (Wilcoxon paired diffs) with Bonferroni over 3 pairs
      - efficacy_vs_zero per variant (one-sample Wilcoxon on block-level deltas)
    """
    from scipy import stats

    run_col = _detect_run_column(combined)
    required = {"CLD", "Variant", "F1_Delta", run_col}
    missing = [c for c in required if c not in combined.columns]
    if missing:
        raise KeyError(f"Missing required columns for blocked tests: {missing}")

    # Block-level mean (in case of duplicates)
    pivot = (
        combined[["CLD", run_col, "Variant", "F1_Delta"]]
        .dropna()
        .groupby(["CLD", run_col, "Variant"], as_index=True)["F1_Delta"]
        .mean()
        .unstack("Variant")
    )

    # Require complete blocks across all variants
    for v in VARIANTS:
        if v not in pivot.columns:
            raise ValueError(f"Variant '{v}' missing from data columns: {list(pivot.columns)}")
    pivot = pivot[VARIANTS].dropna(how="any")

    n_blocks = int(pivot.shape[0])
    k = len(VARIANTS)
    out: Dict[str, Any] = {
        "n_blocks": n_blocks,
        "k": k,
        "friedman_p": np.nan,
        "friedman_chi2": np.nan,
        "kendalls_w": np.nan,
        "posthoc": {"alpha": 0.05, "alpha_adj": 0.05 / 3, "pairs": {}},
        "efficacy_vs_zero": {},
    }

    if n_blocks < 2:
        return out

    # Omnibus: Friedman
    chi2, p = stats.friedmanchisquare(*[pivot[v].to_numpy(dtype=float) for v in VARIANTS])
    w = (chi2 / (n_blocks * (k - 1))) if (n_blocks > 0 and k > 1) else np.nan
    out["friedman_p"] = float(p)
    out["friedman_chi2"] = float(chi2)
    out["kendalls_w"] = float(w)

    # Post-hoc: paired Wilcoxon on diffs, Bonferroni over 3 pairs
    pairs = [("baseline", "cot"), ("baseline", "mechanistic"), ("cot", "mechanistic")]
    m = len(pairs)
    alpha_adj = 0.05 / m
    out["posthoc"]["alpha_adj"] = float(alpha_adj)
    for a, b in pairs:
        diffs = (pivot[a] - pivot[b]).to_numpy(dtype=float)
        diffs = diffs[~np.isnan(diffs)]
        if diffs.size == 0 or np.allclose(diffs, 0.0):
            stat, p_raw = 0.0, 1.0
        else:
            stat, p_raw = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        median_diff = float(np.median(diffs)) if diffs.size else 0.0
        r_rb = _rank_biserial_from_diffs(diffs) if diffs.size else float("nan")
        direction = f"{a}>{b}" if median_diff > 0 else f"{b}>{a}" if median_diff < 0 else f"{a}={b}"
        p_adj = min(float(p_raw) * m, 1.0)
        out["posthoc"]["pairs"][f"{a}__vs__{b}"] = {
            "a": a,
            "b": b,
            "n": int(diffs.size),
            "p_raw": float(p_raw),
            "p_adj": float(p_adj),
            "direction": direction,
            "median_diff": median_diff,
            "r_rb": float(r_rb) if not np.isnan(r_rb) else np.nan,
        }

    # Efficacy vs 0 (per variant) on block-level values
    for v in VARIANTS:
        vals = pivot[v].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            out["efficacy_vs_zero"][v] = {"p": np.nan, "n": 0, "r_rb": np.nan}
            continue
        if np.allclose(vals, 0.0):
            out["efficacy_vs_zero"][v] = {"p": 1.0, "n": int(vals.size), "r_rb": 0.0}
            continue
        _, pv = stats.wilcoxon(vals, alternative="two-sided", zero_method="wilcox")
        out["efficacy_vs_zero"][v] = {
            "p": float(pv),
            "n": int(vals.size),
            "r_rb": float(_rank_biserial_from_diffs(vals)),
        }

    return out


def calculate_per_cld_means(combined: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-CLD means and stds for macro averaging."""
    # Define action columns, handling optional ones that may not exist in older Excel files
    action_cols = ['Actions_Total', 'Actions_Revise', 'Actions_Change']
    optional_action_cols = ['Actions_None', 'Actions_Error', 'Actions_Remove', 'Actions_Flip']
    
    for col in optional_action_cols:
        if col in combined.columns:
            action_cols.append(col)
        else:
            combined[col] = 0  # Default to 0 for missing columns
            action_cols.append(col)
    
    per_cld_agg = combined.groupby(['Variant', 'CLD']).agg({
        'F1_Delta': ['mean', 'std'],
        'Precision_Delta': ['mean', 'std'],
        'Recall_Delta': ['mean', 'std'],
        'Judge_Score_Delta': ['mean', 'std'],
        'Actions_Total': 'sum',
        'Actions_Revise': 'sum',
        'Actions_Change': 'sum',
        'Actions_None': 'sum',
        'Actions_Error': 'sum',
        'Actions_Remove': 'sum',
        'Actions_Flip': 'sum'
    })
    
    # Flatten column names
    per_cld_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                           for col in per_cld_agg.columns.values]
    per_cld_means = per_cld_agg.reset_index()
    
    # Rename for backward compatibility with code expecting 'F1_Delta' as column name
    # Keep both mean and std columns
    per_cld_means.rename(columns={
        'F1_Delta_mean': 'F1_Delta',
        'Precision_Delta_mean': 'Precision_Delta',
        'Recall_Delta_mean': 'Recall_Delta',
        'Judge_Score_Delta_mean': 'Judge_Score_Delta',
    }, inplace=True)
    
    # VERIFY: Total should equal sum of sub-actions
    computed_total = (per_cld_means['Actions_Revise_sum'] + per_cld_means['Actions_Change_sum'] + 
                      per_cld_means['Actions_None_sum'] + per_cld_means['Actions_Error_sum'] + 
                      per_cld_means['Actions_Remove_sum'] + per_cld_means['Actions_Flip_sum'])
    
    mismatch = per_cld_means[per_cld_means['Actions_Total_sum'] != computed_total]
    if len(mismatch) > 0:
        print("\n⚠️  ACTION COUNT MISMATCH DETECTED:")
        for _, row in mismatch.iterrows():
            print(f"   {row['Variant']}/{row['CLD']}: Total={int(row['Actions_Total_sum'])}, "
                  f"Computed={int(computed_total[_])}")
            print(f"      Revise={int(row['Actions_Revise_sum'])}, Change={int(row['Actions_Change_sum'])}, "
                  f"None={int(row['Actions_None_sum'])}, Error={int(row['Actions_Error_sum'])}, "
                  f"Remove={int(row['Actions_Remove_sum'])}, Flip={int(row['Actions_Flip_sum'])}")
    else:
        print("✓ Action counts verified: Total = sum of sub-actions")
    
    return per_cld_means


def calculate_macro_average(per_cld_means: pd.DataFrame) -> pd.DataFrame:
    """Calculate macro-averaged statistics (mean ± std across CLDs)."""
    macro_avg = per_cld_means.groupby('Variant').agg({
        'F1_Delta': ['mean', 'std'],
        'Precision_Delta': ['mean', 'std'],
        'Recall_Delta': ['mean', 'std'],
        'Judge_Score_Delta': ['mean', 'std'],
        'Actions_Total_sum': 'sum',
        'Actions_Revise_sum': 'sum',
        'Actions_Change_sum': 'sum',
        'Actions_None_sum': 'sum',
        'Actions_Error_sum': 'sum',
        'Actions_Remove_sum': 'sum',
        'Actions_Flip_sum': 'sum'
    })
    
    # Flatten column names
    macro_avg.columns = ['_'.join(col).strip('_') for col in macro_avg.columns.values]
    return macro_avg


def generate_latex_table(
    combined: pd.DataFrame,
    experiment_type: str,
    output_dir: Path,
    blocked_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate LaTeX table with macro-averaged aggregate row."""
    
    # CLD order must match LaTeX header: Social Norms, Depressive, Emergency Dept
    clds = ['social_norms', 'depressive', 'emergency_department']
    cld_short_names = {
        'social_norms': 'Social Norms',
        'depressive': 'Depressive',
        'emergency_department': 'Emergency Dept'
    }
    
    # Calculate per-CLD means
    per_cld_means = calculate_per_cld_means(combined)
    
    # Calculate macro averages
    macro_avg = calculate_macro_average(per_cld_means)
    
    # Build table rows
    rows = []
    
    for variant in VARIANTS:
        variant_data = per_cld_means[per_cld_means['Variant'] == variant]
        macro = macro_avg.loc[variant]
        
        # Get per-CLD F1 values with ±std
        cld_f1s = []
        for cld in clds:
            cld_row = variant_data[variant_data['CLD'] == cld]
            if len(cld_row) > 0:
                f1 = cld_row['F1_Delta'].values[0]
                f1_std = cld_row['F1_Delta_std'].values[0] if 'F1_Delta_std' in cld_row.columns else 0.0
                if pd.isna(f1_std):
                    f1_std = 0.0
                sign = '+' if f1 >= 0 else ''
                cld_f1s.append(f"${sign}{f1:.3f} \\pm {f1_std:.3f}$")
            else:
                cld_f1s.append("N/A")
        
        # Format overall F1 (macro-averaged mean ± std) + efficacy marker (Wilcoxon vs 0)
        f1_mean = macro['F1_Delta_mean']
        f1_std = macro['F1_Delta_std']
        overall_f1 = f"${'+' if f1_mean >= 0 else ''}{f1_mean:.3f} \\pm {f1_std:.3f}$"
        if blocked_stats:
            eff_p = (
                blocked_stats.get("efficacy_vs_zero", {})
                .get(variant, {})
                .get("p")
            )
            overall_f1 += _latex_superscript(_p_stars(eff_p))

        # Format overall Judge Δ (macro-averaged mean ± std)
        judge_mean = float(macro['Judge_Score_Delta_mean'])
        judge_std = float(macro['Judge_Score_Delta_std']) if not pd.isna(macro['Judge_Score_Delta_std']) else float("nan")
        if np.isnan(judge_mean):
            overall_judge = "—"
        else:
            # pandas can yield NaN std (e.g., missing values); treat as 0 for display if mean exists
            if np.isnan(judge_std):
                judge_std = 0.0
            overall_judge = f"${'+' if judge_mean >= 0 else ''}{judge_mean:.3f} \\pm {judge_std:.3f}$"
        
        # Action totals - Total = Revise + Change (successful corrections only)
        revise = int(macro['Actions_Revise_sum_sum'])
        change = int(macro['Actions_Change_sum_sum'])
        total = revise + change  # Successful corrections only
        
        # Track None and Error for appendix
        none_count = int(macro.get('Actions_None_sum_sum', 0))
        error_count = int(macro.get('Actions_Error_sum_sum', 0))
        
        # Build row
        variant_name = variant.capitalize() if variant != 'cot' else 'CoT'
        row = f"{variant_name} & {overall_f1} & {overall_judge} & {cld_f1s[0]} & {cld_f1s[1]} & {cld_f1s[2]} & {total} & {revise} & {change} \\\\"
        rows.append(row)
    
    # Add aggregate row (macro-averaged across all variants and CLDs)
    aggregate_data = []
    for metric in ['F1_Delta', 'Precision_Delta', 'Recall_Delta', 'Judge_Score_Delta']:
        values = per_cld_means[metric].values
        aggregate_data.append({
            'metric': metric,
            'mean': values.mean(),
            'std': values.std()
        })
    
    # Total actions across all variants - Total = Revise + Change (successful corrections only)
    total_revise = int(per_cld_means['Actions_Revise_sum'].sum())
    total_change = int(per_cld_means['Actions_Change_sum'].sum())
    total_actions = total_revise + total_change  # Successful corrections only
    
    # Track None and Error for appendix
    total_none = int(per_cld_means['Actions_None_sum'].sum())
    total_error = int(per_cld_means['Actions_Error_sum'].sum())
    
    # Aggregate F1 per CLD (mean ± std across variants)
    aggregate_per_cld = per_cld_means.groupby('CLD')['F1_Delta'].agg(['mean', 'std'])
    agg_cld_f1s = []
    for cld in clds:
        if cld in aggregate_per_cld.index:
            f1 = aggregate_per_cld.loc[cld, 'mean']
            f1_std = aggregate_per_cld.loc[cld, 'std']
            if pd.isna(f1_std):
                f1_std = 0.0
            sign = '+' if f1 >= 0 else ''
            agg_cld_f1s.append(f"${sign}{f1:.3f} \\pm {f1_std:.3f}$")
        else:
            agg_cld_f1s.append("N/A")
    
    # Overall aggregate F1
    agg_f1_mean = per_cld_means['F1_Delta'].mean()
    agg_f1_std = per_cld_means['F1_Delta'].std()
    
    # Overall aggregate Judge Δ
    agg_judge_mean = float(per_cld_means['Judge_Score_Delta'].mean())
    agg_judge_std = float(per_cld_means['Judge_Score_Delta'].std())
    if np.isnan(agg_judge_std) and not np.isnan(agg_judge_mean):
        agg_judge_std = 0.0

    aggregate_row = (
        f"\\midrule\n\\textbf{{Aggregate}} & "
        f"${'+' if agg_f1_mean >= 0 else ''}{agg_f1_mean:.3f}$ $\\pm$ {agg_f1_std:.3f} & "
        f"${'+' if agg_judge_mean >= 0 else ''}{agg_judge_mean:.3f}$ $\\pm$ {agg_judge_std:.3f} & "
        f"{agg_cld_f1s[0]} & {agg_cld_f1s[1]} & {agg_cld_f1s[2]} & {total_actions} & {total_revise} & {total_change} \\\\"
    )
    
    # Build full LaTeX table
    exp_type_title = "Synthetic Corruptions" if experiment_type == 'synthetic' else "Ground Truth"
    
    # Optional blocked stats notes (Friedman + post-hoc Wilcoxon) - no \item since notes are outside tablenotes
    stats_note = ""
    if blocked_stats:
        fried_p = blocked_stats.get("friedman_p")
        stats_note = (
            f"\\textit{{Blocked tests.}} Omnibus prompt effect tested with Friedman using (CLD, run) blocks "
            f"(n={blocked_stats.get('n_blocks', '—')}); p={_format_p(fried_p)}{_latex_superscript(_p_stars(fried_p))}, "
            f"W={blocked_stats.get('kendalls_w', float('nan')):.2f}.\\par\n"
        )
        # Post-hoc summary
        posthoc = blocked_stats.get("posthoc", {})
        alpha_adj = posthoc.get("alpha_adj", 0.05 / 3)
        pair_summaries = []
        for key, r in posthoc.get("pairs", {}).items():
            # Make direction and p-value LaTeX-safe (avoid underscores like "p_adj")
            direction = str(r.get("direction", ""))
            direction_tex = direction.replace(">", "$>$").replace("=", "$=$")
            p_adj = r.get("p_adj")
            p_star = _latex_superscript(_p_stars(p_adj))
            r_rb = r.get("r_rb")
            r_rb_str = "—" if r_rb is None or (isinstance(r_rb, float) and np.isnan(r_rb)) else f"{float(r_rb):+.2f}"
            pair_summaries.append(
                f"{direction_tex} ($p_{{\\text{{adj}}}}={_format_p(p_adj)}{p_star}$, $r_{{rb}}={r_rb_str}$)"
            )
        if pair_summaries:
            stats_note += (
                f"\\textit{{Post-hoc.}} Paired Wilcoxon prompt-pair tests with Bonferroni over 3 pairs "
                f"($\\alpha_{{\\text{{adj}}}}={alpha_adj:.4f}$): " + "; ".join(pair_summaries) + ".\\par\n"
            )

        eff = blocked_stats.get("efficacy_vs_zero", {})
        if eff:
            eff_parts = []
            for v in VARIANTS:
                pv = eff.get(v, {}).get("p")
                rrb = eff.get(v, {}).get("r_rb")
                rrb_str = "—" if rrb is None or (isinstance(rrb, float) and np.isnan(rrb)) else f"{float(rrb):+.2f}"
                eff_parts.append(f"{v}: p={_format_p(pv)}{_latex_superscript(_p_stars(pv))}, $r_{{rb}}={rrb_str}$")
            stats_note += (
                "\\textit{Efficacy.} One-sample Wilcoxon tests of median(F1 $\\Delta$)=0 on block-level values: "
                + ", ".join(eff_parts)
                + ".\n"
            )

    # Caption text (no manual line breaks - minipage handles wrapping)
    caption_text = f"Corrector Prompt Ablation on {exp_type_title}: Performance Comparison"

    # Wrap entire content in minipage to constrain caption + notes to page width
    latex = f"""\\begin{{table}}[H]
\\centering
\\begin{{minipage}}{{\\linewidth}}
\\centering
\\caption{{{caption_text}}}
\\label{{tab:rq1b_prompt_ablation_{experiment_type}}}
\\begin{{threeparttable}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccccc}}
\\toprule
\\textbf{{Prompt}} & \\textbf{{Overall F1 $\\Delta$}} & \\textbf{{Judge $\\Delta$}} & \\textbf{{Social Norms}} & \\textbf{{Depressive}} & \\textbf{{Emergency Dept}} & \\textbf{{Total}} & \\textbf{{Revise}} & \\textbf{{Change Type}} \\\\
\\textbf{{Variant}} & \\textbf{{(mean $\\pm$ std)}} & \\textbf{{(mean $\\pm$ std)}} & \\textbf{{F1 $\\Delta$ (mean $\\pm$ std)}} & \\textbf{{F1 $\\Delta$ (mean $\\pm$ std)}} & \\textbf{{F1 $\\Delta$ (mean $\\pm$ std)}} & \\textbf{{Actions}} & & \\\\
\\midrule
{chr(10).join(rows)}
{aggregate_row}
\\bottomrule
\\end{{tabular}}}}
\\end{{threeparttable}}
\\vspace{{0.5em}}
\\footnotesize
\\textit{{Note.}} Ablation study comparing three corrector prompt variants (3 runs per CLD per prompt, 27 total experiments). 
Aggregate row shows macro-averaged mean $\\pm$ std across CLDs.
F1 $\\Delta$ = change in F1 score from pre- to post-correction; Judge $\\Delta$ = change in LLM judge score.
Total Actions = Revise + Change Type (successful corrections only); Revise = motivation revisions; Change Type = edge type changes.
Edges with no correction needed or API errors excluded (see Appendix for full breakdown).\\par
{stats_note}
\\end{{minipage}}
\\end{{table}}
"""
    
    # Save LaTeX file
    latex_file = output_dir / f'ablation_table_{experiment_type}.tex'
    with open(latex_file, 'w') as f:
        f.write(latex)
    
    print(f"✅ LaTeX table saved to: {latex_file}")
    
    return latex


def generate_appendix_action_table(combined: pd.DataFrame, experiment_type: str, output_dir: Path) -> str:
    """Generate appendix table with full action count breakdown including None and Error."""
    
    # Calculate per-CLD means
    per_cld_means = calculate_per_cld_means(combined)
    
    # Calculate macro averages
    macro_avg = calculate_macro_average(per_cld_means)
    
    # Build table rows
    rows = []
    
    for variant in VARIANTS:
        macro = macro_avg.loc[variant]
        
        # Get all action counts
        revise = int(macro['Actions_Revise_sum_sum'])
        change = int(macro['Actions_Change_sum_sum'])
        none_count = int(macro.get('Actions_None_sum_sum', 0))
        error_count = int(macro.get('Actions_Error_sum_sum', 0))
        total = revise + change + none_count + error_count  # Full total
        successful = revise + change  # Successful only
        
        # Build row
        variant_name = variant.capitalize() if variant != 'cot' else 'CoT'
        row = f"{variant_name} & {total} & {successful} & {revise} & {change} & {none_count} & {error_count} \\\\"
        rows.append(row)
    
    # Aggregate row
    total_revise = int(per_cld_means['Actions_Revise_sum'].sum())
    total_change = int(per_cld_means['Actions_Change_sum'].sum())
    total_none = int(per_cld_means['Actions_None_sum'].sum())
    total_error = int(per_cld_means['Actions_Error_sum'].sum())
    full_total = total_revise + total_change + total_none + total_error
    successful_total = total_revise + total_change
    
    aggregate_row = f"\\midrule\n\\textbf{{Aggregate}} & {full_total} & {successful_total} & {total_revise} & {total_change} & {total_none} & {total_error} \\\\"
    
    # Build full LaTeX table
    exp_type_title = "Synthetic Corruptions" if experiment_type == 'synthetic' else "Ground Truth"
    
    latex = f"""\\begin{{table}}[H]
\\centering
\\caption{{RQ1b Corrector Action Counts - {exp_type_title} (Full Breakdown)}}
\\label{{tab:rq1b_action_counts_{experiment_type}}}
\\begin{{threeparttable}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
\\textbf{{Prompt}} & \\textbf{{Edges}} & \\textbf{{Successful}} & \\textbf{{Revise}} & \\textbf{{Change Type}} & \\textbf{{None}} & \\textbf{{Error}} \\\\
\\textbf{{Variant}} & \\textbf{{Processed}} & \\textbf{{Actions}} & & & & \\\\
\\midrule
{chr(10).join(rows)}
{aggregate_row}
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item \\textit{{Note.}} Full breakdown of corrector actions for verification. 
Edges Processed = total candidate edges for correction.
Successful Actions = Revise + Change (used in main text tables).
None = edges where corrector determined no change needed.
Error = API/parsing errors during correction.
Verification: Edges Processed = Revise + Change + None + Error.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""
    
    # Save LaTeX file
    latex_file = output_dir / f'appendix_action_counts_{experiment_type}.tex'
    with open(latex_file, 'w') as f:
        f.write(latex)
    
    print(f"✅ Appendix action table saved to: {latex_file}")
    
    return latex


def generate_baseline_table(
    combined: pd.DataFrame,
    experiment_type: str,
    output_dir: Path,
    blocked_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate baseline-only table (Tables 7 and 9 in thesis) showing per-CLD breakdown."""
    
    # Filter to baseline only
    baseline_data = combined[combined['Variant'] == 'baseline'].copy()
    
    if baseline_data.empty:
        print(f"⚠️  No baseline data found for {experiment_type}")
        return ""
    
    # CLD order must match LaTeX header
    clds = ['social_norms', 'depressive', 'emergency_department']
    cld_display_names = {
        'social_norms': 'Social Norms',
        'depressive': 'Depressive',
        'emergency_department': 'Emergency Dept'
    }
    
    # Calculate per-CLD means - for actions, compute per-run mean ± std (not totals)
    # First compute Total = Revise + Change for each row
    baseline_data['Actions_Successful'] = baseline_data['Actions_Revise'] + baseline_data['Actions_Change']
    
    per_cld = baseline_data.groupby('CLD').agg({
        'F1_Delta': ['mean', 'std'],
        'Precision_Delta': ['mean', 'std'],
        'Recall_Delta': ['mean', 'std'],
        'Judge_Score_Delta': ['mean', 'std'],
        'Actions_Successful': ['mean', 'std'],  # Per-run mean ± std
        'Actions_Revise': ['mean', 'std'],      # Per-run mean ± std
        'Actions_Change': ['mean', 'std'],      # Per-run mean ± std
        'Actions_None': 'sum',
        'Actions_Error': 'sum'
    })
    
    # Flatten column names
    per_cld.columns = ['_'.join(col).strip('_') for col in per_cld.columns.values]
    
    # Build table rows
    rows = []
    for cld in clds:
        if cld not in per_cld.index:
            continue
        row_data = per_cld.loc[cld]
        
        f1_mean = row_data['F1_Delta_mean']
        f1_std = row_data['F1_Delta_std']
        prec_mean = row_data['Precision_Delta_mean']
        prec_std = row_data['Precision_Delta_std']
        rec_mean = row_data['Recall_Delta_mean']
        rec_std = row_data['Recall_Delta_std']
        judge_mean = row_data['Judge_Score_Delta_mean']
        judge_std = row_data['Judge_Score_Delta_std']
        
        # Actions: Per-run mean ± std (Total = Revise + Change)
        total_mean = row_data['Actions_Successful_mean']
        total_std = row_data['Actions_Successful_std']
        revise_mean = row_data['Actions_Revise_mean']
        revise_std = row_data['Actions_Revise_std']
        change_mean = row_data['Actions_Change_mean']
        change_std = row_data['Actions_Change_std']
        
        # Format with sign
        def fmt(mean, std):
            sign = '+' if mean >= 0 else ''
            return f"${sign}{mean:.3f} \\pm {std:.3f}$"
        
        row = f"{cld_display_names[cld]} & {fmt(f1_mean, f1_std)} & {fmt(prec_mean, prec_std)} & {fmt(rec_mean, rec_std)} & {fmt(judge_mean, judge_std)} & ${total_mean:.1f} \\pm {total_std:.1f}$ & ${revise_mean:.1f} \\pm {revise_std:.1f}$ & ${change_mean:.1f} \\pm {change_std:.1f}$ \\\\"
        rows.append(row)
    
    # Macro average row - mean of per-CLD means ± std across CLDs
    macro_f1_mean = per_cld['F1_Delta_mean'].mean()
    macro_f1_std = per_cld['F1_Delta_mean'].std()
    macro_prec_mean = per_cld['Precision_Delta_mean'].mean()
    macro_prec_std = per_cld['Precision_Delta_mean'].std()
    macro_rec_mean = per_cld['Recall_Delta_mean'].mean()
    macro_rec_std = per_cld['Recall_Delta_mean'].std()
    macro_judge_mean = per_cld['Judge_Score_Delta_mean'].mean()
    macro_judge_std = per_cld['Judge_Score_Delta_mean'].std()
    
    # Aggregate actions - mean of per-CLD per-run means ± std across CLDs
    import numpy as np
    macro_total_mean = per_cld['Actions_Successful_mean'].mean()
    macro_total_std = per_cld['Actions_Successful_mean'].std()
    macro_revise_mean = per_cld['Actions_Revise_mean'].mean()
    macro_revise_std = per_cld['Actions_Revise_mean'].std()
    macro_change_mean = per_cld['Actions_Change_mean'].mean()
    macro_change_std = per_cld['Actions_Change_mean'].std()
    
    def fmt_macro(mean, std):
        sign = '+' if mean >= 0 else ''
        return f"${sign}{mean:.3f}$ $\\pm$ {std:.3f}"
    
    macro_row = f"\\midrule\nMacro Avg & {fmt_macro(macro_f1_mean, macro_f1_std)} & {fmt_macro(macro_prec_mean, macro_prec_std)} & {fmt_macro(macro_rec_mean, macro_rec_std)} & {fmt_macro(macro_judge_mean, macro_judge_std)} & ${macro_total_mean:.1f}$ $\\pm$ {macro_total_std:.1f} & ${macro_revise_mean:.1f}$ $\\pm$ {macro_revise_std:.1f} & ${macro_change_mean:.1f}$ $\\pm$ {macro_change_std:.1f} \\\\"
    
    # Build full LaTeX table
    exp_type_title = "Synthetically Corrupted Data" if experiment_type == 'synthetic' else "Ground Truth Data Using Correctness-Based Validation"
    label_suffix = "synthetic_correctness" if experiment_type == 'synthetic' else "groundtruth_correctness"
    
    # Optional efficacy note for baseline (blocked one-sample Wilcoxon vs 0) - no \item since notes are outside tablenotes
    stats_note = ""
    if blocked_stats:
        eff = blocked_stats.get("efficacy_vs_zero", {})
        base_p = eff.get("baseline", {}).get("p") if isinstance(eff, dict) else None
        base_r = eff.get("baseline", {}).get("r_rb") if isinstance(eff, dict) else None
        n_blocks = blocked_stats.get("n_blocks", None)
        if base_p is not None:
            r_str = "—" if base_r is None or (isinstance(base_r, float) and np.isnan(base_r)) else f"{float(base_r):+.2f}"
            stats_note = (
                f"\\textit{{Efficacy (blocked).}} One-sample Wilcoxon signed-rank test of median(F1 $\\Delta$)=0 "
                f"on block-level values (blocks=(CLD, run), n={n_blocks}): p={_format_p(base_p)}{_latex_superscript(_p_stars(base_p))}, "
                f"$r_{{rb}}={r_str}$.\n"
            )

    baseline_suffix = "(Baseline Prompt)"
    # Wrap entire content in minipage to constrain caption + notes to page width
    latex = f"""\\begin{{table}}[H]
\\centering
\\begin{{minipage}}{{\\linewidth}}
\\centering
\\caption{{Corrector Performance on {exp_type_title} {baseline_suffix}}}
\\label{{tab:rq1b_{label_suffix}}}
\\begin{{threeparttable}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{lccccccc}}
\\toprule
\\textbf{{CLD}} & \\textbf{{F1 $\\Delta$}} & \\textbf{{Precision $\\Delta$}} & \\textbf{{Recall $\\Delta$}} & \\textbf{{Judge Score $\\Delta$}} & \\textbf{{Total Actions}} & \\textbf{{Revise}} & \\textbf{{Change Type}} \\\\
\\midrule
{chr(10).join(rows)}
{macro_row}
\\bottomrule
\\end{{tabular}}}}
\\end{{threeparttable}}
\\vspace{{0.5em}}
\\footnotesize
\\textit{{Note.}} Results show change in performance ($\\Delta$) from pre- to post-correction (3 runs per CLD).
F1 $\\Delta$ = change in F1 score (Post-Pre); Precision $\\Delta$, Recall $\\Delta$, Judge Score $\\Delta$ = corresponding changes.
Total Actions = Revise + Change (successful corrections only); Revise = motivation revisions; Change Type = edge type changes.
Values presented as mean $\\pm$ std across 3 runs per CLD. Macro average shows mean of CLD means $\\pm$ std across CLDs.\\par
{stats_note}
\\end{{minipage}}
\\end{{table}}
"""
    
    # Save LaTeX file
    latex_file = output_dir / f'baseline_table_{experiment_type}.tex'
    with open(latex_file, 'w') as f:
        f.write(latex)
    
    print(f"✅ Baseline table saved to: {latex_file}")
    
    return latex


def analyze_experiment(experiment_type: str, output_dir: Path, judge_suffix: str = ''):
    """Run full analysis for one experiment type."""
    judge_label = f" (Judge: {judge_suffix.replace('_', ' ').strip()})" if judge_suffix else ""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {experiment_type.upper()} ABLATION{judge_label}")
    print(f"{'='*80}")
    
    # Load all results
    all_results = []
    for variant in VARIANTS:
        try:
            df = load_results(variant, experiment_type, judge_suffix)
            all_results.append(df)
            print(f"✅ Loaded {variant}: {len(df)} experiments")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return None
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # 1. Overall Performance by Variant
    print("\n" + "-"*60)
    print("OVERALL PERFORMANCE BY VARIANT")
    print("-"*60)
    
    summary = combined.groupby('Variant').agg({
        'F1_Delta': ['mean', 'std', 'count'],
        'Precision_Delta': ['mean', 'std'],
        'Recall_Delta': ['mean', 'std'],
        'Judge_Score_Delta': ['mean', 'std']
    }).round(4)
    
    print(summary)
    
    # 2. Per-CLD Breakdown
    print("\n" + "-"*60)
    print("PER-CLD BREAKDOWN (F1 Delta)")
    print("-"*60)
    
    for cld in sorted(combined['CLD'].unique()):
        print(f"\n{cld.upper()}:")
        cld_data = combined[combined['CLD'] == cld]
        cld_summary = cld_data.groupby('Variant')['F1_Delta'].agg(['mean', 'std', 'count']).round(4)
        print(cld_summary)
    
    # 3. Macro-Averaged Statistics
    print("\n" + "-"*60)
    print("MACRO-AVERAGED STATISTICS (mean ± std across CLDs)")
    print("-"*60)
    
    per_cld_means = calculate_per_cld_means(combined)
    macro_avg = calculate_macro_average(per_cld_means)
    print(macro_avg.round(4))
    
    # 4. Action Distribution
    print("\n" + "-"*60)
    print("ACTION DISTRIBUTION BY VARIANT")
    print("-"*60)
    
    action_summary = combined.groupby('Variant').agg({
        'Actions_Revise': 'sum',
        'Actions_Change': 'sum',
        'Actions_Remove': 'sum',
        'Actions_Flip': 'sum',
        'Actions_Total': 'sum'
    })
    
    print(action_summary)
    
    # 5. Statistical Tests (blocked design + reproducible)
    print("\n" + "-"*60)
    print("STATISTICAL SIGNIFICANCE")
    print("-"*60)
    
    blocked_stats: Optional[Dict[str, Any]] = None
    try:
        blocked_stats = compute_blocked_prompt_tests(combined)
        print("\nBlocked omnibus (Friedman) on F1_Delta:")
        print(f"  n_blocks = {blocked_stats.get('n_blocks')}, p = {_format_p(blocked_stats.get('friedman_p'))}, W = {blocked_stats.get('kendalls_w', float('nan')):.3f}")
        print("  Post-hoc paired Wilcoxon (Bonferroni over 3 pairs):")
        for key, r in blocked_stats.get("posthoc", {}).get("pairs", {}).items():
            print(f"    {r['a']} vs {r['b']}: p={_format_p(r.get('p_raw'))}, p_adj={_format_p(r.get('p_adj'))}, dir={r.get('direction')}")
        print("  Efficacy vs 0 (one-sample Wilcoxon per variant):")
        for v in VARIANTS:
            print(f"    {v}: p={_format_p(blocked_stats.get('efficacy_vs_zero', {}).get(v, {}).get('p'))}")
    except ImportError:
        print("⚠️  scipy not available, skipping blocked statistical tests")
    except Exception as e:
        print(f"⚠️  Blocked statistical tests failed: {e}")
    
    # 6. Generate LaTeX table with aggregate row
    print("\n" + "-"*60)
    print("GENERATING LATEX TABLE")
    print("-"*60)
    
    latex = generate_latex_table(combined, experiment_type, output_dir, blocked_stats=blocked_stats)
    
    # Also generate appendix table with full action counts
    appendix_latex = generate_appendix_action_table(combined, experiment_type, output_dir)
    
    # Also generate baseline-only table (Tables 7 and 9 in thesis)
    baseline_latex = generate_baseline_table(combined, experiment_type, output_dir, blocked_stats=blocked_stats)
    
    # 7. Export combined results
    output_file = output_dir / f'corrector_ablation_{experiment_type}_results.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        combined.to_excel(writer, sheet_name='All_Results', index=False)
        per_cld_means.to_excel(writer, sheet_name='Per_CLD_Means', index=False)
        macro_avg.to_excel(writer, sheet_name='Macro_Average')
    
    print(f"✅ Results exported to: {output_file}")
    
    return combined


def main():
    parser = argparse.ArgumentParser(description='Analyze corrector ablation study results')
    parser.add_argument('--type', choices=['synthetic', 'groundtruth', 'both'], 
                       default='both',
                       help='Type of experiment to analyze')
    parser.add_argument('--judge-variant', choices=['baseline', 'mechanistic'], 
                       default='baseline',
                       help='Which judge variant was used as input for the corrector')
    args = parser.parse_args()
    
    # Determine folder suffix based on judge variant
    judge_suffix = '' if args.judge_variant == 'baseline' else f'_{args.judge_variant}_judge'
    
    print("="*80)
    print("CORRECTOR ABLATION ANALYSIS")
    if judge_suffix:
        print(f"(Using {args.judge_variant} judge input)")
    print("="*80)
    
    # Create output directory
    output_dir = Path(__file__).resolve().parent.parent.parent / 'final_runs' / 'RQ1b_corrector_ablation_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiments_to_run = []
    if args.type == 'both':
        experiments_to_run = ['synthetic', 'groundtruth']
    else:
        experiments_to_run = [args.type]
    
    results = {}
    for exp_type in experiments_to_run:
        result = analyze_experiment(exp_type, output_dir, judge_suffix)
        if result is not None:
            results[exp_type] = result
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Experiments analyzed: {', '.join(results.keys())}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

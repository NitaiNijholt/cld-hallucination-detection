#!/usr/bin/env python3
"""
Unified RQ3 Deep Research Analysis

Runs all RQ3 analyses and generates reproducible LaTeX tables with embedded statistics.

Usage:
    python run_rq3_analysis.py

Outputs:
    - rq3_main_table.tex (Table with detection rates + statistical tests)
    - rq3_per_cld_table.tex (Per-CLD breakdown)
    - rq3_comprehensive_stats.json (All statistical results)
    - figures/rq3_combined_figure.png (Detection rates + confidence plots)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from datetime import datetime
from typing import Dict, Any, List, Tuple
import os

# Set matplotlib style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300

# ============================================================================
# Configuration
# ============================================================================

# Default INPUT_DIR to the RQ3_deep_research folder (parent of analysis_scripts)
INPUT_DIR = Path(os.environ.get("RQ3_INPUT_DIR", str(Path(__file__).parent.parent))).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("RQ3_OUTPUT_DIR", str(INPUT_DIR))).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

DATA_FILES = [
    (INPUT_DIR / "Data/deep_research_results_depressive_symptoms_20251013_032002_84edges.json", "Depressive"),
    (INPUT_DIR / "Data/deep_research_results_social_norms_20251012_213641_17edges.json", "Social Norms"),
    (INPUT_DIR / "Data/deep_research_results_older_persons_ALL_EDGES_184edges.json", "Older Persons"),
]


# ============================================================================
# Helper Functions
# ============================================================================

def load_results(json_file: str | Path) -> list:
    """Load results from a JSON file."""
    with open(str(json_file), 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    elif isinstance(data, list):
        return data
    else:
        return data.get('results', [])


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute Precision/Recall/F1 from confusion matrix counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compute_enriched_gt_metrics(data: Dict) -> Dict[str, Any]:
    """
    Compute enrichment scenarios from the same RQ3 dataset:
    - Original: (TP, FP, FN) from GT Lit validation
    - Scenario 1: promote FP edges with DR evidence to TP
    - Scenario 2: Scenario 1 + add FN edges with DR evidence (DR-discovered FNs)
    """
    counts = data["counts"]
    per_cld = data["per_cld"]

    # Aggregate confusion matrix counts
    tp = counts["TP"]["total"]
    fp = counts["FP"]["total"]
    fn = counts["FN"]["total"]

    # Aggregate promotions/discoveries from RQ3 edges (binary: DR found direct causal evidence)
    fp_promoted = sum(1 for r in data["_all_results"] if r["classification"] == "FP" and r["found"])
    fn_discovered = sum(1 for r in data["_all_results"] if r["classification"] == "FN" and r["found"])

    original = compute_metrics(tp, fp, fn)
    scenario1 = compute_metrics(tp + fp_promoted, fp - fp_promoted, fn)
    scenario2 = compute_metrics(tp + fp_promoted + fn_discovered, fp - fp_promoted, fn - fn_discovered)

    # Per-CLD breakdown
    cld_rows = {}
    for cld_name, cld_data in per_cld.items():
        tp_c = len(cld_data.get("TP", []))
        fp_c = len(cld_data.get("FP", []))
        fn_c = len(cld_data.get("FN", []))
        fp_prom_c = sum(1 for e in cld_data.get("FP", []) if e["found"])
        fn_disc_c = sum(1 for e in cld_data.get("FN", []) if e["found"])

        cld_rows[cld_name] = {
            "original": compute_metrics(tp_c, fp_c, fn_c),
            "scenario1": compute_metrics(tp_c + fp_prom_c, fp_c - fp_prom_c, fn_c),
            "scenario2": compute_metrics(tp_c + fp_prom_c + fn_disc_c, fp_c - fp_prom_c, fn_c - fn_disc_c),
            "fp_promoted": fp_prom_c,
            "fp_total": fp_c,
            "fn_discovered": fn_disc_c,
            "fn_total": fn_c,
        }

    return {
        "aggregate": {
            "original": original,
            "scenario1": scenario1,
            "scenario2": scenario2,
            "fp_promoted": fp_promoted,
            "fp_total": fp,
            "fn_discovered": fn_discovered,
            "fn_total": fn,
        },
        "per_cld": cld_rows,
    }


def generate_enriched_gt_table(enrichment: Dict[str, Any]) -> str:
    """Generate Table 15 LaTeX (Enriched GT scenarios)."""
    agg = enrichment["aggregate"]

    def fmt_f1(x: float) -> str:
        return f"{x:.3f}"

    def pct_change(new: float, old: float) -> int:
        if old <= 0:
            return 0
        return int(round(100 * (new - old) / old))

    orig_f1 = agg["original"]["f1"]
    s1_f1 = agg["scenario1"]["f1"]
    s2_f1 = agg["scenario2"]["f1"]

    # Helper to make per-CLD row
    def cld_row(label: str, cld_key: str) -> str:
        row = enrichment["per_cld"][cld_key]
        o = row["original"]["f1"]
        s1 = row["scenario1"]["f1"]
        s2 = row["scenario2"]["f1"]
        fp_prom = row["fp_promoted"]
        fp_total = row["fp_total"]
        fn_disc = row["fn_discovered"]
        fn_total = row["fn_total"]
        return (
            f"{label} & {fmt_f1(o)} & {fmt_f1(s1)} (+{pct_change(s1, o)}\\%) & {fp_prom}/{fp_total} ({int(round(100*fp_prom/fp_total)) if fp_total else 0}\\%) "
            f"& {fmt_f1(s2)} (+{pct_change(s2, o)}\\%) & {fn_disc}/{fn_total} ({int(round(100*fn_disc/fn_total)) if fn_total else 0}\\%) \\\\\n"
        )

    latex = r"""\begin{table}[H]
\centering
\begin{minipage}{\linewidth}
\centering
\caption{LLM Generation Metrics: Two Enrichment Scenarios (Aggregate and Per-CLD)}
\label{tab:enriched_gt}
\begin{threeparttable}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
& \textbf{Original} & \multicolumn{2}{c}{\textbf{Scenario 1: Enriched GT}} & \multicolumn{2}{c}{\textbf{Scenario 2: LLM+DR System}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
\textbf{Dataset} & \textbf{F1} & \textbf{F1} & \textbf{FPs $\to$ TP} & \textbf{F1} & \textbf{FNs found} \\
\midrule
"""
    latex += (
        f"\\textbf{{Aggregate}} & {fmt_f1(orig_f1)} & {fmt_f1(s1_f1)} (+{pct_change(s1_f1, orig_f1)}\\%) & "
        f"{agg['fp_promoted']}/{agg['fp_total']} ({int(round(100*agg['fp_promoted']/agg['fp_total'])) if agg['fp_total'] else 0}\\%) & "
        f"{fmt_f1(s2_f1)} (+{pct_change(s2_f1, orig_f1)}\\%) & {agg['fn_discovered']}/{agg['fn_total']} ({int(round(100*agg['fn_discovered']/agg['fn_total'])) if agg['fn_total'] else 0}\\%) \\\\\n"
    )
    latex += r"\midrule" + "\n"
    latex += cld_row("Depressive", "Depressive")
    latex += cld_row("Social Norms", "Social Norms")
    latex += cld_row("Older Persons", "Older Persons")
    latex += r"""\bottomrule
\end{tabular}}
\begin{tablenotes}
\small
\item \textit{Scenario 1 (Enriched GT):} Promotes FP edges with DR evidence to TP. Tests: ``What if experts missed valid relationships?''
\item \textit{Scenario 2 (LLM+DR System):} Scenario 1 + DR discovers FN edges with literature support. Tests: ``What could a complete LLM+DR pipeline achieve?''
\item \textit{Aggregate confusion matrices:} Original: TP=44, FP=178, FN=63. Scenario 1: TP=155, FP=67, FN=63. Scenario 2: TP=182, FP=67, FN=36.
\end{tablenotes}
\end{threeparttable}
\end{minipage}
\end{table}
"""
    return latex

def compute_cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h effect size for proportion difference."""
    p1 = max(0.001, min(0.999, p1))
    p2 = max(0.001, min(0.999, p2))
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2


def cohens_h_ci(p1: float, n1: int, p2: float, n2: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Compute Cohen's h with confidence interval."""
    h = compute_cohens_h(p1, p2)
    se_h = np.sqrt(1/n1 + 1/n2)
    z_crit = stats.norm.ppf(1 - alpha/2)
    lower = h - z_crit * se_h
    upper = h + z_crit * se_h
    return h, lower, upper


def interpret_cohens_h(h: float) -> str:
    """Interpret Cohen's h (Cohen, 1988)."""
    h_abs = abs(h)
    if h_abs < 0.20:
        return "negligible"
    elif h_abs < 0.50:
        return "small"
    elif h_abs < 0.80:
        return "medium"
    else:
        return "large"


def compute_power(p1: float, p2: float, n1: int, n2: int, alpha: float = 0.05) -> float:
    """Compute achieved power for two-proportion z-test."""
    h = abs(compute_cohens_h(p1, p2))
    n_harmonic = 2 / (1/n1 + 1/n2)
    ncp = h * np.sqrt(n_harmonic / 2)
    z_crit = stats.norm.ppf(1 - alpha/2)
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
    return float(power)


def _format_p(p: float) -> str:
    """Format p-value for display."""
    if p is None or np.isnan(p):
        return "—"
    if p >= 1:
        return "1.00"
    if p < 0.001:
        return "<.001"
    # Format as .XXX (e.g., 0.038 -> ".038", 0.382 -> ".38")
    if p < 0.01:
        return f"{p:.3f}".lstrip("0")  # e.g., 0.006 -> ".006"
    return f"{p:.2f}".lstrip("0")  # e.g., 0.38 -> ".38"


def _p_stars(p: float) -> str:
    """Return significance stars."""
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


# ============================================================================
# Data Loading and Processing
# ============================================================================

def load_all_data() -> Dict[str, Dict[str, Any]]:
    """Load all RQ3 data and compute basic statistics."""
    all_results = []
    per_cld_data = {}
    
    for file_path, cld_name in DATA_FILES:
        full_path = OUTPUT_DIR / file_path
        if not full_path.exists():
            print(f"Warning: {full_path} not found")
            continue
        
        results = load_results(str(full_path))
        
        cld_stats = {"TP": [], "FP": [], "FN": []}
        for r in results:
            classification = r.get("classification", "unknown")
            if classification in cld_stats:
                # Handle both field naming conventions
                found = r.get("direct_causal_found", r.get("direct_causal_evidence_found", False))
                confidence = r.get("confidence", r.get("confidence_score", 0))
                edge = f"{r.get('source', 'unknown')} -> {r.get('target', 'unknown')}"
                cld_stats[classification].append({
                    "found": found,
                    "confidence": confidence,
                    "edge": edge
                })
                all_results.append({
                    "classification": classification,
                    "found": found,
                    "confidence": confidence,
                    "cld": cld_name
                })
        
        per_cld_data[cld_name] = cld_stats
    
    # Aggregate counts
    counts = {"TP": {"total": 0, "found": 0}, 
              "FP": {"total": 0, "found": 0}, 
              "FN": {"total": 0, "found": 0}}
    confidences = {"TP": [], "FP": [], "FN": []}
    
    for r in all_results:
        c = r["classification"]
        counts[c]["total"] += 1
        if r["found"]:
            counts[c]["found"] += 1
        confidences[c].append(r["confidence"])
    
    for c in counts:
        counts[c]["rate"] = counts[c]["found"] / counts[c]["total"] if counts[c]["total"] > 0 else 0
        counts[c]["confidence_mean"] = np.mean(confidences[c]) if confidences[c] else 0
    
    return {
        "counts": counts,
        "per_cld": per_cld_data,
        "total_edges": sum(counts[c]["total"] for c in counts),
        "_all_results": all_results,  # internal use for reproducible enrichment table
    }


# ============================================================================
# Statistical Tests
# ============================================================================

def compute_statistical_tests(data: Dict) -> Dict[str, Any]:
    """Compute all statistical tests for RQ3."""
    counts = data["counts"]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_edges": data["total_edges"],
        "counts": counts,
    }
    
    # Pairwise comparisons
    pairs = [("TP", "FP"), ("TP", "FN"), ("FP", "FN")]
    n_tests = len(pairs)
    alpha_adj = 0.05 / n_tests
    
    results["bonferroni"] = {
        "n_tests": n_tests,
        "alpha_original": 0.05,
        "alpha_adjusted": alpha_adj
    }
    
    for a, b in pairs:
        key = f"{a.lower()}_vs_{b.lower()}"
        
        n_a, found_a = counts[a]["total"], counts[a]["found"]
        n_b, found_b = counts[b]["total"], counts[b]["found"]
        p_a, p_b = counts[a]["rate"], counts[b]["rate"]
        
        # Contingency table for Fisher's exact test
        table = [[found_a, n_a - found_a], [found_b, n_b - found_b]]
        
        # Fisher's exact test
        odds_ratio, fisher_p = stats.fisher_exact(table)
        fisher_p_adj = min(fisher_p * n_tests, 1.0)
        
        # Effect size
        h, h_lower, h_upper = cohens_h_ci(p_a, n_a, p_b, n_b)
        
        # Power
        power = compute_power(p_a, p_b, n_a, n_b)
        
        # Odds ratio CI (Woolf method)
        log_or = np.log(odds_ratio) if odds_ratio > 0 else 0
        se_log_or = np.sqrt(1/found_a + 1/(n_a-found_a) + 1/found_b + 1/(n_b-found_b)) if min(found_a, n_a-found_a, found_b, n_b-found_b) > 0 else np.inf
        or_ci_lower = np.exp(log_or - 1.96 * se_log_or)
        or_ci_upper = np.exp(log_or + 1.96 * se_log_or)
        
        results[key] = {
            "gap_pp": (p_a - p_b) * 100,
            "fisher_p": float(fisher_p),
            "fisher_p_adj": float(fisher_p_adj),
            "significant_adj": fisher_p < alpha_adj,
            "odds_ratio": float(odds_ratio),
            "or_ci": [float(or_ci_lower), float(or_ci_upper)],
            "cohens_h": float(h),
            "cohens_h_ci": [float(h_lower), float(h_upper)],
            "cohens_h_interp": interpret_cohens_h(h),
            "power": float(power),
        }
    
    return results


def compute_per_cld_stats(data: Dict) -> Dict[str, Any]:
    """Compute per-CLD detection rates."""
    per_cld_stats = {}
    
    for cld_name, cld_data in data["per_cld"].items():
        cld_stats = {}
        for classification in ["TP", "FP", "FN"]:
            edges = cld_data.get(classification, [])
            n = len(edges)
            found = sum(1 for e in edges if e["found"])
            rate = found / n if n > 0 else 0
            conf_mean = np.mean([e["confidence"] for e in edges]) if edges else 0
            
            cld_stats[classification] = {
                "n": n,
                "found": found,
                "rate": rate,
                "confidence_mean": conf_mean
            }

        # Per-CLD inference: match Table 13 setup (all 3 pairwise comparisons + Bonferroni m=3)
        def _pairwise(a: str, b: str) -> Dict[str, Any] | None:
            aa = cld_stats.get(a, {"n": 0, "found": 0, "rate": 0})
            bb = cld_stats.get(b, {"n": 0, "found": 0, "rate": 0})
            if aa["n"] <= 0 or bb["n"] <= 0:
                return None
            table = [[aa["found"], aa["n"] - aa["found"]], [bb["found"], bb["n"] - bb["found"]]]
            _, p_fisher = stats.fisher_exact(table)
            h, h_lo, h_hi = cohens_h_ci(aa["rate"], aa["n"], bb["rate"], bb["n"])
            return {
                "gap_pp": (aa["rate"] - bb["rate"]) * 100,
                "fisher_p": float(p_fisher),
                "cohens_h": float(h),
                "cohens_h_ci": [float(h_lo), float(h_hi)],
                "cohens_h_interp": interpret_cohens_h(h),
            }

        tp_fp = _pairwise("TP", "FP")
        tp_fn = _pairwise("TP", "FN")
        fp_fn = _pairwise("FP", "FN")
        # Bonferroni correction within CLD for 3 pairwise tests
        m = 3
        for comp in [tp_fp, tp_fn, fp_fn]:
            if comp is not None:
                comp["fisher_p_adj"] = float(min(comp["fisher_p"] * m, 1.0))

        cld_stats["pairwise"] = {
            "tp_vs_fp": tp_fp,
            "tp_vs_fn": tp_fn,
            "fp_vs_fn": fp_fn,
            "bonferroni": {"n_tests": m, "alpha_adjusted": 0.05 / m},
        }

        # Keep TP vs FP convenience field for compact columns
        cld_stats["tp_vs_fp"] = tp_fp
        per_cld_stats[cld_name] = cld_stats
    
    return per_cld_stats


# ============================================================================
# LaTeX Table Generation
# ============================================================================

def generate_main_table(data: Dict, stats: Dict) -> str:
    """Generate the main RQ3 results table with embedded statistics."""
    counts = data["counts"]
    
    tp_fp = stats["tp_vs_fp"]
    tp_fn = stats["tp_vs_fn"]
    fp_fn = stats["fp_vs_fn"]
    bonf = stats["bonferroni"]
    
    latex = r"""\begin{table}[H]
\centering
\begin{minipage}{\linewidth}
\centering
\caption{Deep Research Detection Rate and Discrimination Analysis (n=""" + str(data["total_edges"]) + r""" edges)}
\label{tab:rq3_results}
\begin{threeparttable}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Classification} & \textbf{n} & \textbf{Detection Rate} & \textbf{Confidence} & \textbf{TP--X Gap} & \textbf{Cohen's $h$} & \textbf{$p$-value} \\
\midrule
"""
    
    # TP row
    tp = counts["TP"]
    latex += f"TP (Expert-accepted) & {tp['total']} & {tp['rate']*100:.1f}\\% & {tp['confidence_mean']:.2f} & --- & --- & --- \\\\\n"
    
    # FP row
    fp = counts["FP"]
    h_fp = tp_fp["cohens_h"]
    p_fp = tp_fp["fisher_p"]
    latex += f"FP (LLM hallucinations) & {fp['total']} & {fp['rate']*100:.1f}\\% & {fp['confidence_mean']:.2f} & {tp_fp['gap_pp']:.1f} pp & {h_fp:.2f} & {_format_p(p_fp)} \\\\\n"
    
    # FN row
    fn = counts["FN"]
    h_fn = tp_fn["cohens_h"]
    p_fn = tp_fn["fisher_p"]
    latex += f"FN (Expert-rejected) & {fn['total']} & {fn['rate']*100:.1f}\\% & {fn['confidence_mean']:.2f} & {tp_fn['gap_pp']:.1f} pp & {h_fn:.2f} & {_format_p(p_fn)}{_p_stars(p_fn)} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}}
\end{threeparttable}
\vspace{0.5em}
\footnotesize
"""
    
    # Notes
    latex += r"\textit{Note.} Detection rate = \% edges with direct causal evidence found. "
    latex += r"Confidence = self-assessed (1--10 scale). TP/FP/FN from RQ1a GT Lit validation across all 3 CLDs.\par" + "\n"
    
    # TP-FP test
    latex += r"\textit{TP--FP test.} Fisher's exact $p=" + f"{_format_p(tp_fp['fisher_p'])}$; "
    latex += f"OR = {tp_fp['odds_ratio']:.2f} [95\\% CI: {tp_fp['or_ci'][0]:.2f}, {tp_fp['or_ci'][1]:.2f}]; "
    latex += f"Cohen's $h={tp_fp['cohens_h']:.2f}$ [95\\% CI: {tp_fp['cohens_h_ci'][0]:.2f}, {tp_fp['cohens_h_ci'][1]:.2f}] ({tp_fp['cohens_h_interp']}); "
    latex += f"power = {tp_fp['power']*100:.0f}\\%.\\par\n"
    
    # TP-FN test
    latex += r"\textit{TP--FN test.} Fisher's exact $p=" + f"{_format_p(tp_fn['fisher_p'])}{_p_stars(tp_fn['fisher_p'])}$; "
    latex += f"Cohen's $h={tp_fn['cohens_h']:.2f}$ [95\\% CI: {tp_fn['cohens_h_ci'][0]:.2f}, {tp_fn['cohens_h_ci'][1]:.2f}] ({tp_fn['cohens_h_interp']}); "
    latex += f"power = {tp_fn['power']*100:.0f}\\%.\\par\n"
    
    # FP-FN test
    latex += r"\textit{FP--FN test.} Fisher's exact $p=" + f"{_format_p(fp_fn['fisher_p'])}{_p_stars(fp_fn['fisher_p'])}$; "
    latex += f"Cohen's $h={fp_fn['cohens_h']:.2f}$ [95\\% CI: {fp_fn['cohens_h_ci'][0]:.2f}, {fp_fn['cohens_h_ci'][1]:.2f}] ({fp_fn['cohens_h_interp']}); "
    latex += f"power = {fp_fn['power']*100:.0f}\\%.\\par\n"
    
    # Multiple comparisons
    latex += r"\textit{Multiple comparisons.} Bonferroni correction for " + str(bonf['n_tests']) + " pairwise tests. "
    latex += f"TP--FP $p_{{\\text{{adj}}}}={_format_p(tp_fp['fisher_p_adj'])}{_p_stars(tp_fp['fisher_p_adj'])}$; "
    latex += f"TP--FN $p_{{\\text{{adj}}}}={_format_p(tp_fn['fisher_p_adj'])}{_p_stars(tp_fn['fisher_p_adj'])}$; "
    latex += f"FP--FN $p_{{\\text{{adj}}}}={_format_p(fp_fn['fisher_p_adj'])}{_p_stars(fp_fn['fisher_p_adj'])}$.\n"
    
    latex += r"""
\end{minipage}
\end{table}
"""
    return latex


def generate_per_cld_table(per_cld_stats: Dict) -> str:
    """Generate per-CLD pairwise discrimination tests table (Panel B only; descriptives shown in Figure 13)."""
    
    latex = r"""\begin{table}[H]
\centering
\begin{minipage}{\linewidth}
\centering
\caption{Per-CLD Pairwise Discrimination Tests (Fisher's Exact, Bonferroni $m=3$)}
\label{tab:rq3_per_cld}
\begin{threeparttable}
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llccccc}
\toprule
\textbf{CLD} & \textbf{Comparison} & \textbf{$\Delta$pp} & \textbf{Cohen's $h$} & \textbf{Interpretation} & \textbf{$p$} & \textbf{$p_{\text{adj}}$} \\
\midrule
"""
    
    for cld_name in ["Depressive", "Social Norms", "Older Persons"]:
        if cld_name not in per_cld_stats:
            continue
        
        pw = per_cld_stats[cld_name].get("pairwise", {})
        comparisons = [
            ("TP--FP", pw.get("tp_vs_fp")),
            ("TP--FN", pw.get("tp_vs_fn")),
            ("FP--FN", pw.get("fp_vs_fn")),
        ]
        
        first_row = True
        for comp_name, comp in comparisons:
            if not comp:
                continue
            cld_display = cld_name if first_row else ""
            gap = comp["gap_pp"]
            h = comp["cohens_h"]
            interp = comp.get("cohens_h_interp", interpret_cohens_h(h))
            p = comp["fisher_p"]
            p_adj = comp.get("fisher_p_adj", min(p * 3, 1.0))
            
            latex += f"{cld_display} & {comp_name} & {gap:+.1f} & {h:.2f} & {interp} & {_format_p(p)}{_p_stars(p)} & {_format_p(p_adj)}{_p_stars(p_adj)} \\\\\n"
            first_row = False
        
        latex += r"\midrule" + "\n"
    
    # Remove last \midrule and add bottomrule
    latex = latex.rsplit(r"\midrule", 1)[0]
    
    latex += r"""\bottomrule
\end{tabular}}
\end{threeparttable}

\vspace{0.25em}
\scriptsize
\textit{Note.} All pairwise comparisons (TP--FP, TP--FN, FP--FN) within each CLD using Fisher's exact test; Bonferroni correction applied within each CLD ($m=3$, $\alpha_{\text{adj}}=0.017$). Effect size interpretation: $|h|<0.2$ negligible, $0.2$--$0.5$ small, $0.5$--$0.8$ medium, $>0.8$ large. Detection rates and confidence distributions shown in Figure~\ref{fig:rq3_combined}.

\end{minipage}
\end{table}
"""
    return latex


# ============================================================================
# Figure Generation
# ============================================================================

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute Wilson score 95% confidence interval for a proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) / n) + (z**2) / (4 * (n**2)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def normalize_cld(name: str) -> str:
    """Normalize CLD name to canonical form."""
    s = (name or "").lower()
    if "depressive" in s:
        return "Depressive"
    if "social" in s:
        return "Social Norms"
    return "Older Persons"


def generate_all_figures(data: Dict, per_cld_stats: Dict, output_dir: Path):
    """Generate all RQ3 figures (combined 2-panel figure)."""
    
    clds = ["Depressive", "Social Norms", "Older Persons"]
    cls_order = ["TP", "FP", "FN"]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']  # Green, Red, Orange
    color_map = {"TP": colors[0], "FP": colors[1], "FN": colors[2]}
    markers = {"TP": "o", "FP": "s", "FN": "D"}
    offsets = {"TP": -0.22, "FP": 0.0, "FN": 0.22}
    
    # Collect raw edge data for confidence plot
    all_edges = []
    for fn, cld_label in DATA_FILES:
        fp = Path(__file__).parent / fn
        edges = load_results(str(fp))
        for e in edges:
            e["CLD_normalized"] = cld_label
        all_edges.extend(edges)
    
    # Aggregate counts by (CLD, class)
    counts = {(cld, cls): 0 for cld in clds for cls in cls_order}
    found = {(cld, cls): 0 for cld in clds for cls in cls_order}
    conf_by_cld_cls = {(cld, cls): [] for cld in clds for cls in cls_order}
    
    for r in all_edges:
        cls = r.get("classification")
        if cls not in cls_order:
            continue
        cld = r.get("CLD_normalized", normalize_cld(r.get("CLD", "")))
        counts[(cld, cls)] += 1
        if r.get("direct_causal_found"):
            found[(cld, cls)] += 1
        conf = r.get("confidence")
        if conf is not None:
            conf_by_cld_cls[(cld, cls)].append(conf)
    
    # Create combined figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # =========================================================================
    # Panel A: Detection Rates by CLD (Wilson 95% CI)
    # =========================================================================
    ax = axes[0]
    x = np.arange(len(clds), dtype=float)
    
    for cls in cls_order:
        y = []
        yerr_lower = []
        yerr_upper = []
        for cld in clds:
            k = found[(cld, cls)]
            n = counts[(cld, cls)]
            lo, hi = wilson_ci(k, n)
            p = (k / n) if n else 0.0
            y.append(100 * p)
            yerr_lower.append(100 * (p - lo))
            yerr_upper.append(100 * (hi - p))
        
        y = np.array(y)
        yerr = np.vstack([yerr_lower, yerr_upper])
        ax.errorbar(
            x + offsets[cls],
            y,
            yerr=yerr,
            fmt=markers[cls],
            markersize=7,
            color=color_map[cls],
            ecolor="black",
            elinewidth=1.2,
            capsize=3,
            label=f"{cls}",
        )
    
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Detection Rate (%)", fontweight="bold")
    ax.set_xlabel("CLD (with TP/FP/FN)", fontweight="bold")
    ax.set_title("(A) Detection Rates by CLD (Wilson 95% CI)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(clds)
    ax.set_ylim(0, 100)
    ax.legend(title="Class", loc="lower left")
    
    # =========================================================================
    # Panel B: Confidence by CLD and classification (Mean ± SD)
    # =========================================================================
    ax = axes[1]
    
    for cls in cls_order:
        y = []
        yerr = []
        for cld in clds:
            vals = conf_by_cld_cls[(cld, cls)]
            mean_val = np.mean(vals) if vals else 0.0
            std_val = np.std(vals) if len(vals) > 1 else 0.0
            y.append(mean_val)
            yerr.append(std_val)
        
        y = np.array(y)
        yerr = np.array(yerr)
        ax.errorbar(
            x + offsets[cls],
            y,
            yerr=yerr,
            fmt=markers[cls],
            markersize=7,
            color=color_map[cls],
            ecolor="black",
            elinewidth=1.2,
            capsize=3,
            label=f"{cls}",
        )
    
    ax.set_ylabel("Confidence (1–10)", fontweight="bold")
    ax.set_xlabel("CLD (with TP/FP/FN)", fontweight="bold")
    ax.set_title("(B) Confidence Scores by CLD (Mean ± SD)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(clds)
    ax.set_ylim(0, 10)
    ax.legend(title="Class", loc="lower left")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq3_combined_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'rq3_combined_figure.pdf', bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir / 'rq3_combined_figure.png'}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("RQ3 DEEP RESEARCH UNIFIED ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    data = load_all_data()
    print(f"   Total edges: {data['total_edges']}")
    for c in ["TP", "FP", "FN"]:
        print(f"   {c}: n={data['counts'][c]['total']}, rate={data['counts'][c]['rate']*100:.1f}%")
    
    # Compute statistical tests
    print("\n2. Computing statistical tests...")
    stats_results = compute_statistical_tests(data)
    
    print(f"   TP vs FP: gap={stats_results['tp_vs_fp']['gap_pp']:.1f}pp, p={stats_results['tp_vs_fp']['fisher_p']:.3f}, h={stats_results['tp_vs_fp']['cohens_h']:.2f} ({stats_results['tp_vs_fp']['cohens_h_interp']})")
    print(f"   TP vs FN: gap={stats_results['tp_vs_fn']['gap_pp']:.1f}pp, p={stats_results['tp_vs_fn']['fisher_p']:.3f}, h={stats_results['tp_vs_fn']['cohens_h']:.2f} ({stats_results['tp_vs_fn']['cohens_h_interp']})")
    print(f"   FP vs FN: gap={stats_results['fp_vs_fn']['gap_pp']:.1f}pp, p={stats_results['fp_vs_fn']['fisher_p']:.3f}, h={stats_results['fp_vs_fn']['cohens_h']:.2f} ({stats_results['fp_vs_fn']['cohens_h_interp']})")
    
    # Per-CLD stats
    print("\n3. Computing per-CLD statistics...")
    per_cld_stats = compute_per_cld_stats(data)
    for cld_name, cld in per_cld_stats.items():
        tp_rate = cld.get("TP", {}).get("rate", 0) * 100
        fp_rate = cld.get("FP", {}).get("rate", 0) * 100
        print(f"   {cld_name}: TP={tp_rate:.0f}%, FP={fp_rate:.0f}%, gap={tp_rate-fp_rate:+.0f}pp")
    
    # Generate LaTeX tables
    print("\n4. Generating LaTeX tables...")
    
    main_table = generate_main_table(data, stats_results)
    main_table_path = OUTPUT_DIR / "rq3_main_table.tex"
    with open(main_table_path, 'w') as f:
        f.write(main_table)
    print(f"   Saved: {main_table_path}")
    
    per_cld_table = generate_per_cld_table(per_cld_stats)
    per_cld_table_path = OUTPUT_DIR / "rq3_per_cld_table.tex"
    with open(per_cld_table_path, 'w') as f:
        f.write(per_cld_table)
    print(f"   Saved: {per_cld_table_path}")

    # Table 15: Enriched GT scenarios (reproducible)
    enrichment = compute_enriched_gt_metrics(data)
    enriched_table = generate_enriched_gt_table(enrichment)
    enriched_table_path = OUTPUT_DIR / "rq3_enriched_gt_table.tex"
    with open(enriched_table_path, "w") as f:
        f.write(enriched_table)
    print(f"   Saved: {enriched_table_path}")
    
    # Save comprehensive stats JSON
    print("\n5. Saving comprehensive statistics...")
    comprehensive_stats = {
        "timestamp": datetime.now().isoformat(),
        "data_summary": {
            "total_edges": data["total_edges"],
            "counts": data["counts"]
        },
        "statistical_tests": stats_results,
        "per_cld_stats": per_cld_stats
    }
    stats_path = OUTPUT_DIR / "rq3_comprehensive_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(comprehensive_stats, f, indent=2, default=str)
    print(f"   Saved: {stats_path}")
    
    # Generate figures
    print("\n6. Generating figures...")
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)
    generate_all_figures(data, per_cld_stats, figures_dir)
    
    print("\n" + "=" * 80)
    print("RQ3 ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  Tables:")
    print(f"    - {main_table_path}")
    print(f"    - {per_cld_table_path}")
    print(f"    - {enriched_table_path}")
    print(f"  Figures:")
    print(f"    - {figures_dir / 'rq3_combined_figure.png'}")
    print(f"  Stats:")
    print(f"    - {stats_path}")
    print(f"\nTo include in thesis, use:")
    print(f"  \\input{{../../final_runs/RQ3_deep_research/rq3_main_table.tex}}")
    print(f"  \\input{{../../final_runs/RQ3_deep_research/rq3_per_cld_table.tex}}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
RQ3 Per-CLD Detection Rates Analysis.

This script generates a detailed breakdown of Deep Research detection rates
by CLD (Depressive, Social Norms, Older Persons) and classification (TP, FP, FN).

Generated: 2025-12-19
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime
from collections import defaultdict
import os

# Default INPUT_DIR to the RQ3_deep_research folder (parent of analysis_scripts)
INPUT_DIR = Path(os.environ.get("RQ3_INPUT_DIR", str(Path(__file__).parent.parent))).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("RQ3_OUTPUT_DIR", str(INPUT_DIR))).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data files (kept in final_runs/RQ3_deep_research_validation/data)
DATA_FILES = [
    (INPUT_DIR / "Data/deep_research_results_depressive_symptoms_20251013_032002_84edges.json", "Depressive"),
    (INPUT_DIR / "Data/deep_research_results_social_norms_20251012_213641_17edges.json", "Social Norms"),
    (INPUT_DIR / "Data/deep_research_results_older_persons_ALL_EDGES_184edges.json", "Older Persons"),
]


def load_results(json_file: str) -> list:
    """Load results from a JSON file, handling both formats."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    elif isinstance(data, list):
        return data
    else:
        return data.get('results', [])


def compute_cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h effect size for proportion difference."""
    p1 = max(0, min(1, p1))
    p2 = max(0, min(1, p2))
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2


def main():
    """Generate per-CLD detection rates analysis."""
    
    print("=" * 90)
    print("RQ3 PER-CLD DETECTION RATES ANALYSIS")
    print("=" * 90)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load all results, grouped by CLD
    cld_data = {}
    all_results = []
    
    for filename, cld_name in DATA_FILES:
        filepath = Path(filename)
        if filepath.exists():
            results = load_results(str(filepath))
            for r in results:
                r['CLD'] = cld_name
            cld_data[cld_name] = results
            all_results.extend(results)
            print(f"Loaded {len(results)} edges from {cld_name}")
        else:
            print(f"WARNING: File not found: {filepath}")
    
    print(f"\nTotal edges: {len(all_results)}")
    
    # ========================================================================
    # PER-CLD BREAKDOWN
    # ========================================================================
    print("\n" + "=" * 90)
    print("DETECTION RATES BY CLD AND CLASSIFICATION")
    print("=" * 90)
    
    # Structure: cld_stats[cld][classification] = {total, found, rate}
    cld_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'found': 0}))
    
    for r in all_results:
        cld = r['CLD']
        cls = r['classification']
        cld_stats[cld][cls]['total'] += 1
        if r.get('direct_causal_found', False):
            cld_stats[cld][cls]['found'] += 1
    
    # Calculate rates
    for cld in cld_stats:
        for cls in cld_stats[cld]:
            s = cld_stats[cld][cls]
            s['rate'] = s['found'] / s['total'] if s['total'] > 0 else 0
            s['rate_pct'] = s['rate'] * 100
    
    # Print table
    print(f"\n{'CLD':<15} {'Class':<6} {'Total':>6} {'Found':>6} {'Rate':>8} {'TP-FP Gap':>10}")
    print("-" * 60)
    
    cld_order = ['Depressive', 'Social Norms', 'Older Persons']
    
    for cld in cld_order:
        if cld in cld_stats:
            for cls in ['TP', 'FP', 'FN']:
                if cls in cld_stats[cld]:
                    s = cld_stats[cld][cls]
                    # Calculate TP-FP gap for this CLD
                    if cls == 'TP' and 'FP' in cld_stats[cld]:
                        gap = s['rate_pct'] - cld_stats[cld]['FP']['rate_pct']
                        gap_str = f"{gap:+.1f} pp"
                    else:
                        gap_str = ""
                    print(f"{cld:<15} {cls:<6} {s['total']:>6} {s['found']:>6} {s['rate_pct']:>7.1f}% {gap_str:>10}")
            print()
    
    # ========================================================================
    # AGGREGATE COMPARISON
    # ========================================================================
    print("\n" + "=" * 90)
    print("AGGREGATE STATISTICS")
    print("=" * 90)
    
    agg = defaultdict(lambda: {'total': 0, 'found': 0})
    for r in all_results:
        cls = r['classification']
        agg[cls]['total'] += 1
        if r.get('direct_causal_found', False):
            agg[cls]['found'] += 1
    
    for cls in agg:
        agg[cls]['rate'] = agg[cls]['found'] / agg[cls]['total']
        agg[cls]['rate_pct'] = agg[cls]['rate'] * 100
    
    print(f"\n{'Class':<6} {'Total':>6} {'Found':>6} {'Rate':>8}")
    print("-" * 30)
    for cls in ['TP', 'FP', 'FN']:
        if cls in agg:
            s = agg[cls]
            print(f"{cls:<6} {s['total']:>6} {s['found']:>6} {s['rate_pct']:>7.1f}%")
    
    # Overall TP-FP gap
    tp_rate = agg['TP']['rate_pct']
    fp_rate = agg['FP']['rate_pct']
    fn_rate = agg['FN']['rate_pct']
    
    print(f"\nAggregate TP-FP Gap: {tp_rate - fp_rate:.1f} pp")
    print(f"Aggregate TP-FN Gap: {tp_rate - fn_rate:.1f} pp")
    print(f"Aggregate FP-FN Gap: {fp_rate - fn_rate:.1f} pp")
    
    # ========================================================================
    # DOMAIN VARIATION ANALYSIS
    # ========================================================================
    print("\n" + "=" * 90)
    print("DOMAIN VARIATION ANALYSIS")
    print("=" * 90)
    
    print("\nFP Detection Rate Variation:")
    fp_rates = []
    for cld in cld_order:
        if cld in cld_stats and 'FP' in cld_stats[cld]:
            rate = cld_stats[cld]['FP']['rate_pct']
            fp_rates.append(rate)
            print(f"  {cld}: {rate:.1f}%")
    
    if len(fp_rates) >= 2:
        print(f"\n  Range: {min(fp_rates):.1f}% - {max(fp_rates):.1f}% ({max(fp_rates) - min(fp_rates):.1f} pp spread)")
        print(f"  Std Dev: {np.std(fp_rates):.1f} pp")
    
    print("\nTP-FP Gap by CLD:")
    gaps = []
    for cld in cld_order:
        if cld in cld_stats and 'TP' in cld_stats[cld] and 'FP' in cld_stats[cld]:
            gap = cld_stats[cld]['TP']['rate_pct'] - cld_stats[cld]['FP']['rate_pct']
            gaps.append((cld, gap))
            direction = "TP > FP" if gap > 0 else "FP > TP"
            print(f"  {cld}: {gap:+.1f} pp ({direction})")
    
    # ========================================================================
    # LATEX TABLE OUTPUT
    # ========================================================================
    print("\n" + "=" * 90)
    print("LATEX TABLE FOR THESIS")
    print("=" * 90)
    
    print(r"""
\begin{table}[H]
\centering
\caption{Deep Research Detection Rates by CLD and Classification}
\label{tab:rq3_per_cld}
\begin{threeparttable}
\begin{tabular}{llcccc}
\toprule
\textbf{CLD} & \textbf{Class} & \textbf{n} & \textbf{Found} & \textbf{Detection Rate} & \textbf{TP--FP Gap} \\
\midrule""")
    
    for cld in cld_order:
        if cld in cld_stats:
            for i, cls in enumerate(['TP', 'FP', 'FN']):
                if cls in cld_stats[cld]:
                    s = cld_stats[cld][cls]
                    cld_display = cld if i == 0 else ""
                    
                    # TP-FP gap only on TP row
                    if cls == 'TP' and 'FP' in cld_stats[cld]:
                        gap = s['rate_pct'] - cld_stats[cld]['FP']['rate_pct']
                        gap_str = f"{gap:+.1f} pp"
                    else:
                        gap_str = "---"
                    
                    print(f"{cld_display} & {cls} & {s['total']} & {s['found']} & {s['rate_pct']:.1f}\\% & {gap_str} \\\\")
            print(r"\midrule")
    
    # Aggregate row
    print(r"\textit{Aggregate} & TP & " + f"{agg['TP']['total']} & {agg['TP']['found']} & {agg['TP']['rate_pct']:.1f}\\% & {tp_rate - fp_rate:+.1f} pp \\\\")
    print(r" & FP & " + f"{agg['FP']['total']} & {agg['FP']['found']} & {agg['FP']['rate_pct']:.1f}\\% & --- \\\\")
    print(r" & FN & " + f"{agg['FN']['total']} & {agg['FN']['found']} & {agg['FN']['rate_pct']:.1f}\\% & --- \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} Detection Rate = percentage of edges where Deep Research found direct causal evidence.
\item TP = True Positives (expert-accepted), FP = False Positives (LLM hallucinations), FN = False Negatives (expert-included but LLM-missed).
\item \textbf{Key finding:} FP detection rate varies substantially by CLD (28\%--76\%), suggesting domain-specific patterns in hallucination plausibility.
\item Depressive shows lowest FP detection (28\%), Older Persons highest (76\%).
\end{tablenotes}
\end{threeparttable}
\end{table}
""")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_edges": len(all_results),
        "per_cld": {
            cld: {cls: dict(cld_stats[cld][cls]) for cls in cld_stats[cld]}
            for cld in cld_stats
        },
        "aggregate": {cls: dict(agg[cls]) for cls in agg},
        "domain_variation": {
            "fp_rate_min": min(fp_rates) if fp_rates else None,
            "fp_rate_max": max(fp_rates) if fp_rates else None,
            "fp_rate_range_pp": max(fp_rates) - min(fp_rates) if len(fp_rates) >= 2 else None,
            "fp_rate_std": float(np.std(fp_rates)) if len(fp_rates) >= 2 else None,
            "tp_fp_gaps": {cld: gap for cld, gap in gaps}
        }
    }
    
    output_file = OUTPUT_DIR / f"rq3_per_cld_detection_rates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    latest_file = OUTPUT_DIR / "rq3_per_cld_detection_rates_latest.json"
    with open(latest_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Latest results: {latest_file}")


if __name__ == "__main__":
    main()







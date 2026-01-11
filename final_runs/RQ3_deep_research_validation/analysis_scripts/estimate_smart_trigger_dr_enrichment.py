#!/usr/bin/env python3
"""
Estimate expected F1 improvement if we use a cheap trigger (RQ1a GT Correctness judge score)
to decide which edges get expensive Deep Research validation.

Approach:
1. Parse per-edge Mechanistic GT correctness Aggregate Score from the 9 judged XLSX files.
2. Join to the 285 DR edges using canonical (CLD, source, target) keys.
3. Compute triggered set (mean Aggregate Score ≤ 0.5).
4. Count N_triggered_and_dr_found per class (FP, FN).
5. Calibrate promotion/discovery rates from human validation at t=0.7 (Wilson 95% CI).
6. Compute expected F1 for Scenario 1 (Enriched GT) and Scenario 2 (LLM+DR system).
7. Output LaTeX macros and tables.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# ---------------------------------------------------------------------------
# Resolve inputs robustly (avoid hard-coding timestamped folders)
# ---------------------------------------------------------------------------
def _resolve_latest_rq1a_gt_lit_correctness_xlsx() -> Path:
    """
    Resolve the most recent RQ1a GT Lit correctness 'enhanced results' workbook.

    This repo often contains multiple timestamped analysis outputs; rather than hard-coding
    one, select the newest file by modification time.
    """
    root = BASE_DIR / "final_runs/RQ1a_gt_lit_correctness"
    candidates = list(root.glob("**/rq1a_ground_truth_enhanced_results.xlsx"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find rq1a_ground_truth_enhanced_results.xlsx under {root}. "
            "Run the RQ1a enhanced analysis first or set the path explicitly in the script."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


RQ1A_ENHANCED = _resolve_latest_rq1a_gt_lit_correctness_xlsx()
DR_FILES = [
    BASE_DIR / "final_runs/RQ3_deep_research_validation/Data/deep_research_results_depressive_symptoms_20251013_032002_84edges.json",
    BASE_DIR / "final_runs/RQ3_deep_research_validation/Data/deep_research_results_social_norms_20251012_213641_17edges.json",
    BASE_DIR / "final_runs/RQ3_deep_research_validation/Data/deep_research_results_older_persons_ALL_EDGES_184edges.json",
]
VALIDATION_FILE = BASE_DIR / "final_runs/RQ3_deep_research_validation/validation/deep_research_edge_validation_sample_20251226_055509_nitai_validated.xlsx"
OUTPUT_DIR = BASE_DIR / "final_runs/RQ3_deep_research_validation/validation"
LATEX_DIR = BASE_DIR / "thesis/final_thesis/generated"

# Threshold for trigger
TRIGGER_THRESHOLD = 0.5

# Threshold for applicability (from previous analysis)
APPLICABILITY_THRESHOLD = 0.7

# CLD mapping: from file path keywords to DR CLD names
CLD_MAP = {
    "depressive": "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults",
    "social_norms": "Social_norms_and_obesity_prevalence",
    "emergency_department": "older_persons_emergency_department_visits_and_interactionstitled_spreadsheet",
}


def norm_str(s: str | None) -> str:
    """Normalize string for matching: lowercase, strip whitespace."""
    if s is None:
        return ""
    return str(s).lower().strip()


def detect_cld_from_path(file_path: str) -> str:
    """Detect CLD from file path."""
    fp_lower = file_path.lower()
    if "depressive" in fp_lower:
        return CLD_MAP["depressive"]
    elif "social_norms" in fp_lower:
        return CLD_MAP["social_norms"]
    elif "emergency" in fp_lower or "older_persons" in fp_lower:
        return CLD_MAP["emergency_department"]
    else:
        raise ValueError(f"Unknown CLD in path: {file_path}")


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score interval for proportion k/n. Returns (point, lower, upper)."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return p, max(0, centre - margin), min(1, centre + margin)


# ---------------------------------------------------------------------------
# Step 1: Parse per-edge Mechanistic Aggregate Scores
# ---------------------------------------------------------------------------
def parse_mechanistic_scores() -> pd.DataFrame:
    """Parse per-edge Aggregate Scores from the 9 mechanistic judged XLSX files."""
    df_raw = pd.read_excel(RQ1A_ENHANCED, sheet_name="Raw Data")
    mech = df_raw[df_raw["prompt"].str.lower() == "mechanistic"]
    
    all_edges = []
    for _, row in mech.iterrows():
        file_path = row["file_path"]
        cld = detect_cld_from_path(file_path)
        
        # Read All Edges sheet
        df_edges = pd.read_excel(file_path, sheet_name="All Edges")
        
        for _, e in df_edges.iterrows():
            source = norm_str(e.get("Source"))
            target = norm_str(e.get("Target"))
            score = e.get("Aggregate Score")
            classification = e.get("Classification")
            
            if pd.isna(score) or source == "" or target == "":
                continue
            
            all_edges.append({
                "cld": cld,
                "source": source,
                "target": target,
                "classification": classification,
                "aggregate_score": float(score),
            })
    
    df = pd.DataFrame(all_edges)
    print(f"Parsed {len(df)} edge-run observations from 9 mechanistic judged files")
    
    # Aggregate across runs: mean score per edge (ignore classification from judge files, use DR classification)
    df_agg = df.groupby(["cld", "source", "target"]).agg(
        mean_score=("aggregate_score", "mean"),
        n_runs=("aggregate_score", "count"),
    ).reset_index()
    
    print(f"Aggregated to {len(df_agg)} unique edges (mean across runs)")
    return df_agg


# ---------------------------------------------------------------------------
# Step 2: Load DR edges
# ---------------------------------------------------------------------------
def load_dr_edges() -> pd.DataFrame:
    """Load all 285 DR edges."""
    all_edges = []
    for f in DR_FILES:
        with open(f) as fp:
            data = json.load(fp)
        cld = data.get("cld", "")
        for r in data.get("results", []):
            all_edges.append({
                "cld": cld,
                "source": norm_str(r.get("source")),
                "target": norm_str(r.get("target")),
                "classification": r.get("classification"),
                "direct_causal_found": r.get("direct_causal_found", False),
            })
    df = pd.DataFrame(all_edges)
    print(f"Loaded {len(df)} DR edges")
    return df


# ---------------------------------------------------------------------------
# Step 3: Join trigger scores to DR edges
# ---------------------------------------------------------------------------
def join_trigger_to_dr(df_scores: pd.DataFrame, df_dr: pd.DataFrame) -> pd.DataFrame:
    """Join per-edge trigger scores to DR edges."""
    # Merge on (cld, source, target)
    df_merged = df_dr.merge(
        df_scores[["cld", "source", "target", "mean_score"]],
        on=["cld", "source", "target"],
        how="left",
    )
    
    # Compute triggered indicator
    df_merged["triggered"] = df_merged["mean_score"] <= TRIGGER_THRESHOLD
    
    # Count joins
    n_joined = df_merged["mean_score"].notna().sum()
    n_triggered = df_merged["triggered"].sum()
    print(f"Joined {n_joined}/{len(df_merged)} DR edges to trigger scores")
    print(f"Triggered (score ≤ {TRIGGER_THRESHOLD}): {n_triggered} edges")
    
    return df_merged


# ---------------------------------------------------------------------------
# Step 4: Count triggered and DR-found per class
# ---------------------------------------------------------------------------
def count_triggered_dr_found(df: pd.DataFrame) -> dict:
    """Count N_total, N_triggered, N_triggered_and_dr_found per class."""
    counts = {}
    for cls in ["TP", "FP", "FN"]:
        subset = df[df["classification"] == cls]
        n_total = len(subset)
        n_triggered = subset["triggered"].sum()
        n_triggered_dr_found = ((subset["triggered"]) & (subset["direct_causal_found"])).sum()
        counts[cls] = {
            "n_total": int(n_total),
            "n_triggered": int(n_triggered),
            "n_triggered_dr_found": int(n_triggered_dr_found),
        }
        print(f"  {cls}: total={n_total}, triggered={n_triggered}, triggered+dr_found={n_triggered_dr_found}")
    return counts


# ---------------------------------------------------------------------------
# Step 5: Calibrate from human validation
# ---------------------------------------------------------------------------
def calibrate_from_validation() -> dict:
    """Compute p_FP and p_FN from human validation at applicability >= 0.7."""
    df = pd.read_excel(VALIDATION_FILE)
    
    # Clean human_verdict: consider 'causal' or ' causal' as causal
    df["is_causal"] = df["human_verdict"].str.strip().str.lower().isin(["causal"])
    
    # Filter to Fully applicable >= threshold
    df_high = df[df["Fully applicable"] >= APPLICABILITY_THRESHOLD]
    
    rates = {}
    for cls in ["FP", "FN"]:
        subset = df_high[df_high["classification"] == cls]
        n = len(subset)
        k = subset["is_causal"].sum()
        p, p_low, p_high = wilson_ci(k, n)
        rates[cls] = {
            "n": int(n),
            "k": int(k),
            "p": float(p),
            "p_low": float(p_low),
            "p_high": float(p_high),
        }
        print(f"  {cls}: {k}/{n} = {p:.1%} causal (95% CI: [{p_low:.1%}, {p_high:.1%}])")
    
    return rates


# ---------------------------------------------------------------------------
# Step 6: Compute expected F1
# ---------------------------------------------------------------------------
def compute_prf(tp: float, fp: float, fn: float) -> dict:
    """Compute Precision, Recall, F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def estimate_enriched_f1(counts: dict, rates: dict) -> dict:
    """Estimate F1 for original, Scenario 1, and Scenario 2."""
    # Original counts (from DR dataset)
    orig_tp = 44
    orig_fp = 178
    orig_fn = 63
    
    # Expected promotions/discoveries
    n_fp_trig_dr = counts["FP"]["n_triggered_dr_found"]
    n_fn_trig_dr = counts["FN"]["n_triggered_dr_found"]
    
    p_fp, p_fp_low, p_fp_high = rates["FP"]["p"], rates["FP"]["p_low"], rates["FP"]["p_high"]
    p_fn, p_fn_low, p_fn_high = rates["FN"]["p"], rates["FN"]["p_low"], rates["FN"]["p_high"]
    
    # Point estimates
    fp_promote = n_fp_trig_dr * p_fp
    fn_discover = n_fn_trig_dr * p_fn
    
    # Conservative estimates (use lower bound of CI)
    fp_promote_cons = n_fp_trig_dr * p_fp_low
    fn_discover_cons = n_fn_trig_dr * p_fn_low
    
    # Optimistic estimates (use upper bound of CI)
    fp_promote_opt = n_fp_trig_dr * p_fp_high
    fn_discover_opt = n_fn_trig_dr * p_fn_high
    
    results = {
        "original": {
            "tp": orig_tp, "fp": orig_fp, "fn": orig_fn,
            **compute_prf(orig_tp, orig_fp, orig_fn),
        },
        "scenario1": {  # Enriched GT: FP promotions only
            "point": {
                "fp_promote": fp_promote,
                "tp": orig_tp + fp_promote,
                "fp": orig_fp - fp_promote,
                "fn": orig_fn,
                **compute_prf(orig_tp + fp_promote, orig_fp - fp_promote, orig_fn),
            },
            "lower": {
                "fp_promote": fp_promote_cons,
                **compute_prf(orig_tp + fp_promote_cons, orig_fp - fp_promote_cons, orig_fn),
            },
            "upper": {
                "fp_promote": fp_promote_opt,
                **compute_prf(orig_tp + fp_promote_opt, orig_fp - fp_promote_opt, orig_fn),
            },
        },
        "scenario2": {  # LLM+DR system: FP promotions + FN discoveries
            "point": {
                "fp_promote": fp_promote,
                "fn_discover": fn_discover,
                "tp": orig_tp + fp_promote + fn_discover,
                "fp": orig_fp - fp_promote,
                "fn": orig_fn - fn_discover,
                **compute_prf(orig_tp + fp_promote + fn_discover, orig_fp - fp_promote, orig_fn - fn_discover),
            },
            "lower": {
                "fp_promote": fp_promote_cons,
                "fn_discover": fn_discover_cons,
                **compute_prf(orig_tp + fp_promote_cons + fn_discover_cons, orig_fp - fp_promote_cons, orig_fn - fn_discover_cons),
            },
            "upper": {
                "fp_promote": fp_promote_opt,
                "fn_discover": fn_discover_opt,
                **compute_prf(orig_tp + fp_promote_opt + fn_discover_opt, orig_fp - fp_promote_opt, orig_fn - fn_discover_opt),
            },
        },
        "trigger_counts": {
            "n_fp_triggered_dr_found": n_fp_trig_dr,
            "n_fn_triggered_dr_found": n_fn_trig_dr,
        },
        "rates": rates,
    }
    
    return results


# ---------------------------------------------------------------------------
# Step 7: Output LaTeX
# ---------------------------------------------------------------------------
def write_latex_outputs(results: dict, output_dir: Path) -> None:
    """Write LaTeX macros and table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    orig = results["original"]
    s1 = results["scenario1"]
    s2 = results["scenario2"]
    trigger = results["trigger_counts"]
    rates = results["rates"]
    
    # Macros
    macros = f"""% Auto-generated by estimate_smart_trigger_dr_enrichment.py
% Smart-trigger DR enrichment estimates

% Trigger parameters
\\newcommand{{\\rqThreeTriggerThreshold}}{{{TRIGGER_THRESHOLD}}}
\\newcommand{{\\rqThreeApplicabilityThreshold}}{{{APPLICABILITY_THRESHOLD}}}

% Triggered counts
\\newcommand{{\\rqThreeFPTriggeredDR}}{{{trigger['n_fp_triggered_dr_found']}}}
\\newcommand{{\\rqThreeFNTriggeredDR}}{{{trigger['n_fn_triggered_dr_found']}}}

% Calibration rates
\\newcommand{{\\rqThreePFP}}{{{rates['FP']['p']*100:.1f}\\%}}
\\newcommand{{\\rqThreePFPCI}}{{[{rates['FP']['p_low']*100:.1f}\\%, {rates['FP']['p_high']*100:.1f}\\%]}}
\\newcommand{{\\rqThreePFN}}{{{rates['FN']['p']*100:.1f}\\%}}
\\newcommand{{\\rqThreePFNCI}}{{[{rates['FN']['p_low']*100:.1f}\\%, {rates['FN']['p_high']*100:.1f}\\%]}}

% Original metrics
\\newcommand{{\\rqThreeOrigF}}{{{orig['f1']:.3f}}}
\\newcommand{{\\rqThreeOrigP}}{{{orig['precision']:.3f}}}
\\newcommand{{\\rqThreeOrigR}}{{{orig['recall']:.3f}}}

% Scenario 1: Enriched GT (FP promotions only)
\\newcommand{{\\rqThreeSTFpPromote}}{{{s1['point']['fp_promote']:.1f}}}
\\newcommand{{\\rqThreeSTEnrichF}}{{{s1['point']['f1']:.3f}}}
\\newcommand{{\\rqThreeSTEnrichFCI}}{{[{s1['lower']['f1']:.3f}, {s1['upper']['f1']:.3f}]}}
\\newcommand{{\\rqThreeSTEnrichP}}{{{s1['point']['precision']:.3f}}}
\\newcommand{{\\rqThreeSTEnrichR}}{{{s1['point']['recall']:.3f}}}

% Scenario 2: LLM+DR system (FP + FN)
\\newcommand{{\\rqThreeSTFnDiscover}}{{{s2['point']['fn_discover']:.1f}}}
\\newcommand{{\\rqThreeSTSystemF}}{{{s2['point']['f1']:.3f}}}
\\newcommand{{\\rqThreeSTSystemFCI}}{{[{s2['lower']['f1']:.3f}, {s2['upper']['f1']:.3f}]}}
\\newcommand{{\\rqThreeSTSystemP}}{{{s2['point']['precision']:.3f}}}
\\newcommand{{\\rqThreeSTSystemR}}{{{s2['point']['recall']:.3f}}}

% F1 improvement
\\newcommand{{\\rqThreeSTDeltaF}}{{{s1['point']['f1'] - orig['f1']:.3f}}}
\\newcommand{{\\rqThreeSTDeltaFSys}}{{{s2['point']['f1'] - orig['f1']:.3f}}}
"""
    
    with open(output_dir / "rq3_smart_trigger_numbers.tex", "w") as f:
        f.write(macros)
    print(f"Wrote {output_dir / 'rq3_smart_trigger_numbers.tex'}")
    
    # Table
    table = f"""% Auto-generated by estimate_smart_trigger_dr_enrichment.py
\\begin{{table}}[H]
\\centering
\\caption{{Smart-trigger Deep Research enrichment estimates. Trigger: Mechanistic GT correctness score $\\leq {TRIGGER_THRESHOLD}$. Calibration: human-validated applicability $\\geq {APPLICABILITY_THRESHOLD}$.}}
\\label{{tab:rq3_smart_trigger}}
\\begin{{tabular}}{{lccc}}
\\toprule
Scenario & Precision & Recall & F1 (95\\% CI) \\\\
\\midrule
Original & {orig['precision']:.3f} & {orig['recall']:.3f} & {orig['f1']:.3f} \\\\
Enriched GT (+{s1['point']['fp_promote']:.1f} FP$\\to$TP) & {s1['point']['precision']:.3f} & {s1['point']['recall']:.3f} & {s1['point']['f1']:.3f} [{s1['lower']['f1']:.3f}, {s1['upper']['f1']:.3f}] \\\\
LLM+DR (+{s2['point']['fn_discover']:.1f} FN discovered) & {s2['point']['precision']:.3f} & {s2['point']['recall']:.3f} & {s2['point']['f1']:.3f} [{s2['lower']['f1']:.3f}, {s2['upper']['f1']:.3f}] \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open(output_dir / "rq3_smart_trigger_table.tex", "w") as f:
        f.write(table)
    print(f"Wrote {output_dir / 'rq3_smart_trigger_table.tex'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Smart-trigger DR enrichment estimate")
    print("=" * 60)
    
    # Step 1: Parse mechanistic scores
    print("\n[Step 1] Parsing mechanistic judge scores...")
    df_scores = parse_mechanistic_scores()
    
    # Step 2: Load DR edges
    print("\n[Step 2] Loading DR edges...")
    df_dr = load_dr_edges()
    
    # Step 3: Join trigger scores to DR edges
    print("\n[Step 3] Joining trigger scores to DR edges...")
    df_merged = join_trigger_to_dr(df_scores, df_dr)
    
    # Step 4: Count triggered and DR-found per class
    print("\n[Step 4] Counting triggered + DR-found per class...")
    counts = count_triggered_dr_found(df_merged)
    
    # Step 5: Calibrate from human validation
    print(f"\n[Step 5] Calibrating from human validation (applicability >= {APPLICABILITY_THRESHOLD})...")
    rates = calibrate_from_validation()
    
    # Step 6: Estimate enriched F1
    print("\n[Step 6] Estimating enriched F1...")
    results = estimate_enriched_f1(counts, rates)
    
    # Print summary
    orig = results["original"]
    s1 = results["scenario1"]
    s2 = results["scenario2"]
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nOriginal: F1={orig['f1']:.3f} (P={orig['precision']:.3f}, R={orig['recall']:.3f})")
    print(f"\nScenario 1 (Enriched GT):")
    print(f"  Expected FP promotions: {s1['point']['fp_promote']:.1f}")
    print(f"  F1={s1['point']['f1']:.3f} (95% CI: [{s1['lower']['f1']:.3f}, {s1['upper']['f1']:.3f}])")
    print(f"  ΔF1 = {s1['point']['f1'] - orig['f1']:+.3f}")
    print(f"\nScenario 2 (LLM+DR system):")
    print(f"  Expected FN discoveries: {s2['point']['fn_discover']:.1f}")
    print(f"  F1={s2['point']['f1']:.3f} (95% CI: [{s2['lower']['f1']:.3f}, {s2['upper']['f1']:.3f}])")
    print(f"  ΔF1 = {s2['point']['f1'] - orig['f1']:+.3f}")
    
    # Step 7: Output LaTeX
    print("\n[Step 7] Writing LaTeX outputs...")
    write_latex_outputs(results, LATEX_DIR)
    
    # Save JSON summary
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "smart_trigger_enrichment_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {json_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()


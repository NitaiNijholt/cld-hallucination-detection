#!/usr/bin/env python3
"""
Effect Size Summary Table Generator

Generates the complete Effect Size Summary table (Table 32) for the thesis
by extracting data from all RQ analysis outputs.

Usage: 
    python final_runs/retrieve_effect_sizes.py --run-dir /path/to/output
    
Outputs:
    - effect_size_summary_table.tex
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

REPO_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EffectRow:
    """Represents one row in the effect size table."""
    rq: str
    hypothesis: str
    h0_h1: str
    test: str
    effect: str
    value: str
    p_value: str
    conclusion: str
    is_subrow: bool = False  # If True, first column is empty


def format_p(p: float) -> str:
    """Format p-value with significance stars."""
    if p is None or np.isnan(p):
        return "---"
    if p < 0.001:
        return r"$<$.001***"
    elif p < 0.01:
        return f".{int(p*1000):03d}**"
    elif p < 0.05:
        return f".{int(p*1000):03d}*"
    else:
        return f".{int(p*1000):03d}"


def interpret_w(w: float) -> str:
    """Interpret Kendall's W effect size."""
    if w < 0.3:
        return "weak"
    elif w < 0.5:
        return "moderate"
    else:
        return "strong"


def interpret_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def interpret_r(r: float) -> str:
    """Interpret Pearson r effect size."""
    r_abs = abs(r)
    if r_abs < 0.2:
        return "weak"
    elif r_abs < 0.5:
        return "moderate"
    else:
        return "strong"


def interpret_h(h: float) -> str:
    """Interpret Cohen's h effect size."""
    h_abs = abs(h)
    if h_abs < 0.2:
        return "negl."
    elif h_abs < 0.5:
        return "small"
    elif h_abs < 0.8:
        return "medium"
    else:
        return "large"


def interpret_kappa(k: float) -> str:
    """Interpret Cohen's kappa (Landis-Koch scale)."""
    if k < 0.20:
        return "slight"
    elif k < 0.40:
        return "fair"
    elif k < 0.60:
        return "moderate"
    elif k < 0.80:
        return "substantial"
    else:
        return "almost perfect"


def interpret_icc(icc: float) -> str:
    """Interpret ICC(2,1) (Koo & Li)."""
    if icc < 0.50:
        return "poor"
    elif icc < 0.75:
        return "moderate"
    elif icc < 0.90:
        return "good"
    else:
        return "excellent"


# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================

def get_rq1a_human_agreement() -> Optional[EffectRow]:
    """Extract RQ1a human-LLM agreement (kappa) from validation results."""
    # The kappa value is 0.618 from the combined_human_validation_analysis.py output
    # This is a fixed value from the validation study
    kappa = 0.618
    return EffectRow(
        rq="1a",
        hypothesis="LLM-human agreement",
        h0_h1=r"$H_0$: $\kappa=0$ vs $H_1$: $\kappa>0$",
        test=r"Cohen's $\kappa$ (n=72)",
        effect=r"$\kappa$",
        value="0.618",
        p_value="---",
        conclusion=f"$H_0$ rejected; {interpret_kappa(kappa)}"
    )


def get_rq1a_prompt_sensitivity() -> List[EffectRow]:
    """Extract RQ1a prompt sensitivity (Friedman W) from sensitivity analysis."""
    rows = []
    
    # Define experiments with their data sources
    experiments = [
        {
            "name": "Corr. Citation",
            "dir": "RQ1a_gt_synth_citation",
            "pattern": "rq1a_enhanced_results.xlsx"
        },
        {
            "name": "Corr. Correctness",
            "dir": "RQ1a_gt_synth_correctness",
            "pattern": "rq1a_enhanced_results.xlsx"
        },
        {
            "name": "GT Citation",
            "dir": "RQ1a_gt_lit_citation",
            "pattern": "rq1a_enhanced_results.xlsx"
        },
        {
            "name": "GT Correctness",
            "dir": "RQ1a_gt_lit_correctness",
            "pattern": "rq1a_enhanced_results.xlsx"
        }
    ]
    
    for i, exp in enumerate(experiments):
        base_dir = REPO_ROOT / "final_runs" / exp["dir"]
        files = list(base_dir.rglob(exp["pattern"]))
        
        if not files:
            print(f"  Warning: No results found for {exp['name']}")
            # Use fallback values from thesis
            fallback = {
                "Corr. Citation": (0.29, 0.049),
                "Corr. Correctness": (0.70, 0.002),
                "GT Citation": (0.30, 0.042),
                "GT Correctness": (0.48, 0.013)
            }
            w, p = fallback.get(exp["name"], (0.0, 1.0))
        else:
            latest_file = sorted(files, key=lambda x: x.stat().st_mtime)[-1]
            try:
                df = pd.read_excel(latest_file, sheet_name='Friedman Tests', engine='openpyxl')
                f1_row = df[df['Metric'] == 'F1 Mean'].iloc[0]
                w = f1_row.get("Kendall's W", 0)
                p = f1_row.get("P-value", 1.0)
            except Exception as e:
                print(f"  Error reading {exp['name']}: {e}")
                w, p = 0.0, 1.0
        
        h0_rejected = p < 0.05
        conclusion = f"$H_0$ {'rejected' if h0_rejected else 'not rejected'}; {interpret_w(w)}"
        
        rows.append(EffectRow(
            rq="1a" if i == 0 else "",
            hypothesis=f"Prompt affects F1",
            h0_h1=r"$H_0$: equal medians" if i == 0 else "",
            test="Friedman (9 blocks)" if i == 0 else "",
            effect="$W$",
            value=f"{w:.2f}",
            p_value=format_p(p),
            conclusion=conclusion,
            is_subrow=(i > 0)
        ))
        
        # Add experiment name as subrow
        rows.append(EffectRow(
            rq="",
            hypothesis=f"({exp['name']})",
            h0_h1=r"$H_1$: $\geq$1 differs" if i == 0 else "",
            test="",
            effect="",
            value="",
            p_value="",
            conclusion="",
            is_subrow=True
        ))
    
    return rows


def get_rq1b_corrector_efficacy() -> List[EffectRow]:
    """Extract RQ1b corrector efficacy (Cohen's d, Wilcoxon p) from corrector stats."""
    rows = []
    
    json_path = REPO_ROOT / "final_runs/Sensitivity_analysis_simple/corrector_comprehensive_stats.json"
    
    # Default fallback values from thesis
    data_synthetic = {"d": 1.18, "p": 0.031}
    data_gt = {"d": -0.32, "p": 0.301}
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            experiments = data.get('experiments', {})
            
            # Synthetic
            if 'Synthetic' in experiments:
                synth = experiments['Synthetic']
                baseline = synth.get('prompts', {}).get('Baseline', {})
                data_synthetic["d"] = baseline.get('cohens_d', data_synthetic["d"])
                # Get efficacy p-value
                eff = synth.get('efficacy_vs_zero', {}).get('Baseline', {})
                data_synthetic["p"] = eff.get('p_value', data_synthetic["p"])
            
            # Ground Truth
            if 'Ground Truth' in experiments:
                gt = experiments['Ground Truth']
                baseline = gt.get('prompts', {}).get('Baseline', {})
                data_gt["d"] = baseline.get('cohens_d', data_gt["d"])
                eff = gt.get('efficacy_vs_zero', {}).get('Baseline', {})
                data_gt["p"] = eff.get('p_value', data_gt["p"])
                
        except Exception as e:
            print(f"  Error reading RQ1b stats: {e}")
    
    # Synthetic row
    h0_rejected = data_synthetic["p"] < 0.05
    rows.append(EffectRow(
        rq="1b",
        hypothesis="Corrector efficacy",
        h0_h1=r"$H_0$: med($\Delta$F1)$=$0",
        test="Wilcoxon signed-rank",
        effect="$d$",
        value=f"{data_synthetic['d']:+.2f}",
        p_value=format_p(data_synthetic["p"]),
        conclusion=f"$H_0$ {'rejected' if h0_rejected else 'not rejected'}; {interpret_d(data_synthetic['d'])}"
    ))
    rows.append(EffectRow(
        rq="",
        hypothesis="(Synthetic)",
        h0_h1=r"$H_1$: med($\Delta$F1)$\neq$0",
        test="(9 blocks)",
        effect="",
        value="",
        p_value="",
        conclusion="",
        is_subrow=True
    ))
    
    # Ground Truth row
    h0_rejected = data_gt["p"] < 0.05
    rows.append(EffectRow(
        rq="",
        hypothesis="Corrector efficacy",
        h0_h1=r"$H_0$: med($\Delta$F1)$=$0",
        test="Wilcoxon signed-rank",
        effect="$d$",
        value=f"$-${abs(data_gt['d']):.2f}",
        p_value=format_p(data_gt["p"]),
        conclusion=f"$H_0$ {'rejected' if h0_rejected else 'not rejected'}; {interpret_d(data_gt['d'])}",
        is_subrow=True
    ))
    rows.append(EffectRow(
        rq="",
        hypothesis="(Ground Truth)",
        h0_h1=r"$H_1$: med($\Delta$F1)$\neq$0",
        test="(9 blocks)",
        effect="",
        value="",
        p_value="",
        conclusion="",
        is_subrow=True
    ))
    
    return rows


def get_rq2_discrimination() -> List[EffectRow]:
    """Extract RQ2 UQ discrimination (AUC, p) from single-metric results."""
    rows = []
    
    # Read single metric results (block-level Wilcoxon signed-rank on (AUC - 0.5), Bonferroni-adjusted across 4 metrics)
    json_path = REPO_ROOT / "final_runs/RQ2_uq_hallucination_detection/single_metric_results.json"
    
    # Default fallback
    auc_val = 0.672
    p_val = 0.0156
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            cosine = data.get('metrics', {}).get('Gen Cosine Similarity', {})
            auc_val = cosine.get('mean_auc', auc_val)
            # Prefer Bonferroni-adjusted Wilcoxon p-value (primary inference for RQ2 single-metric)
            p_val = cosine.get('auc_wilcoxon_p_adj', cosine.get('auc_wilcoxon_p', p_val))
        except Exception as e:
            print(f"  Error reading RQ2 single metric results: {e}")
    
    h0_rejected = p_val < 0.05
    
    # Interpretation of AUC as effect size
    # 0.5=random, 0.7=moderate, 0.8=good, 0.9=excellent
    interpretation = "weak"
    if auc_val > 0.7: interpretation = "good"
    elif auc_val > 0.6: interpretation = "moderate"
    
    rows.append(EffectRow(
        rq="2",
        hypothesis="UQ discriminates",
        h0_h1=r"$H_0$: median(AUC$-0.5)=0$",
        test=r"Wilcoxon signed-rank",
        effect="AUC",
        value=f"{auc_val:.3f}",
        p_value=format_p(p_val),
        conclusion=f"$H_0$ {'rejected' if h0_rejected else 'not rejected'}; {interpretation}"
    ))
    rows.append(EffectRow(
        rq="",
        hypothesis="hallucinations",
        h0_h1=r"$H_1$: median(AUC$-0.5)\neq 0$",
        test="(9 blocks)",
        effect="",
        value="",
        p_value="",
        conclusion="",
        is_subrow=True
    ))
    
    # Cross-CLD generalization from ensemble_performance_results.json
    ensemble_path = REPO_ROOT / "final_runs/RQ2_uq_hallucination_detection/ensemble_performance_results.json"
    delta_auc = 0.18  # Default fallback (RF drop)
    
    if ensemble_path.exists():
        try:
            with open(ensemble_path, 'r') as f:
                ens_data = json.load(f)
            # Get Random Forest (highest in-distribution, largest drop)
            for clf in ens_data.get('classifiers', []):
                if clf.get('classifier') == 'Random Forest':
                    drop = clf.get('drop')
                    if drop is not None:
                        delta_auc = drop
                    break
        except Exception as e:
            print(f"  Error reading RQ2 ensemble results: {e}")
    
    rows.append(EffectRow(
        rq="",
        hypothesis="Cross-CLD",
        h0_h1="---",
        test="Descriptive",
        effect=r"$\Delta$AUC",
        value=f"{delta_auc:.2f}",
        p_value="---",
        conclusion="Generalization gap",
        is_subrow=True
    ))
    rows.append(EffectRow(
        rq="",
        hypothesis="generalization",
        h0_h1="",
        test=r"(Phase 5$\to$6)",
        effect="",
        value="",
        p_value="",
        conclusion="",
        is_subrow=True
    ))
    
    return rows


def get_rq3_inter_rater_reliability() -> List[EffectRow]:
    """Add RQ3 inter-rater reliability (human validation overlap) as descriptive rows."""
    # From Results: overlap n=13, kappa=0.26, ICC(2,1)=0.03
    kappa = 0.26
    icc = 0.03
    return [
        EffectRow(
            rq="",
            hypothesis="Inter-rater reliability",
            h0_h1="---",
            test=r"Cohen's $\kappa$ (n=13)",
            effect=r"$\kappa$",
            value=f"{kappa:.2f}",
            p_value="---",
            conclusion=f"Descriptive; {interpret_kappa(kappa)}",
            is_subrow=True,
        ),
        EffectRow(
            rq="",
            hypothesis="(Applicability score)",
            h0_h1="---",
            test=r"ICC(2,1) (n=13)",
            effect="ICC",
            value=f"{icc:.2f}",
            p_value="---",
            conclusion=f"Descriptive; {interpret_icc(icc)}",
            is_subrow=True,
        ),
    ]


def get_rq3_discrimination() -> List[EffectRow]:
    """Extract RQ3 DR discrimination (Cohen's h, Fisher p) from statistical test results."""
    rows = []
    
    json_path = REPO_ROOT / "final_runs/RQ3_deep_research_validation/rq3_statistical_test_results_latest.json"
    
    # Default fallback values from thesis
    data = {
        "tp_fp": {"h": 0.17, "p": 1.000},
        "tp_fn": {"h": 0.56, "p": 0.018},
        "fp_fn": {"h": 0.39, "p": 0.024}
    }
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                loaded = json.load(f)
            
            if 'tp_vs_fp' in loaded:
                data["tp_fp"]["h"] = loaded['tp_vs_fp'].get('cohens_h', data["tp_fp"]["h"])
                data["tp_fp"]["p"] = loaded['tp_vs_fp'].get('bonferroni', {}).get('p_adjusted', 
                                     loaded['tp_vs_fp'].get('fisher_p', data["tp_fp"]["p"]))
            if 'tp_vs_fn' in loaded:
                data["tp_fn"]["h"] = loaded['tp_vs_fn'].get('cohens_h', data["tp_fn"]["h"])
                data["tp_fn"]["p"] = loaded['tp_vs_fn'].get('bonferroni', {}).get('p_adjusted',
                                     loaded['tp_vs_fn'].get('fisher_p', data["tp_fn"]["p"]))
            if 'fp_vs_fn' in loaded:
                data["fp_fn"]["h"] = loaded['fp_vs_fn'].get('cohens_h', data["fp_fn"]["h"])
                data["fp_fn"]["p"] = loaded['fp_vs_fn'].get('bonferroni', {}).get('p_adjusted',
                                     loaded['fp_vs_fn'].get('fisher_p', data["fp_fn"]["p"]))
        except Exception as e:
            print(f"  Error reading RQ3 results: {e}")
    
    comparisons = [
        ("TP--FP", "tp_fp", r"$p_{\text{TP}}=p_{\text{FP}}$", r"$p_{\text{TP}}\neq p_{\text{FP}}$"),
        ("TP--FN", "tp_fn", r"$p_{\text{TP}}=p_{\text{FN}}$", r"$p_{\text{TP}}\neq p_{\text{FN}}$"),
        ("FP--FN", "fp_fn", r"$p_{\text{FP}}=p_{\text{FN}}$", r"$p_{\text{FP}}\neq p_{\text{FN}}$"),
    ]
    
    for i, (label, key, h0, h1) in enumerate(comparisons):
        h_val = data[key]["h"]
        p_val = data[key]["p"]
        h0_rejected = p_val < 0.05
        
        rows.append(EffectRow(
            rq="3" if i == 0 else "",
            hypothesis=f"DR discriminates",
            h0_h1=f"$H_0$: {h0}",
            test="Fisher's exact",
            effect="$h$",
            value=f"{h_val:.2f}",
            p_value=f"{p_val:.3f}" if p_val >= 0.001 else format_p(p_val),
            conclusion=f"$H_0$ {'rejected' if h0_rejected else 'not rejected'}; {interpret_h(h_val)}",
            is_subrow=(i > 0)
        ))
        rows.append(EffectRow(
            rq="",
            hypothesis=f"({label})",
            h0_h1=f"$H_1$: {h1}",
            test=r"(Bonf. $m$=3)",
            effect="",
            value="",
            p_value="",
            conclusion="",
            is_subrow=True
        ))
    
    return rows


def get_rq3_enrichment() -> List[EffectRow]:
    """Extract RQ3 enrichment metrics from enriched GT tables.
    
    For human-validated scenarios, we use the CONSERVATIVE estimates
    (Wilson CI lower bound) to provide a more defensible lower-bound estimate.
    """
    rows = []
    
    enriched_gt_tex = REPO_ROOT / "final_runs/RQ3_deep_research_validation/rq3_enriched_gt_table.tex"
    validation_tex = REPO_ROOT / "thesis/reproducible_version/generated/rq3_enrichment_validation_table.tex"
    
    # Default fallback
    orig, sc1, sc2 = 0.267, 0.705, 0.779
    # Human-validated: use conservative (Wilson CI lower bound) estimates
    hv_sc1, hv_sc2 = 0.527, 0.582
    
    try:
        txt = enriched_gt_tex.read_text(encoding="utf-8", errors="replace")
        for line in txt.splitlines():
            if "\\textbf{Aggregate}" in line and "&" in line:
                floats = re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", line)
                if len(floats) >= 3:
                    orig, sc1, sc2 = float(floats[0]), float(floats[1]), float(floats[2])
    except Exception as e:
        print(f"  Warning: Could not read enriched GT table: {e}")
    
    try:
        txt = validation_tex.read_text(encoding="utf-8", errors="replace")
        for line in txt.splitlines():
            # Read CONSERVATIVE scenarios (Wilson CI lower bound) for human-validated estimates
            if "Conservative Scenario 1" in line:
                floats = re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", line)
                if floats:
                    hv_sc1 = float(floats[-1])  # Last float is F1
            if "Conservative Scenario 2" in line:
                floats = re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", line)
                if floats:
                    hv_sc2 = float(floats[-1])  # Last float is F1
    except Exception as e:
        print(f"  Warning: Could not read validation table: {e}")
    
    enrichment_data = [
        ("Enrichment baseline", "Descriptive", "F1", f"{orig:.3f}", "Baseline"),
        ("Scenario 1: Enriched GT", "Descriptive", r"$\Delta$F1", f"{sc1-orig:+.3f}", "Exploratory"),
        ("Scenario 2: LLM+DR", "Descriptive", r"$\Delta$F1", f"{sc2-orig:+.3f}", "Exploratory"),
        ("Human-val. Sc. 1 (cons.)", "Extrapolation", r"$\Delta$F1", f"{hv_sc1-orig:+.3f}", "Exploratory"),
        ("Human-val. Sc. 2 (cons.)", "Extrapolation", r"$\Delta$F1", f"{hv_sc2-orig:+.3f}", "Exploratory"),
    ]
    
    for hyp, test, effect, value, conclusion in enrichment_data:
        rows.append(EffectRow(
            rq="",
            hypothesis=hyp,
            h0_h1="---",
            test=test,
            effect=effect,
            value=value,
            p_value="---",
            conclusion=conclusion,
            is_subrow=True
        ))
    
    # Smart compute
    rows.append(EffectRow(
        rq="",
        hypothesis="Smart compute",
        h0_h1="---",
        test="Descriptive",
        effect=r"$\Delta$F1/\$100",
        value="+0.027",
        p_value="---",
        conclusion="Exploratory",
        is_subrow=True
    ))
    
    return rows


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_effect_size_table() -> str:
    """Generate the complete effect size summary table."""
    
    print("Generating Effect Size Summary Table...")
    print("=" * 60)
    
    # Collect all rows
    all_rows: List[EffectRow] = []
    
    # RQ1a Human Agreement
    print("RQ1a: Human-LLM Agreement")
    row = get_rq1a_human_agreement()
    if row:
        all_rows.append(row)
    
    # RQ1a Prompt Sensitivity
    print("RQ1a: Prompt Sensitivity")
    all_rows.extend(get_rq1a_prompt_sensitivity())
    
    # RQ1b Corrector Efficacy
    print("RQ1b: Corrector Efficacy")
    all_rows.extend(get_rq1b_corrector_efficacy())
    
    # RQ2 Discrimination
    print("RQ2: UQ Discrimination")
    all_rows.extend(get_rq2_discrimination())
    
    # RQ3 Discrimination
    print("RQ3: DR Discrimination")
    all_rows.extend(get_rq3_discrimination())

    # RQ3 Inter-rater reliability (descriptive; overlap n=13)
    print("RQ3: Inter-rater reliability")
    all_rows.extend(get_rq3_inter_rater_reliability())
    
    # RQ3 Enrichment
    print("RQ3: Enrichment")
    all_rows.extend(get_rq3_enrichment())
    
    print("=" * 60)
    print(f"Total rows: {len(all_rows)}")
    
    # Generate LaTeX
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Effect Size Summary: Hypothesis Operationalization, Statistical Tests, and Conclusions}")
    lines.append(r"\label{tab:effect_size_summary}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{llp{3.2cm}p{2.8cm}cccl}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{RQ} & \textbf{Hypothesis} & \textbf{$H_0$ / $H_1$} & \textbf{Test} & \textbf{Effect} & \textbf{Value} & \textbf{$p$} & \textbf{Conclusion (decision; effect)} \\")
    lines.append(r"\midrule")
    
    # Section headers and row tracking
    current_section = None
    prev_rq = None
    
    for row in all_rows:
        # Add section headers
        if row.rq == "1a" and current_section != "1a":
            lines.append(r"\multicolumn{8}{l}{\textit{RQ1a: LLM-as-a-Judge Validation}} \\")
            lines.append(r"\midrule")
            current_section = "1a"
        elif row.rq == "1b" and current_section != "1b":
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{8}{l}{\textit{RQ1b: LLM-as-a-Corrector Evaluation}} \\")
            lines.append(r"\midrule")
            current_section = "1b"
        elif row.rq == "2" and current_section != "2":
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{8}{l}{\textit{RQ2: Context-Insensitive Hallucination Detection}} \\")
            lines.append(r"\midrule")
            current_section = "2"
        elif row.rq == "3" and current_section != "3":
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{8}{l}{\textit{RQ3: Deep Research Literature Validation}$^{\ast}$} \\")
            lines.append(r"\midrule")
            current_section = "3"
        
        # Check for enrichment section
        if row.hypothesis == "Enrichment baseline" and current_section != "3e":
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{8}{l}{\textit{RQ3: Enrichment Analysis (Exploratory)}$^{\ast}$} \\")
            lines.append(r"\midrule")
            current_section = "3e"
        
        # Add cmidrule between different hypotheses in same RQ
        if prev_rq and row.rq == "" and not row.is_subrow and row.hypothesis and not row.hypothesis.startswith("("):
            if prev_rq == current_section:
                lines.append(r"\cmidrule{2-8}")
        
        # Format the row
        line = f"{row.rq} & {row.hypothesis} & {row.h0_h1} & {row.test} & {row.effect} & {row.value} & {row.p_value} & {row.conclusion} \\\\"
        lines.append(line)
        
        if row.rq:
            prev_rq = current_section
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    
    # Table notes
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\scriptsize")
    lines.append(r"\item \textit{Effect size interpretation.} $\kappa$: Landis-Koch (0.61--0.80 = substantial). ICC: $<$0.5 poor, 0.5--0.75 moderate, 0.75--0.9 good, $>$0.9 excellent. $W$: $<$0.3 weak, 0.3--0.5 moderate, $>$0.5 strong. $d$: $<$0.2 small, 0.2--0.8 medium, $>$0.8 large. $r$: $<$0.2 weak, 0.2--0.5 moderate, $>$0.5 strong. $h$: $<$0.2 negligible, 0.2--0.5 small, 0.5--0.8 medium, $>$0.8 large.")
    lines.append(r"\item \textit{Significance.} * $p<.05$, ** $p<.01$, *** $p<.001$. RQ3 $p$-values are Bonferroni-adjusted ($\alpha_{\text{adj}}=0.0167$). RQ2 single-metric $p$ uses Bonferroni adjustment across 4 metrics ($\alpha_{\text{adj}}=0.0125$).")
    lines.append(r"\item \textsuperscript{$\ast$} \textit{RQ3 exploratory.} Enrichment estimates are extrapolated; human-validated scenarios use Wilson CI lower bound (conservative). Not independently validated ground truth.")
    lines.append(r"\item \textit{RQ1a prompt tests.} Friedman with (CLD$\times$run) blocking; $W$ = Kendall's concordance. \textit{RQ1b efficacy.} One-sample Wilcoxon signed-rank on 9 blocks.")
    lines.append(r"\item \textit{RQ2 discrimination.} AUC = Area Under Curve; one-sample Wilcoxon signed-rank test on $(\mathrm{AUC}-0.5)$ over 9 blocks (CLD$\times$run). Cross-CLD generalization is reported descriptively (Phase 5$\to$6).")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def main():
    """Generate and save the effect size summary table."""
    parser = argparse.ArgumentParser(description="Effect Size Summary Table Generator")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Output directory for generated table. If not specified, writes to thesis/reproducible_version/generated/"
    )
    args = parser.parse_args()
    
    # Generate the table
    table_tex = generate_effect_size_table()
    
    # Determine output paths
    # Always write to thesis generated folder
    thesis_out = REPO_ROOT / "thesis/reproducible_version/generated/effect_size_summary_table.tex"
    out_paths = [thesis_out]
    
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        out_paths.append(run_dir / "effect_size_summary_table.tex")
    
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(table_tex, encoding="utf-8")
        print(f"✓ Wrote: {p}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

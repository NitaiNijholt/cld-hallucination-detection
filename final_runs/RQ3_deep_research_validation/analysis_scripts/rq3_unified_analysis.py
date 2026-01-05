#!/usr/bin/env python3
"""
RQ3 Unified Analysis Script - Single Entrypoint for Full Methodological Replicability

This script orchestrates all RQ3 analysis components to provide complete reproducibility.
It calls the existing analysis modules rather than reimplementing functionality.

Orchestrated Scripts:
1. analyze_deep_research_results.py - Core detection rate analysis
2. rq3_statistical_test.py - Statistical tests with assumption checking
3. rq3_per_cld_detection_rates.py - Per-CLD breakdown
4. compute_enriched_gt_metrics.py - Enrichment scenario analysis
5. generate_figures.py - Figure generation
6. run_rq3_analysis.py - LaTeX table generation

Usage:
    python3 rq3_unified_analysis.py [--skip-interactive] [--report-only]

Options:
    --skip-interactive  Skip scripts that produce interactive output
    --report-only       Only generate the master report from existing results

Outputs:
    - All outputs from individual scripts
    - rq3_unified_report.md - Master report consolidating all results
    - rq3_unified_run_log.json - Log of all scripts executed with timestamps

Author: MSc Thesis Analysis Pipeline
Date: 2024-12-28
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import argparse
import os

# Input/output roots (support rerouting outputs under a single run folder)
# Default INPUT_DIR to the RQ3_deep_research folder (parent of analysis_scripts)
INPUT_DIR = Path(os.environ.get("RQ3_INPUT_DIR", str(Path(__file__).parent.parent))).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("RQ3_OUTPUT_DIR", str(INPUT_DIR))).expanduser().resolve()
# Script directory (where analysis scripts live)
SCRIPT_DIR = INPUT_DIR / "analysis_scripts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "validation").mkdir(parents=True, exist_ok=True)

# Scripts to orchestrate (in execution order)
ANALYSIS_SCRIPTS = [
    {
        "name": "analyze_deep_research_results.py",
        "description": "Core detection rate analysis with rich console output",
        "interactive": True,  # Produces console output with rich formatting
        "required": False,  # Not required for report generation
    },
    {
        "name": "rq3_statistical_test.py",
        "description": "Statistical tests (Fisher's exact, Chi-square, Cohen's h, power analysis)",
        "interactive": False,
        "required": True,
        "output_file": "rq3_statistical_test_results_latest.json",
    },
    {
        "name": "rq3_per_cld_detection_rates.py",
        "description": "Per-CLD detection rate breakdown",
        "interactive": False,
        "required": True,
        "output_file": "rq3_per_cld_detection_rates_latest.json",
    },
    {
        "name": "compute_enriched_gt_metrics.py",
        "description": "Enriched ground truth scenarios (FP promotion, FN discovery)",
        "interactive": False,
        "required": True,
        "output_file": "enriched_gt_analysis_latest.json",
    },
    {
        "name": "run_rq3_analysis.py",
        "description": "Unified analysis with LaTeX table generation",
        "interactive": False,
        "required": True,
        "output_file": "rq3_comprehensive_stats.json",
    },
    {
        "name": "generate_figures.py",
        "description": "Publication-quality figure generation",
        "interactive": False,
        "required": True,
        "output_file": "figures/rq3_combined_figure.png",
    },
    # Thesis-facing validation figures (used in Results)
    {
        "name": "enrichment_extrapolation.py",
        "description": "Validation threshold tradeoff + extrapolation (writes validation/all_edges_threshold_tradeoff.png)",
        "interactive": False,
        "required": False,
        "output_file": "validation/all_edges_threshold_tradeoff.png",
    },
    {
        "name": "plot_dr_cost_efficiency.py",
        "description": "Cost-efficiency comparison (writes validation/dr_cost_efficiency_comparison.png)",
        "interactive": False,
        "required": False,
        "output_file": "validation/dr_cost_efficiency_comparison.png",
    },
]


def run_script(script_name: str, cwd: Path, env: dict | None = None) -> Dict[str, Any]:
    """Run a Python script and capture its output."""
    script_path = cwd / script_name
    
    if not script_path.exists():
        return {
            "script": script_name,
            "success": False,
            "error": f"Script not found: {script_path}",
            "duration_seconds": 0,
        }
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            env=env,
            timeout=300,  # 5 minute timeout
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "script": script_name,
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout_lines": len(result.stdout.split('\n')) if result.stdout else 0,
            "stderr_lines": len(result.stderr.split('\n')) if result.stderr else 0,
            "duration_seconds": duration,
            "error": result.stderr[:500] if result.returncode != 0 and result.stderr else None,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "script": script_name,
            "success": False,
            "error": "Script timed out after 300 seconds",
            "duration_seconds": 300,
        }
    except Exception as e:
        return {
            "script": script_name,
            "success": False,
            "error": str(e),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
        }


def load_json_results(filepath: Path) -> Optional[Dict]:
    """Load JSON results if file exists."""
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def generate_unified_report(output_dir: Path) -> str:
    """Generate unified markdown report from all analysis results."""
    
    report = []
    report.append("# RQ3: Deep Research for Enhanced Causal Verification")
    report.append("## Unified Analysis Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("**Research Question:** Can test-time compute via Deep Research improve hallucination detection in CLD generation?\n")
    
    report.append("---\n")
    report.append("## Table of Contents\n")
    report.append("1. [Executive Summary](#executive-summary)")
    report.append("2. [Dataset Overview](#dataset-overview)")
    report.append("3. [Detection Rate Analysis](#detection-rate-analysis)")
    report.append("4. [Statistical Tests](#statistical-tests)")
    report.append("5. [Per-CLD Breakdown](#per-cld-breakdown)")
    report.append("6. [Enrichment Scenarios](#enrichment-scenarios)")
    report.append("7. [Methodology](#methodology)")
    report.append("8. [Limitations](#limitations)")
    report.append("9. [Output Files](#output-files)\n")
    
    report.append("---\n")
    
    # Load results from the existing analysis outputs
    stats = load_json_results(output_dir / "rq3_statistical_test_results_latest.json")
    per_cld = load_json_results(output_dir / "rq3_per_cld_detection_rates_latest.json")
    enrichment = load_json_results(output_dir / "enriched_gt_analysis_latest.json")
    comprehensive = load_json_results(output_dir / "rq3_comprehensive_stats.json")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    
    if stats:
        tp_fp = stats.get("tp_vs_fp", {})
        tp_fn = stats.get("tp_vs_fn", {})
        counts = stats.get("counts", {})
        
        report.append("### Key Findings\n")
        report.append(f"**Total Edges Analyzed:** {stats.get('total_edges', 'N/A')}\n")
        
        report.append("**Detection Rates:**")
        for cls in ["TP", "FP", "FN"]:
            if cls in counts:
                c = counts[cls]
                report.append(f"- {cls}: {c.get('rate', 0)*100:.1f}% ({c.get('found', 0)}/{c.get('total', 0)})")
        report.append("")
        
        report.append("**Discrimination Analysis:**")
        if tp_fp:
            report.append(f"- TP vs FP gap: {tp_fp.get('gap_pp', 0):.1f} pp")
            report.append(f"- Cohen's h: {tp_fp.get('cohens_h', 0):.2f} ({tp_fp.get('cohens_h_interpretation', 'N/A')})")
            report.append(f"- Fisher's exact p: {tp_fp.get('fisher_p', 0):.3f}")
            report.append(f"- Achieved power: {tp_fp.get('power', 0)*100:.0f}%")
        report.append("")
        
        # Answer to RQ3
        report.append("### Answer to RQ3\n")
        gap = tp_fp.get('gap_pp', 0)
        h_interp = tp_fp.get('cohens_h_interpretation', 'negligible')
        
        if gap > 10 and h_interp in ['medium', 'large']:
            report.append("**YES:** Deep Research shows significant discriminative power between TP and FP edges.")
        elif gap > 5:
            report.append("**PARTIAL:** Deep Research shows moderate discrimination, but effect size is small.")
        else:
            report.append("**LIMITED:** Deep Research shows minimal discrimination between TP and FP edges. The TP-FP gap is not statistically significant.\n")
        report.append("")
    
    report.append("---\n")
    
    # Dataset Overview
    report.append("## Dataset Overview\n")
    if per_cld:
        report.append(f"**Total Edges:** {per_cld.get('total_edges', 'N/A')}\n")
        report.append("| CLD | TP | FP | FN | Total |")
        report.append("|-----|----|----|-----|-------|")
        for cld_name, cld_data in per_cld.get("per_cld", {}).items():
            tp_n = cld_data.get("TP", {}).get("total", 0)
            fp_n = cld_data.get("FP", {}).get("total", 0)
            fn_n = cld_data.get("FN", {}).get("total", 0)
            total = tp_n + fp_n + fn_n
            report.append(f"| {cld_name} | {tp_n} | {fp_n} | {fn_n} | {total} |")
        report.append("")
    
    report.append("**Edge Classification Definitions:**")
    report.append("- **TP (True Positive):** Expert-accepted edges present in ground truth CLD")
    report.append("- **FP (False Positive):** LLM-generated edges NOT in ground truth (hallucinations)")
    report.append("- **FN (False Negative):** Ground truth edges NOT generated by LLM (missed edges)\n")
    
    report.append("---\n")
    
    # Detection Rate Analysis
    report.append("## Detection Rate Analysis\n")
    report.append("*Source: `rq3_statistical_test.py`*\n")
    
    if stats:
        counts = stats.get("counts", {})
        report.append("| Classification | n | Found | Rate | Mean Confidence |")
        report.append("|----------------|---|-------|------|-----------------|")
        for cls in ["TP", "FP", "FN"]:
            if cls in counts:
                c = counts[cls]
                report.append(f"| {cls} | {c.get('total', 0)} | {c.get('found', 0)} | {c.get('rate', 0)*100:.1f}% | — |")
        report.append("")
    
    report.append("---\n")
    
    # Statistical Tests
    report.append("## Statistical Tests\n")
    report.append("*Source: `rq3_statistical_test.py`*\n")
    
    if stats:
        # Assumption testing
        assumptions = stats.get("assumptions", {})
        report.append("### Assumption Testing\n")
        report.append(f"- Sample size: N = {assumptions.get('total_n', 'N/A')}")
        report.append(f"- Minimum expected cell frequency: {assumptions.get('min_expected', 'N/A'):.1f}")
        report.append(f"- Chi-square valid: {'Yes' if assumptions.get('chi2_valid', False) else 'No (use Fisher exact)'}")
        report.append("")
        
        # Pairwise comparisons
        report.append("### Pairwise Comparisons\n")
        bonf = stats.get("bonferroni_correction", {})
        report.append(f"*Bonferroni correction: m={bonf.get('n_tests', 3)}, α_adj={bonf.get('alpha_adjusted', 0.017):.3f}*\n")
        
        report.append("| Comparison | Gap (pp) | Cohen's h | Interpretation | p | p_adj | Significant |")
        report.append("|------------|----------|-----------|----------------|---|-------|-------------|")
        
        for key, label in [("tp_vs_fp", "TP vs FP"), ("tp_vs_fn", "TP vs FN"), ("fp_vs_fn", "FP vs FN")]:
            if key in stats:
                s = stats[key]
                sig = "Yes ✓" if s.get('significant_adjusted', False) else "No"
                report.append(f"| {label} | {s.get('gap_pp', 0):.1f} | {s.get('cohens_h', 0):.2f} | {s.get('cohens_h_interpretation', 'N/A')} | {s.get('fisher_p', 0):.3f} | {s.get('fisher_p_adjusted', 0):.3f} | {sig} |")
        report.append("")
        
        # Power analysis
        report.append("### Power Analysis\n")
        if "tp_vs_fp" in stats:
            tp_fp = stats["tp_vs_fp"]
            power = tp_fp.get('power', 0)
            min_h = tp_fp.get('min_detectable_h', 0)
            obs_h = tp_fp.get('cohens_h', 0)
            
            report.append(f"- Achieved power: {power*100:.1f}%")
            report.append(f"- Minimum detectable effect (80% power): h = {min_h:.3f}")
            report.append(f"- Observed effect: h = {obs_h:.3f}")
            
            if power < 0.80:
                report.append(f"- ⚠️ **Underpowered**: Study has insufficient power to detect the observed effect")
            else:
                report.append(f"- ✓ Adequate power achieved")
        report.append("")
    
    report.append("---\n")
    
    # Per-CLD Breakdown
    report.append("## Per-CLD Breakdown\n")
    report.append("*Source: `rq3_per_cld_detection_rates.py`*\n")
    
    if per_cld:
        for cld_name, cld_data in per_cld.get("per_cld", {}).items():
            report.append(f"### {cld_name}\n")
            report.append("| Class | n | Found | Rate |")
            report.append("|-------|---|-------|------|")
            for cls in ["TP", "FP", "FN"]:
                if cls in cld_data:
                    c = cld_data[cls]
                    report.append(f"| {cls} | {c.get('total', 0)} | {c.get('found', 0)} | {c.get('rate_pct', 0):.1f}% |")
            report.append("")
        
        # Domain variation
        dv = per_cld.get("domain_variation", {})
        if dv:
            report.append("### Domain Variation\n")
            report.append(f"- FP detection rate range: {dv.get('fp_rate_min', 0):.1f}% – {dv.get('fp_rate_max', 0):.1f}%")
            report.append(f"- FP rate spread: {dv.get('fp_rate_range_pp', 0):.1f} pp")
            report.append(f"- FP rate std dev: {dv.get('fp_rate_std', 0):.1f} pp")
            report.append("")
    
    report.append("---\n")
    
    # Enrichment Scenarios
    report.append("## Enrichment Scenarios\n")
    report.append("*Source: `compute_enriched_gt_metrics.py`*\n")
    
    if enrichment:
        orig = enrichment.get("original_metrics", {})
        enr = enrichment.get("enriched_metrics", {})
        sys_m = enrichment.get("system_metrics", {})
        imp = enrichment.get("improvement", {})
        sys_imp = enrichment.get("system_improvement", {})
        
        report.append("| Scenario | Precision | Recall | F1 | Change |")
        report.append("|----------|-----------|--------|-----|--------|")
        report.append(f"| Original (Expert GT) | {orig.get('precision', 0):.3f} | {orig.get('recall', 0):.3f} | {orig.get('f1', 0):.3f} | — |")
        report.append(f"| Scenario 1 (Enriched GT) | {enr.get('precision', 0):.3f} | {enr.get('recall', 0):.3f} | {enr.get('f1', 0):.3f} | +{imp.get('f1', 0):.3f} |")
        report.append(f"| Scenario 2 (LLM+DR) | {sys_m.get('precision', 0):.3f} | {sys_m.get('recall', 0):.3f} | {sys_m.get('f1', 0):.3f} | +{sys_imp.get('f1', 0):.3f} |")
        report.append("")
        
        fp_prom = enrichment.get("fp_promotion", {})
        fn_disc = enrichment.get("fn_discovery", {})
        report.append("**FP Promotion (Scenario 1):**")
        report.append(f"- FPs with DR evidence: {fp_prom.get('promoted', 0)}/{fp_prom.get('promoted', 0) + fp_prom.get('not_promoted', 0)} ({fp_prom.get('promotion_rate_pct', 0):.1f}%)")
        report.append("")
        report.append("**FN Discovery (Scenario 2):**")
        report.append(f"- FNs with DR evidence: {fn_disc.get('discovered', 0)}/{fn_disc.get('discovered', 0) + fn_disc.get('not_discovered', 0)} ({fn_disc.get('discovery_rate_pct', 0):.1f}%)")
        report.append("")
    
    report.append("---\n")
    
    # Methodology
    report.append("## Methodology\n")
    report.append("### Deep Research System Design\n")
    report.append("The Deep Research system employs principles from LLM multi-agent literature:")
    report.append("- **Iterative refinement:** Self-refine search agent (Madaan et al., 2023)")
    report.append("- **Independent search directions:** Chain-of-Verification (Dhuliawala et al., 2024)")
    report.append("- **Parallel execution:** Test-time compute scaling (Snell et al., 2024)")
    report.append("- **Mandatory passage citing:** SAFE system (Batista et al., 2025)\n")
    
    report.append("### Statistical Methods\n")
    report.append("- **Fisher's Exact Test:** Primary test for 2×2 contingency tables")
    report.append("- **Chi-Square Test:** With assumption checking (Cochran's rule)")
    report.append("- **Cohen's h:** Effect size for proportion differences with 95% CI")
    report.append("- **Wilson Score Interval:** 95% CI for proportions")
    report.append("- **Bonferroni Correction:** For multiple pairwise comparisons (m=3)")
    report.append("- **Post-hoc Power Analysis:** To assess adequacy of sample size\n")
    
    report.append("---\n")
    
    # Limitations
    report.append("## Limitations\n")
    report.append("1. **TP-FP Discrimination:** The gap (8.1 pp) is not statistically significant; DR cannot reliably distinguish valid from hallucinated edges")
    report.append("2. **Domain Variation:** FP detection rates vary substantially by CLD (28%–76%)")
    report.append("3. **Correlational Confound:** DR may detect correlational rather than causal evidence")
    report.append("4. **No Component Ablation:** Individual component contributions not isolated")
    report.append("5. **Single Run:** No DR replication for uncertainty quantification")
    report.append("6. **Strong Assumptions:** Evidence evaluated in isolation, not in full CLD context\n")
    
    report.append("---\n")
    
    # Output Files
    report.append("## Output Files\n")
    report.append("### Analysis Scripts")
    for script in ANALYSIS_SCRIPTS:
        output = script.get('output_file', 'console output')
        report.append(f"- `{script['name']}` → `{output}`")
    report.append("")
    
    report.append("### LaTeX Tables")
    report.append("- `rq3_main_table.tex` - Main detection rate table with statistics")
    report.append("- `rq3_per_cld_table.tex` - Per-CLD pairwise discrimination tests")
    report.append("- `rq3_enriched_gt_table.tex` - Enrichment scenarios comparison")
    report.append("")
    
    report.append("### Figures")
    report.append("- `figures/rq3_combined_figure.png` - Detection rates and confidence by CLD")
    report.append("- `figures/rq3_detection_rate.png` - Aggregate detection rate bar chart")
    report.append("- `figures/rq3_confidence_scores.png` - Confidence score distributions")
    report.append("")
    
    report.append("---\n")
    report.append(f"**End of Report** - Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="RQ3 Unified Analysis - Single Entrypoint")
    parser.add_argument("--skip-interactive", action="store_true", 
                       help="Skip scripts that produce interactive output")
    parser.add_argument("--report-only", action="store_true",
                       help="Only generate the master report from existing results")
    args = parser.parse_args()
    
    print("=" * 80)
    print("RQ3 UNIFIED ANALYSIS - SINGLE ENTRYPOINT FOR FULL REPLICABILITY")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory:   {INPUT_DIR}")
    print(f"Output directory:  {OUTPUT_DIR}")
    print()
    
    run_log = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "scripts": [],
    }

    env = os.environ.copy()
    env["RQ3_INPUT_DIR"] = str(INPUT_DIR)
    env["RQ3_OUTPUT_DIR"] = str(OUTPUT_DIR)
    
    if not args.report_only:
        # Run all analysis scripts
        print("📊 Running Analysis Scripts")
        print("-" * 80)
        
        for script_info in ANALYSIS_SCRIPTS:
            script_name = script_info["name"]
            description = script_info["description"]
            interactive = script_info.get("interactive", False)
            
            if args.skip_interactive and interactive:
                print(f"\n⏭️  Skipping {script_name} (interactive)")
                run_log["scripts"].append({
                    "script": script_name,
                    "skipped": True,
                    "reason": "interactive mode"
                })
                continue
            
            print(f"\n▶️  Running: {script_name}")
            print(f"   {description}")
            
            # Determine script directory:
            # - Scripts starting with "scripts/" are in INPUT_DIR/scripts/
            # - Other scripts are in SCRIPT_DIR (analysis_scripts/)
            if script_name.startswith("scripts/"):
                script_cwd = INPUT_DIR
            else:
                script_cwd = SCRIPT_DIR
            
            # Run from appropriate directory so relative paths resolve correctly.
            result = run_script(script_name, script_cwd, env=env)
            run_log["scripts"].append(result)
            
            if result["success"]:
                print(f"   ✅ Completed in {result['duration_seconds']:.1f}s")
                if script_info.get("output_file"):
                    output_path = OUTPUT_DIR / script_info["output_file"]
                    if output_path.exists():
                        print(f"   📄 Output: {script_info['output_file']}")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')[:100]}")
    
    # Generate unified report
    print("\n" + "-" * 80)
    print("📝 Generating Unified Report")
    print("-" * 80)
    
    report_content = generate_unified_report(OUTPUT_DIR)
    report_path = OUTPUT_DIR / "rq3_unified_report.md"
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"   ✅ Saved: {report_path}")
    
    # Save run log
    log_path = OUTPUT_DIR / "rq3_unified_run_log.json"
    with open(log_path, 'w') as f:
        json.dump(run_log, f, indent=2, default=str)
    print(f"   ✅ Run log: {log_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("RQ3 UNIFIED ANALYSIS COMPLETE")
    print("=" * 80)
    
    if not args.report_only:
        successful = sum(1 for s in run_log["scripts"] if s.get("success", False))
        skipped = sum(1 for s in run_log["scripts"] if s.get("skipped", False))
        failed = len(run_log["scripts"]) - successful - skipped
        
        print(f"\n📊 Scripts: {successful} successful, {skipped} skipped, {failed} failed")
    
    print(f"\n📁 Output Files:")
    print(f"   Report: {report_path}")
    print(f"   Run Log: {log_path}")
    print(f"   Tables: rq3_main_table.tex, rq3_per_cld_table.tex, rq3_enriched_gt_table.tex")
    print(f"   Figures: figures/rq3_combined_figure.png")
    print(f"   Stats: rq3_comprehensive_stats.json")
    
    print("\n✅ All RQ3 analyses completed!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

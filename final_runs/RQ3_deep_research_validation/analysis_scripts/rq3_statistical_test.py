#!/usr/bin/env python3
"""
RQ3 Statistical Test: Chi-square test for TP-FP discrimination gap.

This script computes the statistical significance of the TP-FP detection rate gap
using chi-square and Fisher's exact tests, with:
- Assumption testing for chi-square (expected cell frequencies)
- Cohen's h effect size with 95% confidence intervals
- Post-hoc power analysis
- Odds ratio with confidence intervals

Generated: 2025-12-19
Updated: 2025-12-19 (added assumption testing, effect size CIs, power analysis)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime
import math
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
    """
    Compute Cohen's h effect size for proportion difference.
    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    
    Cohen (1988) interpretation:
    - |h| < 0.20: negligible
    - 0.20 <= |h| < 0.50: small
    - 0.50 <= |h| < 0.80: medium
    - |h| >= 0.80: large
    """
    p1 = max(0.001, min(0.999, p1))  # Avoid edge cases
    p2 = max(0.001, min(0.999, p2))
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2


def cohens_h_ci(p1: float, n1: int, p2: float, n2: int, alpha: float = 0.05) -> tuple:
    """
    Compute confidence interval for Cohen's h using the delta method.
    
    The variance of phi = 2*arcsin(sqrt(p)) is approximately 1/n.
    Therefore, SE(h) = sqrt(1/n1 + 1/n2).
    
    Returns: (h, lower_ci, upper_ci)
    """
    h = compute_cohens_h(p1, p2)
    se_h = np.sqrt(1/n1 + 1/n2)
    z_crit = stats.norm.ppf(1 - alpha/2)
    lower = h - z_crit * se_h
    upper = h + z_crit * se_h
    return h, lower, upper, se_h


def interpret_cohens_h(h: float) -> str:
    """Interpret Cohen's h effect size (Cohen, 1988)."""
    h_abs = abs(h)
    if h_abs < 0.20:
        return "negligible"
    elif h_abs < 0.50:
        return "small"
    elif h_abs < 0.80:
        return "medium"
    else:
        return "large"


def odds_ratio_ci(contingency: np.ndarray, alpha: float = 0.05) -> tuple:
    """
    Compute odds ratio and 95% CI using Woolf's method.
    
    For 2x2 table:
        [[a, b],
         [c, d]]
    
    OR = (a*d) / (b*c)
    SE(ln(OR)) = sqrt(1/a + 1/b + 1/c + 1/d)
    
    Returns: (odds_ratio, lower_ci, upper_ci)
    """
    a, b = contingency[0]
    c, d = contingency[1]
    
    # Add 0.5 if any cell is zero (Haldane-Anscombe correction)
    if any(x == 0 for x in [a, b, c, d]):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    
    odds_ratio = (a * d) / (b * c)
    ln_or = np.log(odds_ratio)
    se_ln_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    lower = np.exp(ln_or - z_crit * se_ln_or)
    upper = np.exp(ln_or + z_crit * se_ln_or)
    
    return odds_ratio, lower, upper


def check_chi2_assumptions(contingency: np.ndarray, expected: np.ndarray) -> dict:
    """
    Check assumptions for chi-square test.
    
    Assumptions:
    1. Independence: Observations are independent (assumed by design)
    2. Sample size: Total N should be reasonably large (>20)
    3. Expected frequencies: 
       - Cochran's rule: No expected cell < 1, and no more than 20% < 5
       - Conservative: All expected cells >= 5
    
    Returns dict with assumption check results.
    """
    n_cells = expected.size
    cells_below_1 = np.sum(expected < 1)
    cells_below_5 = np.sum(expected < 5)
    pct_below_5 = 100 * cells_below_5 / n_cells
    min_expected = expected.min()
    total_n = contingency.sum()
    
    # Cochran's rule
    cochran_violated = (cells_below_1 > 0) or (pct_below_5 > 20)
    
    # Conservative rule (all >= 5)
    conservative_violated = min_expected < 5
    
    assumptions_met = not cochran_violated
    use_fisher = cochran_violated or conservative_violated
    
    return {
        "total_n": int(total_n),
        "n_cells": int(n_cells),
        "expected_frequencies": [[float(x) for x in row] for row in expected.tolist()],
        "min_expected": float(min_expected),
        "cells_below_1": int(cells_below_1),
        "cells_below_5": int(cells_below_5),
        "pct_below_5": float(pct_below_5),
        "cochran_rule_met": True if not cochran_violated else False,
        "conservative_rule_met": True if not conservative_violated else False,
        "chi2_valid": True if assumptions_met else False,
        "recommend_fisher": True if use_fisher else False
    }


def compute_power(h: float, n1: int, n2: int, alpha: float = 0.05) -> float:
    """
    Compute post-hoc power for two-proportion z-test.
    
    Uses the formula for power of a z-test comparing two proportions.
    Power = P(reject H0 | H1 true) = Φ(z_h - z_α/2)
    
    where z_h = h / sqrt(1/n1 + 1/n2)
    """
    se = np.sqrt(1/n1 + 1/n2)
    z_effect = abs(h) / se
    z_crit = stats.norm.ppf(1 - alpha/2)
    power = stats.norm.cdf(z_effect - z_crit)
    return power


def required_sample_size(h: float, power: float = 0.80, alpha: float = 0.05, ratio: float = 1.0) -> int:
    """
    Compute required sample size per group to detect effect h with given power.
    
    For equal groups: n = 2 * ((z_α/2 + z_β) / h)^2
    """
    if abs(h) < 0.01:
        return float('inf')
    
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # For two groups with ratio k = n2/n1
    # n1 = ((z_alpha + z_beta)^2 * (1 + 1/k)) / h^2
    n1 = ((z_alpha + z_beta)**2 * (1 + 1/ratio)) / (h**2)
    
    return int(np.ceil(n1))


def bonferroni_correction(p_values: dict, alpha: float = 0.05) -> dict:
    """
    Apply Bonferroni correction to a family of p-values.
    
    Args:
        p_values: Dict of {test_name: p_value}
        alpha: Significance level (default 0.05)
    
    Returns:
        Dict with correction results including adjusted p-values and significance
    """
    m = len(p_values)
    alpha_adj = alpha / m
    
    results = {}
    for name, p in p_values.items():
        p_adj = min(p * m, 1.0)  # Adjusted p-value
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
        'n_significant_original': sum(1 for r in results.values() if r['significant_original']),
        'n_significant_adjusted': sum(1 for r in results.values() if r['significant_adjusted'])
    }


def main():
    """Run statistical tests for RQ3 discrimination analysis."""
    
    print("=" * 90)
    print("RQ3 STATISTICAL TEST: TP-FP Discrimination Gap")
    print("With Assumption Testing, Effect Size CIs, and Power Analysis")
    print("=" * 90)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load all results
    all_results = []
    for filename, cld_name in DATA_FILES:
        filepath = Path(filename)
        if filepath.exists():
            results = load_results(str(filepath))
            for r in results:
                if 'CLD' not in r:
                    r['CLD'] = cld_name
            all_results.extend(results)
            print(f"Loaded {len(results)} edges from {cld_name}")
    
    print(f"\nTotal edges: {len(all_results)}")
    
    # Count by classification
    counts = {}
    for r in all_results:
        cls = r['classification']
        if cls not in counts:
            counts[cls] = {'total': 0, 'found': 0}
        counts[cls]['total'] += 1
        if r.get('direct_causal_found', False):
            counts[cls]['found'] += 1
    
    # Calculate detection rates
    for cls in counts:
        counts[cls]['rate'] = counts[cls]['found'] / counts[cls]['total']
        counts[cls]['not_found'] = counts[cls]['total'] - counts[cls]['found']
    
    print("\n" + "-" * 90)
    print("DETECTION RATES")
    print("-" * 90)
    for cls in ['TP', 'FP', 'FN']:
        if cls in counts:
            c = counts[cls]
            print(f"{cls}: {c['found']}/{c['total']} = {c['rate']*100:.1f}%")
    
    # ========================================================================
    # MAIN TEST: TP vs FP discrimination
    # ========================================================================
    print("\n" + "=" * 90)
    print("STATISTICAL TEST: TP vs FP Discrimination")
    print("=" * 90)
    
    tp = counts['TP']
    fp = counts['FP']
    
    # 2x2 contingency table
    contingency = np.array([
        [tp['found'], tp['not_found']],  # TP row
        [fp['found'], fp['not_found']]   # FP row
    ])
    
    print(f"\nContingency Table:")
    print(f"                      Found    Not Found    Total")
    print(f"TP (Expert-accepted)    {tp['found']:3d}          {tp['not_found']:3d}        {tp['total']:3d}")
    print(f"FP (Hallucinations)    {fp['found']:3d}           {fp['not_found']:3d}        {fp['total']:3d}")
    print(f"Total                  {tp['found']+fp['found']:3d}           {tp['not_found']+fp['not_found']:3d}        {tp['total']+fp['total']:3d}")
    
    # ========================================================================
    # ASSUMPTION TESTING
    # ========================================================================
    print("\n" + "-" * 90)
    print("ASSUMPTION TESTING FOR CHI-SQUARE")
    print("-" * 90)
    
    # Chi-square test (to get expected frequencies)
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
    
    assumptions = check_chi2_assumptions(contingency, expected)
    
    print(f"\n1. Sample Size: N = {assumptions['total_n']} (should be > 20)")
    print(f"   ✓ PASS" if assumptions['total_n'] > 20 else "   ✗ FAIL")
    
    print(f"\n2. Expected Cell Frequencies:")
    print(f"   Expected table:")
    print(f"                      Found    Not Found")
    print(f"   TP                  {expected[0,0]:.1f}         {expected[0,1]:.1f}")
    print(f"   FP                 {expected[1,0]:.1f}         {expected[1,1]:.1f}")
    print(f"\n   Minimum expected frequency: {assumptions['min_expected']:.2f}")
    print(f"   Cells < 5: {assumptions['cells_below_5']}/{assumptions['n_cells']} ({assumptions['pct_below_5']:.1f}%)")
    print(f"   Cells < 1: {assumptions['cells_below_1']}/{assumptions['n_cells']}")
    
    print(f"\n   Cochran's Rule (no cell < 1, ≤20% cells < 5): ", end="")
    print("✓ PASS" if assumptions['cochran_rule_met'] else "✗ FAIL")
    print(f"   Conservative Rule (all cells ≥ 5): ", end="")
    print("✓ PASS" if assumptions['conservative_rule_met'] else "✗ FAIL")
    
    print(f"\n3. Independence: Assumed by design (each edge evaluated once)")
    print(f"   ✓ PASS (by design)")
    
    print(f"\n→ Chi-square test valid: {'YES' if assumptions['chi2_valid'] else 'NO - use Fisher exact'}")
    print(f"→ Recommended test: {'Fisher exact' if assumptions['recommend_fisher'] else 'Chi-square'}")
    
    # ========================================================================
    # STATISTICAL TESTS
    # ========================================================================
    print("\n" + "-" * 90)
    print("STATISTICAL TESTS")
    print("-" * 90)
    
    # Fisher's exact test (always valid for 2x2)
    odds_ratio, p_fisher = stats.fisher_exact(contingency)
    or_val, or_lower, or_upper = odds_ratio_ci(contingency)
    
    # Gap in percentage points
    gap_pp = (tp['rate'] - fp['rate']) * 100
    
    print(f"\nDetection Rates:")
    print(f"  TP: {tp['rate']*100:.1f}% ({tp['found']}/{tp['total']})")
    print(f"  FP: {fp['rate']*100:.1f}% ({fp['found']}/{fp['total']})")
    print(f"  Gap: {gap_pp:.1f} percentage points")
    
    print(f"\nChi-Square Test:")
    print(f"  χ²({dof}) = {chi2:.3f}, p = {p_chi2:.4f}")
    if not assumptions['chi2_valid']:
        print(f"  ⚠ WARNING: Chi-square assumptions violated; interpret with caution")
    
    print(f"\nFisher's Exact Test (preferred for 2x2 tables):")
    print(f"  p = {p_fisher:.4f}")
    print(f"  Odds Ratio = {or_val:.3f} [95% CI: {or_lower:.3f}, {or_upper:.3f}]")
    
    # Interpretation of odds ratio
    if or_lower <= 1.0 <= or_upper:
        print(f"  → CI includes 1.0: No significant association")
    else:
        direction = "higher" if or_val > 1 else "lower"
        print(f"  → TP has {direction} odds of detection than FP")
    
    # ========================================================================
    # EFFECT SIZE WITH CONFIDENCE INTERVAL
    # ========================================================================
    print("\n" + "-" * 90)
    print("EFFECT SIZE ANALYSIS")
    print("-" * 90)
    
    h, h_lower, h_upper, h_se = cohens_h_ci(tp['rate'], tp['total'], fp['rate'], fp['total'])
    h_interp = interpret_cohens_h(h)
    
    print(f"\nCohen's h Effect Size:")
    print(f"  h = {h:.3f} [95% CI: {h_lower:.3f}, {h_upper:.3f}]")
    print(f"  SE(h) = {h_se:.3f}")
    print(f"  Interpretation: {h_interp.upper()}")
    print(f"\n  Cohen (1988) benchmarks:")
    print(f"    |h| < 0.20: negligible")
    print(f"    0.20 ≤ |h| < 0.50: small")
    print(f"    0.50 ≤ |h| < 0.80: medium")
    print(f"    |h| ≥ 0.80: large")
    
    # Check if CI includes zero
    if h_lower <= 0 <= h_upper:
        print(f"\n  → 95% CI includes 0: Effect not significantly different from zero")
    else:
        print(f"\n  → 95% CI excludes 0: Significant effect")
    
    # ========================================================================
    # POWER ANALYSIS
    # ========================================================================
    print("\n" + "-" * 90)
    print("POWER ANALYSIS")
    print("-" * 90)
    
    power = compute_power(h, tp['total'], fp['total'])
    
    print(f"\nPost-hoc Power Analysis:")
    print(f"  Observed effect: h = {h:.3f}")
    print(f"  Sample sizes: n₁ = {tp['total']}, n₂ = {fp['total']}")
    print(f"  Alpha = 0.05 (two-tailed)")
    print(f"  Achieved power = {power:.3f} ({power*100:.1f}%)")
    
    if power < 0.80:
        print(f"  ⚠ UNDERPOWERED: Power < 80% to detect observed effect")
        
        # What sample size would be needed?
        n_needed = required_sample_size(h, power=0.80)
        ratio = fp['total'] / tp['total']
        n1_needed = required_sample_size(h, power=0.80, ratio=ratio)
        n2_needed = int(n1_needed * ratio)
        
        print(f"\n  To achieve 80% power at h = {h:.3f}:")
        print(f"    Equal groups: n = {n_needed} per group")
        print(f"    Current ratio ({ratio:.1f}:1): n₁ = {n1_needed}, n₂ = {n2_needed}")
    else:
        print(f"  ✓ Adequate power (≥80%)")
    
    # What effect size could we detect?
    # Solve for h given power = 0.80
    z_alpha = stats.norm.ppf(0.975)
    z_beta = stats.norm.ppf(0.80)
    se = np.sqrt(1/tp['total'] + 1/fp['total'])
    h_detectable = (z_alpha + z_beta) * se
    
    print(f"\n  Minimum detectable effect (80% power, α=0.05):")
    print(f"    h = {h_detectable:.3f} ({interpret_cohens_h(h_detectable)})")
    print(f"    Observed h = {h:.3f} is {'detectable' if abs(h) >= h_detectable else 'below detection threshold'}")
    
    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("\n" + "=" * 90)
    print("CONCLUSION")
    print("=" * 90)
    
    sig = p_fisher < 0.05
    
    print(f"\n1. Statistical Significance:")
    print(f"   Fisher's exact p = {p_fisher:.4f}")
    print(f"   → {'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'} at α = 0.05")
    
    print(f"\n2. Effect Size:")
    print(f"   Cohen's h = {h:.3f} [{h_lower:.3f}, {h_upper:.3f}]")
    print(f"   → {h_interp.upper()} effect")
    
    print(f"\n3. Practical Interpretation:")
    if not sig and h_interp == "negligible":
        print(f"   Deep Research does NOT significantly discriminate between TP and FP edges.")
        print(f"   The 8.1 pp gap is neither statistically significant (p = {p_fisher:.2f})")
        print(f"   nor practically meaningful (Cohen's h = {h:.2f}, negligible).")
    elif sig:
        print(f"   Deep Research shows significant discrimination (p < 0.05).")
    
    print(f"\n4. Power:")
    print(f"   Achieved power = {power*100:.1f}%")
    if power < 0.80:
        print(f"   Study is underpowered; Type II error is possible.")
    
    # ========================================================================
    # SECONDARY TESTS (abbreviated)
    # ========================================================================
    print("\n" + "=" * 90)
    print("SECONDARY TESTS (TP vs FN, FP vs FN)")
    print("=" * 90)
    
    fn = counts['FN']
    
    # TP vs FN
    contingency_tf = np.array([[tp['found'], tp['not_found']], [fn['found'], fn['not_found']]])
    chi2_tf, p_tf, dof_tf, expected_tf = stats.chi2_contingency(contingency_tf)
    _, p_fisher_tf = stats.fisher_exact(contingency_tf)
    h_tf, h_tf_lo, h_tf_hi, _ = cohens_h_ci(tp['rate'], tp['total'], fn['rate'], fn['total'])
    gap_tf = (tp['rate'] - fn['rate']) * 100
    power_tf = compute_power(h_tf, tp['total'], fn['total'])
    
    print(f"\nTP vs FN:")
    print(f"  Gap: {gap_tf:.1f} pp")
    print(f"  Fisher's p = {p_fisher_tf:.4f} {'**' if p_fisher_tf < 0.01 else '*' if p_fisher_tf < 0.05 else ''}")
    print(f"  Cohen's h = {h_tf:.3f} [{h_tf_lo:.3f}, {h_tf_hi:.3f}] ({interpret_cohens_h(h_tf)})")
    print(f"  Power = {power_tf*100:.1f}%")
    print(f"  → {'SIGNIFICANT' if p_fisher_tf < 0.05 else 'NOT SIGNIFICANT'}")
    
    # FP vs FN
    contingency_ff = np.array([[fp['found'], fp['not_found']], [fn['found'], fn['not_found']]])
    chi2_ff, p_ff, dof_ff, expected_ff = stats.chi2_contingency(contingency_ff)
    _, p_fisher_ff = stats.fisher_exact(contingency_ff)
    h_ff, h_ff_lo, h_ff_hi, _ = cohens_h_ci(fp['rate'], fp['total'], fn['rate'], fn['total'])
    gap_ff = (fp['rate'] - fn['rate']) * 100
    power_ff = compute_power(h_ff, fp['total'], fn['total'])
    
    print(f"\nFP vs FN:")
    print(f"  Gap: {gap_ff:.1f} pp")
    print(f"  Fisher's p = {p_fisher_ff:.4f} {'**' if p_fisher_ff < 0.01 else '*' if p_fisher_ff < 0.05 else ''}")
    print(f"  Cohen's h = {h_ff:.3f} [{h_ff_lo:.3f}, {h_ff_hi:.3f}] ({interpret_cohens_h(h_ff)})")
    print(f"  Power = {power_ff*100:.1f}%")
    print(f"  → {'SIGNIFICANT' if p_fisher_ff < 0.05 else 'NOT SIGNIFICANT'}")
    
    # ========================================================================
    # BONFERRONI CORRECTION (3 pairwise tests within RQ3)
    # ========================================================================
    print("\n" + "=" * 90)
    print("BONFERRONI CORRECTION (3 pairwise tests within RQ3)")
    print("=" * 90)
    
    p_values_dict = {
        "TP_vs_FP": p_fisher,
        "TP_vs_FN": p_fisher_tf,
        "FP_vs_FN": p_fisher_ff
    }
    
    correction = bonferroni_correction(p_values_dict, alpha=0.05)
    
    print(f"\nFamily: RQ3 pairwise discrimination tests")
    print(f"Number of tests: {correction['n_tests']}")
    print(f"α_original = {correction['alpha_original']:.3f}")
    print(f"α_adjusted = {correction['alpha_adjusted']:.4f} (Bonferroni)")
    print()
    
    print(f"{'Comparison':<15} {'p (raw)':<12} {'p (adj)':<12} {'Sig (raw)':<12} {'Sig (adj)':<12} {'Changed'}")
    print("-" * 75)
    for name, res in correction['results'].items():
        sig_raw = "Yes *" if res['significant_original'] else "No"
        sig_adj = "Yes †" if res['significant_adjusted'] else "No"
        changed = "← CHANGED" if res['changed'] else ""
        print(f"{name:<15} {res['p_original']:<12.4f} {res['p_adjusted']:<12.4f} {sig_raw:<12} {sig_adj:<12} {changed}")
    
    print()
    print(f"Significant before correction: {correction['n_significant_original']}/{correction['n_tests']}")
    print(f"Significant after correction: {correction['n_significant_adjusted']}/{correction['n_tests']}")
    
    # ========================================================================
    # LATEX OUTPUT
    # ========================================================================
    print("\n" + "=" * 90)
    print("LATEX OUTPUT FOR THESIS")
    print("=" * 90)
    
    # Get adjusted p-values for LaTeX
    p_adj_tp_fp = correction['results']['TP_vs_FP']['p_adjusted']
    p_adj_tp_fn = correction['results']['TP_vs_FN']['p_adjusted']
    p_adj_fp_fn = correction['results']['FP_vs_FN']['p_adjusted']
    sig_adj_tp_fn = correction['results']['TP_vs_FN']['significant_adjusted']
    sig_adj_fp_fn = correction['results']['FP_vs_FN']['significant_adjusted']
    
    print(f"""
% RQ3 Statistical Test Results (generated {datetime.now().strftime('%Y-%m-%d')})
% TP vs FP discrimination with full assumption testing

% Assumption Testing:
% - Sample size: N = {assumptions['total_n']} (adequate)
% - Expected cell frequencies: min = {assumptions['min_expected']:.1f} ({"all ≥5" if assumptions['conservative_rule_met'] else "some <5"})
% - Chi-square valid: {"Yes" if assumptions['chi2_valid'] else "No, use Fisher"}

% Main Results:
% TP detection rate: {tp['rate']*100:.1f}% ({tp['found']}/{tp['total']})
% FP detection rate: {fp['rate']*100:.1f}% ({fp['found']}/{fp['total']})
% Gap: {gap_pp:.1f} pp

% Statistical Tests (with Bonferroni correction, m=3, α_adj=0.0167):
% TP vs FP: Fisher's exact p = {p_fisher:.3f}, p_adj = {p_adj_tp_fp:.3f}
% TP vs FN: Fisher's exact p = {p_fisher_tf:.3f}, p_adj = {p_adj_tp_fn:.3f} {'†' if sig_adj_tp_fn else ''}
% FP vs FN: Fisher's exact p = {p_fisher_ff:.3f}, p_adj = {p_adj_fp_fn:.3f} {'†' if sig_adj_fp_fn else ''}

% Effect Size:
% Cohen's h = {h:.3f} [95\\% CI: {h_lower:.3f}, {h_upper:.3f}] ({h_interp})

% Power Analysis:
% Achieved power = {power*100:.1f}%
% Minimum detectable effect = {h_detectable:.3f}

% For thesis table note:
% \\textit{{Statistical tests:}} Fisher's exact $p={p_fisher:.2f}$, $p_{{\\text{{adj}}}}={p_adj_tp_fp:.2f}$; 
% Cohen's $h={h:.2f}$ [95\\% CI: {h_lower:.2f}, {h_upper:.2f}] ({h_interp}); 
% achieved power = {power*100:.0f}\\%.
% \\textit{{Multiple comparisons:}} Bonferroni-corrected for 3 pairwise tests ($\\alpha_{{\\text{{adj}}}} = 0.017$).
% TP--FN $p_{{\\text{{adj}}}}={p_adj_tp_fn:.3f}${'†' if sig_adj_tp_fn else ''}, FP--FN $p_{{\\text{{adj}}}}={p_adj_fp_fn:.3f}${'†' if sig_adj_fp_fn else ''}.
""")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_edges": len(all_results),
        "counts": {cls: dict(c) for cls, c in counts.items()},
        "assumptions": assumptions,
        "bonferroni_correction": {
            "method": correction['method'],
            "n_tests": correction['n_tests'],
            "alpha_original": correction['alpha_original'],
            "alpha_adjusted": correction['alpha_adjusted'],
            "n_significant_original": correction['n_significant_original'],
            "n_significant_adjusted": correction['n_significant_adjusted']
        },
        "tp_vs_fp": {
            "gap_pp": float(gap_pp),
            "chi2": float(chi2),
            "chi2_dof": int(dof),
            "chi2_p": float(p_chi2),
            "fisher_p": float(p_fisher),
            "fisher_p_adjusted": float(correction['results']['TP_vs_FP']['p_adjusted']),
            "significant_adjusted": bool(correction['results']['TP_vs_FP']['significant_adjusted']),
            "odds_ratio": float(or_val),
            "odds_ratio_ci_lower": float(or_lower),
            "odds_ratio_ci_upper": float(or_upper),
            "cohens_h": float(h),
            "cohens_h_ci_lower": float(h_lower),
            "cohens_h_ci_upper": float(h_upper),
            "cohens_h_se": float(h_se),
            "cohens_h_interpretation": h_interp,
            "power": float(power),
            "min_detectable_h": float(h_detectable),
            "significant_p05": bool(p_fisher < 0.05)
        },
        "tp_vs_fn": {
            "gap_pp": float(gap_tf),
            "chi2_p": float(p_tf),
            "fisher_p": float(p_fisher_tf),
            "fisher_p_adjusted": float(correction['results']['TP_vs_FN']['p_adjusted']),
            "significant_adjusted": bool(correction['results']['TP_vs_FN']['significant_adjusted']),
            "cohens_h": float(h_tf),
            "cohens_h_ci": [float(h_tf_lo), float(h_tf_hi)],
            "cohens_h_interpretation": interpret_cohens_h(h_tf),
            "power": float(power_tf),
            "significant_p05": bool(p_fisher_tf < 0.05)
        },
        "fp_vs_fn": {
            "gap_pp": float(gap_ff),
            "chi2_p": float(p_ff),
            "fisher_p": float(p_fisher_ff),
            "fisher_p_adjusted": float(correction['results']['FP_vs_FN']['p_adjusted']),
            "significant_adjusted": bool(correction['results']['FP_vs_FN']['significant_adjusted']),
            "cohens_h": float(h_ff),
            "cohens_h_ci": [float(h_ff_lo), float(h_ff_hi)],
            "cohens_h_interpretation": interpret_cohens_h(h_ff),
            "power": float(power_ff),
            "significant_p05": bool(p_fisher_ff < 0.05)
        }
    }
    
    output_file = OUTPUT_DIR / f"rq3_statistical_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    latest_file = OUTPUT_DIR / "rq3_statistical_test_results_latest.json"
    with open(latest_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Latest results: {latest_file}")


if __name__ == "__main__":
    main()







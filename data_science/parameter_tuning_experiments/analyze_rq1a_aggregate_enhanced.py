#!/usr/bin/env python3
"""
RQ1a Enhanced Aggregate Analysis - Corruption Detection with Scientific Rigor

Comprehensive analysis including:
- AUC-ROC metrics
- Corruption-type-specific performance
- Statistical significance testing (Friedman, Wilcoxon, Mann-Whitney)
- Assumption testing (Shapiro-Wilk, Levene's)
- Multi-dimensional aggregation (per-prompt, per-CLD, per-corruption-type)
- Confidence intervals and effect sizes
- Enhanced visualizations

Usage:
    python analyze_rq1a_aggregate_enhanced.py [--base_dir PATH]
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.stats import shapiro, levene, friedmanchisquare, wilcoxon, mannwhitneyu
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn for AUC-ROC
try:
    from sklearn.metrics import roc_auc_score, roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. AUC-ROC metrics will be skipped.")
    SKLEARN_AVAILABLE = False

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

def format_mean_minmax(values):
    """Format values as mean [min, max] per uncertainty reporting rule (N<30)."""
    if len(values) == 0:
        return "N/A"
    mean_val = np.mean(values)
    min_val = np.min(values)
    max_val = np.max(values)
    return f"{mean_val:.3f} [{min_val:.3f}, {max_val:.3f}]"

# Global prompt display names for consistent labeling
PROMPT_DISPLAY = {
    'baseline': 'Baseline',
    'mechanistic': 'Mechanistic',
    'mechanistic_original': 'Mechanistic',
    'cot': 'CoT',
    'mechanistic_lit': 'Mech-Lit',
}

def get_prompt_label(prompt: str) -> str:
    """Get consistent display label for a prompt name."""
    return PROMPT_DISPLAY.get(prompt.lower(), prompt.replace('_', '-').title())


def collect_all_run_results(base_dir: Path) -> dict:
    """Collect results from all run directories.
    
    Supports new folder structure: Data/CLD/run_X/prompt/*.xlsx
    Falls back to old structure: CLD/run_X/judged_*_prompt_*.xlsx
    """
    results = {}
    
    cld_mappings = {
        'depressive': 'Depressive symptoms',
        'social_norms': 'Social norms', 
        'emergency_department': 'Emergency department'
    }
    
    print("="*80)
    print("COLLECTING RUN RESULTS")
    print("="*80)
    print(f"Base directory: {base_dir}\n")
    
    # Check for new structure (Data/ subfolder)
    data_dir = base_dir / "Data"
    if data_dir.exists():
        search_dir = data_dir
        use_new_structure = True
        print("Using NEW folder structure: Data/CLD/run_X/prompt/*.xlsx\n")
    else:
        search_dir = base_dir
        use_new_structure = False
        print("Using OLD folder structure: CLD/run_X/judged_*_prompt_*.xlsx\n")
    
    for cld_dir in search_dir.iterdir():
        if not cld_dir.is_dir():
            continue
        
        cld_name = cld_dir.name
        if cld_name not in cld_mappings:
            continue
        
        results[cld_name] = {}
        run_dirs = sorted([d for d in cld_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
        
        for run_dir in run_dirs:
            run_name = run_dir.name
            
            if use_new_structure:
                # New structure: run_X/prompt/*.xlsx
                prompt_dict = {}
                for prompt in ['baseline', 'cot', 'mechanistic', 'mechanistic_lit']:
                    prompt_dir = run_dir / prompt
                    if prompt_dir.exists():
                        xlsx_files = list(prompt_dir.glob("*.xlsx"))
                        if xlsx_files:
                            prompt_dict[prompt] = xlsx_files[0]
                
                if len(prompt_dict) >= 3:
                    results[cld_name][run_name] = prompt_dict
                    if 'mechanistic_lit' in prompt_dict:
                        print(f"  ✓ {cld_name}/{run_name}: Found all 4 prompt folders (incl. mech_lit)")
                    else:
                        print(f"  ✓ {cld_name}/{run_name}: Found 3 prompt folders")
                else:
                    print(f"  ✗ {cld_name}/{run_name}: Missing prompt folders (found {len(prompt_dict)})")
            else:
                # Old structure: run_X/judged_*_prompt_*.xlsx
                baseline_files = list(run_dir.glob("judged_*_baseline_*.xlsx"))
                mechanistic_files = [f for f in run_dir.glob("judged_*_mechanistic_*.xlsx") 
                                     if '_mechanistic_lit_' not in f.name]
                cot_files = list(run_dir.glob("judged_*_cot_*.xlsx"))
                mechanistic_lit_files = list(run_dir.glob("judged_*_mechanistic_lit_*.xlsx"))
                
                if baseline_files and mechanistic_files and cot_files:
                    results[cld_name][run_name] = {
                        'baseline': baseline_files[0],
                        'mechanistic': mechanistic_files[0],
                        'cot': cot_files[0]
                    }
                    if mechanistic_lit_files:
                        results[cld_name][run_name]['mechanistic_lit'] = mechanistic_lit_files[0]
                        print(f"  ✓ {cld_name}/{run_name}: Found all 4 prompt files (incl. mech_lit)")
                    else:
                        print(f"  ✓ {cld_name}/{run_name}: Found 3 prompt files")
                else:
                    print(f"  ✗ {cld_name}/{run_name}: Missing files")
    
    total_runs = sum(len(runs) for runs in results.values())
    print(f"\nSummary: {len(results)} CLDs, {total_runs} total runs")
    return results


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


def calculate_confidence_interval(data: np.ndarray, confidence=0.95) -> Tuple[float, float]:
    """Calculate confidence interval for a dataset."""
    if len(data) < 2:
        return (np.nan, np.nan)
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return (mean - ci, mean + ci)


def calculate_ci_halfwidth(data: np.ndarray, confidence=0.95) -> float:
    """Calculate the half-width of a confidence interval (for error bars)."""
    if len(data) < 2:
        return 0.0
    sem = stats.sem(data)
    ci_halfwidth = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return ci_halfwidth


def _sig_stars(p_value: float) -> str:
    """Return significance stars for a p-value."""
    if p_value is None or np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _format_p(p_value: float) -> str:
    if p_value is None or np.isnan(p_value):
        return "—"
    return f"{p_value:.3f}" if p_value >= 0.001 else "<0.001"


def compute_prompt_effect_test(
    all_results: dict,
    metric_key: str,
    prompt_names: List[str],
) -> dict:
    """
    Compute prompt effect using Friedman test with (CLD, run) as blocks,
    plus post-hoc pairwise Wilcoxon signed-rank tests with Bonferroni correction.
    
    This is a repeated-measures design where:
    - Block = (CLD, run/seed) — the same underlying instance
    - Treatment = prompt variant
    - Within each block, we compare prompts on the same data
    
    Friedman test is used because:
    1. It's the non-parametric equivalent of repeated-measures ANOVA
    2. It controls for block-level differences (CLD difficulty, run variation)
    3. It's robust to non-normality with small n per block
    
    Returns dict with:
    - Friedman test results and Kendall's W effect size
    - Post-hoc pairwise comparisons (Wilcoxon signed-rank with Bonferroni)
    """
    # Build a table: rows = blocks (CLD, run), columns = prompts
    block_data = []  # List of dicts: {prompt1: val1, prompt2: val2, ...}
    block_labels = []
    
    for cld_name, runs in all_results.items():
        for run_name, files in runs.items():
            row = {}
            for prompt_name in prompt_names:
                if prompt_name in files:
                    metrics = extract_detailed_performance(files[prompt_name])
                    if metrics:
                        v = metrics.get(metric_key, np.nan)
                        if v is not None and not np.isnan(v):
                            row[prompt_name] = float(v)
            # Only include block if it has data for all prompts (complete case)
            if len(row) == len(prompt_names):
                block_data.append(row)
                block_labels.append(f"{cld_name}/{run_name}")
    
    n_blocks = len(block_data)
    k_prompts = len(prompt_names)
    
    if n_blocks < 2:
        return {
            "ok": False,
            "reason": f"insufficient complete blocks ({n_blocks})",
            "test": None,
            "p": np.nan,
            "kendall_w": np.nan,
            "n_blocks": n_blocks,
            "k_prompts": k_prompts,
            "posthoc": {},
        }
    
    # Convert to matrix for Friedman: shape (n_blocks, k_prompts)
    # Each row is a block, each column is a prompt
    data_matrix = np.array([[row[p] for p in prompt_names] for row in block_data])
    
    # Friedman test
    try:
        stat, p_value = friedmanchisquare(*[data_matrix[:, i] for i in range(k_prompts)])
    except Exception as e:
        return {
            "ok": False,
            "reason": str(e),
            "test": None,
            "p": np.nan,
            "kendall_w": np.nan,
            "n_blocks": n_blocks,
            "k_prompts": k_prompts,
            "posthoc": {},
        }
    
    # Effect size: Kendall's W = χ² / (n * (k - 1))
    # Interpretation: 0.1 = small, 0.3 = medium, 0.5 = large
    kendall_w = float(stat) / (n_blocks * (k_prompts - 1)) if n_blocks > 0 and k_prompts > 1 else np.nan
    
    # Post-hoc pairwise Wilcoxon signed-rank tests with Bonferroni correction
    posthoc = {}
    n_comparisons = k_prompts * (k_prompts - 1) // 2
    alpha_bonf = 0.05 / n_comparisons if n_comparisons > 0 else 0.05
    
    for i in range(k_prompts):
        for j in range(i + 1, k_prompts):
            p1, p2 = prompt_names[i], prompt_names[j]
            data1 = data_matrix[:, i]
            data2 = data_matrix[:, j]
            
            try:
                # Wilcoxon signed-rank test (paired, non-parametric)
                w_stat, w_p = wilcoxon(data1, data2)
                
                # Effect size: r = Z / sqrt(N), where Z = (W - mean) / std
                # For Wilcoxon, approximate r using: r = Z / sqrt(2N)
                # We use the p-value to get approximate Z
                from scipy.stats import norm
                z_score = norm.ppf(1 - w_p / 2) if w_p > 0 and w_p < 1 else 0
                effect_r = abs(z_score) / np.sqrt(2 * n_blocks)
                
                # Mean difference (for direction)
                mean_diff = float(np.mean(data1) - np.mean(data2))
                
                posthoc[f"{p1}_vs_{p2}"] = {
                    "p": float(w_p),
                    "p_adj": float(min(w_p * n_comparisons, 1.0)),  # Bonferroni adjusted
                    "sig": bool(w_p < alpha_bonf),
                    "effect_r": float(effect_r),
                    "mean_diff": mean_diff,
                    "winner": p1 if mean_diff > 0 else p2,
                }
            except Exception as e:
                posthoc[f"{p1}_vs_{p2}"] = {"p": np.nan, "sig": False, "error": str(e)}
    
    # Per-CLD pairwise tests (Wilcoxon across runs within each CLD)
    per_cld_posthoc = {}
    cld_short = {"depressive": "DE", "emergency_department": "ED", "social_norms": "SN"}
    
    for cld_name, runs in all_results.items():
        cld_block_data = []
        for run_name, files in runs.items():
            row = {}
            for prompt_name in prompt_names:
                if prompt_name in files:
                    metrics = extract_detailed_performance(files[prompt_name])
                    if metrics:
                        v = metrics.get(metric_key, np.nan)
                        if v is not None and not np.isnan(v):
                            row[prompt_name] = float(v)
            if len(row) == len(prompt_names):
                cld_block_data.append(row)
        
        if len(cld_block_data) >= 2:
            cld_matrix = np.array([[row[p] for p in prompt_names] for row in cld_block_data])
            cld_posthoc = {}
            
            for i in range(k_prompts):
                for j in range(i + 1, k_prompts):
                    p1, p2 = prompt_names[i], prompt_names[j]
                    data1 = cld_matrix[:, i]
                    data2 = cld_matrix[:, j]
                    
                    try:
                        w_stat, w_p = wilcoxon(data1, data2)
                        mean_diff = float(np.mean(data1) - np.mean(data2))
                        # Use uncorrected p<0.05 for per-CLD (exploratory)
                        cld_posthoc[f"{p1}_vs_{p2}"] = {
                            "p": float(w_p),
                            "sig": bool(w_p < 0.05),
                            "mean_diff": mean_diff,
                            "winner": p1 if mean_diff > 0 else p2,
                        }
                    except:
                        cld_posthoc[f"{p1}_vs_{p2}"] = {"p": np.nan, "sig": False}
            
            per_cld_posthoc[cld_name] = cld_posthoc
    
    return {
        "ok": True,
        "test": "Friedman",
        "p": float(p_value),
        "stat": float(stat),
        "kendall_w": float(kendall_w) if not np.isnan(kendall_w) else np.nan,
        "n_blocks": n_blocks,
        "k_prompts": k_prompts,
        "block_labels": block_labels,
        "posthoc": posthoc,
        "per_cld_posthoc": per_cld_posthoc,
        "alpha_bonf": alpha_bonf,
        "n_comparisons": n_comparisons,
    }

def get_prompt_names_from_results(all_results: dict) -> List[str]:
    """
    Determine which prompt variants are available in the run files, in a stable order.
    This must match the prompt selection logic used for figure annotations to keep
    the statistical tables consistent with the displayed results.
    """
    available_prompts: set[str] = set()
    for _cld_name, runs in all_results.items():
        for _run_name, files in runs.items():
            available_prompts.update(files.keys())
    # Stable preference order (include mechanistic_original and mechanistic_lit if present)
    prompt_order_full = ['baseline', 'mechanistic', 'mechanistic_original', 'cot', 'mechanistic_lit']
    prompt_names = [p for p in prompt_order_full if p in available_prompts]
    # If we have both mechanistic and mechanistic_original, prefer mechanistic_original for blocking
    if 'mechanistic' in prompt_names and 'mechanistic_original' in prompt_names:
        prompt_names.remove('mechanistic')
    return prompt_names


def perform_blocked_statistical_tests(all_results: dict, prompt_names: List[str]) -> dict:
    """
    Perform prompt-effect testing using the *correct* repeated-measures unit:
      - Block = (CLD, run/seed)
      - Treatment = prompt variant
    
    This mirrors the approach used in `analyze_rq1a_ground_truth_enhanced.py`:
      - Global Friedman test (blocked)
      - Post-hoc paired Wilcoxon (Bonferroni-adjusted)
      - Per-CLD paired Wilcoxon across runs (exploratory, uncorrected)
    
    Returns a dict compatible with downstream reporting:
      - friedman_tests: list of rows (metric, chi2, p, W, n_blocks)
      - posthoc_tests: list of rows (metric, comparison, p, p_adj, sig, effect_r, cohens_dz, winner)
      - per_cld_posthoc_tests: list of rows (metric, cld, comparison, p, sig, winner)
    """
    results = {
        'friedman_tests': [],
        'posthoc_tests': [],
        'per_cld_posthoc_tests': [],
    }
    if not all_results or not prompt_names or len(prompt_names) < 2:
        return results
    
    metric_map = {
        'Precision Mean': 'precision',
        'Recall Mean': 'recall',
        'F1 Mean': 'f1',
        'Accuracy Mean': 'accuracy',
        'AUC-ROC Mean': 'auc_roc',
    }
    
    for metric_label, metric_key in metric_map.items():
        test = compute_prompt_effect_test(all_results, metric_key=metric_key, prompt_names=prompt_names)
        if not test.get("ok", False):
            continue
        
        results['friedman_tests'].append({
            'Metric': metric_label,
            'Chi-square': float(test.get("stat", np.nan)),
            'P-value': float(test.get("p", np.nan)),
            "Kendall's W": float(test.get("kendall_w", np.nan)),
            'N Blocks': int(test.get("n_blocks", 0)),
            'K Prompts': int(test.get("k_prompts", 0)),
            'Alpha (Bonferroni)': float(test.get("alpha_bonf", np.nan)),
        })
        
        # Global post-hoc (Bonferroni-adjusted)
        posthoc = test.get("posthoc", {}) or {}
        for pair_name, pair_data in posthoc.items():
            # Pair names are like "baseline_vs_cot"
            try:
                p1, p2 = pair_name.split("_vs_")
            except Exception:
                p1, p2 = pair_name, ""
            comparison = f"{p1} vs {p2}"
            
            # Compute paired Cohen's dz from the block data (mean(diff)/std(diff))
            cohens_dz = np.nan
            try:
                # Reconstruct paired vectors from the stored block data matrix by re-running
                # a minimal extraction for this metric/pair.
                # (Keeps code simple and avoids storing raw matrices in the dict.)
                block_vals_1 = []
                block_vals_2 = []
                for cld_name, runs in all_results.items():
                    for run_name, files in runs.items():
                        if p1 in files and p2 in files:
                            m1 = extract_detailed_performance(files[p1]).get(metric_key, np.nan)
                            m2 = extract_detailed_performance(files[p2]).get(metric_key, np.nan)
                            if not np.isnan(m1) and not np.isnan(m2):
                                block_vals_1.append(float(m1))
                                block_vals_2.append(float(m2))
                if len(block_vals_1) >= 2 and len(block_vals_1) == len(block_vals_2):
                    diffs = np.array(block_vals_1) - np.array(block_vals_2)
                    sd = np.std(diffs, ddof=1)
                    cohens_dz = float(np.mean(diffs) / sd) if sd > 0 else np.nan
            except Exception:
                cohens_dz = np.nan
            
            results['posthoc_tests'].append({
                'Metric': metric_label,
                'Comparison': comparison,
                'P-value': float(pair_data.get("p", np.nan)),
                'P-adj (Bonferroni)': float(pair_data.get("p_adj", np.nan)),
                'Significant (Bonferroni)': bool(pair_data.get("sig", False)),
                'Wilcoxon r': float(pair_data.get("effect_r", np.nan)),
                # Backwards compatible column name expected by existing visualization code:
                # prior implementation reported "Cohens d" from CLD-only tests.
                # Here we report paired Cohen's dz (within-block standardized mean difference).
                'Cohens d': cohens_dz,
                "Cohen's dz": cohens_dz,
                'Mean diff (p1 - p2)': float(pair_data.get("mean_diff", np.nan)),
                'Winner': pair_data.get("winner", ""),
            })
        
        # Per-CLD post-hoc (exploratory, uncorrected p<0.05)
        per_cld = test.get("per_cld_posthoc", {}) or {}
        for cld_name, pairs in per_cld.items():
            for pair_name, pair_data in pairs.items():
                try:
                    p1, p2 = pair_name.split("_vs_")
                except Exception:
                    p1, p2 = pair_name, ""
                results['per_cld_posthoc_tests'].append({
                    'Metric': metric_label,
                    'CLD': cld_name,
                    'Comparison': f"{p1} vs {p2}",
                    'P-value': float(pair_data.get("p", np.nan)),
                    'Significant (p<0.05)': bool(pair_data.get("sig", False)),
                    'Winner': pair_data.get("winner", ""),
                    'Mean diff (p1 - p2)': float(pair_data.get("mean_diff", np.nan)),
                })
    
    return results


def extract_detailed_performance(excel_path: Path) -> dict:
    """
    Extract comprehensive performance metrics including AUC-ROC.
    
    Returns dict with:
    - Basic metrics: precision, recall, F1, accuracy
    - AUC-ROC (if sklearn available)
    - Point-biserial correlation
    - Confusion matrix
    - Per-edge data for further analysis
    """
    try:
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        
        # Extract data
        data = []
        for _, row in df.iterrows():
            is_corrupted = row.get('Is Corrupted', False)
            aggregate_score = row.get('Aggregate Score', np.nan)
            corruption_subtype = row.get('Corruption Subtype', '')
            
            judge_msg = row.get('Judge Message', '')
            try:
                judge_data = json.loads(judge_msg)
                aggregate_verdict = judge_data.get('aggregate_verdict', 'UNKNOWN')
                if pd.isna(aggregate_score):
                    aggregate_score = judge_data.get('aggregate_score', np.nan)
            except:
                aggregate_verdict = 'UNKNOWN'
            
            data.append({
                'is_corrupted': is_corrupted,
                'aggregate_score': aggregate_score,
                'aggregate_verdict': aggregate_verdict,
                'corruption_subtype': corruption_subtype
            })
        
        df_data = pd.DataFrame(data)
        df_data['ground_truth'] = df_data['is_corrupted'].fillna(False).astype(int)
        
        # Score-based threshold: score <= 0.5 = hallucination (1), score > 0.5 = correct (0)
        # Handle NaN scores by defaulting to 0.5 (treated as hallucination)
        df_data['aggregate_score'] = df_data['aggregate_score'].fillna(0.5)
        df_data['judge_pred'] = (df_data['aggregate_score'] <= 0.5).astype(int)
        
        # Confusion matrix
        tp = ((df_data['ground_truth'] == 1) & (df_data['judge_pred'] == 1)).sum()
        tn = ((df_data['ground_truth'] == 0) & (df_data['judge_pred'] == 0)).sum()
        fp = ((df_data['ground_truth'] == 0) & (df_data['judge_pred'] == 1)).sum()
        fn = ((df_data['ground_truth'] == 1) & (df_data['judge_pred'] == 0)).sum()
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(df_data) if len(df_data) > 0 else 0.0
        
        # AUC-ROC
        auc_roc = np.nan
        if SKLEARN_AVAILABLE:
            scores_clean = df_data[df_data['aggregate_score'].notna()].copy()
            if len(scores_clean) >= 2 and scores_clean['ground_truth'].nunique() == 2:
                try:
                    # For AUC, we want P(corrupted), so use 1 - score if lower scores indicate corruption
                    auc_roc = roc_auc_score(scores_clean['ground_truth'], 
                                           1 - scores_clean['aggregate_score'])
                except:
                    auc_roc = np.nan
        
        # Point-biserial correlation
        scores_clean = df_data[df_data['aggregate_score'].notna()].copy()
        if len(scores_clean) >= 2:
            r_pb, pb_p = stats.pointbiserialr(scores_clean['ground_truth'], 
                                             scores_clean['aggregate_score'])
        else:
            r_pb, pb_p = 0.0, 1.0
        
        # Calculate mean scores by classification type
        tp_scores = df_data[(df_data['ground_truth'] == 1) & (df_data['judge_pred'] == 1)]['aggregate_score']
        tn_scores = df_data[(df_data['ground_truth'] == 0) & (df_data['judge_pred'] == 0)]['aggregate_score']
        fp_scores = df_data[(df_data['ground_truth'] == 0) & (df_data['judge_pred'] == 1)]['aggregate_score']
        fn_scores = df_data[(df_data['ground_truth'] == 1) & (df_data['judge_pred'] == 0)]['aggregate_score']
        
        tp_mean = tp_scores.mean() if len(tp_scores) > 0 else np.nan
        tn_mean = tn_scores.mean() if len(tn_scores) > 0 else np.nan
        fp_mean = fp_scores.mean() if len(fp_scores) > 0 else np.nan
        fn_mean = fn_scores.mean() if len(fn_scores) > 0 else np.nan
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp_mean_score': tp_mean,
            'tn_mean_score': tn_mean,
            'fp_mean_score': fp_mean,
            'fn_mean_score': fn_mean,
            'n_total': len(df_data),
            'n_corrupted': int(df_data['ground_truth'].sum()),
            'n_clean': int((df_data['ground_truth'] == 0).sum()),
            'point_biserial_r': r_pb,
            'pb_p_value': pb_p,
            'file_path': str(excel_path),
            'per_edge_data': df_data  # Keep for corruption-type analysis
        }
    
    except Exception as e:
        print(f"Error extracting from {excel_path}: {e}")
        return {}


def calculate_corruption_type_metrics(df_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate metrics per corruption type.
    
    Returns DataFrame with columns:
    - corruption_type, precision, recall, f1, accuracy, n_total, n_detected, n_missed
    """
    corruption_metrics = []
    
    # Filter to corrupted edges only
    corrupted = df_data[df_data['ground_truth'] == 1].copy()
    
    if len(corrupted) == 0:
        return pd.DataFrame()
    
    for corruption_type in corrupted['corruption_subtype'].unique():
        if not corruption_type or pd.isna(corruption_type):
            continue
        
        subset = corrupted[corrupted['corruption_subtype'] == corruption_type]
        n_total = len(subset)
        n_detected = subset['judge_pred'].sum()
        n_missed = n_total - int(n_detected)
        recall = n_detected / n_total if n_total > 0 else 0.0
        
        corruption_metrics.append({
            'corruption_type': corruption_type,
            'n_total': n_total,
            'n_detected': int(n_detected),
            'n_missed': n_missed,
            'recall': recall
        })
    
    return pd.DataFrame(corruption_metrics)


def aggregate_by_cld_and_prompt(all_results: dict) -> pd.DataFrame:
    """
    Aggregate metrics across runs for each CLD/prompt combination.
    Includes AUC-ROC, confidence intervals, and sample sizes.
    """
    aggregate_data = []
    
    prompt_labels = {'baseline': 'Baseline', 'mechanistic': 'Mechanistic', 'mechanistic_original': 'Mechanistic', 'cot': 'CoT', 'mechanistic_lit': 'Mech-Lit'}
    
    for cld_name, runs in all_results.items():
        for prompt_name in ['baseline', 'mechanistic', 'mechanistic_original', 'cot', 'mechanistic_lit']:
            metrics_list = []
            
            for run_name, files in runs.items():
                if prompt_name in files:
                    metrics = extract_detailed_performance(files[prompt_name])
                    if metrics:
                        metrics_list.append(metrics)
            
            if not metrics_list:
                continue
            
            n_runs = len(metrics_list)
            
            # Calculate means and std devs
            precision_vals = [m['precision'] for m in metrics_list]
            recall_vals = [m['recall'] for m in metrics_list]
            f1_vals = [m['f1'] for m in metrics_list]
            accuracy_vals = [m['accuracy'] for m in metrics_list]
            auc_vals = [m['auc_roc'] for m in metrics_list if not np.isnan(m['auc_roc'])]
            rpb_vals = [m['point_biserial_r'] for m in metrics_list]
            rpb_p_vals = [m['pb_p_value'] for m in metrics_list]
            
            # Classification score means
            tp_score_vals = [m['tp_mean_score'] for m in metrics_list if not np.isnan(m.get('tp_mean_score', np.nan))]
            tn_score_vals = [m['tn_mean_score'] for m in metrics_list if not np.isnan(m.get('tn_mean_score', np.nan))]
            fp_score_vals = [m['fp_mean_score'] for m in metrics_list if not np.isnan(m.get('fp_mean_score', np.nan))]
            fn_score_vals = [m['fn_mean_score'] for m in metrics_list if not np.isnan(m.get('fn_mean_score', np.nan))]
            
            # Classification counts
            tp_counts = [m['tp'] for m in metrics_list]
            tn_counts = [m['tn'] for m in metrics_list]
            fp_counts = [m['fp'] for m in metrics_list]
            fn_counts = [m['fn'] for m in metrics_list]
            
            # Confidence intervals for ALL metrics
            prec_ci = calculate_confidence_interval(np.array(precision_vals))
            rec_ci = calculate_confidence_interval(np.array(recall_vals))
            f1_ci = calculate_confidence_interval(np.array(f1_vals))
            acc_ci = calculate_confidence_interval(np.array(accuracy_vals))
            auc_ci = calculate_confidence_interval(np.array(auc_vals)) if auc_vals else (np.nan, np.nan)
            rpb_ci = calculate_confidence_interval(np.array(rpb_vals))
            
            # CIs for classification scores
            tp_score_ci = calculate_confidence_interval(np.array(tp_score_vals)) if tp_score_vals else (np.nan, np.nan)
            tn_score_ci = calculate_confidence_interval(np.array(tn_score_vals)) if tn_score_vals else (np.nan, np.nan)
            fp_score_ci = calculate_confidence_interval(np.array(fp_score_vals)) if fp_score_vals else (np.nan, np.nan)
            fn_score_ci = calculate_confidence_interval(np.array(fn_score_vals)) if fn_score_vals else (np.nan, np.nan)
            
            # Calculate CI halfwidths for error bars
            prec_ci_hw = calculate_ci_halfwidth(np.array(precision_vals))
            rec_ci_hw = calculate_ci_halfwidth(np.array(recall_vals))
            f1_ci_hw = calculate_ci_halfwidth(np.array(f1_vals))
            acc_ci_hw = calculate_ci_halfwidth(np.array(accuracy_vals))
            auc_ci_hw = calculate_ci_halfwidth(np.array(auc_vals)) if auc_vals else 0.0
            rpb_ci_hw = calculate_ci_halfwidth(np.array(rpb_vals))
            
            # Calculate aggregate p-value for point-biserial r using t-test against 0
            # This tests if the mean r across runs is significantly different from 0
            if len(rpb_vals) > 1:
                try:
                    _, rpb_agg_p_value = stats.ttest_1samp(rpb_vals, 0)
                except Exception:
                    # Fallback if t-test fails (e.g. all values identical)
                    rpb_agg_p_value = np.mean(rpb_p_vals)
            else:
                rpb_agg_p_value = np.mean(rpb_p_vals)  # Fallback to mean of single run
                
            tp_score_ci_hw = calculate_ci_halfwidth(np.array(tp_score_vals)) if tp_score_vals else 0.0
            tn_score_ci_hw = calculate_ci_halfwidth(np.array(tn_score_vals)) if tn_score_vals else 0.0
            fp_score_ci_hw = calculate_ci_halfwidth(np.array(fp_score_vals)) if fp_score_vals else 0.0
            fn_score_ci_hw = calculate_ci_halfwidth(np.array(fn_score_vals)) if fn_score_vals else 0.0
            
            aggregate_data.append({
                'CLD': cld_name,
                'Prompt': prompt_labels[prompt_name],
                'N Runs': n_runs,
                'N Edges': int(np.mean([m['n_total'] for m in metrics_list])),
                'N Corrupted': int(np.mean([m['n_corrupted'] for m in metrics_list])),
                'N Clean': int(np.mean([m['n_clean'] for m in metrics_list])),
                'Precision Mean': np.mean(precision_vals),
                'Precision Std': np.std(precision_vals, ddof=1) if len(precision_vals) > 1 else 0,
                'Precision CI Low': prec_ci[0],
                'Precision CI High': prec_ci[1],
                'Precision CI Halfwidth': prec_ci_hw,
                'Recall Mean': np.mean(recall_vals),
                'Recall Std': np.std(recall_vals, ddof=1) if len(recall_vals) > 1 else 0,
                'Recall CI Low': rec_ci[0],
                'Recall CI High': rec_ci[1],
                'Recall CI Halfwidth': rec_ci_hw,
                'F1 Mean': np.mean(f1_vals),
                'F1 Std': np.std(f1_vals, ddof=1) if len(f1_vals) > 1 else 0,
                'F1 CI Low': f1_ci[0],
                'F1 CI High': f1_ci[1],
                'F1 CI Halfwidth': f1_ci_hw,
                'Accuracy Mean': np.mean(accuracy_vals),
                'Accuracy Std': np.std(accuracy_vals, ddof=1) if len(accuracy_vals) > 1 else 0,
                'Accuracy CI Low': acc_ci[0],
                'Accuracy CI High': acc_ci[1],
                'Accuracy CI Halfwidth': acc_ci_hw,
                'AUC-ROC Mean': np.mean(auc_vals) if auc_vals else np.nan,
                'AUC-ROC Std': np.std(auc_vals, ddof=1) if len(auc_vals) > 1 else 0,
                'AUC-ROC CI Low': auc_ci[0],
                'AUC-ROC CI High': auc_ci[1],
                'AUC-ROC CI Halfwidth': auc_ci_hw,
                'Point-Biserial r Mean': np.mean(rpb_vals),
                'Point-Biserial r Std': np.std(rpb_vals, ddof=1) if len(rpb_vals) > 1 else 0,
                'Point-Biserial r CI Low': rpb_ci[0],
                'Point-Biserial r CI High': rpb_ci[1],
                'Point-Biserial r CI Halfwidth': rpb_ci_hw,
                'Point-Biserial r P-value Mean': np.mean(rpb_p_vals),
                'Point-Biserial r Aggregate P-value': rpb_agg_p_value,
                'Point-Biserial r Significant': all(p < 0.001 for p in rpb_p_vals),  # All runs p < 0.001
                'TP Mean Score Mean': np.mean(tp_score_vals) if tp_score_vals else np.nan,
                'TP Mean Score Std': np.std(tp_score_vals, ddof=1) if len(tp_score_vals) > 1 else 0,
                'TP Mean Score CI Low': tp_score_ci[0],
                'TP Mean Score CI High': tp_score_ci[1],
                'TP Mean Score CI Halfwidth': tp_score_ci_hw,
                'TN Mean Score Mean': np.mean(tn_score_vals) if tn_score_vals else np.nan,
                'TN Mean Score Std': np.std(tn_score_vals, ddof=1) if len(tn_score_vals) > 1 else 0,
                'TN Mean Score CI Low': tn_score_ci[0],
                'TN Mean Score CI High': tn_score_ci[1],
                'TN Mean Score CI Halfwidth': tn_score_ci_hw,
                'FP Mean Score Mean': np.mean(fp_score_vals) if fp_score_vals else np.nan,
                'FP Mean Score Std': np.std(fp_score_vals, ddof=1) if len(fp_score_vals) > 1 else 0,
                'FP Mean Score CI Low': fp_score_ci[0],
                'FP Mean Score CI High': fp_score_ci[1],
                'FP Mean Score CI Halfwidth': fp_score_ci_hw,
                'FN Mean Score Mean': np.mean(fn_score_vals) if fn_score_vals else np.nan,
                'FN Mean Score Std': np.std(fn_score_vals, ddof=1) if len(fn_score_vals) > 1 else 0,
                'FN Mean Score CI Low': fn_score_ci[0],
                'FN Mean Score CI High': fn_score_ci[1],
                'FN Mean Score CI Halfwidth': fn_score_ci_hw,
                'N_TP': int(np.mean(tp_counts)),
                'N_TN': int(np.mean(tn_counts)),
                'N_FP': int(np.mean(fp_counts)),
                'N_FN': int(np.mean(fn_counts))
            })
    
    return pd.DataFrame(aggregate_data)


def aggregate_by_prompt(all_results: dict) -> pd.DataFrame:
    """Aggregate metrics across all CLDs for each prompt."""
    aggregate_data = []
    prompt_labels = {'baseline': 'Baseline', 'mechanistic': 'Mechanistic', 'mechanistic_original': 'Mechanistic', 'cot': 'CoT', 'mechanistic_lit': 'Mech-Lit'}
    
    for prompt_name in ['baseline', 'mechanistic', 'mechanistic_original', 'cot', 'mechanistic_lit']:
        all_metrics = []
        
        for cld_name, runs in all_results.items():
            for run_name, files in runs.items():
                if prompt_name in files:
                    metrics = extract_detailed_performance(files[prompt_name])
                    if metrics:
                        all_metrics.append(metrics)
        
        if not all_metrics:
            continue
        
        # Calculate aggregates
        precision_vals = [m['precision'] for m in all_metrics]
        recall_vals = [m['recall'] for m in all_metrics]
        f1_vals = [m['f1'] for m in all_metrics]
        accuracy_vals = [m['accuracy'] for m in all_metrics]
        auc_vals = [m['auc_roc'] for m in all_metrics if not np.isnan(m['auc_roc'])]
        rpb_vals = [m['point_biserial_r'] for m in all_metrics]
        
        n_clds = len(set(cld for cld in all_results.keys()))
        n_runs = len(all_metrics)
        
        aggregate_data.append({
            'Prompt': prompt_labels[prompt_name],
            'N CLDs': n_clds,
            'N Runs': n_runs,
            'N Edges': int(np.mean([m['n_total'] for m in all_metrics])),
            'Precision': format_mean_minmax(precision_vals),
            'Recall': format_mean_minmax(recall_vals),
            'F1': format_mean_minmax(f1_vals),
            'Accuracy': format_mean_minmax(accuracy_vals),
            'AUC-ROC': format_mean_minmax(auc_vals) if auc_vals else "N/A",
            'Point-Biserial r': format_mean_minmax(rpb_vals)
        })
    
    return pd.DataFrame(aggregate_data)


def aggregate_by_cld(all_results: dict) -> pd.DataFrame:
    """Aggregate metrics across all prompts for each CLD."""
    aggregate_data = []
    
    for cld_name, runs in all_results.items():
        all_metrics = []
        
        for run_name, files in runs.items():
            for prompt_name in ['baseline', 'mechanistic', 'mechanistic_original', 'cot', 'mechanistic_lit']:
                if prompt_name in files:
                    metrics = extract_detailed_performance(files[prompt_name])
                    if metrics:
                        all_metrics.append(metrics)
        
        if not all_metrics:
            continue
        
        precision_vals = [m['precision'] for m in all_metrics]
        recall_vals = [m['recall'] for m in all_metrics]
        f1_vals = [m['f1'] for m in all_metrics]
        accuracy_vals = [m['accuracy'] for m in all_metrics]
        auc_vals = [m['auc_roc'] for m in all_metrics if not np.isnan(m['auc_roc'])]
        rpb_vals = [m['point_biserial_r'] for m in all_metrics]
        
        n_prompts = 4  # baseline, mechanistic, cot, mechanistic_lit
        n_runs = len(all_metrics) // n_prompts
        
        aggregate_data.append({
            'CLD': cld_name,
            'N Prompts': n_prompts,
            'N Runs': n_runs,
            'N Edges': int(np.mean([m['n_total'] for m in all_metrics])),
            'Precision': format_mean_minmax(precision_vals),
            'Recall': format_mean_minmax(recall_vals),
            'F1': format_mean_minmax(f1_vals),
            'Accuracy': format_mean_minmax(accuracy_vals),
            'AUC-ROC': format_mean_minmax(auc_vals) if auc_vals else "N/A",
            'Point-Biserial r': format_mean_minmax(rpb_vals)
        })
    
    return pd.DataFrame(aggregate_data)


def calculate_overall_aggregate(all_results: dict) -> dict:
    """Calculate grand mean across all CLDs and prompts."""
    all_metrics = []
    
    for cld_name, runs in all_results.items():
        for run_name, files in runs.items():
            for prompt_name in ['baseline', 'mechanistic', 'mechanistic_original', 'cot', 'mechanistic_lit']:
                if prompt_name in files:
                    metrics = extract_detailed_performance(files[prompt_name])
                    if metrics:
                        all_metrics.append(metrics)
    
    if not all_metrics:
        return {}
    
    precision_vals = [m['precision'] for m in all_metrics]
    recall_vals = [m['recall'] for m in all_metrics]
    f1_vals = [m['f1'] for m in all_metrics]
    accuracy_vals = [m['accuracy'] for m in all_metrics]
    auc_vals = [m['auc_roc'] for m in all_metrics if not np.isnan(m['auc_roc'])]
    rpb_vals = [m['point_biserial_r'] for m in all_metrics]
    
    return {
        'N Total Runs': len(all_metrics),
        'N Total Edges': int(np.mean([m['n_total'] for m in all_metrics])),
        'Precision': format_mean_minmax(precision_vals),
        'Recall': format_mean_minmax(recall_vals),
        'F1': format_mean_minmax(f1_vals),
        'Accuracy': format_mean_minmax(accuracy_vals),
        'AUC-ROC': format_mean_minmax(auc_vals) if auc_vals else "N/A",
        'Point-Biserial r': format_mean_minmax(rpb_vals)
    }


def aggregate_corruption_type_detailed(all_results: dict) -> pd.DataFrame:
    """Aggregate per-corruption-type metrics across all runs and CLDs."""
    corruption_data = []
    
    prompt_labels = {'baseline': 'Baseline', 'mechanistic': 'Mechanistic', 'mechanistic_original': 'Mechanistic', 'cot': 'CoT', 'mechanistic_lit': 'Mech-Lit'}
    
    for prompt_name in ['baseline', 'mechanistic', 'mechanistic_original', 'cot', 'mechanistic_lit']:
        corruption_metrics_by_type = {}
        
        for cld_name, runs in all_results.items():
            for run_name, files in runs.items():
                if prompt_name in files:
                    metrics = extract_detailed_performance(files[prompt_name])
                    if metrics and 'per_edge_data' in metrics:
                        type_metrics = calculate_corruption_type_metrics(metrics['per_edge_data'])
                        
                        for _, row in type_metrics.iterrows():
                            ctype = row['corruption_type']
                            if ctype not in corruption_metrics_by_type:
                                corruption_metrics_by_type[ctype] = []
                            corruption_metrics_by_type[ctype].append(row.to_dict())
        
        # Aggregate per type
        for ctype, metrics_list in corruption_metrics_by_type.items():
            recalls = [m['recall'] for m in metrics_list]
            n_total = sum(m['n_total'] for m in metrics_list)
            n_detected = sum(m['n_detected'] for m in metrics_list)
            
            # Calculate 95% CI
            recall_mean = np.mean(recalls)
            recall_ci_low, recall_ci_high = calculate_confidence_interval(np.array(recalls))
            
            corruption_data.append({
                'Prompt': prompt_labels[prompt_name],
                'Corruption Type': ctype,
                'N Corrupted': n_total,
                'Recall Mean': recall_mean,
                'Recall Std': np.std(recalls, ddof=1) if len(recalls) > 1 else 0,
                'Recall CI Low': recall_ci_low,
                'Recall CI High': recall_ci_high,
                'Total Detected': n_detected,
                'Total Missed': n_total - n_detected,
                'N Runs': len(metrics_list)
            })
    
    return pd.DataFrame(corruption_data)


# Statistical testing functions will be continued in the next section...


def perform_assumption_tests(aggregate_df: pd.DataFrame) -> dict:
    """
    Test statistical assumptions for parametric vs non-parametric tests.
    
    Tests:
    1. Shapiro-Wilk for normality
    2. Levene's test for homogeneity of variances
    """
    results = {
        'normality_tests': [],
        'variance_tests': []
    }
    
    print("\n" + "="*80)
    print("ASSUMPTION TESTING")
    print("="*80)
    
    # Test normality for each metric per prompt
    # Keep this list aligned with the key outcome metrics used throughout RQ1a,
    # especially F1 and AUC-ROC, so both RQ1a analysis scripts report the same assumption checks.
    metrics = ['Precision Mean', 'Recall Mean', 'F1 Mean', 'Accuracy Mean', 'AUC-ROC Mean']
    prompts = aggregate_df['Prompt'].unique()
    
    print("\n1. SHAPIRO-WILK TEST FOR NORMALITY")
    print("-" * 80)
    print("H0: Data is normally distributed (if p < 0.05, reject normality)")
    
    for metric in metrics:
        print(f"\n{metric}:")
        for prompt in prompts:
            data = aggregate_df[aggregate_df['Prompt'] == prompt][metric].values
            if len(data) >= 3:
                stat, p_value = shapiro(data)
                is_normal = p_value >= 0.05
                results['normality_tests'].append({
                    'Metric': metric,
                    'Prompt': prompt,
                    'W-statistic': stat,
                    'P-value': p_value,
                    'Normal': is_normal
                })
                status = "✓ Normal" if is_normal else "✗ Not normal"
                print(f"  {prompt}: W={stat:.4f}, p={p_value:.4f} {status}")
    
    # Test homogeneity of variances
    print("\n2. LEVENE'S TEST FOR HOMOGENEITY OF VARIANCES")
    print("-" * 80)
    print("H0: Variances are equal across prompts (if p < 0.05, reject homogeneity)")
    
    for metric in metrics:
        groups = [aggregate_df[aggregate_df['Prompt'] == prompt][metric].values 
                 for prompt in prompts]
        groups = [g for g in groups if len(g) >= 2]
        
        if len(groups) >= 2:
            stat, p_value = levene(*groups)
            is_homogeneous = p_value >= 0.05
            results['variance_tests'].append({
                'Metric': metric,
                'Statistic': stat,
                'P-value': p_value,
                'Homogeneous': is_homogeneous
            })
            status = "✓ Homogeneous" if is_homogeneous else "✗ Heterogeneous"
            print(f"{metric}: W={stat:.4f}, p={p_value:.4f} {status}")
    
    return results


def perform_statistical_tests(aggregate_df: pd.DataFrame) -> dict:
    """
    Perform statistical significance testing.
    
    Tests:
    1. Friedman test (non-parametric repeated measures ANOVA)
    2. Post-hoc Wilcoxon signed-rank tests
    3. Effect sizes
    """
    results = {
        'friedman_tests': [],
        'posthoc_tests': [],
        'effect_sizes': []
    }
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    
    metrics = ['Precision Mean', 'Recall Mean', 'F1 Mean', 'Accuracy Mean', 'AUC-ROC Mean']
    prompts = ['Baseline', 'Mechanistic', 'SCE']
    
    # Friedman test
    print("\n1. FRIEDMAN TEST (Non-parametric Repeated Measures ANOVA)")
    print("-" * 80)
    print("H0: No difference in prompt performance (if p < 0.05, prompts differ)")
    
    for metric in metrics:
        if metric not in aggregate_df.columns:
            continue
        
        # Get data for each prompt across CLDs
        data_by_prompt = {}
        for prompt in prompts:
            data = aggregate_df[aggregate_df['Prompt'] == prompt][metric].dropna().values
            if len(data) > 0:
                data_by_prompt[prompt] = data
        
        if len(data_by_prompt) >= 3:
            groups = [data_by_prompt[p] for p in prompts if p in data_by_prompt]
            # Ensure all groups have same length for Friedman
            min_len = min(len(g) for g in groups)
            groups = [g[:min_len] for g in groups]
            
            if min_len >= 2:
                try:
                    stat, p_value = friedmanchisquare(*groups)
                    results['friedman_tests'].append({
                        'Metric': metric,
                        'Chi-square': stat,
                        'P-value': p_value,
                        'Significant': p_value < 0.05
                    })
                    sig_str = "*** SIGNIFICANT" if p_value < 0.001 else "** SIGNIFICANT" if p_value < 0.01 else "* SIGNIFICANT" if p_value < 0.05 else "ns (not significant)"
                    print(f"{metric}: χ²={stat:.4f}, p={p_value:.4f} {sig_str}")
                except Exception as e:
                    print(f"{metric}: Error in Friedman test - {e}")
    
    # Post-hoc pairwise comparisons
    print("\n2. POST-HOC PAIRWISE COMPARISONS (Wilcoxon Signed-Rank)")
    print("-" * 80)
    pairs = [('Baseline', 'Mechanistic'), ('Baseline', 'CoT'), ('Mechanistic', 'CoT')]
    n_pairs = len(pairs)
    alpha_adj_pairs = 0.05 / n_pairs
    print(f"Bonferroni correction: α = 0.05/{n_pairs} = {alpha_adj_pairs:.4f}")
    
    for metric in metrics:
        if metric not in aggregate_df.columns:
            continue
        
        print(f"\n{metric}:")
        for p1, p2 in pairs:
            data1 = aggregate_df[aggregate_df['Prompt'] == p1][metric].dropna().values
            data2 = aggregate_df[aggregate_df['Prompt'] == p2][metric].dropna().values
            
            min_len = min(len(data1), len(data2))
            if min_len >= 2:
                try:
                    stat, p_value = wilcoxon(data1[:min_len], data2[:min_len])
                    
                    # Effect size (Wilcoxon r = Z / sqrt(N))
                    z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 3.0
                    effect_size = z_score / np.sqrt(min_len)
                    
                    # Cohen's d
                    mean_diff = np.mean(data1[:min_len]) - np.mean(data2[:min_len])
                    pooled_std = np.sqrt((np.std(data1[:min_len], ddof=1)**2 + np.std(data2[:min_len], ddof=1)**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    results['posthoc_tests'].append({
                        'Metric': metric,
                        'Comparison': f"{p1} vs {p2}",
                        'W-statistic': stat,
                        'P-value': p_value,
                        'Significant (α=0.0167)': p_value < 0.0167,
                        'Wilcoxon r': effect_size,
                        'Cohens d': cohens_d
                    })
                    
                    sig_str = "*** SIG" if p_value < 0.001 else "** SIG" if p_value < 0.01 else "* SIG" if p_value < 0.0167 else "ns"
                    print(f"  {p1} vs {p2}: W={stat:.1f}, p={p_value:.4f} {sig_str}, d={cohens_d:.3f}")
                except Exception as e:
                    print(f"  {p1} vs {p2}: Error - {e}")
    
    return results


# This will be inserted after the perform_statistical_tests function

def perform_parametric_tests(aggregate_df: pd.DataFrame) -> dict:
    """
    Perform parametric statistical tests (Repeated Measures ANOVA + Paired t-tests).
    
    Tests:
    1. Repeated Measures ANOVA (one-way within-subjects)
    2. Paired t-tests with Bonferroni correction
    3. Effect sizes (Cohen's d, Eta-squared)
    """
    from scipy.stats import f_oneway, ttest_rel
    
    results = {
        'rm_anova_tests': [],
        'paired_t_tests': [],
        'effect_sizes': []
    }
    
    print("\n" + "="*80)
    print("PARAMETRIC STATISTICAL TESTING")
    print("="*80)
    print("(Using parametric tests since assumptions are met)")
    
    metrics = ['Precision Mean', 'Recall Mean', 'F1 Mean', 'Accuracy Mean', 'AUC-ROC Mean']
    prompts = ['Baseline', 'Mechanistic', 'SCE']
    
    # 1. Repeated Measures ANOVA (approximated with one-way ANOVA on differences)
    print("\n1. REPEATED MEASURES ANOVA")
    print("-" * 80)
    print("H0: No difference in prompt performance (if p < 0.05, prompts differ)")
    
    for metric in metrics:
        if metric not in aggregate_df.columns:
            continue
        
        # Get data for each prompt across CLDs
        data_by_prompt = {}
        for prompt in prompts:
            data = aggregate_df[aggregate_df['Prompt'] == prompt][metric].dropna().values
            if len(data) > 0:
                data_by_prompt[prompt] = data
        
        if len(data_by_prompt) >= 3:
            groups = [data_by_prompt[p] for p in prompts if p in data_by_prompt]
            # Ensure all groups have same length
            min_len = min(len(g) for g in groups)
            groups = [g[:min_len] for g in groups]
            
            if min_len >= 2:
                try:
                    # One-way ANOVA (treating as between-subjects for simplicity)
                    f_stat, p_value = f_oneway(*groups)
                    
                    # Calculate eta-squared (effect size for ANOVA)
                    all_data = np.concatenate(groups)
                    grand_mean = np.mean(all_data)
                    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
                    ss_total = sum((x - grand_mean)**2 for x in all_data)
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    
                    results['rm_anova_tests'].append({
                        'Metric': metric,
                        'F-statistic': f_stat,
                        'P-value': p_value,
                        'Eta-squared': eta_squared,
                        'Significant': p_value < 0.05
                    })
                    
                    sig_str = "*** SIGNIFICANT" if p_value < 0.001 else "** SIGNIFICANT" if p_value < 0.01 else "* SIGNIFICANT" if p_value < 0.05 else "ns (not significant)"
                    print(f"{metric}: F={f_stat:.4f}, p={p_value:.4f}, η²={eta_squared:.4f} {sig_str}")
                except Exception as e:
                    print(f"{metric}: Error in ANOVA - {e}")
    
    # 2. Paired t-tests
    print("\n2. PAIRED T-TESTS (Post-hoc Comparisons)")
    print("-" * 80)
    pairs = [('Baseline', 'Mechanistic'), ('Baseline', 'CoT'), ('Mechanistic', 'CoT')]
    n_pairs = len(pairs)
    alpha_adj_pairs = 0.05 / n_pairs
    print(f"Bonferroni correction: α = 0.05/{n_pairs} = {alpha_adj_pairs:.4f}")
    
    for metric in metrics:
        if metric not in aggregate_df.columns:
            continue
        
        print(f"\n{metric}:")
        for p1, p2 in pairs:
            data1 = aggregate_df[aggregate_df['Prompt'] == p1][metric].dropna().values
            data2 = aggregate_df[aggregate_df['Prompt'] == p2][metric].dropna().values
            
            min_len = min(len(data1), len(data2))
            if min_len >= 2:
                try:
                    # Paired t-test
                    t_stat, p_value = ttest_rel(data1[:min_len], data2[:min_len])
                    
                    # Cohen's d for paired samples
                    mean_diff = np.mean(data1[:min_len] - data2[:min_len])
                    std_diff = np.std(data1[:min_len] - data2[:min_len], ddof=1)
                    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
                    
                    # 95% confidence interval for mean difference
                    from scipy import stats as sp_stats
                    ci = sp_stats.t.interval(0.95, min_len-1, 
                                            loc=mean_diff, 
                                            scale=std_diff/np.sqrt(min_len))
                    
                    results['paired_t_tests'].append({
                        'Metric': metric,
                        'Comparison': f"{p1} vs {p2}",
                        'T-statistic': t_stat,
                        'P-value': p_value,
                        'Significant (α=0.0167)': p_value < 0.0167,
                        'Cohens d': cohens_d,
                        'Mean Diff': mean_diff,
                        'CI Lower': ci[0],
                        'CI Upper': ci[1]
                    })
                    
                    sig_str = "*** SIG" if p_value < 0.001 else "** SIG" if p_value < 0.01 else "* SIG" if p_value < 0.0167 else "ns"
                    print(f"  {p1} vs {p2}: t={t_stat:.3f}, p={p_value:.4f} {sig_str}, d={cohens_d:.3f}, Δ={mean_diff:.4f} [95% CI: {ci[0]:.4f}, {ci[1]:.4f}]")
                except Exception as e:
                    print(f"  {p1} vs {p2}: Error - {e}")
    
    return results


def compare_parametric_nonparametric(friedman_results: dict, anova_results: dict) -> pd.DataFrame:
    """Create comparison table of parametric vs non-parametric results."""
    comparison_data = []
    
    # Map Friedman to ANOVA results by metric
    for friedman in friedman_results.get('friedman_tests', []):
        metric = friedman['Metric']
        
        # Find matching ANOVA result
        anova_match = None
        for anova in anova_results.get('rm_anova_tests', []):
            if anova['Metric'] == metric:
                anova_match = anova
                break
        
        if anova_match:
            comparison_data.append({
                'Metric': metric,
                'Friedman χ²': friedman['Chi-square'],
                'Friedman p': friedman['P-value'],
                'ANOVA F': anova_match['F-statistic'],
                'ANOVA p': anova_match['P-value'],
                'η²': anova_match['Eta-squared'],
                'Friedman Sig': '✓' if friedman['Significant'] else '✗',
                'ANOVA Sig': '✓' if anova_match['Significant'] else '✗'
            })
    
    return pd.DataFrame(comparison_data)



def generate_enhanced_latex_tables(aggregate_df: pd.DataFrame, 
                                   per_prompt_df: pd.DataFrame,
                                   per_cld_df: pd.DataFrame,
                                   corruption_type_df: pd.DataFrame,
                                   stat_results: dict) -> Dict[str, str]:
    """Generate all LaTeX tables."""
    tables = {}
    
    # Table 1: Main Results (CLD × Prompt)
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{RQ1a: Judge Performance on Corruption Detection (by CLD and Prompt)}")
    latex_lines.append("\\label{tab:rq1a_main}")
    latex_lines.append("\\begin{tabular}{llcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{CLD} & \\textbf{Prompt} & \\textbf{N} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{AUC} & \\textbf{r$_{pb}$} \\\\")
    latex_lines.append("\\midrule")
    
    for cld in aggregate_df['CLD'].unique():
        cld_data = aggregate_df[aggregate_df['CLD'] == cld].sort_values('Prompt')
        for idx, row in cld_data.iterrows():
            cld_label = cld.replace('_', ' ') if idx == cld_data.index[0] else ''
            p = f"{row['Precision Mean']:.3f}±{row['Precision Std']:.3f}"
            r = f"{row['Recall Mean']:.3f}±{row['Recall Std']:.3f}"
            f1 = f"{row['F1 Mean']:.3f}±{row['F1 Std']:.3f}"
            auc = f"{row['AUC-ROC Mean']:.3f}" if not pd.isna(row['AUC-ROC Mean']) else "N/A"
            rpb = f"{row['Point-Biserial r Mean']:.3f}"
            
            latex_lines.append(f"{cld_label} & {row['Prompt']} & {row['N Runs']} & {p} & {r} & {f1} & {auc} & {rpb} \\\\")
        latex_lines.append("\\midrule")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    tables['main'] = "\n".join(latex_lines)
    
    # Table 2: Per-Prompt Aggregate
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{RQ1a: Judge Performance Aggregated by Prompt (across all CLDs)}")
    latex_lines.append("\\label{tab:rq1a_by_prompt}")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Prompt} & \\textbf{N} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{AUC} & \\textbf{r$_{pb}$} \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in per_prompt_df.iterrows():
        latex_lines.append(f"{row['Prompt']} & {row['N Runs']} & {row['Precision']} & {row['Recall']} & {row['F1']} & {row['AUC-ROC']} & {row['Point-Biserial r']} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    tables['per_prompt'] = "\n".join(latex_lines)
    
    # Table 3: Per-CLD Aggregate
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{RQ1a: Judge Performance Aggregated by CLD (across all prompts)}")
    latex_lines.append("\\label{tab:rq1a_by_cld}")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{CLD} & \\textbf{N} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{AUC} & \\textbf{r$_{pb}$} \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in per_cld_df.iterrows():
        cld_label = row['CLD'].replace('_', ' ')
        latex_lines.append(f"{cld_label} & {row['N Runs']} & {row['Precision']} & {row['Recall']} & {row['F1']} & {row['AUC-ROC']} & {row['Point-Biserial r']} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    tables['per_cld'] = "\n".join(latex_lines)
    
    # Table 4: Corruption Type Performance
    if not corruption_type_df.empty:
        latex_lines = []
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{RQ1a: Detection Recall by Corruption Type}")
        latex_lines.append("\\label{tab:rq1a_corruption_type}")
        latex_lines.append("\\begin{tabular}{llcccc}")
        latex_lines.append("\\toprule")
        latex_lines.append("\\textbf{Prompt} & \\textbf{Corruption Type} & \\textbf{N} & \\textbf{Recall} & \\textbf{Detected} & \\textbf{Missed} \\\\")
        latex_lines.append("\\midrule")
        
        for _, row in corruption_type_df.iterrows():
            recall_str = f"{row['Recall Mean']:.3f}±{row['Recall Std']:.3f}"
            latex_lines.append(f"{row['Prompt']} & {row['Corruption Type']} & {row['N Corrupted']} & {recall_str} & {row['Total Detected']} & {row['Total Missed']} \\\\")
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        tables['corruption_type'] = "\n".join(latex_lines)
    
    return tables




def create_enhanced_visualizations(aggregate_df: pd.DataFrame,
                                  corruption_type_df: pd.DataFrame,
                                  stat_results: dict,
                                  output_dir: Path,
                                  all_results: dict):
    """Create comprehensive visualizations."""
    
    # Map Prompt column to display names for consistent legends
    if 'Prompt' in aggregate_df.columns:
        aggregate_df = aggregate_df.copy()
        print(f"[DEBUG] Prompt values BEFORE mapping: {aggregate_df['Prompt'].unique().tolist()}")
        aggregate_df['Prompt'] = aggregate_df['Prompt'].apply(get_prompt_label)
        print(f"[DEBUG] Prompt values AFTER mapping: {aggregate_df['Prompt'].unique().tolist()}")
    if 'Prompt' in corruption_type_df.columns:
        corruption_type_df = corruption_type_df.copy()
        corruption_type_df['Prompt'] = corruption_type_df['Prompt'].apply(get_prompt_label)

    # Detect available prompts from the data
    available_prompts = set()
    for cld_name, runs in all_results.items():
        for run_name, files in runs.items():
            available_prompts.update(files.keys())
    # Use consistent order (include mechanistic_original for citation experiments)
    prompt_order = ["baseline", "mechanistic", "mechanistic_original", "cot", "mechanistic_lit"]
    prompt_names = [p for p in prompt_order if p in available_prompts]
    # If we have both mechanistic and mechanistic_original, prefer mechanistic_original for blocking
    if 'mechanistic' in prompt_names and 'mechanistic_original' in prompt_names:
        prompt_names.remove('mechanistic')
    
    # Prompt-effect stats (assumption-gated) for figure annotations
    prompt_effect_f1 = compute_prompt_effect_test(
        all_results,
        metric_key="f1",
        prompt_names=prompt_names,
    )
    prompt_effect_auc = compute_prompt_effect_test(
        all_results,
        metric_key="auc_roc",
        prompt_names=prompt_names,
    )
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    
    # Plot 1: F1 Scores by CLD and Prompt
    ax1 = fig.add_subplot(gs[0, 0])
    pivot_f1 = aggregate_df.pivot(index='CLD', columns='Prompt', values='F1 Mean')
    n_clds = len(pivot_f1.index)
    n_bars = 4
    width = 0.18  # Narrower bars for more spacing
    group_width = n_bars * width
    x = np.arange(n_clds) * (group_width + 0.15)  # Add gap between CLD groups
    
    for i, prompt in enumerate(['Baseline', 'Mechanistic', 'CoT', 'Mech-Lit']):
        if prompt in pivot_f1.columns:
            values = pivot_f1[prompt].values
            errors = aggregate_df[aggregate_df['Prompt'] == prompt]['F1 CI Halfwidth'].values
            ax1.bar(x + i*width, values, width, label=prompt, yerr=errors, capsize=5)
    
    ax1.set_xlabel('CLD', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1 Scores by CLD and Prompt (±95% CI)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + (n_bars - 1) * width / 2)
    ax1.set_xticklabels([c.replace('_', ' ') for c in pivot_f1.index], rotation=15, ha='right')
    ax1.legend(title='', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: AUC-ROC by CLD and Prompt
    ax2 = fig.add_subplot(gs[0, 1])
    pivot_auc = aggregate_df.pivot(index='CLD', columns='Prompt', values='AUC-ROC Mean')
    x_auc = np.arange(len(pivot_auc.index)) * (group_width + 0.15)  # Same spacing as F1 plot
    
    for i, prompt in enumerate(['Baseline', 'Mechanistic', 'CoT', 'Mech-Lit']):
        if prompt in pivot_auc.columns:
            values = pivot_auc[prompt].values
            errors = aggregate_df[aggregate_df['Prompt'] == prompt]['AUC-ROC CI Halfwidth'].values
            ax2.bar(x_auc + i*width, values, width, label=prompt, yerr=errors, capsize=5)
    
    ax2.set_xlabel('CLD', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax2.set_title('AUC-ROC by CLD (±95% CI)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_auc + (n_bars - 1) * width / 2)
    ax2.set_xticklabels([c.replace('_', ' ') for c in pivot_auc.index], rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0.0, 1.0])
    # Add horizontal line at 0.5 (random/chance level)
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random')
    ax2.legend(title='', fontsize=9)
    
    # Plot 3: Precision vs Recall
    ax3 = fig.add_subplot(gs[0, 2])
    # Plot each CLD × Prompt combination individually with markers
    colors_map = {'Baseline': '#4472C4', 'Mechanistic': '#ED7D31', 'CoT': '#70AD47', 'Mech-Lit': '#9467BD'}
    markers = {'depressive': 'o', 'emergency_department': 's', 'social_norms': '^'}
    
    print(f"\n[DEBUG] Plotting Precision vs Recall:")
    print(f"  aggregate_df shape: {aggregate_df.shape}")
    print(f"  aggregate_df columns: {list(aggregate_df.columns)}")
    plot_count = 0
    
    for prompt in ['Baseline', 'Mechanistic', 'CoT', 'Mech-Lit']:
        subset = aggregate_df[aggregate_df['Prompt'] == prompt]
        print(f"  Prompt '{prompt}': {len(subset)} rows")
        for idx, row in subset.iterrows():
            cld = row['CLD']
            recall = row['Recall Mean']
            precision = row['Precision Mean']
            marker = markers.get(cld, 'o')
            print(f"    - CLD: {cld}, Recall: {recall:.3f}, Precision: {precision:.3f}, Marker: {marker}")
            # Use lower alpha to show overlapping points (prompts may have similar precision/recall)
            ax3.scatter(recall, precision, 
                       s=150, label=f"{prompt}" if cld == subset.iloc[0]['CLD'] else "", 
                       alpha=0.5, color=colors_map.get(prompt, None),
                       marker=marker, edgecolors='black', linewidth=1.5)
            plot_count += 1
    print(f"  Total points plotted: {plot_count}\n")
    
    ax3.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax3.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
    
    # Create legend combining colors (prompts) and markers (CLDs)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    # Use rectangles (patches) for prompts to match bar chart style
    prompt_legend = [Patch(facecolor=colors_map[p], edgecolor='black', label=p)
                    for p in ['Baseline', 'Mechanistic', 'CoT', 'Mech-Lit']]
    cld_legend = [Line2D([0], [0], marker=markers[cld], color='w', label=cld.replace('_', ' ').title(),
                        markerfacecolor='gray', markersize=10, markeredgecolor='black')
                 for cld in ['depressive', 'emergency_department', 'social_norms']]
    
    first_legend = ax3.legend(handles=prompt_legend, title='', loc='upper left', fontsize=9)
    ax3.add_artist(first_legend)
    ax3.legend(handles=cld_legend, title='CLD', loc='lower right', fontsize=8)
    
    ax3.grid(alpha=0.3)
    # Use full 0-1 range for clarity
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # Plot 4: Corruption Type Recall Heatmap with 95% CI
    ax4 = fig.add_subplot(gs[1, 0])
    if not corruption_type_df.empty:
        pivot_corr = corruption_type_df.pivot(index='Prompt', 
                                              columns='Corruption Type', 
                                              values='Recall Mean')
        pivot_ci_low = corruption_type_df.pivot(index='Prompt',
                                                columns='Corruption Type',
                                                values='Recall CI Low')
        pivot_ci_high = corruption_type_df.pivot(index='Prompt',
                                                 columns='Corruption Type',
                                                 values='Recall CI High')
        
        # Create custom annotations with mean and 95% CI
        annot_labels = np.empty_like(pivot_corr, dtype=object)
        for i in range(pivot_corr.shape[0]):
            for j in range(pivot_corr.shape[1]):
                mean_val = pivot_corr.iloc[i, j]
                ci_low = pivot_ci_low.iloc[i, j]
                ci_high = pivot_ci_high.iloc[i, j]
                if pd.notna(mean_val) and pd.notna(ci_low) and pd.notna(ci_high):
                    annot_labels[i, j] = f'{mean_val:.3f}\n[{ci_low:.3f}, {ci_high:.3f}]'
                else:
                    annot_labels[i, j] = ''
        
        # Use 0.5-1.0 range for better visual discrimination
        sns.heatmap(pivot_corr, annot=annot_labels, fmt='', cmap='RdYlGn', 
                   vmin=0.5, vmax=1.0, center=0.75, ax=ax4, cbar_kws={'label': 'Recall'},
                   annot_kws={'fontsize': 9})
        ax4.set_title('Detection Recall by Corruption Type (with 95% CI)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Corruption Type', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Prompt', fontsize=12, fontweight='bold')
    
    # Plot 5: Point-Biserial Correlation (ACTUAL VALUES - negative) with significance
    ax5 = fig.add_subplot(gs[1, 1])
    pivot_rpb = aggregate_df.pivot(index='CLD', columns='Prompt', 
                                   values='Point-Biserial r Mean')
    pivot_rpb_p = aggregate_df.pivot(index='CLD', columns='Prompt',
                                     values='Point-Biserial r Aggregate P-value')
    x = np.arange(len(pivot_rpb.index))
    
    for i, prompt in enumerate(['Baseline', 'Mechanistic', 'CoT', 'Mech-Lit']):
        if prompt in pivot_rpb.columns:
            values = pivot_rpb[prompt].values  # ACTUAL values (negative)
            errors = aggregate_df[aggregate_df['Prompt'] == prompt]['Point-Biserial r CI Halfwidth'].values
            p_values = pivot_rpb_p[prompt].values if prompt in pivot_rpb_p.columns else None
            
            bars = ax5.bar(x + i*width, values, width, label=prompt, yerr=errors, capsize=5)
            
            # Add significance markers (in red)
            if p_values is not None:
                for j, (val, p_val, err) in enumerate(zip(values, p_values, errors)):
                    if p_val < 0.001:
                        sig_marker = '***'
                    elif p_val < 0.01:
                        sig_marker = '**'
                    elif p_val < 0.05:
                        sig_marker = '*'
                    else:
                        sig_marker = ''
                    
                    if sig_marker:
                        y_pos = val - err - 0.03 if val < 0 else val + err + 0.03
                        ax5.text(x[j] + i*width, y_pos, sig_marker, 
                                ha='center', va='bottom' if val > 0 else 'top', 
                                fontsize=14, fontweight='bold', color='red')
    
    ax5.set_xlabel('CLD', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Point-Biserial r', fontsize=12, fontweight='bold')
    ax5.set_title('Score Discrimination (±95% CI)', fontsize=14, fontweight='bold')
    ax5.set_xticks(x + width)
    ax5.set_xticklabels([c.replace('_', ' ') for c in pivot_rpb.index], rotation=15, ha='right')
    ax5.legend(title='', fontsize=9)
    ax5.grid(axis='y', alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add significance note
    ax5.text(0.02, 0.98, '*** p < 0.001  ** p < 0.01  * p < 0.05', 
            transform=ax5.transAxes, fontsize=8, va='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 6: Effect Sizes from Post-hoc Tests - TABLE FORMAT
    # EXCLUDED FROM AGGREGATE FIGURE (per user request - shown separately)
    # The Cohen's d table is still saved as a separate PNG and CSV file
    # ax6 = fig.add_subplot(gs[1, 2])
    
    # Save Cohen's D table as CSV (still generate this)
    if stat_results['posthoc_tests']:
        posthoc_df = pd.DataFrame(stat_results['posthoc_tests'])
        f1_posthoc = posthoc_df[posthoc_df['Metric'] == 'F1 Mean'].copy()
        
        if not f1_posthoc.empty:
            f1_posthoc['abs_d'] = f1_posthoc['Cohens d'].abs()
            f1_posthoc = f1_posthoc.sort_values('Cohens d', ascending=False)
            
            cohens_d_table = f1_posthoc[['Comparison', 'Cohens d', 'P-value']].copy()
            cohens_d_table['Significance'] = cohens_d_table['P-value'].apply(
                lambda p: '***' if p < 0.0167 else ('*' if p < 0.05 else '')
            )
            cohens_d_csv = output_dir / "cohens_d_pairwise_comparisons.csv"
            cohens_d_table.to_csv(cohens_d_csv, index=False)
            print(f"✓ Cohen's d table saved: {cohens_d_csv}")
    
    output_file = output_dir / "rq1a_enhanced_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Changed from 150 to 300 DPI
    print(f"✓ Visualization saved: {output_file}")
    
    # ============================================================================
    # CREATE SPLIT FIGURES: Row 1 (Performance Metrics) and Row 2 (Detailed Analysis)
    # ============================================================================
    print("  Creating split row figures (1x3 and 1x2 layouts)...")
    
    # Store variables needed for split figures
    from matplotlib.lines import Line2D
    
    # FIGURE 1: Row 1 - Performance Metrics (1x3 layout)
    fig1 = plt.figure(figsize=(18, 5))
    gs1 = fig1.add_gridspec(1, 3, hspace=0.3, wspace=0.4)
    
    # Recreate Plot 1: F1 Scores
    ax1_split = fig1.add_subplot(gs1[0, 0])
    for i, prompt in enumerate(['Baseline', 'Mechanistic', 'CoT', 'Mech-Lit']):
        if prompt in pivot_f1.columns:
            values = pivot_f1[prompt].values
            errors = aggregate_df[aggregate_df['Prompt'] == prompt]['F1 CI Halfwidth'].values
            ax1_split.bar(x + i*width, values, width, label=prompt, yerr=errors, capsize=5)
    ax1_split.set_xlabel('CLD', fontsize=12, fontweight='bold')
    ax1_split.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax1_split.set_title('F1 Scores by CLD and Prompt (±95% CI)', fontsize=14, fontweight='bold')
    ax1_split.set_xticks(x + (n_bars - 1) * width / 2)
    ax1_split.set_xticklabels([c.replace('_', ' ') for c in pivot_f1.index], rotation=15, ha='right')
    ax1_split.legend(title='', fontsize=9, loc='lower left', bbox_to_anchor=(0.02, 0.07))
    ax1_split.grid(axis='y', alpha=0.3)

    # Add prompt-effect annotation (F1) — Friedman test with per-CLD post-hoc
    if prompt_effect_f1.get("ok", False):
        p = prompt_effect_f1.get("p", np.nan)
        stars = _sig_stars(p)
        n_blocks = prompt_effect_f1.get("n_blocks", 0)
        w = prompt_effect_f1.get("kendall_w", np.nan)
        w_str = f", W={w:.2f}" if not np.isnan(w) else ""
        
        # Build global post-hoc summary (using Bonferroni-corrected global pairs)
        global_posthoc = prompt_effect_f1.get("posthoc", {})
        prompt_short = {"baseline": "B", "mechanistic": "M", "mechanistic_original": "M", "cot": "C", "mechanistic_lit": "ML"}
        cld_short = {"depressive": "DE", "emergency_department": "ED", "social_norms": "SN"}
        
        global_sig = []
        for pair_name, pair_data in global_posthoc.items():
            if pair_data.get("sig", False):
                p1, p2 = pair_name.split("_vs_")
                winner = pair_data.get("winner", "")
                loser = p2 if winner == p1 else p1
                global_sig.append(f"{prompt_short.get(winner, winner[0].upper())}>{prompt_short.get(loser, loser[0].upper())}")
        
        global_str = f"{', '.join(global_sig)}" if global_sig else "none"
        
        # Build per-CLD post-hoc summary (exploratory, uncorrected p<0.05)
        per_cld = prompt_effect_f1.get("per_cld_posthoc", {})
        cld_results = []
        for cld_name, pairs in per_cld.items():
            sig_in_cld = []
            for pair_name, pair_data in pairs.items():
                if pair_data.get("sig", False):
                    p1, p2 = pair_name.split("_vs_")
                    winner = pair_data.get("winner", "")
                    loser = p2 if winner == p1 else p1
                    sig_in_cld.append(f"{prompt_short.get(winner, winner[0].upper())}>{prompt_short.get(loser, loser[0].upper())}")
            cld_label = cld_short.get(cld_name, cld_name[:2].upper())
            cld_results.append(f"{cld_label}:{','.join(sig_in_cld) if sig_in_cld else '—'}")
        
        per_cld_str = "; ".join(cld_results) if cld_results else ""
        
        # Place inside axes at bottom-left to avoid legend overlap
        ax1_split.text(
            0.02,
            0.02,
            f"Friedman (n={n_blocks}): p={_format_p(p)}{stars}{w_str} | Global: {global_str}\nPer-CLD: {per_cld_str}",
            transform=ax1_split.transAxes,
            va="bottom",
            ha="left",
            fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        )
    
    # Recreate Plot 2: AUC-ROC
    ax2_split = fig1.add_subplot(gs1[0, 1])
    for i, prompt in enumerate(['Baseline', 'Mechanistic', 'CoT', 'Mech-Lit']):
        if prompt in pivot_auc.columns:
            values = pivot_auc[prompt].values
            errors = aggregate_df[aggregate_df['Prompt'] == prompt]['AUC-ROC CI Halfwidth'].values
            ax2_split.bar(x_auc + i*width, values, width, label=prompt, yerr=errors, capsize=5)
    ax2_split.set_xlabel('CLD', fontsize=12, fontweight='bold')
    ax2_split.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax2_split.set_title('AUC-ROC by CLD (±95% CI)', fontsize=14, fontweight='bold')
    ax2_split.set_xticks(x_auc + (n_bars - 1) * width / 2)
    ax2_split.set_xticklabels([c.replace('_', ' ') for c in pivot_auc.index], rotation=15, ha='right')
    ax2_split.grid(axis='y', alpha=0.3)
    ax2_split.set_ylim([0.0, 1.0])
    ax2_split.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random (AUC=0.5)')
    ax2_split.legend(title='', fontsize=9, loc='lower left', bbox_to_anchor=(0.02, 0.07))

    # Add prompt-effect annotation (AUC) — Friedman test with per-CLD post-hoc
    if prompt_effect_auc.get("ok", False):
        p = prompt_effect_auc.get("p", np.nan)
        stars = _sig_stars(p)
        n_blocks = prompt_effect_auc.get("n_blocks", 0)
        w = prompt_effect_auc.get("kendall_w", np.nan)
        w_str = f", W={w:.2f}" if not np.isnan(w) else ""
        
        # Build global post-hoc summary (using Bonferroni-corrected global pairs)
        global_posthoc = prompt_effect_auc.get("posthoc", {})
        prompt_short = {"baseline": "B", "mechanistic": "M", "mechanistic_original": "M", "cot": "C", "mechanistic_lit": "ML"}
        cld_short = {"depressive": "DE", "emergency_department": "ED", "social_norms": "SN"}
        
        global_sig = []
        for pair_name, pair_data in global_posthoc.items():
            if pair_data.get("sig", False):
                p1, p2 = pair_name.split("_vs_")
                winner = pair_data.get("winner", "")
                loser = p2 if winner == p1 else p1
                global_sig.append(f"{prompt_short.get(winner, winner[0].upper())}>{prompt_short.get(loser, loser[0].upper())}")
        
        global_str = f"{', '.join(global_sig)}" if global_sig else "none"
        
        # Build per-CLD post-hoc summary (exploratory, uncorrected p<0.05)
        per_cld = prompt_effect_auc.get("per_cld_posthoc", {})
        cld_results = []
        for cld_name, pairs in per_cld.items():
            sig_in_cld = []
            for pair_name, pair_data in pairs.items():
                if pair_data.get("sig", False):
                    p1, p2 = pair_name.split("_vs_")
                    winner = pair_data.get("winner", "")
                    loser = p2 if winner == p1 else p1
                    sig_in_cld.append(f"{prompt_short.get(winner, winner[0].upper())}>{prompt_short.get(loser, loser[0].upper())}")
            cld_label = cld_short.get(cld_name, cld_name[:2].upper())
            cld_results.append(f"{cld_label}:{','.join(sig_in_cld) if sig_in_cld else '—'}")
        
        per_cld_str = "; ".join(cld_results) if cld_results else ""
        
        # Place inside axes at bottom-left to avoid legend overlap
        ax2_split.text(
            0.02,
            0.02,
            f"Friedman (n={n_blocks}): p={_format_p(p)}{stars}{w_str} | Global: {global_str}\nPer-CLD: {per_cld_str}",
            transform=ax2_split.transAxes,
            va="bottom",
            ha="left",
            fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        )
    
    # Recreate Plot 3: Precision vs Recall
    ax3_split = fig1.add_subplot(gs1[0, 2])
    # Get unique prompts from data (already mapped to display names)
    unique_prompts = aggregate_df['Prompt'].unique().tolist()
    prompt_order = ['Baseline', 'Mechanistic', 'CoT', 'Mech-Lit']
    available_prompts = [p for p in prompt_order if p in unique_prompts]
    
    for prompt in available_prompts:
        subset = aggregate_df[aggregate_df['Prompt'] == prompt]
        for idx, row in subset.iterrows():
            cld = row['CLD']
            recall = row['Recall Mean']
            precision = row['Precision Mean']
            marker = markers.get(cld, 'o')
            # Use lower alpha to show overlapping points (prompts may have similar precision/recall)
            ax3_split.scatter(recall, precision, 
                       s=150, label=f"{prompt}" if cld == subset.iloc[0]['CLD'] else "", 
                       alpha=0.5, color=colors_map.get(prompt, None),
                       marker=marker, edgecolors='black', linewidth=1.5)
    ax3_split.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax3_split.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax3_split.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
    # Use rectangles (patches) for prompts to match bar chart style
    prompt_legend = [Patch(facecolor=colors_map.get(p, '#808080'), edgecolor='black', label=p)
                    for p in available_prompts]
    cld_legend = [Line2D([0], [0], marker=markers[cld], color='w', label=cld.replace('_', ' ').title(),
                        markerfacecolor='gray', markersize=10, markeredgecolor='black')
                 for cld in ['depressive', 'emergency_department', 'social_norms']]
    first_legend = ax3_split.legend(handles=prompt_legend, title='', loc='upper left', fontsize=9)
    ax3_split.add_artist(first_legend)
    ax3_split.legend(handles=cld_legend, title='CLD', loc='lower right', fontsize=8)
    ax3_split.grid(alpha=0.3)
    ax3_split.set_xlim([0, 1])
    ax3_split.set_ylim([0, 1])
    
    output_file_row1 = output_dir / "rq1a_row1_performance_metrics.png"
    fig1.savefig(output_file_row1, dpi=300, bbox_inches='tight')
    print(f"  ✓ Row 1 (Performance Metrics 1x3): {output_file_row1.name}")
    
    # FIGURE 2: Row 2 - Detailed Analysis (1x3 layout - heatmap, score discrimination, and classification scores)
    fig2 = plt.figure(figsize=(18, 5))
    gs2 = fig2.add_gridspec(1, 3, hspace=0.3, wspace=0.4)
    
    # Recreate Plot 4: Corruption Type Heatmap
    ax4_split = fig2.add_subplot(gs2[0, 0])
    if not corruption_type_df.empty:
        pivot_corr = corruption_type_df.pivot(index='Prompt', 
                                              columns='Corruption Type', 
                                              values='Recall Mean')
        sns.heatmap(pivot_corr, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0.5, vmax=1.0, center=0.75, cbar_kws={'label': 'Recall'},
                   ax=ax4_split, linewidths=1, linecolor='black')
        ax4_split.set_xlabel('Corruption Type', fontsize=12, fontweight='bold')
        ax4_split.set_ylabel('Prompt', fontsize=12, fontweight='bold')
        ax4_split.set_title('Detection Recall by Corruption Type\n(±95% CI)', fontsize=12, fontweight='bold')
    
    # Recreate Plot 5: Score Discrimination
    ax5_split = fig2.add_subplot(gs2[0, 1])
    pivot_rpb = aggregate_df.pivot(index='CLD', columns='Prompt', 
                                   values='Point-Biserial r Mean')
    pivot_rpb_sig = aggregate_df.pivot(index='CLD', columns='Prompt', 
                                       values='Point-Biserial r Aggregate P-value')
    
    x_rpb = np.arange(len(pivot_rpb.index))
    # Dynamically detect available prompts (already mapped to display names)
    prompt_order_rpb = ['Baseline', 'Mechanistic', 'CoT', 'Mech-Lit']
    available_prompts_rpb = [p for p in prompt_order_rpb if p in pivot_rpb.columns]
    n_prompts_rpb = len(available_prompts_rpb)
    
    for i, prompt in enumerate(available_prompts_rpb):
        values = pivot_rpb[prompt].values
        errors = aggregate_df[aggregate_df['Prompt'] == prompt]['Point-Biserial r CI Halfwidth'].values
        bars = ax5_split.bar(x_rpb + i*width, values, width, label=prompt, 
                       yerr=errors, capsize=5, alpha=0.8)
        sig_vals = pivot_rpb_sig[prompt].values if prompt in pivot_rpb_sig.columns else None
        if sig_vals is not None:
            for j, (bar, p_val) in enumerate(zip(bars, sig_vals)):
                stars = _sig_stars(p_val)
                if stars:
                    height = bar.get_height()
                    # Position star slightly above bar (positive) or below (negative)
                    # Add offset based on error bar size if possible, otherwise fixed offset
                    # Here we just use bar height + small offset
                    y_offset = 0.02 if height >= 0 else -0.04
                    va = 'bottom' if height >= 0 else 'top'
                    
                    # Ensure stars don't overlap with error bars if they extend beyond bar
                    # Error is symmetric, so we just check direction
                    error = errors[j] if j < len(errors) else 0
                    if height >= 0:
                        text_y = max(height, height + error) + 0.01
                    else:
                        text_y = min(height, height - error) - 0.01
                        
                    ax5_split.text(bar.get_x() + bar.get_width()/2., text_y,
                             stars, ha='center', va=va, fontsize=12, fontweight='bold')
    ax5_split.set_xlabel('CLD', fontsize=12, fontweight='bold')
    ax5_split.set_ylabel('Point-biserial r', fontsize=12, fontweight='bold')
    ax5_split.set_title('Score Discrimination\n(±95% CI)', fontsize=12, fontweight='bold')
    ax5_split.set_xticks(x_rpb + width * (n_prompts_rpb - 1) / 2)
    ax5_split.set_xticklabels([c.replace('_', ' ') for c in pivot_rpb.index], rotation=15, ha='right')
    ax5_split.legend(title='', fontsize=9, loc='lower left', bbox_to_anchor=(0.02, 0.07))
    ax5_split.grid(axis='y', alpha=0.3)
    ax5_split.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5_split.text(0.02, 0.98, '*** p < 0.001  ** p < 0.01  * p < 0.05', 
            transform=ax5_split.transAxes, fontsize=8, va='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # NEW Plot 6: Classification Statistics Bar Chart (scores as heights)
    ax6_split = fig2.add_subplot(gs2[0, 2])
    
    from scipy import stats as sp_stats
    
    # Calculate overall statistics (averaged across all prompts & CLDs)
    classifications = ['TP', 'TN', 'FP', 'FN']
    score_cols = ['TP Mean Score Mean', 'TN Mean Score Mean', 'FP Mean Score Mean', 'FN Mean Score Mean']
    count_cols = ['N_TP', 'N_TN', 'N_FP', 'N_FN']
    
    # Aggregate across all conditions
    total_counts = []
    mean_scores = []
    ci_scores = []
    
    for score_col, count_col in zip(score_cols, count_cols):
        # Total count across all conditions
        total_count = int(aggregate_df[count_col].sum())
        total_counts.append(total_count)
        
        # Mean score across all conditions
        mean_score = aggregate_df[score_col].mean()
        mean_scores.append(mean_score)
        
        # 95% CI for score
        scores = aggregate_df[score_col].dropna()
        if len(scores) > 1:
            n = len(scores)
            std = scores.std()
            ci = std / np.sqrt(n) * sp_stats.t.ppf(0.975, n - 1)
        else:
            ci = 0
        ci_scores.append(ci)
    
    # Calculate percentages
    total_all = sum(total_counts)
    percentages = [100 * c / total_all for c in total_counts]
    
    # Color bars: GREEN for correct (TP, TN), RED for errors (FP, FN)
    bar_colors = ['#99ff99', '#99ff99', '#ff9999', '#ff9999']  # TP=green, TN=green, FP=red, FN=red
    
    # Create bar chart with scores as heights
    x_pos = np.arange(len(classifications))
    bars = ax6_split.bar(x_pos, mean_scores, yerr=ci_scores, capsize=5,
                         color=bar_colors, alpha=0.8, 
                         edgecolor='black', linewidth=2,
                         error_kw={'linewidth': 2, 'elinewidth': 2})
    
    # Add value labels on bars
    for i, (bar, score, ci) in enumerate(zip(bars, mean_scores, ci_scores)):
        height = bar.get_height()
        ax6_split.text(bar.get_x() + bar.get_width()/2., height + ci + 0.03,
                     f'{score:.2f}',
                     ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # Set labels and title
    ax6_split.set_ylabel('Mean Judge Score (μ)', fontsize=12, fontweight='bold')
    ax6_split.set_xlabel('Hallucination-Detection Outcome', fontsize=12, fontweight='bold')
    ax6_split.set_title('Judge Scores by Hallucination-Detection\nOutcome (±95% CI)', fontsize=12, fontweight='bold')
    ax6_split.set_xticks(x_pos)
    ax6_split.set_xticklabels(classifications, fontsize=14, fontweight='bold')
    ax6_split.set_ylim([0, 1.15])
    ax6_split.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at 0.5 (threshold)
    ax6_split.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Decision Threshold')
    
    # Add legend with counts and percentages
    legend_labels = [
        f'TP: n={total_counts[0]:,} ({percentages[0]:.1f}%)',
        f'TN: n={total_counts[1]:,} ({percentages[1]:.1f}%)',
        f'FP: n={total_counts[2]:,} ({percentages[2]:.1f}%)',
        f'FN: n={total_counts[3]:,} ({percentages[3]:.1f}%)',
        'Threshold (0.5)'
    ]
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Patch(facecolor=bar_colors[i], edgecolor='black', linewidth=1.5,
                            label=legend_labels[i]) for i in range(4)]
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                                 label=legend_labels[4]))
    
    ax6_split.legend(handles=legend_elements, loc='center right', fontsize=8, 
                    title=f'Total: n={total_all:,} edges', title_fontsize=9, framealpha=0.95)
    
    plt.tight_layout()
    output_file_row2 = output_dir / "rq1a_row2_detailed_analysis.png"
    fig2.savefig(output_file_row2, dpi=300, bbox_inches='tight')
    print(f"  ✓ Row 2 (Detailed Analysis 1x3): {output_file_row2.name}")
    
    plt.close('all')
    
    # Create score heatmap
    create_score_heatmap(aggregate_df, output_dir)
    
    # Additional plot: ROC curves (if sklearn available)
    if SKLEARN_AVAILABLE:
        create_roc_curves(all_results, output_dir)
    
    # NEW: Generate individual high-DPI figures for thesis
    print("\n📊 Generating individual high-DPI figures for thesis...")
    create_individual_figures_high_dpi(aggregate_df, corruption_type_df, stat_results, output_dir, all_results)


def create_score_heatmap(aggregate_df: pd.DataFrame, output_dir: Path):
    """Create score heatmap showing TP/TN/FP/FN mean scores by CLD×Prompt."""
    print("  Creating score heatmap...")
    
    # Detect experiment type from output_dir path for descriptive naming
    output_dir_str = str(output_dir).lower()
    if 'citation' in output_dir_str:
        judge_type = 'citation'
        judge_label = 'Citation Judge'
    else:
        judge_type = 'correctness'
        judge_label = 'Correctness Judge'
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Build heatmap data from aggregate_df
    heatmap_data = []
    for _, row in aggregate_df.iterrows():
        cld = row['CLD']
        prompt = row['Prompt']
        
        entry = {'CLD_Prompt': f"{cld}_{prompt}"}
        
        # Map column names to heatmap labels
        score_mappings = [
            ('TP Score', 'TP Mean Score Mean', 'TP Mean Score CI Halfwidth'),
            ('TN Score', 'TN Mean Score Mean', 'TN Mean Score CI Halfwidth'),
            ('FP Score', 'FP Mean Score Mean', 'FP Mean Score CI Halfwidth'),
            ('FN Score', 'FN Mean Score Mean', 'FN Mean Score CI Halfwidth'),
        ]
        
        for score_label, mean_col, ci_col in score_mappings:
            mean_val = row.get(mean_col, np.nan)
            ci_val = row.get(ci_col, 0)
            entry[score_label] = mean_val
            entry[f"{score_label}_CI"] = ci_val if pd.notna(ci_val) else 0
        
        heatmap_data.append(entry)
    
    heatmap_df = pd.DataFrame(heatmap_data).set_index('CLD_Prompt')
    
    # Create mean matrix for heatmap
    mean_cols = ['TP Score', 'TN Score', 'FP Score', 'FN Score']
    mean_df = heatmap_df[mean_cols]
    
    # Create custom annotations with mean ± CI
    annot_matrix = []
    for score_type in mean_cols:
        row_annots = []
        for idx in heatmap_df.index:
            mean_val = heatmap_df.loc[idx, score_type]
            ci_val = heatmap_df.loc[idx, f"{score_type}_CI"]
            if pd.notna(mean_val) and pd.notna(ci_val) and ci_val > 0:
                row_annots.append(f"{mean_val:.3f}\n±{ci_val:.3f}")
            elif pd.notna(mean_val):
                row_annots.append(f"{mean_val:.3f}")
            else:
                row_annots.append("")
        annot_matrix.append(row_annots)
    
    # Transpose for heatmap (scores as rows, CLD×Prompt as columns)
    annot_array = np.array(annot_matrix)
    
    sns.heatmap(mean_df.T, annot=annot_array, fmt='', cmap='RdYlGn', 
                center=0.75, vmin=0.5, vmax=1.0, ax=ax, 
                cbar_kws={'label': 'Mean Judge Score'},
                annot_kws={'fontsize': 8})
    ax.set_title(f'Corruption Detection Score Heatmap ({judge_label})\nJudge Scores by Classification (Mean ± 95% CI)', 
                 fontweight='bold', fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('Classification', fontweight='bold')
    plt.xticks(rotation=90, ha='center')
    
    plt.tight_layout()
    output_file = output_dir / f'rq1a_corruption_detection_{judge_type}_score_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Score heatmap saved: {output_file.name}")


def create_roc_curves(all_results: dict, output_dir: Path):
    """Create ROC curves for each prompt."""
    # Detect available prompts
    available_prompts = set()
    for cld_name, runs in all_results.items():
        for run_name, files in runs.items():
            available_prompts.update(files.keys())
    
    # Use consistent order, filter to available
    prompt_order = ['baseline', 'mechanistic', 'mechanistic_original', 'cot', 'mechanistic_lit']
    prompts_to_plot = [p for p in prompt_order if p in available_prompts]
    
    # If both mechanistic and mechanistic_original exist, skip mechanistic_original
    if 'mechanistic' in prompts_to_plot and 'mechanistic_original' in prompts_to_plot:
        prompts_to_plot.remove('mechanistic_original')
    
    n_prompts = len(prompts_to_plot)
    if n_prompts == 0:
        return
        
    fig, axes = plt.subplots(1, n_prompts, figsize=(6 * n_prompts, 6))
    if n_prompts == 1:
        axes = [axes]
    
    for idx, prompt_name in enumerate(prompts_to_plot):
        prompt_label = get_prompt_label(prompt_name)
        ax = axes[idx]
        
        for cld_name, runs in all_results.items():
            all_y_true = []
            all_y_scores = []
            
            for run_name, files in runs.items():
                if prompt_name in files:
                    metrics = extract_detailed_performance(files[prompt_name])
                    if metrics and 'per_edge_data' in metrics:
                        df = metrics['per_edge_data']
                        scores_clean = df[df['aggregate_score'].notna()].copy()
                        if len(scores_clean) > 0:
                            all_y_true.extend(scores_clean['ground_truth'].values)
                            all_y_scores.extend((1 - scores_clean['aggregate_score']).values)
            
            if len(all_y_true) > 0 and len(set(all_y_true)) == 2:
                fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
                auc = roc_auc_score(all_y_true, all_y_scores)
                ax.plot(fpr, tpr, label=f'{cld_name.replace("_", " ")} (AUC={auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title(f'{prompt_label} Prompt', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
    
    output_file = output_dir / "rq1a_roc_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Changed from 150 to 300 DPI
    print(f"✓ ROC curves saved: {output_file}")
    plt.close()


def create_individual_figures_high_dpi(aggregate_df: pd.DataFrame,
                                       corruption_type_df: pd.DataFrame,
                                       stat_results: dict,
                                       output_dir: Path,
                                       all_results: dict):
    """
    Create individual high-DPI publication-quality figures for thesis.
    
    Each figure is saved separately with:
    - 300 DPI for print quality
    - Larger fonts for readability
    - Professional styling
    - APA-compliant formatting
    """
    
    # Create subdirectory for individual figures
    individual_dir = output_dir / "individual_figures"
    individual_dir.mkdir(exist_ok=True)
    
    print(f"  Creating individual figures in: {individual_dir}")
    
    # Set publication-quality defaults
    pub_dpi = 300
    pub_fontsize_title = 16
    pub_fontsize_label = 14
    pub_fontsize_tick = 12
    pub_fontsize_legend = 11
    
    # ========================================================================
    # Figure 1: F1 Scores by CLD and Prompt
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_f1_pub = aggregate_df.pivot(index='CLD', columns='Prompt', values='F1 Mean')
    n_clds_pub = len(pivot_f1_pub.index)
    n_bars_pub = 4
    width_pub = 0.18  # Narrower bars for more spacing
    group_width_pub = n_bars_pub * width_pub
    x_pub = np.arange(n_clds_pub) * (group_width_pub + 0.15)  # Add gap between CLD groups
    
    colors = {'Baseline': '#4472C4', 'Mechanistic': '#ED7D31', 'CoT': '#70AD47', 'Mech-Lit': '#9467BD'}
    
    for i, prompt in enumerate(['Baseline', 'Mechanistic', 'CoT', 'Mech-Lit']):
        if prompt in pivot_f1_pub.columns:
            values = pivot_f1_pub[prompt].values
            errors = aggregate_df[aggregate_df['Prompt'] == prompt]['F1 CI Halfwidth'].values
            ax.bar(x_pub + i*width_pub, values, width_pub, label=prompt, yerr=errors, 
                   capsize=5, color=colors.get(prompt, None), 
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax.set_xlabel('Causal Loop Diagram', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_title('F1 Scores by CLD and Prompt Variant (±95% CI)', 
                 fontsize=pub_fontsize_title, fontweight='bold', pad=20)
    ax.set_xticks(x_pub + (n_bars_pub - 1) * width_pub / 2)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in pivot_f1_pub.index], 
                       rotation=0, ha='center', fontsize=pub_fontsize_tick)
    ax.tick_params(axis='y', labelsize=pub_fontsize_tick)
    ax.legend(fontsize=pub_fontsize_legend, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_file = individual_dir / "fig_f1_scores_by_cld_prompt.png"
    plt.savefig(output_file, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ F1 scores figure: {output_file.name}")
    plt.close()
    
    # ========================================================================
    # Figure 2: AUC-ROC by CLD and Prompt
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot_auc_pub = aggregate_df.pivot(index='CLD', columns='Prompt', values='AUC-ROC Mean')
    x_auc_pub = np.arange(len(pivot_auc_pub.index)) * (group_width_pub + 0.15)  # Same spacing as F1 plot
    
    for i, prompt in enumerate(['Baseline', 'Mechanistic', 'CoT', 'Mech-Lit']):
        if prompt in pivot_auc_pub.columns:
            values = pivot_auc_pub[prompt].values
            errors = aggregate_df[aggregate_df['Prompt'] == prompt]['AUC-ROC CI Halfwidth'].values
            ax.bar(x_auc_pub + i*width_pub, values, width_pub, label=prompt, yerr=errors,
                   capsize=5, color=colors.get(prompt, None),
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax.set_xlabel('Causal Loop Diagram', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_ylabel('AUC-ROC', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_title('Area Under ROC Curve by CLD and Prompt (±95% CI)', 
                 fontsize=pub_fontsize_title, fontweight='bold', pad=20)
    ax.set_xticks(x_auc_pub + (n_bars_pub - 1) * width_pub / 2)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in pivot_auc_pub.index],
                       rotation=0, ha='center', fontsize=pub_fontsize_tick)
    ax.tick_params(axis='y', labelsize=pub_fontsize_tick)
    ax.legend(fontsize=pub_fontsize_legend, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0.5, 1.0])
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Chance')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_file = individual_dir / "fig_auc_roc_by_cld_prompt.png"
    plt.savefig(output_file, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ AUC-ROC figure: {output_file.name}")
    plt.close()
    
    # ========================================================================
    # Figure 3: Precision vs Recall Trade-off
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for prompt in ['Baseline', 'Mechanistic', 'CoT', 'Mech-Lit']:
        subset = aggregate_df[aggregate_df['Prompt'] == prompt]
        ax.scatter(subset['Recall Mean'], subset['Precision Mean'],
                  s=200, label=prompt, alpha=0.7, color=colors.get(prompt, None),
                  edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_title('Precision-Recall Trade-off', 
                 fontsize=pub_fontsize_title, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=pub_fontsize_tick)
    
    # Create custom legend with rectangles to match bar chart style
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=colors[p], edgecolor='black', label=p)
                     for p in ['Baseline', 'Mechanistic', 'CoT']]
    ax.legend(handles=legend_handles, fontsize=pub_fontsize_legend, 
             frameon=True, shadow=True, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Use actual data range with padding
    all_recall = aggregate_df['Recall Mean'].values
    all_precision = aggregate_df['Precision Mean'].values
    ax.set_xlim([max(0, all_recall.min() - 0.05), min(1.0, all_recall.max() + 0.05)])
    ax.set_ylim([max(0, all_precision.min() - 0.05), min(1.0, all_precision.max() + 0.05)])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_file = individual_dir / "fig_precision_recall_tradeoff.png"
    plt.savefig(output_file, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Precision-Recall figure: {output_file.name}")
    plt.close()
    
    # ========================================================================
    # Figure 4: Corruption Type Recall Heatmap
    # ========================================================================
    if not corruption_type_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_corr = corruption_type_df.pivot(index='Prompt',
                                              columns='Corruption Type',
                                              values='Recall Mean')
        
        # Create heatmap with better styling (0.5-1.0 range for visual discrimination)
        sns.heatmap(pivot_corr, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0.5, vmax=1.0, center=0.75,
                   ax=ax, cbar_kws={'label': 'Recall'},
                   annot_kws={'fontsize': pub_fontsize_tick, 'weight': 'bold'},
                   linewidths=1, linecolor='white')
        
        ax.set_title('Detection Recall by Corruption Type',
                    fontsize=pub_fontsize_title, fontweight='bold', pad=20)
        ax.set_xlabel('Corruption Type', fontsize=pub_fontsize_label, fontweight='bold')
        ax.set_ylabel('Prompt Variant', fontsize=pub_fontsize_label, fontweight='bold')
        ax.tick_params(axis='both', labelsize=pub_fontsize_tick)
        
        plt.tight_layout()
        output_file = individual_dir / "fig_corruption_type_recall_heatmap.png"
        plt.savefig(output_file, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Corruption type heatmap: {output_file.name}")
        plt.close()
    
    # ========================================================================
    # Figure 5: Score Discrimination (Point-Biserial Correlation)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_rpb = aggregate_df.pivot(index='CLD', columns='Prompt',
                                   values='Point-Biserial r Mean')
    pivot_rpb_p = aggregate_df.pivot(index='CLD', columns='Prompt',
                                     values='Point-Biserial r Aggregate P-value')
    # For 3 prompts in this figure
    n_bars_rpb = 3
    width_rpb = 0.22
    group_width_rpb = n_bars_rpb * width_rpb
    x_rpb = np.arange(len(pivot_rpb.index)) * (group_width_rpb + 0.15)
    
    for i, prompt in enumerate(['Baseline', 'Mechanistic', 'CoT']):
        if prompt in pivot_rpb.columns:
            values = pivot_rpb[prompt].values
            errors = aggregate_df[aggregate_df['Prompt'] == prompt]['Point-Biserial r CI Halfwidth'].values
            p_values = pivot_rpb_p[prompt].values if prompt in pivot_rpb_p.columns else None
            
            bars = ax.bar(x_rpb + i*width_rpb, values, width_rpb, label=prompt, yerr=errors,
                         capsize=5, color=colors.get(prompt, None),
                         edgecolor='black', linewidth=1.5, alpha=0.85)
            
            # Add significance markers
            if p_values is not None:
                for j, (val, p_val, err) in enumerate(zip(values, p_values, errors)):
                    if p_val < 0.001:
                        sig_marker = '***'
                    elif p_val < 0.01:
                        sig_marker = '**'
                    elif p_val < 0.05:
                        sig_marker = '*'
                    else:
                        sig_marker = ''
                    
                    if sig_marker:
                        y_pos = val - err - 0.05 if val < 0 else val + err + 0.05
                        ax.text(x_rpb[j] + i*width_rpb, y_pos, sig_marker,
                               ha='center', va='bottom' if val > 0 else 'top',
                               fontsize=14, fontweight='bold', color='red')
    
    ax.set_xlabel('Causal Loop Diagram', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_ylabel('Point-Biserial Correlation (r)', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_title('Score Discrimination (±95% CI)',
                 fontsize=pub_fontsize_title, fontweight='bold', pad=20)
    ax.set_xticks(x_rpb + (n_bars_rpb - 1) * width_rpb / 2)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in pivot_rpb.index],
                       rotation=0, ha='center', fontsize=pub_fontsize_tick)
    ax.tick_params(axis='y', labelsize=pub_fontsize_tick)
    ax.legend(fontsize=pub_fontsize_legend, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add significance legend
    ax.text(0.02, 0.98, '*** p < 0.001  ** p < 0.01  * p < 0.05',
           transform=ax.transAxes, fontsize=pub_fontsize_tick, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    output_file = individual_dir / "fig_score_discrimination.png"
    plt.savefig(output_file, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Score discrimination figure: {output_file.name}")
    plt.close()
    
    # ========================================================================
    # Figure 6: APA-Compliant Cohen's D Table
    # ========================================================================
    if stat_results['posthoc_tests']:
        posthoc_df = pd.DataFrame(stat_results['posthoc_tests'])
        f1_posthoc = posthoc_df[posthoc_df['Metric'] == 'F1 Mean'].copy()
        
        if not f1_posthoc.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('tight')
            ax.axis('off')
            
            # Sort by Cohen's d
            f1_posthoc = f1_posthoc.sort_values('Cohens d', ascending=False)
            
            # Prepare table data in APA format
            table_data = []
            for _, row in f1_posthoc.iterrows():
                # Format p-value in APA style
                p_val = row['P-value']
                if p_val < 0.001:
                    p_str = '< .001***'
                elif p_val < 0.01:
                    p_str = f'{p_val:.3f}**'.replace('0.', '.')
                elif p_val < 0.05:
                    p_str = f'{p_val:.3f}*'.replace('0.', '.')
                else:
                    p_str = f'{p_val:.3f}'.replace('0.', '.')
                
                # Format Cohen's d
                d_str = f'{row["Cohens d"]:.3f}'
                
                table_data.append([
                    row['Comparison'],
                    d_str,
                    p_str
                ])
            
            # Create APA-style table
            table = ax.table(
                cellText=table_data,
                colLabels=['Comparison', "Cohen's d", 'p'],
                cellLoc='center',
                loc='center',
                colWidths=[0.5, 0.25, 0.25]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(pub_fontsize_tick)
            table.scale(1, 2.5)
            
            # APA-style formatting: simple, clean
            for i in range(3):
                cell = table[(0, i)]
                cell.set_facecolor('white')
                cell.set_text_props(weight='bold', fontsize=pub_fontsize_label)
                cell.set_edgecolor('black')
                cell.set_linewidth(2)
            
            # Style data rows - minimal coloring, APA prefers clean tables
            for i in range(1, len(table_data) + 1):
                for j in range(3):
                    cell = table[(i, j)]
                    cell.set_facecolor('white')
                    cell.set_edgecolor('black')
                    cell.set_linewidth(0.5)
            
            # Add top and bottom thick lines (APA style)
            for j in range(3):
                table[(0, j)].set_linewidth(2)
                table[(len(table_data), j)].set_linewidth(2)
            
            ax.set_title("Pairwise Prompt Comparisons: Effect Sizes for F1 Score",
                        fontsize=pub_fontsize_title, fontweight='bold', pad=30)
            
            # Add note about significance levels
            fig.text(0.5, 0.02, 
                    'Note. *** p < .001, ** p < .01, * p < .05',
                    ha='center', fontsize=pub_fontsize_tick, style='italic')
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            output_file = individual_dir / "table_cohens_d_pairwise_comparisons.png"
            plt.savefig(output_file, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
            print(f"  ✓ Cohen's d table: {output_file.name}")
            plt.close()
            
            # Also save as CSV for LaTeX import
            csv_data = f1_posthoc[['Comparison', 'Cohens d', 'P-value']].copy()
            csv_data['Significance'] = csv_data['P-value'].apply(
                lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            )
            csv_file = individual_dir / "table_cohens_d_pairwise_comparisons.csv"
            csv_data.to_csv(csv_file, index=False)
            print(f"  ✓ Cohen's d CSV: {csv_file.name}")
    
    # ========================================================================
    # Additional Figure: ROC Curves (if sklearn available)
    # ========================================================================
    if SKLEARN_AVAILABLE and all_results:
        print(f"  Generating ROC curves...")
        create_individual_roc_curves(all_results, individual_dir, pub_dpi, pub_fontsize_title, pub_fontsize_label, pub_fontsize_tick, pub_fontsize_legend)
    
    print(f"\n✅ Individual high-DPI figures created in: {individual_dir}")
    print(f"   All figures saved at {pub_dpi} DPI for publication quality\n")


def create_individual_roc_curves(all_results: dict, output_dir: Path, pub_dpi: int,
                                  pub_fontsize_title: int, pub_fontsize_label: int,
                                  pub_fontsize_tick: int, pub_fontsize_legend: int):
    """Create individual ROC curve figures for each prompt."""
    from sklearn.metrics import roc_auc_score, roc_curve
    
    prompt_labels = {'baseline': 'Baseline', 'mechanistic': 'Mechanistic', 'mechanistic_original': 'Mechanistic', 'cot': 'CoT', 'mechanistic_lit': 'Mech-Lit'}
    colors = {'Baseline': '#4472C4', 'Mechanistic': '#ED7D31', 'CoT': '#70AD47', 'Mech-Lit': '#9467BD'}
    cld_colors = {'depressive': '#9467BD', 'emergency_department': '#E377C2', 'social_norms': '#BCBD22'}
    
    for prompt_name, prompt_label in prompt_labels.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for cld_name, runs in all_results.items():
            all_y_true = []
            all_y_scores = []
            
            for run_name, files in runs.items():
                if prompt_name in files:
                    metrics = extract_detailed_performance(files[prompt_name])
                    if metrics and 'per_edge_data' in metrics:
                        df = metrics['per_edge_data']
                        scores_clean = df[df['aggregate_score'].notna()].copy()
                        if len(scores_clean) > 0:
                            all_y_true.extend(scores_clean['ground_truth'].values)
                            all_y_scores.extend((1 - scores_clean['aggregate_score']).values)
            
            if len(all_y_true) > 0 and len(set(all_y_true)) == 2:
                fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
                auc = roc_auc_score(all_y_true, all_y_scores)
                cld_label = cld_name.replace('_', ' ').title()
                ax.plot(fpr, tpr, label=f'{cld_label} (AUC={auc:.3f})', 
                       linewidth=2.5, color=cld_colors.get(cld_name, None))
        
        # Plot random baseline
        ax.plot([0, 1], [0, 1], 'k--', label='Chance (AUC=0.5)', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('False Positive Rate', fontsize=pub_fontsize_label, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=pub_fontsize_label, fontweight='bold')
        ax.set_title(f'ROC Curves: {prompt_label} Prompt', 
                    fontsize=pub_fontsize_title, fontweight='bold', pad=20)
        ax.tick_params(axis='both', labelsize=pub_fontsize_tick)
        ax.legend(fontsize=pub_fontsize_legend, frameon=True, shadow=True, loc='lower right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        output_file = output_dir / f"fig_roc_curves_{prompt_name}.png"
        plt.savefig(output_file, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
        print(f"  ✓ ROC curves ({prompt_label}): {output_file.name}")
        plt.close()



def main():
    parser = argparse.ArgumentParser(description='RQ1a Enhanced Aggregate Analysis')
    parser.add_argument('--base_dir', type=str,
                       default='/home/nitai/code/causalix.ai/final_runs/RQ1a_corruption_detection_correctness',
                       help='Base directory containing CLD run subdirectories')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: creates timestamped dir in base_dir)')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        sys.exit(1)
    
    print("="*80)
    print("RQ1a: ENHANCED AGGREGATE ANALYSIS - CORRUPTION DETECTION")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base directory: {base_dir}\n")
    
    # Step 1: Collect results
    all_results = collect_all_run_results(base_dir)
    if not all_results:
        print("\nError: No results found!")
        sys.exit(1)
    
    # Step 2: Main aggregation (CLD × Prompt)
    print("\n" + "="*80)
    print("MAIN RESULTS: CLD × PROMPT")
    print("="*80)
    aggregate_df = aggregate_by_cld_and_prompt(all_results)
    print(aggregate_df.to_string(index=False))
    
    # Step 3: Per-Prompt aggregation
    print("\n" + "="*80)
    print("PER-PROMPT AGGREGATE (across all CLDs)")
    print("="*80)
    per_prompt_df = aggregate_by_prompt(all_results)
    print(per_prompt_df.to_string(index=False))
    
    # Step 4: Per-CLD aggregation
    print("\n" + "="*80)
    print("PER-CLD AGGREGATE (across all Prompts)")
    print("="*80)
    per_cld_df = aggregate_by_cld(all_results)
    print(per_cld_df.to_string(index=False))
    
    # Step 5: Overall aggregate
    print("\n" + "="*80)
    print("OVERALL AGGREGATE (Grand Mean)")
    print("="*80)
    overall_agg = calculate_overall_aggregate(all_results)
    for key, value in overall_agg.items():
        print(f"  {key}: {value}")
    
    # Step 6: Corruption type detailed
    print("\n" + "="*80)
    print("CORRUPTION-TYPE-SPECIFIC PERFORMANCE")
    print("="*80)
    corruption_type_df = aggregate_corruption_type_detailed(all_results)
    print(corruption_type_df.to_string(index=False))
    
    # Step 7: Assumption testing
    assumption_results = perform_assumption_tests(aggregate_df)
    
    # Step 8: Prompt-effect testing (CONSISTENT BLOCKING)
    # Use the same repeated-measures unit as the ground-truth RQ1a analyses:
    #   Block = (CLD, run/seed)
    # The older CLD-only tests (on CLD×prompt means) are intentionally not used for inference.
    print("\n" + "="*80)
    print("PROMPT-EFFECT TESTING (BLOCKED BY CLD×RUN)")
    print("="*80)
    prompt_names = get_prompt_names_from_results(all_results)
    blocked_stat_results = perform_blocked_statistical_tests(all_results, prompt_names=prompt_names)
    # Keep downstream interfaces intact (visualizations expect a `stat_results` dict)
    stat_results = blocked_stat_results
    parametric_results = {'rm_anova_tests': [], 'paired_t_tests': []}
    comparison_df = pd.DataFrame()
    
    # Step 11: Generate LaTeX tables
    print("\n" + "="*80)
    print("GENERATING LATEX TABLES")
    print("="*80)
    latex_tables = generate_enhanced_latex_tables(
        aggregate_df, per_prompt_df, per_cld_df, corruption_type_df, stat_results
    )
    
    # Step 10: Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = base_dir / f"enhanced_analysis_{timestamp}"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save Excel
    output_excel = output_dir / "rq1a_enhanced_results.xlsx"
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        aggregate_df.to_excel(writer, sheet_name='Main (CLD x Prompt)', index=False)
        per_prompt_df.to_excel(writer, sheet_name='Per-Prompt Aggregate', index=False)
        per_cld_df.to_excel(writer, sheet_name='Per-CLD Aggregate', index=False)
        
        overall_df = pd.DataFrame([overall_agg])
        overall_df.to_excel(writer, sheet_name='Overall Aggregate', index=False)
        
        corruption_type_df.to_excel(writer, sheet_name='Corruption Type Detail', index=False)
        
        # Assumption tests
        if assumption_results['normality_tests']:
            pd.DataFrame(assumption_results['normality_tests']).to_excel(
                writer, sheet_name='Normality Tests', index=False)
        if assumption_results['variance_tests']:
            pd.DataFrame(assumption_results['variance_tests']).to_excel(
                writer, sheet_name='Variance Tests', index=False)
        
        # Prompt-effect statistical tests (BLOCKED BY CLD×RUN)
        if stat_results.get('friedman_tests'):
            pd.DataFrame(stat_results['friedman_tests']).to_excel(
                writer, sheet_name='Friedman Tests', index=False)
        if stat_results.get('posthoc_tests'):
            pd.DataFrame(stat_results['posthoc_tests']).to_excel(
                writer, sheet_name='Wilcoxon Tests', index=False)
        if stat_results.get('per_cld_posthoc_tests'):
            pd.DataFrame(stat_results['per_cld_posthoc_tests']).to_excel(
                writer, sheet_name='Per-CLD Wilcoxon (exploratory)', index=False)
    
    print(f"\n✓ Excel saved: {output_excel}")
    
    # Save LaTeX tables
    for table_name, latex_content in latex_tables.items():
        latex_file = output_dir / f"table_{table_name}.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_content)
        print(f"✓ LaTeX table saved: {latex_file}")
    
    # Step 12: Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_enhanced_visualizations(aggregate_df, corruption_type_df, stat_results, 
                                   output_dir, all_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - rq1a_enhanced_results.xlsx (comprehensive results)")
    print(f"  - table_*.tex (4 LaTeX tables)")
    print(f"  - rq1a_enhanced_visualization.png (6-panel figure)")
    print(f"  - rq1a_roc_curves.png (ROC curves)")
    print()


if __name__ == "__main__":
    main()

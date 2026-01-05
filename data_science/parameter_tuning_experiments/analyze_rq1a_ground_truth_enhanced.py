#!/usr/bin/env python3
"""
RQ1a Ground Truth Enhanced Aggregate Analysis

Groups by CLD and prompt (averaging across runs) and analyzes:
- Judge performance on TP/TN/FP/FN edges
- Correlation between classification and judge scores
- Performance aggregated by CLD and by prompt
- Enhanced visualizations with statistical rigor

Usage:
    python analyze_rq1a_ground_truth_enhanced.py [--base_dir PATH] [--output-dir DIR] [--judge-type {correctness,citation}]
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
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Tuple
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

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

def build_assumption_test_df(by_cld_prompt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a CLD×Prompt aggregate dataframe with *consistent column names*
    for assumption testing across RQ1a scripts.
    
    Output columns match `analyze_rq1a_aggregate_enhanced.py`:
      - CLD, Prompt
      - Precision Mean, Recall Mean, F1 Mean, Accuracy Mean, AUC-ROC Mean
    """
    df = by_cld_prompt_df.copy()
    if df.empty:
        return pd.DataFrame(columns=[
            'CLD', 'Prompt',
            'Precision Mean', 'Recall Mean', 'F1 Mean', 'Accuracy Mean', 'AUC-ROC Mean'
        ])
    
    # Ensure expected columns exist (fail soft with NaNs)
    for col in ['precision_mean', 'recall_mean', 'f1_mean', 'accuracy_mean', 'roc_auc_mean']:
        if col not in df.columns:
            df[col] = np.nan
    
    out = pd.DataFrame({
        'CLD': df['cld'],
        'Prompt': df['prompt'].apply(get_prompt_label),
        'Precision Mean': df['precision_mean'],
        'Recall Mean': df['recall_mean'],
        'F1 Mean': df['f1_mean'],
        'Accuracy Mean': df['accuracy_mean'],
        'AUC-ROC Mean': df['roc_auc_mean'],
    })
    return out


def perform_assumption_tests(aggregate_df: pd.DataFrame) -> dict:
    """
    Test statistical assumptions for parametric vs non-parametric tests.
    
    Tests (kept consistent with `analyze_rq1a_aggregate_enhanced.py`):
    1. Shapiro-Wilk for normality (per metric per prompt)
    2. Levene's test for homogeneity of variances (per metric across prompts)
    """
    results = {
        'normality_tests': [],
        'variance_tests': []
    }
    
    print("\n" + "="*80)
    print("ASSUMPTION TESTING")
    print("="*80)
    
    if aggregate_df is None or aggregate_df.empty:
        print("No data available for assumption testing.")
        return results
    
    # Keep metric list aligned with key RQ1a outcomes (F1 and AUC-ROC included)
    metrics = ['Precision Mean', 'Recall Mean', 'F1 Mean', 'Accuracy Mean', 'AUC-ROC Mean']
    prompts = aggregate_df['Prompt'].unique()
    
    print("\n1. SHAPIRO-WILK TEST FOR NORMALITY")
    print("-" * 80)
    print("H0: Data is normally distributed (if p < 0.05, reject normality)")
    
    for metric in metrics:
        if metric not in aggregate_df.columns:
            continue
        print(f"\n{metric}:")
        for prompt in prompts:
            data = aggregate_df[aggregate_df['Prompt'] == prompt][metric].dropna().values
            # Shapiro requires at least 3 observations; here that's typically 3 CLDs
            if len(data) >= 3:
                stat, p_value = stats.shapiro(data)
                is_normal = p_value >= 0.05
                results['normality_tests'].append({
                    'Metric': metric,
                    'Prompt': prompt,
                    'W-statistic': float(stat),
                    'P-value': float(p_value),
                    'Normal': bool(is_normal),
                    'N': int(len(data)),
                })
                status = "✓ Normal" if is_normal else "✗ Not normal"
                print(f"  {prompt}: W={stat:.4f}, p={p_value:.4f} {status} (n={len(data)})")
    
    print("\n2. LEVENE'S TEST FOR HOMOGENEITY OF VARIANCES")
    print("-" * 80)
    print("H0: Variances are equal across prompts (if p < 0.05, reject homogeneity)")
    
    for metric in metrics:
        if metric not in aggregate_df.columns:
            continue
        groups = [
            aggregate_df[aggregate_df['Prompt'] == prompt][metric].dropna().values
            for prompt in prompts
        ]
        # Need at least two groups with 2+ observations each
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            stat, p_value = stats.levene(*groups)
            is_homogeneous = p_value >= 0.05
            results['variance_tests'].append({
                'Metric': metric,
                'Statistic': float(stat),
                'P-value': float(p_value),
                'Homogeneous': bool(is_homogeneous),
                'N Groups': int(len(groups)),
            })
            status = "✓ Homogeneous" if is_homogeneous else "✗ Heterogeneous"
            print(f"{metric}: W={stat:.4f}, p={p_value:.4f} {status}")
    
    return results


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
    judge_type: str,
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
    from scipy.stats import wilcoxon, norm
    
    # Build a table: rows = blocks (CLD, run), columns = prompts
    block_data = []  # List of dicts: {prompt1: val1, prompt2: val2, ...}
    block_labels = []
    
    for cld_name, runs in all_results.items():
        for run_name, files in runs.items():
            row = {}
            for prompt_name in prompt_names:
                if prompt_name in files:
                    metrics = extract_detailed_performance(files[prompt_name], judge_type=judge_type)
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
        stat, p_value = stats.friedmanchisquare(*[data_matrix[:, i] for i in range(k_prompts)])
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
                
                # Effect size: r = Z / sqrt(N)
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
    
    for cld_name, runs in all_results.items():
        cld_block_data = []
        for run_name, files in runs.items():
            row = {}
            for prompt_name in prompt_names:
                if prompt_name in files:
                    metrics = extract_detailed_performance(files[prompt_name], judge_type=judge_type)
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


def collect_all_run_results(base_dir: Path) -> dict:
    """Collect results from all run directories.
    
    Supports new folder structure: Data/CLD/run_X/prompt/*.xlsx
    Falls back to old structure: CLD/run_X/judged_*_prompt_*.xlsx
    """
    results = {}
    
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
    
    cld_names = ['depressive', 'social_norms', 'emergency_department']
    
    for cld_dir in search_dir.iterdir():
        if not cld_dir.is_dir():
            continue
        
        cld_name = cld_dir.name
        if cld_name not in cld_names:
            continue
            
        results[cld_name] = {}
        run_dirs = sorted([d for d in cld_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
        
        for run_dir in run_dirs:
            run_name = run_dir.name
            
            if use_new_structure:
                # New structure: run_X/prompt/*.xlsx
                prompt_dict = {}
                for prompt in ['baseline', 'cot', 'mechanistic', 'mechanistic_lit', 'mechanistic_original']:
                    prompt_dir = run_dir / prompt
                    if prompt_dir.exists():
                        xlsx_files = list(prompt_dir.glob("*.xlsx"))
                        if xlsx_files:
                            prompt_dict[prompt] = xlsx_files[0]
                
                if prompt_dict:
                    results[cld_name][run_name] = prompt_dict
                    prompts_found = ', '.join(prompt_dict.keys())
                    print(f"  ✓ {cld_name}/{run_name}: Found {len(prompt_dict)} prompts ({prompts_found})")
                else:
                    print(f"  ✗ {cld_name}/{run_name}: No prompt folders found")
            else:
                # Old structure: run_X/judged_*_prompt_*.xlsx
                baseline_files = list(run_dir.glob("judged_*_baseline_*.xlsx"))
                cot_files = list(run_dir.glob("judged_*_cot_*.xlsx"))
                mechanistic_lit_files = list(run_dir.glob("judged_*_mechanistic_lit_*.xlsx"))
                mechanistic_original_files = list(run_dir.glob("judged_*_mechanistic_original_*.xlsx"))
                mechanistic_files = [f for f in run_dir.glob("judged_*_mechanistic_*.xlsx") 
                                    if 'mechanistic_original' not in f.name and 'mechanistic_lit' not in f.name]
                
                prompt_dict = {}
                if baseline_files:
                    prompt_dict['baseline'] = baseline_files[0]
                if cot_files:
                    prompt_dict['cot'] = cot_files[0]
                if mechanistic_files:
                    prompt_dict['mechanistic'] = mechanistic_files[0]
                if mechanistic_lit_files:
                    prompt_dict['mechanistic_lit'] = mechanistic_lit_files[0]
                if mechanistic_original_files:
                    prompt_dict['mechanistic_original'] = mechanistic_original_files[0]
                
                if prompt_dict:
                    results[cld_name][run_name] = prompt_dict
                    prompts_found = ', '.join(prompt_dict.keys())
                    print(f"  ✓ {cld_name}/{run_name}: Found {len(prompt_dict)} prompts ({prompts_found})")
                else:
                    print(f"  ✗ {cld_name}/{run_name}: No prompt files found")
    
    total_runs = sum(len(runs) for runs in results.values())
    print(f"\nSummary: {len(results)} CLDs, {total_runs} total runs")
    return results


def extract_detailed_performance(excel_path: Path, judge_type: str = 'correctness') -> dict:
    """
    Extract comprehensive performance metrics.
    
    Args:
        excel_path: Path to judged Excel file
        judge_type: 'correctness' or 'citation' - determines verdict mapping
    
    Returns dict with:
    - TP/TN/FP/FN counts
    - Precision, Recall, F1, Accuracy (for hallucination detection)
    - Point-biserial correlation (classification vs score)
    - Per-classification score statistics
    """
    try:
        # Check if 'All Edges' sheet exists
        xl_file = pd.ExcelFile(excel_path)
        if 'All Edges' not in xl_file.sheet_names:
            print(f"⚠️  Skipping {excel_path.name}: 'All Edges' sheet not found")
            return None
        
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        
        # Filter to only TP/TN/FP/FN edges
        df = df[df['Classification'].isin(['TP', 'TN', 'FP', 'FN'])].copy()
        
        if len(df) == 0:
            return None
        
        # Extract data
        data = []
        for _, row in df.iterrows():
            classification = row.get('Classification', 'UNKNOWN')
            # FP = hallucination (incorrectly included), FN = missed correct detection (also error)
            is_hallucination = (classification in ['FP', 'FN'])
            aggregate_score = row.get('Aggregate Score', np.nan)
            
            # Try to get verdict from Judge Verdict column first, then from Judge Message JSON
            aggregate_verdict = row.get('Judge Verdict', None)
            if pd.isna(aggregate_verdict) or aggregate_verdict is None:
                judge_msg = row.get('Judge Message', '')
                try:
                    judge_data = json.loads(judge_msg)
                    aggregate_verdict = judge_data.get('aggregate_verdict', 'UNKNOWN')
                    if pd.isna(aggregate_score):
                        aggregate_score = judge_data.get('aggregate_score', np.nan)
                except:
                    aggregate_verdict = 'UNKNOWN'
            
            data.append({
                'classification': classification,
                'is_hallucination': is_hallucination,
                'aggregate_score': aggregate_score,
                'aggregate_verdict': aggregate_verdict
            })
        
        df_data = pd.DataFrame(data)
        df_data['ground_truth'] = df_data['is_hallucination'].astype(int)
        
        # For ground truth, use numeric score threshold instead of verdict mapping
        # Score <= 0.5 means predicted as hallucination (judge_pred = 1)
        # Score > 0.5 means predicted as clean (judge_pred = 0)
        df_data['judge_pred'] = (df_data['aggregate_score'] <= 0.5).astype(int)
        
        # Confusion matrix based on score threshold
        # Confusion matrix based on score threshold (0.5)
        tp = ((df_data['ground_truth'] == 1) & (df_data['judge_pred'] == 1)).sum()
        tn = ((df_data['ground_truth'] == 0) & (df_data['judge_pred'] == 0)).sum()
        fp = ((df_data['ground_truth'] == 0) & (df_data['judge_pred'] == 1)).sum()
        fn = ((df_data['ground_truth'] == 1) & (df_data['judge_pred'] == 0)).sum()
        
        # F1, Precision, Recall at threshold=0.5 (lenient)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(df_data) if len(df_data) > 0 else 0.0
        
        # ROC-AUC (using scores: 0.0/0.5/1.0)
        # For hallucination detection, we need detection_score = 1 - judge_score
        # (lower judge score = higher detection score)
        scores_for_roc = df_data[df_data['aggregate_score'].notna()].copy()
        if len(scores_for_roc) >= 2 and scores_for_roc['ground_truth'].nunique() == 2:
            # Detection score: invert so higher = more likely hallucination
            detection_scores = 1 - scores_for_roc['aggregate_score']
            roc_auc = roc_auc_score(scores_for_roc['ground_truth'], detection_scores)
        else:
            roc_auc = np.nan
        
        # Point-biserial correlation (is_hallucination vs score)
        scores_clean = df_data[df_data['aggregate_score'].notna()].copy()
        if len(scores_clean) >= 2:
            r_pb, pb_p = stats.pointbiserialr(scores_clean['is_hallucination'], 
                                             scores_clean['aggregate_score'])
        else:
            r_pb, pb_p = 0.0, 1.0
        
        # ---------------------------------------------------------------------
        # Score distributions by *detection confusion type* (aligned with F1/AUC)
        # ---------------------------------------------------------------------
        # IMPORTANT: In GT Lit, df_data['classification'] refers to generator-vs-GT
        # (TP/TN/FP/FN w.r.t. expert CLD), whereas the plotted F1/AUC treat
        # "hallucination" as the positive class (GT label = FP or FN) and use a
        # judge-score threshold (<=0.5) as the prediction. To keep the "Judge Scores
        # by Classification" panel consistent with the F1 plot, we compute the
        # confusion type using (ground_truth, judge_pred).
        df_data['detect_confusion'] = np.select(
            [
                (df_data['ground_truth'] == 1) & (df_data['judge_pred'] == 1),  # hallucination detected
                (df_data['ground_truth'] == 0) & (df_data['judge_pred'] == 1),  # false alarm
                (df_data['ground_truth'] == 1) & (df_data['judge_pred'] == 0),  # missed hallucination
                (df_data['ground_truth'] == 0) & (df_data['judge_pred'] == 0),  # correct non-hallucination
            ],
            ['TP', 'FP', 'FN', 'TN'],
            default='NA'
        )
        tp_scores = df_data[df_data['detect_confusion'] == 'TP']['aggregate_score'].dropna()
        tn_scores = df_data[df_data['detect_confusion'] == 'TN']['aggregate_score'].dropna()
        fp_scores = df_data[df_data['detect_confusion'] == 'FP']['aggregate_score'].dropna()
        fn_scores = df_data[df_data['detect_confusion'] == 'FN']['aggregate_score'].dropna()
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            # Confusion matrix counts (positive class = hallucination)
            'tp_detect': int(tp),
            'tn_detect': int(tn),
            'fp_detect': int(fp),
            'fn_detect': int(fn),
            # Keep legacy keys used by plotting code: these now refer to detection
            # confusion counts (aligned with the F1/ROC computation above).
            'n_tp_edges': int(tp),
            'n_tn_edges': int(tn),
            'n_fp_edges': int(fp),
            'n_fn_edges': int(fn),
            'n_total': len(df_data),
            'n_hallucinations': int(df_data['is_hallucination'].sum()),
            'point_biserial_r': r_pb,
            'pb_p_value': pb_p,
            'tp_mean_score': float(tp_scores.mean()) if len(tp_scores) > 0 else np.nan,
            'tn_mean_score': float(tn_scores.mean()) if len(tn_scores) > 0 else np.nan,
            'fp_mean_score': float(fp_scores.mean()) if len(fp_scores) > 0 else np.nan,
            'fn_mean_score': float(fn_scores.mean()) if len(fn_scores) > 0 else np.nan,
            'tp_std_score': float(tp_scores.std()) if len(tp_scores) > 0 else np.nan,
            'tn_std_score': float(tn_scores.std()) if len(tn_scores) > 0 else np.nan,
            'fp_std_score': float(fp_scores.std()) if len(fp_scores) > 0 else np.nan,
            'fn_std_score': float(fn_scores.std()) if len(fn_scores) > 0 else np.nan,
            'file_path': str(excel_path)
        }
    
    except Exception as e:
        print(f"Error analyzing {excel_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_raw_edge_data(excel_path: Path, judge_type: str = 'correctness') -> pd.DataFrame:
    """
    Extract raw edge-level data (is_hallucination, aggregate_score) from Excel file.
    
    Returns DataFrame with columns: is_hallucination, aggregate_score
    """
    try:
        xl_file = pd.ExcelFile(excel_path)
        if 'All Edges' not in xl_file.sheet_names:
            return pd.DataFrame()
        
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        df = df[df['Classification'].isin(['TP', 'TN', 'FP', 'FN'])].copy()
        
        if len(df) == 0:
            return pd.DataFrame()
        
        data = []
        for _, row in df.iterrows():
            classification = row.get('Classification', 'UNKNOWN')
            is_hallucination = (classification in ['FP', 'FN'])
            aggregate_score = row.get('Aggregate Score', np.nan)
            
            if pd.isna(aggregate_score):
                judge_msg = row.get('Judge Message', '')
                try:
                    judge_data = json.loads(judge_msg)
                    aggregate_score = judge_data.get('aggregate_score', np.nan)
                except:
                    pass
            
            if pd.notna(aggregate_score):
                data.append({
                    'is_hallucination': int(is_hallucination),
                    'aggregate_score': aggregate_score
                })
        
        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame()


def aggregate_results(results: dict, judge_type: str = 'correctness') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate results by CLD and by prompt.
    
    For point-biserial correlation, pools raw edge data across runs and computes
    correlation on pooled data (proper statistical test) instead of averaging.
    
    Args:
        results: Dict of CLD runs
        judge_type: 'correctness' or 'citation'
    
    Returns:
    - by_cld_prompt_df: Aggregated by CLD and prompt (averaged across runs, except point-biserial)
    - raw_df: Raw data for all runs
    """
    raw_data = []
    
    for cld_name, runs in results.items():
        for run_name, prompt_files in runs.items():
            for prompt_name, excel_path in prompt_files.items():
                perf = extract_detailed_performance(excel_path, judge_type=judge_type)
                if perf:
                    perf['cld'] = cld_name
                    perf['run'] = run_name
                    perf['prompt'] = prompt_name
                    raw_data.append(perf)
    
    raw_df = pd.DataFrame(raw_data)
    
    # If both 'mechanistic' and 'mechanistic_original' are present, rename 'mechanistic' to 'mechanistic_lit'
    unique_prompts = raw_df['prompt'].unique()
    if 'mechanistic' in unique_prompts and 'mechanistic_original' in unique_prompts:
        print("\n📝 Note: Both 'mechanistic' and 'mechanistic_original' detected.")
        print("   Renaming 'mechanistic' → 'mechanistic_lit' for clarity in results.")
        raw_df['prompt'] = raw_df['prompt'].replace('mechanistic', 'mechanistic_lit')
    
    # Aggregate by CLD and prompt (average across runs)
    groupby_cols = ['cld', 'prompt']
    agg_funcs = {
        'roc_auc': ['mean', 'std', 'count'],
        'precision': ['mean', 'std', 'count'],
        'recall': ['mean', 'std', 'count'],
        'f1': ['mean', 'std', 'count'],
        'accuracy': ['mean', 'std', 'count'],
        'point_biserial_r': ['mean', 'std', 'count'],  # Will be replaced with pooled
        'pb_p_value': 'mean',  # Will be replaced with pooled
        'tp_mean_score': 'mean',
        'tn_mean_score': 'mean',
        'fp_mean_score': 'mean',
        'fn_mean_score': 'mean',
        'n_tp_edges': 'sum',
        'n_tn_edges': 'sum',
        'n_fp_edges': 'sum',
        'n_fn_edges': 'sum',
        'n_total': 'sum',
        'n_hallucinations': 'sum'
    }
    
    by_cld_prompt_df = raw_df.groupby(groupby_cols).agg(agg_funcs).reset_index()
    by_cld_prompt_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                                  for col in by_cld_prompt_df.columns.values]
    
    # Compute point-biserial correlation on pooled data across runs (proper statistical test)
    print("\n📊 Computing point-biserial correlations on pooled data across runs...")
    pooled_correlations = []
    for _, row in by_cld_prompt_df.iterrows():
        cld = row['cld']
        prompt = row['prompt']
        
        # Collect all edge data for this CLD×prompt combination across all runs
        pooled_edges = []
        for cld_name, runs in results.items():
            if cld_name != cld:
                continue
            for run_name, prompt_files in runs.items():
                for prompt_name, excel_path in prompt_files.items():
                    if prompt_name == prompt or (prompt_name == 'mechanistic' and prompt == 'mechanistic_lit'):
                        edge_data = extract_raw_edge_data(excel_path, judge_type=judge_type)
                        if len(edge_data) > 0:
                            pooled_edges.append(edge_data)
        
        if pooled_edges:
            pooled_df = pd.concat(pooled_edges, ignore_index=True)
            pooled_df = pooled_df[pooled_df['aggregate_score'].notna()]
            
            if len(pooled_df) >= 2 and pooled_df['is_hallucination'].nunique() == 2:
                r_pb, pb_p = stats.pointbiserialr(
                    pooled_df['is_hallucination'],
                    pooled_df['aggregate_score']
                )
                pooled_correlations.append({
                    'cld': cld,
                    'prompt': prompt,
                    'point_biserial_r_pooled': r_pb,
                    'pb_p_value_pooled': pb_p,
                    'n_edges_pooled': len(pooled_df)
                })
            else:
                pooled_correlations.append({
                    'cld': cld,
                    'prompt': prompt,
                    'point_biserial_r_pooled': np.nan,
                    'pb_p_value_pooled': np.nan,
                    'n_edges_pooled': len(pooled_df) if pooled_edges else 0
                })
        else:
            pooled_correlations.append({
                'cld': cld,
                'prompt': prompt,
                'point_biserial_r_pooled': np.nan,
                'pb_p_value_pooled': np.nan,
                'n_edges_pooled': 0
            })
    
    pooled_df = pd.DataFrame(pooled_correlations)
    
    # Merge pooled correlations into aggregated dataframe
    by_cld_prompt_df = by_cld_prompt_df.merge(
        pooled_df[['cld', 'prompt', 'point_biserial_r_pooled', 'pb_p_value_pooled']],
        on=['cld', 'prompt'],
        how='left'
    )
    
    # Replace averaged correlations with pooled ones
    by_cld_prompt_df['point_biserial_r_mean'] = by_cld_prompt_df['point_biserial_r_pooled']
    by_cld_prompt_df['pb_p_value_mean'] = by_cld_prompt_df['pb_p_value_pooled']
    
    # Calculate 95% confidence intervals (CI = 1.96 * SEM, where SEM = std / sqrt(n))
    from scipy import stats as sp_stats
    
    for metric in ['roc_auc', 'precision', 'recall', 'f1', 'accuracy']:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        count_col = f'{metric}_count'
        ci_col = f'{metric}_ci95'
        
        # Calculate 95% CI using t-distribution for small samples
        by_cld_prompt_df[ci_col] = by_cld_prompt_df.apply(
            lambda row: row[std_col] / np.sqrt(row[count_col]) * sp_stats.t.ppf(0.975, row[count_col] - 1)
            if row[count_col] > 1 and pd.notna(row[std_col]) else 0,
            axis=1
        )
    
    # For point-biserial correlation, compute CI from pooled data using standard error
    # SE = sqrt((1 - r^2) / (n - 2))
    # 95% CI = r ± t(0.975, n-2) * SE
    def compute_pb_ci(row):
        r = row['point_biserial_r_mean']
        n = row.get('n_total_sum', 0)  # Use n_total_sum column
        if pd.notna(r) and n > 2:
            se = np.sqrt((1 - r**2) / (n - 2))
            t_crit = sp_stats.t.ppf(0.975, n - 2)
            return se * t_crit
        return 0
    
    by_cld_prompt_df['point_biserial_r_ci95'] = by_cld_prompt_df.apply(compute_pb_ci, axis=1)
    
    return by_cld_prompt_df, raw_df


def create_enhanced_visualizations(by_cld_prompt_df: pd.DataFrame, 
                                 raw_df: pd.DataFrame,
                                 output_dir: Path,
                                 judge_type: str,
                                 all_results: dict = None):
    """Create comprehensive visualizations."""
    # Map prompt identifiers to display labels for all downstream plots/legends
    by_cld_prompt_df = by_cld_prompt_df.copy()
    if 'prompt' in by_cld_prompt_df.columns:
        by_cld_prompt_df['prompt'] = by_cld_prompt_df['prompt'].apply(get_prompt_label)
    raw_df = raw_df.copy()
    if 'prompt' in raw_df.columns:
        raw_df['prompt'] = raw_df['prompt'].apply(get_prompt_label)
    
    # Figure 1: Main performance comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Panel 1: ROC-AUC by CLD and Prompt (with 95% CI)
    ax1 = axes[0, 0]
    pivot_roc_mean = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='roc_auc_mean')
    pivot_roc_ci = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='roc_auc_ci95')
    # Rename columns to consistent display names
    pivot_roc_mean.columns = [get_prompt_label(c) for c in pivot_roc_mean.columns]
    pivot_roc_ci.columns = [get_prompt_label(c) for c in pivot_roc_ci.columns]
    
    pivot_roc_mean.plot(kind='bar', ax=ax1, width=0.8, edgecolor='black', alpha=0.8,
                       yerr=pivot_roc_ci, capsize=4, error_kw={'linewidth': 1.5})
    ax1.set_ylabel('ROC-AUC', fontsize=14, fontweight='bold')
    ax1.set_title('ROC-AUC by CLD and Prompt (±95% CI)', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (AUC=0.5)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15, ha='right')
    ax1.legend(title='', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: F1 Score at threshold=0.5 by CLD and Prompt (with 95% CI)
    ax2 = axes[0, 1]
    pivot_f1_mean = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='f1_mean')
    pivot_f1_ci = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='f1_ci95')
    # Rename columns to consistent display names
    pivot_f1_mean.columns = [get_prompt_label(c) for c in pivot_f1_mean.columns]
    pivot_f1_ci.columns = [get_prompt_label(c) for c in pivot_f1_ci.columns]
    
    pivot_f1_mean.plot(kind='bar', ax=ax2, width=0.8, edgecolor='black', alpha=0.8,
                       yerr=pivot_f1_ci, capsize=4, error_kw={'linewidth': 1.5})
    ax2.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax2.set_title('F1 Scores by CLD and Prompt (±95% CI)', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15, ha='right')
    ax2.legend(title='', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Panel 3: Performance by Prompt (Avg. Across CLDs) - SWAPPED from bottom right
    ax3 = axes[0, 2]
    
    from scipy import stats as sp_stats
    
    per_prompt_avg = by_cld_prompt_df.groupby('prompt').agg({
        'roc_auc_mean': 'mean', 'f1_mean': 'mean', 'precision_mean': 'mean', 
        'recall_mean': 'mean', 'point_biserial_r_mean': 'mean'
    })
    per_prompt_std = by_cld_prompt_df.groupby('prompt').agg({
        'roc_auc_mean': 'std', 'f1_mean': 'std', 'precision_mean': 'std', 
        'recall_mean': 'std', 'point_biserial_r_mean': 'std'
    })
    n_clds = by_cld_prompt_df.groupby('prompt').size()
    
    # Calculate CI for each metric separately
    per_prompt_ci = pd.DataFrame(index=per_prompt_avg.index, columns=per_prompt_avg.columns)
    for metric_col in per_prompt_avg.columns:
        for prompt in per_prompt_avg.index:
            std_val = per_prompt_std.loc[prompt, metric_col]
            n = n_clds.loc[prompt]
            ci_val = std_val / np.sqrt(n) * sp_stats.t.ppf(0.975, n - 1) if pd.notna(std_val) and n > 1 else 0
            per_prompt_ci.loc[prompt, metric_col] = ci_val
    
    metrics = ['ROC-AUC', 'F1', 'Precision', 'Recall']
    metric_cols = ['roc_auc_mean', 'f1_mean', 'precision_mean', 'recall_mean']
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, prompt in enumerate(per_prompt_avg.index):
        values = [per_prompt_avg.loc[prompt, col] for col in metric_cols]
        errors = [per_prompt_ci.loc[prompt, col] for col in metric_cols]
        ax3.bar(x + i * width, values, width, label=get_prompt_label(prompt), 
               yerr=errors, capsize=4, alpha=0.8, edgecolor='black')
    
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(metrics)
    ax3.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax3.set_title('Performance by Prompt (±95% CI)', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.legend(title='', loc='upper right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Precision vs Recall
    ax4 = axes[1, 0]
    prompts = by_cld_prompt_df['prompt'].unique()
    colors = plt.cm.tab10(range(len(prompts)))
    for i, prompt in enumerate(prompts):
        data = by_cld_prompt_df[by_cld_prompt_df['prompt'] == prompt]
        ax4.scatter(data['recall_mean'], data['precision_mean'], 
                   s=200, label=get_prompt_label(prompt), color=colors[i],
                   alpha=0.7, edgecolors='black', linewidth=1.5)
    ax4.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax4.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    ax4.legend(title='', fontsize=9)
    ax4.grid(alpha=0.3)
    
    # Panel 5: Score distributions by classification (with 95% CI and counts)
    ax5 = axes[1, 1]
    score_cols = ['tp_mean_score_mean', 'tn_mean_score_mean', 'fp_mean_score_mean', 'fn_mean_score_mean']
    count_cols = ['n_tp_edges_sum', 'n_tn_edges_sum', 'n_fp_edges_sum', 'n_fn_edges_sum']
    classifications = ['TP', 'TN', 'FP', 'FN']
    
    # Calculate mean scores and total counts
    avg_scores = [by_cld_prompt_df[col].mean() for col in score_cols]
    std_scores = [by_cld_prompt_df[col].std() for col in score_cols]
    total_counts = [int(by_cld_prompt_df[col].sum()) for col in count_cols]
    n_samples = len(by_cld_prompt_df)  # Should be 12 (3 CLDs × 4 prompts)
    
    from scipy import stats as sp_stats
    ci_scores = [std / np.sqrt(n_samples) * sp_stats.t.ppf(0.975, n_samples - 1) 
                 if pd.notna(std) else 0 for std in std_scores]
    
    # Calculate percentages
    total_all = sum(total_counts)
    percentages = [100 * c / total_all for c in total_counts]
    
    # Color bars: GREEN for correct (TP, TN), RED for errors (FP, FN)
    bar_colors = ['#99ff99', '#99ff99', '#ff9999', '#ff9999']  # TP=green, TN=green, FP=red, FN=red
    
    bars = ax5.bar(range(len(classifications)), avg_scores, yerr=ci_scores, capsize=5,
                   color=bar_colors, alpha=0.8, edgecolor='black', linewidth=2,
                   error_kw={'linewidth': 2, 'elinewidth': 2})
    
    # Add value labels on bars
    for i, (bar, score, ci) in enumerate(zip(bars, avg_scores, ci_scores)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + ci + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax5.set_xticks(range(len(classifications)))
    ax5.set_xticklabels(classifications, fontsize=14, fontweight='bold')
    ax5.set_ylabel('Mean Judge Score (μ)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Classification Type', fontsize=14, fontweight='bold')
    ax5.set_title('Judge Scores by Hallucination-Detection Outcome (±95% CI)', fontsize=14, fontweight='bold')
    ax5.set_ylim([0, 1.15])
    ax5.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at 0.5 (threshold)
    ax5.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add legend with counts
    legend_labels = [
        f'TP: n={total_counts[0]:,} ({percentages[0]:.1f}%)',
        f'TN: n={total_counts[1]:,} ({percentages[1]:.1f}%)',
        f'FP: n={total_counts[2]:,} ({percentages[2]:.1f}%)',
        f'FN: n={total_counts[3]:,} ({percentages[3]:.1f}%)',
        'Threshold (0.5)'
    ]
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Patch(facecolor=bar_colors[i], edgecolor='black', linewidth=1.5,
                            label=legend_labels[i]) for i in range(4)]
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                                 label=legend_labels[4]))
    ax5.legend(handles=legend_elements, loc='center right', fontsize=8, 
              title=f'Total: n={total_all:,} edges', title_fontsize=9, framealpha=0.95)
    
    # Panel 6: Performance by prompt (averaged across CLDs) with CI
    ax6 = axes[1, 2]
    by_prompt = by_cld_prompt_df.groupby('prompt').agg({
        'roc_auc_mean': 'mean',
        'f1_mean': 'mean',
        'precision_mean': 'mean',
        'recall_mean': 'mean',
        'roc_auc_ci95': 'mean',
        'f1_ci95': 'mean',
        'precision_ci95': 'mean',
        'recall_ci95': 'mean'
    }).reset_index()
    
    x = np.arange(len(by_prompt))
    width = 0.2
    ax6.bar(x - 1.5*width, by_prompt['roc_auc_mean'], width, label='ROC-AUC', alpha=0.8,
            yerr=by_prompt['roc_auc_ci95'], capsize=3, error_kw={'linewidth': 1.2})
    ax6.bar(x - 0.5*width, by_prompt['precision_mean'], width, label='Precision', alpha=0.8,
            yerr=by_prompt['precision_ci95'], capsize=3, error_kw={'linewidth': 1.2})
    ax6.bar(x + 0.5*width, by_prompt['recall_mean'], width, label='Recall', alpha=0.8,
            yerr=by_prompt['recall_ci95'], capsize=3, error_kw={'linewidth': 1.2})
    ax6.bar(x + 1.5*width, by_prompt['f1_mean'], width, label='F1', alpha=0.8,
            yerr=by_prompt['f1_ci95'], capsize=3, error_kw={'linewidth': 1.2})
    
    ax6.set_xticks(x)
    ax6.set_xticklabels([get_prompt_label(p) for p in by_prompt['prompt']], rotation=15, ha='right')
    ax6.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Prompt', fontsize=14, fontweight='bold')
    ax6.set_title('Performance by Prompt (±95% CI)', fontsize=14, fontweight='bold')
    ax6.set_ylim([0, 1])
    ax6.legend(title='Metric', loc='upper right', fontsize=9)
    ax6.grid(axis='y', alpha=0.3)
    
    # Add significance stars for Point-Biserial correlation (using p-value from pooled test)
    pivot_pb_mean = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='point_biserial_r_mean')
    pivot_pb_pval = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='pb_p_value_mean')
    # Rename columns to display names for consistent legends
    pivot_pb_mean.columns = [get_prompt_label(c) for c in pivot_pb_mean.columns]
    pivot_pb_pval.columns = [get_prompt_label(c) for c in pivot_pb_pval.columns]
    
    for i, cld in enumerate(pivot_pb_mean.index):
        for j, prompt in enumerate(pivot_pb_mean.columns):
            r_val = pivot_pb_mean.loc[cld, prompt]
            p_val = pivot_pb_pval.loc[cld, prompt]
            
            if pd.notna(r_val) and pd.notna(p_val):
                # Use p-value from pooled statistical test
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                
                if sig:
                    y_pos = r_val + (0.02 if r_val > 0 else -0.05)
                    ax6.text(i + (j - 1) * 0.27, y_pos, sig, ha='center', va='center', 
                            fontsize=14, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq1a_ground_truth_enhanced_visualization.png', 
                dpi=300, bbox_inches='tight')  # Changed from 300 to ensure consistency
    print(f"✓ Enhanced visualization saved")
    
    # ============================================================================
    # CREATE SPLIT FIGURES: Row 1 (Performance Metrics) and Row 2 (Detailed Analysis)
    # ============================================================================
    print("  Creating split row figures (1x3 layouts)...")

    # Detect available prompts from the data
    available_prompts = set()
    if all_results is not None:
        for cld_name, runs in all_results.items():
            for run_name, files in runs.items():
                available_prompts.update(files.keys())
    # Use consistent order (include mechanistic_original for citation experiments)
    prompt_order_full = ['baseline', 'mechanistic', 'mechanistic_original', 'cot', 'mechanistic_lit']
    prompt_names = [p for p in prompt_order_full if p in available_prompts]
    # If we have both mechanistic and mechanistic_original, prefer mechanistic_original for blocking
    if 'mechanistic' in prompt_names and 'mechanistic_original' in prompt_names:
        prompt_names.remove('mechanistic')
    
    # Prompt-effect stats (assumption-gated) for figure annotations
    prompt_effect_f1 = None
    prompt_effect_auc = None
    if all_results is not None and len(prompt_names) >= 2:
        prompt_effect_f1 = compute_prompt_effect_test(
            all_results,
            metric_key="f1",
            prompt_names=prompt_names,
            judge_type=judge_type,
        )
        prompt_effect_auc = compute_prompt_effect_test(
            all_results,
            metric_key="roc_auc",
            prompt_names=prompt_names,
            judge_type=judge_type,
        )
    
    # FIGURE 1: Row 1 - Performance Metrics (1x3 layout)
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    
    # Recreate Panel 1: F1 Score (Left)
    ax1_split = axes1[0]
    pivot_f1_mean.plot(kind='bar', ax=ax1_split, width=0.8, edgecolor='black', alpha=0.8,
                       yerr=pivot_f1_ci, capsize=4, error_kw={'linewidth': 1.5})
    ax1_split.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax1_split.set_title('F1 Scores by CLD and Prompt (±95% CI)', fontsize=14, fontweight='bold')
    ax1_split.set_ylim([0, 1])
    ax1_split.set_xticklabels(ax1_split.get_xticklabels(), rotation=15, ha='right')
    ax1_split.legend(title='', fontsize=9)
    ax1_split.grid(axis='y', alpha=0.3)

    if prompt_effect_f1 and prompt_effect_f1.get("ok", False):
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
    
    # Recreate Panel 2: ROC-AUC (Middle)
    ax2_split = axes1[1]
    pivot_roc_mean.plot(kind='bar', ax=ax2_split, width=0.8, edgecolor='black', alpha=0.8,
                       yerr=pivot_roc_ci, capsize=4, error_kw={'linewidth': 1.5})
    ax2_split.set_ylabel('ROC-AUC', fontsize=14, fontweight='bold')
    ax2_split.set_title('ROC-AUC by CLD and Prompt (±95% CI)', fontsize=14, fontweight='bold')
    ax2_split.set_ylim([0, 1])
    ax2_split.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (AUC=0.5)')
    ax2_split.set_xticklabels(ax2_split.get_xticklabels(), rotation=15, ha='right')
    ax2_split.legend(title='', fontsize=9, loc='lower left', bbox_to_anchor=(0.02, 0.07))
    ax2_split.grid(axis='y', alpha=0.3)

    if prompt_effect_auc and prompt_effect_auc.get("ok", False):
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
    
    # Recreate Panel 3: Performance by Prompt (SWAPPED)
    ax3_split = axes1[2]
    metrics = ['ROC-AUC', 'F1', 'Precision', 'Recall']
    metric_cols = ['roc_auc_mean', 'f1_mean', 'precision_mean', 'recall_mean']
    x = np.arange(len(metrics))
    width_bar = 0.2
    
    for i, prompt in enumerate(per_prompt_avg.index):
        values = [per_prompt_avg.loc[prompt, col] for col in metric_cols]
        errors = [per_prompt_ci.loc[prompt, col] for col in metric_cols]
        ax3_split.bar(x + i * width_bar, values, width_bar, label=get_prompt_label(prompt), 
               yerr=errors, capsize=4, alpha=0.8, edgecolor='black')
    
    ax3_split.set_xticks(x + width_bar * 1.5)
    ax3_split.set_xticklabels(metrics)
    ax3_split.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax3_split.set_title('Performance by Prompt (±95% CI)', fontsize=14, fontweight='bold')
    ax3_split.set_ylim([0, 1])
    ax3_split.legend(title='', loc='upper right', fontsize=9)
    ax3_split.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file_row1 = output_dir / "rq1a_ground_truth_row1_performance_metrics.png"
    fig1.savefig(output_file_row1, dpi=300, bbox_inches='tight')
    print(f"  ✓ Row 1 (Performance Metrics 1x3): {output_file_row1.name}")
    plt.close(fig1)
    
    # FIGURE 2: Row 2 - Detailed Analysis (1x3 layout)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    # Recreate Panel 4: Precision vs Recall (with CLD markers)
    ax4_split = axes2[0]
    
    # Define colors for prompts (mapped to display labels)
    base_prompt_colors = {
        'baseline': '#4472C4',  # Blue
        'mechanistic': '#ED7D31',  # Orange
        'cot': '#70AD47',  # Green
        'mechanistic_lit': '#FFC000',  # Yellow/Amber
        'mechanistic_original': '#5B9BD5',  # Additional mech variant if present
    }
    prompt_colors = {get_prompt_label(k).lower(): v for k, v in base_prompt_colors.items()}
    
    # Define markers for CLDs
    cld_markers = {
        'depressive': 'o',  # Circle
        'emergency_department': 's',  # Square
        'social_norms': '^'  # Triangle
    }
    
    # Plot each point with color (prompt) and marker (CLD)
    for idx, row in by_cld_prompt_df.iterrows():
        prompt = row['prompt']
        cld = row['cld']
        recall = row['recall_mean']
        precision = row['precision_mean']
        
        color = prompt_colors.get(prompt.lower(), '#808080')  # Default gray if not found
        marker = cld_markers.get(cld.lower(), 'o')  # Default circle if not found
        
        ax4_split.scatter(recall, precision, 
                   s=200, alpha=0.7, color=color, marker=marker,
                   edgecolors='black', linewidth=1.5)
    
    ax4_split.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax4_split.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax4_split.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
    ax4_split.set_xlim([0, 1])
    ax4_split.set_ylim([0, 1])
    ax4_split.grid(alpha=0.3)
    
    # Create dual legend: prompts (colors) and CLDs (markers)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Prompt legend (top-left) - using colored squares
    prompts_in_data = sorted(by_cld_prompt_df['prompt'].unique())
    # Display names with proper capitalization (use display-label keys)
    prompt_display_names = {get_prompt_label(k).lower(): get_prompt_label(k) 
                            for k in base_prompt_colors.keys()}
    prompt_legend = [Patch(facecolor=prompt_colors.get(p.lower(), '#808080'), 
                          edgecolor='black', label=prompt_display_names.get(p.lower(), get_prompt_label(p)))
                    for p in prompts_in_data]
    
    # CLD legend (bottom-right) - using gray markers
    clds_in_data = sorted(by_cld_prompt_df['cld'].unique())
    cld_legend = [Line2D([0], [0], marker=cld_markers.get(c.lower(), 'o'), 
                        color='w', label=c.replace('_', ' ').title(),
                        markerfacecolor='gray', markersize=10, markeredgecolor='black')
                 for c in clds_in_data]
    
    first_legend = ax4_split.legend(handles=prompt_legend, title='', 
                                    loc='upper left', fontsize=9)
    ax4_split.add_artist(first_legend)
    ax4_split.legend(handles=cld_legend, title='CLD', loc='upper right', fontsize=8)
    
    # Recreate Panel 5: Point-Biserial Correlation (swapped to middle)
    ax5_split = axes2[1]
    pivot_pb_mean = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='point_biserial_r_mean')
    pivot_pb_ci = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='point_biserial_r_ci95')
    pivot_pb_pval = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='pb_p_value_mean')
    # Rename columns to display names for consistent legends
    pivot_pb_mean.columns = [get_prompt_label(c) for c in pivot_pb_mean.columns]
    pivot_pb_ci.columns = [get_prompt_label(c) for c in pivot_pb_ci.columns]
    pivot_pb_pval.columns = [get_prompt_label(c) for c in pivot_pb_pval.columns]
    # Rename columns to display names for consistent legends
    pivot_pb_mean.columns = [get_prompt_label(c) for c in pivot_pb_mean.columns]
    pivot_pb_ci.columns = [get_prompt_label(c) for c in pivot_pb_ci.columns]
    pivot_pb_pval.columns = [get_prompt_label(c) for c in pivot_pb_pval.columns]
    
    pivot_pb_mean.plot(kind='bar', ax=ax5_split, width=0.8, edgecolor='black', alpha=0.8,
                       yerr=pivot_pb_ci, capsize=4, error_kw={'linewidth': 1.5},
                       color=[prompt_colors.get(c.lower(), '#808080') for c in pivot_pb_mean.columns])
    ax5_split.set_ylabel('Point-Biserial r', fontsize=14, fontweight='bold')
    ax5_split.set_title('Score Discrimination (±95% CI)', fontsize=14, fontweight='bold')
    ax5_split.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax5_split.set_xticklabels(ax5_split.get_xticklabels(), rotation=15, ha='right')
    ax5_split.legend(title='', fontsize=9)
    ax5_split.grid(axis='y', alpha=0.3)
    
    # Add significance stars based on p-value from pooled statistical test
    # Use p-value from scipy.stats.pointbiserialr computed on pooled data
    n_prompts = len(pivot_pb_mean.columns)
    bar_width = 0.8 / n_prompts  # Each bar's width within the group
    
    # Track min/max y positions for axis limits
    y_positions = []
    
    for i, cld in enumerate(pivot_pb_mean.index):
        for j, prompt in enumerate(pivot_pb_mean.columns):
            r_val = pivot_pb_mean.loc[cld, prompt]
            p_val = pivot_pb_pval.loc[cld, prompt]
            ci_val = pivot_pb_ci.loc[cld, prompt] if pd.notna(pivot_pb_ci.loc[cld, prompt]) else 0
            
            if pd.notna(r_val) and pd.notna(p_val):
                # Use p-value from pooled statistical test
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                
                if sig:
                    # Calculate x position: group center (i) - half group width + bar center
                    # For pandas bar plot: bars are centered at i, distributed within width=0.8
                    x_pos = i - 0.4 + (j + 0.5) * bar_width
                    y_pos = r_val + ci_val + 0.02 if r_val > 0 else r_val - ci_val - 0.05
                    y_positions.append(y_pos)
                    ax5_split.text(x_pos, y_pos, sig, ha='center', va='bottom' if r_val > 0 else 'top', 
                                  fontsize=12, fontweight='bold', color='black')
    
    # Adjust y-axis limits to include significance markers with padding
    if y_positions:
        current_ylim = ax5_split.get_ylim()
        y_min = min(current_ylim[0], min(y_positions) - 0.08)
        y_max = max(current_ylim[1], max(y_positions) + 0.08)
        ax5_split.set_ylim([y_min, y_max])
    
    # Recreate Panel 6: Score distributions by classification (swapped to right)
    ax6_split = axes2[2]
    score_cols = ['tp_mean_score_mean', 'tn_mean_score_mean', 'fp_mean_score_mean', 'fn_mean_score_mean']
    count_cols = ['n_tp_edges_sum', 'n_tn_edges_sum', 'n_fp_edges_sum', 'n_fn_edges_sum']
    classifications = ['TP', 'TN', 'FP', 'FN']
    
    avg_scores = [by_cld_prompt_df[col].mean() for col in score_cols]
    std_scores = [by_cld_prompt_df[col].std() for col in score_cols]
    total_counts = [int(by_cld_prompt_df[col].sum()) for col in count_cols]
    n_samples = len(by_cld_prompt_df)
    
    from scipy import stats as sp_stats
    ci_scores = [std / np.sqrt(n_samples) * sp_stats.t.ppf(0.975, n_samples - 1) 
                 if pd.notna(std) else 0 for std in std_scores]
    
    # Calculate percentages
    total_all = sum(total_counts)
    percentages = [100 * c / total_all for c in total_counts]
    
    # Color bars: GREEN for correct (TP, TN), RED for errors (FP, FN)
    bar_colors = ['#99ff99', '#99ff99', '#ff9999', '#ff9999']
    
    bars = ax6_split.bar(range(len(classifications)), avg_scores, yerr=ci_scores, capsize=5,
                   color=bar_colors, alpha=0.8, edgecolor='black', linewidth=2,
                   error_kw={'linewidth': 2, 'elinewidth': 2})
    
    # Add value labels on bars
    for i, (bar, score, ci) in enumerate(zip(bars, avg_scores, ci_scores)):
        height = bar.get_height()
        ax6_split.text(bar.get_x() + bar.get_width()/2., height + ci + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax6_split.set_xticks(range(len(classifications)))
    ax6_split.set_xticklabels(classifications, fontsize=14, fontweight='bold')
    ax6_split.set_ylabel('Mean Judge Score (μ)', fontsize=12, fontweight='bold')
    ax6_split.set_xlabel('Classification Type', fontsize=12, fontweight='bold')
    ax6_split.set_title('Judge Scores by Hallucination-Detection\nOutcome (±95% CI)', fontsize=12, fontweight='bold')
    ax6_split.set_ylim([0, 1.15])
    ax6_split.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at 0.5 (threshold)
    ax6_split.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add legend with counts
    legend_labels = [
        f'TP: n={total_counts[0]:,} ({percentages[0]:.1f}%)',
        f'TN: n={total_counts[1]:,} ({percentages[1]:.1f}%)',
        f'FP: n={total_counts[2]:,} ({percentages[2]:.1f}%)',
        f'FN: n={total_counts[3]:,} ({percentages[3]:.1f}%)',
        'Threshold (0.5)'
    ]
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Patch(facecolor=bar_colors[i], edgecolor='black', linewidth=1.5,
                            label=legend_labels[i]) for i in range(4)]
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                                 label=legend_labels[4]))
    ax6_split.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                    fontsize=8, title=f'Total: n={total_all:,} edges', title_fontsize=9, framealpha=0.95)
    
    plt.tight_layout()
    output_file_row2 = output_dir / "rq1a_ground_truth_row2_detailed_analysis.png"
    fig2.savefig(output_file_row2, dpi=300, bbox_inches='tight')
    print(f"  ✓ Row 2 (Detailed Analysis 1x3): {output_file_row2.name}")
    plt.close(fig2)
    
    plt.close('all')
    
    # Figure 2: Correlation heatmap with CIs
    # Detect judge type from output_dir for descriptive naming
    output_dir_str = str(output_dir).lower()
    if 'citation' in output_dir_str:
        judge_type = 'citation'
        judge_label = 'Citation Judge'
    else:
        judge_type = 'correctness'
        judge_label = 'Correctness Judge'
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # For each CLD×Prompt combination, get the raw scores across runs to compute CI
    # We need to go back to raw_df to compute proper CIs
    heatmap_data = []
    for _, row in by_cld_prompt_df.iterrows():
        cld = row['cld']
        prompt = row['prompt']
        
        # Get the 3 runs for this CLD×prompt combination
        runs_data = raw_df[(raw_df['cld'] == cld) & (raw_df['prompt'] == prompt)]
        
        # Compute mean and 95% CI for each classification score type
        from scipy import stats as sp_stats
        entry = {'CLD_Prompt': f"{cld}_{prompt}"}
        
        for score_type, col_name in [('TP Score', 'tp_mean_score'), 
                                      ('TN Score', 'tn_mean_score'),
                                      ('FP Score', 'fp_mean_score'), 
                                      ('FN Score', 'fn_mean_score')]:
            scores = runs_data[col_name].dropna()
            if len(scores) > 1:
                mean = scores.mean()
                std = scores.std()
                n = len(scores)
                ci = std / np.sqrt(n) * sp_stats.t.ppf(0.975, n - 1)
                entry[score_type] = mean
                entry[f"{score_type}_CI"] = ci
            else:
                entry[score_type] = scores.iloc[0] if len(scores) == 1 else np.nan
                entry[f"{score_type}_CI"] = 0
        
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
            if pd.notna(mean_val) and pd.notna(ci_val):
                row_annots.append(f"{mean_val:.3f}\n±{ci_val:.3f}")
            else:
                row_annots.append(f"{mean_val:.3f}")
        annot_matrix.append(row_annots)
    
    # Transpose for heatmap (scores as rows, CLD×Prompt as columns)
    annot_array = np.array(annot_matrix)
    
    sns.heatmap(mean_df.T, annot=annot_array, fmt='', cmap='RdYlGn', 
                center=0.75, vmin=0.5, vmax=1.0, ax=ax, 
                cbar_kws={'label': 'Mean Judge Score'},
                annot_kws={'fontsize': 8})
    ax.set_title(f'Ground Truth Validation Score Heatmap ({judge_label})\nJudge Scores by Classification (Mean ± 95% CI)', 
                 fontweight='bold', fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('Classification', fontweight='bold')
    
    plt.tight_layout()
    heatmap_filename = f'rq1a_ground_truth_{judge_type}_score_heatmap.png'
    plt.savefig(output_dir / heatmap_filename, 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Score heatmap saved: {heatmap_filename}")
    
    # NEW: Generate ROC curves
    if all_results:
        create_ground_truth_roc_curves(all_results, output_dir)
    
    # NEW: Generate individual high-DPI figures for thesis
    print("\n📊 Generating individual high-DPI figures for thesis...")
    create_individual_ground_truth_figures_high_dpi(by_cld_prompt_df, raw_df, output_dir)


def create_ground_truth_roc_curves(all_results: dict, output_dir: Path):
    """Create ROC curves for ground truth experiments."""
    print("  Creating ROC curves...")
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Map prompt names to display labels
    prompt_labels = {
        'baseline': 'Baseline', 
        'mechanistic': 'Mechanistic', 
        'mechanistic_original': 'Mechanistic',
        'cot': 'CoT', 
        'mechanistic_lit': 'Mech-Lit'
    }
    
    # Get unique prompts from the data
    all_prompts = set()
    for cld_name, runs in all_results.items():
        for run_name, files in runs.items():
            all_prompts.update(files.keys())
    
    print(f"    Prompts found in data: {all_prompts}")
    
    # Select prompts to plot - prioritize order but use what's available
    preferred_order = ['baseline', 'cot', 'mechanistic', 'mechanistic_lit', 'mechanistic_original']
    prompts_to_plot = [p for p in preferred_order if p in all_prompts][:4]
    
    print(f"    Plotting prompts: {prompts_to_plot}")
    
    # Adjust figure size based on number of prompts
    n_prompts = len(prompts_to_plot)
    if n_prompts == 0:
        print("    ⚠️  No prompts found, skipping ROC curves")
        plt.close()
        return
    
    # Pad to fill remaining subplots
    while len(prompts_to_plot) < 4:
        prompts_to_plot.append(None)
    
    for idx, prompt_name in enumerate(prompts_to_plot):
        ax = axes[idx]
        
        if prompt_name is None:
            ax.set_visible(False)
            continue
        
        prompt_label = prompt_labels.get(prompt_name, prompt_name.replace('_', ' ').title())
        
        for cld_name, runs in all_results.items():
            all_y_true = []
            all_y_scores = []
            
            for run_name, files in runs.items():
                # Handle mechanistic/mechanistic_original as the same
                prompt_variants = [prompt_name]
                if prompt_name == 'mechanistic':
                    prompt_variants = ['mechanistic', 'mechanistic_original']
                elif prompt_name == 'mechanistic_original':
                    prompt_variants = ['mechanistic_original', 'mechanistic']
                
                # Find which variant exists
                actual_prompt = None
                for pv in prompt_variants:
                    if pv in files:
                        actual_prompt = pv
                        break
                
                if actual_prompt:
                    excel_path = files[actual_prompt]
                    try:
                        df = pd.read_excel(excel_path, sheet_name='All Edges')
                        
                        # Extract ground truth and scores
                        for _, row in df.iterrows():
                            # Determine ground truth based on Classification column
                            # FP and FN are hallucinations (wrong edges), TP and TN are correct
                            classification = row.get('Classification', '')
                            if classification in ['FP', 'FN']:
                                is_hallucination = 1
                            elif classification in ['TP', 'TN']:
                                is_hallucination = 0
                            else:
                                continue  # Skip unknown classifications
                            
                            # Get aggregate score
                            agg_score = row.get('Aggregate Score', np.nan)
                            if pd.isna(agg_score):
                                judge_msg = row.get('Judge Message', '')
                                if isinstance(judge_msg, str) and judge_msg.startswith('{'):
                                    try:
                                        import json
                                        judge_data = json.loads(judge_msg)
                                        agg_score = judge_data.get('aggregate_score', np.nan)
                                    except:
                                        pass
                            
                            if pd.notna(agg_score):
                                all_y_true.append(is_hallucination)
                                # Lower score = more likely hallucination, so invert for ROC
                                all_y_scores.append(1 - agg_score)
                    except Exception as e:
                        pass
            
            if len(all_y_true) > 0 and len(set(all_y_true)) == 2:
                fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
                auc = roc_auc_score(all_y_true, all_y_scores)
                cld_label = cld_name.replace('_', ' ').title()
                ax.plot(fpr, tpr, label=f'{cld_label} (AUC={auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'{prompt_label} Prompt', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    output_file = output_dir / "rq1a_ground_truth_roc_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curves saved: {output_file}")
    plt.close()


def create_individual_ground_truth_figures_high_dpi(by_cld_prompt_df: pd.DataFrame,
                                                     raw_df: pd.DataFrame,
                                                     output_dir: Path):
    """
    Create individual high-DPI publication-quality figures for ground truth experiments.
    
    Each figure is saved separately with:
    - 300 DPI for print quality
    - Larger, BOLD fonts for readability
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
    
    base_colors = {'baseline': '#4472C4', 'mechanistic': '#ED7D31', 'cot': '#70AD47', 
                   'mechanistic_lit': '#FFC000', 'mechanistic_original': '#5B9BD5'}
    # Map colors to display labels (after prompt renaming)
    colors = {get_prompt_label(k): v for k, v in base_colors.items()}
    
    # Get actual prompts from data
    prompts_in_data = sorted(by_cld_prompt_df['prompt'].unique())
    
    # ========================================================================
    # Figure 1: ROC-AUC by CLD and Prompt
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot_roc_mean = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='roc_auc_mean')
    pivot_roc_ci = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='roc_auc_ci95')
    
    # Use pandas plot which handles NaN values automatically
    pivot_roc_mean.plot(kind='bar', ax=ax, width=0.7, edgecolor='black', alpha=0.85,
                       yerr=pivot_roc_ci, capsize=5, error_kw={'linewidth': 1.5},
                       color=[colors.get(p, None) for p in pivot_roc_mean.columns],
                       linewidth=1.5)
    
    ax.set_xlabel('Causal Loop Diagram', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_ylabel('AUC-ROC', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_title('ROC-AUC: Overall Discrimination Performance', 
                 fontsize=pub_fontsize_title, fontweight='bold', pad=20)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in pivot_roc_mean.index],
                       rotation=0, fontsize=pub_fontsize_tick)
    ax.tick_params(axis='y', labelsize=pub_fontsize_tick)
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    # Update legend labels to be title case
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [l.replace('_', ' ').title() for l in labels],
             fontsize=pub_fontsize_legend, frameon=True, shadow=True, loc='best', title='')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_file = individual_dir / "fig_roc_auc_by_cld_prompt.png"
    plt.savefig(output_file, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ ROC-AUC figure: {output_file.name}")
    plt.close()
    
    # ========================================================================
    # Figure 2: F1 Score by CLD and Prompt
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot_f1_mean = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='f1_mean')
    pivot_f1_ci = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='f1_ci95')
    
    # Use pandas plot which handles NaN values automatically
    pivot_f1_mean.plot(kind='bar', ax=ax, width=0.7, edgecolor='black', alpha=0.85,
                      yerr=pivot_f1_ci, capsize=5, error_kw={'linewidth': 1.5},
                      color=[colors.get(p, None) for p in pivot_f1_mean.columns],
                      linewidth=1.5)
    
    ax.set_xlabel('Causal Loop Diagram', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_title('F1 Score at Threshold ≤0.5 (PC+INC Flagged)', 
                 fontsize=pub_fontsize_title, fontweight='bold', pad=20)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in pivot_f1_mean.index],
                       rotation=0, fontsize=pub_fontsize_tick)
    ax.tick_params(axis='y', labelsize=pub_fontsize_tick)
    ax.set_ylim([0, 1])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [l.replace('_', ' ').title() for l in labels],
             fontsize=pub_fontsize_legend, frameon=True, shadow=True, loc='best', title='')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_file = individual_dir / "fig_f1_score_by_cld_prompt.png"
    plt.savefig(output_file, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ F1 score figure: {output_file.name}")
    plt.close()
    
    # ========================================================================
    # Figure 3: Point-Biserial Correlation
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot_pb_mean = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='point_biserial_r_mean')
    pivot_pb_ci = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='point_biserial_r_ci95')
    pivot_pb_pval = by_cld_prompt_df.pivot(index='cld', columns='prompt', values='pb_p_value_mean')
    
    # Use pandas plot which handles NaN values automatically  
    pivot_pb_mean.plot(kind='bar', ax=ax, width=0.7, edgecolor='black', alpha=0.85,
                      yerr=pivot_pb_ci, capsize=5, error_kw={'linewidth': 1.5},
                      color=[colors.get(p, '#808080') for p in pivot_pb_mean.columns],
                      linewidth=1.5)
    
    # Add significance markers manually after plotting (using p-value from pooled test)
    for i, cld in enumerate(pivot_pb_mean.index):
        for j, prompt in enumerate(pivot_pb_mean.columns):
            r_val = pivot_pb_mean.loc[cld, prompt]
            p_val = pivot_pb_pval.loc[cld, prompt]
            ci_val = pivot_pb_ci.loc[cld, prompt]
            
            if pd.notna(r_val) and pd.notna(p_val):
                # Use p-value from pooled statistical test
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                
                if sig:
                    y_pos = r_val + (ci_val + 0.03 if r_val > 0 else -(ci_val + 0.03))
                    # Position based on bar groups
                    n_bars = len(pivot_pb_mean.columns)
                    bar_width = 0.7 / n_bars
                    x_pos = i + (j - n_bars/2 + 0.5) * bar_width
                    ax.text(x_pos, y_pos, sig, ha='center', va='bottom' if r_val > 0 else 'top',
                           fontsize=14, fontweight='bold', color='red')
    
    ax.set_xlabel('Causal Loop Diagram', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_ylabel('Point-Biserial Correlation (r)', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_title('Correlation: Hallucination vs Judge Score', 
                 fontsize=pub_fontsize_title, fontweight='bold', pad=20)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in pivot_pb_mean.index],
                       rotation=0, fontsize=pub_fontsize_tick)
    ax.tick_params(axis='y', labelsize=pub_fontsize_tick)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [l.replace('_', ' ').title() for l in labels],
             fontsize=pub_fontsize_legend, frameon=True, shadow=True, loc='best', title='')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add significance legend
    ax.text(0.02, 0.98, '*** p < .001  ** p < .01  * p < .05',
           transform=ax.transAxes, fontsize=pub_fontsize_tick, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    output_file = individual_dir / "fig_point_biserial_correlation.png"
    plt.savefig(output_file, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Point-biserial correlation figure: {output_file.name}")
    plt.close()
    
    # ========================================================================
    # Figure 4: Precision vs Recall
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for prompt in prompts_in_data:
        data = by_cld_prompt_df[by_cld_prompt_df['prompt'] == prompt]
        ax.scatter(data['recall_mean'], data['precision_mean'],
                  s=200, label=get_prompt_label(prompt), alpha=0.7,
                  color=colors.get(prompt, None),
                  edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_title('Precision vs Recall Trade-off (threshold=0.5)', 
                 fontsize=pub_fontsize_title, fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.tick_params(axis='both', labelsize=pub_fontsize_tick)
    ax.legend(fontsize=pub_fontsize_legend, frameon=True, shadow=True)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_file = individual_dir / "fig_precision_recall_tradeoff.png"
    plt.savefig(output_file, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Precision-Recall figure: {output_file.name}")
    plt.close()
    
    # ========================================================================
    # Figure 5: Score Distributions by Classification Type
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    score_cols = ['tp_mean_score_mean', 'tn_mean_score_mean', 'fp_mean_score_mean', 'fn_mean_score_mean']
    score_labels = ['TP\n(Correct)', 'TN\n(Correct)', 'FP\n(Hallucination)', 'FN\n(Hallucination)']
    score_colors = ['darkgreen', 'lightgreen', 'red', 'darkred']
    
    # Calculate mean and 95% CI across all CLDs and prompts
    avg_scores = [by_cld_prompt_df[col].mean() for col in score_cols]
    std_scores = [by_cld_prompt_df[col].std() for col in score_cols]
    n_samples = len(by_cld_prompt_df)
    
    from scipy import stats as sp_stats
    ci_scores = [std / np.sqrt(n_samples) * sp_stats.t.ppf(0.975, n_samples - 1) 
                 for std in std_scores]
    
    bars = ax.bar(range(4), avg_scores, yerr=ci_scores, capsize=8,
                  color=score_colors, edgecolor='black', linewidth=2, alpha=0.85)
    
    ax.set_ylabel('Mean Judge Score', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_xlabel('Classification Type', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_title('Judge Scores by Hallucination-Detection Outcome (Ground Truth)', 
                 fontsize=pub_fontsize_title, fontweight='bold', pad=20)
    ax.set_xticks(range(4))
    ax.set_xticklabels(score_labels, fontsize=pub_fontsize_tick)
    ax.tick_params(axis='y', labelsize=pub_fontsize_tick)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold')
    ax.legend(fontsize=pub_fontsize_legend, frameon=True, shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_file = individual_dir / "fig_score_distributions_by_classification.png"
    plt.savefig(output_file, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Score distributions figure: {output_file.name}")
    plt.close()
    
    # ========================================================================
    # Figure 6: Score Heatmap (CLD × Prompt)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap data
    heatmap_data = []
    for _, row in by_cld_prompt_df.iterrows():
        cld = row['cld']
        prompt = row['prompt']
        runs_data = raw_df[(raw_df['cld'] == cld) & (raw_df['prompt'] == prompt)]
        
        entry = {'CLD_Prompt': f"{cld.replace('_', ' ').title()}\\n{prompt.title()}"}
        
        for score_type, col_name in [('TP', 'tp_mean_score'), 
                                      ('TN', 'tn_mean_score'),
                                      ('FP', 'fp_mean_score'), 
                                      ('FN', 'fn_mean_score')]:
            scores = runs_data[col_name].dropna()
            entry[score_type] = scores.mean() if len(scores) > 0 else np.nan
        
        heatmap_data.append(entry)
    
    heatmap_df = pd.DataFrame(heatmap_data).set_index('CLD_Prompt')
    mean_cols = ['TP', 'TN', 'FP', 'FN']
    mean_df = heatmap_df[mean_cols]
    
    sns.heatmap(mean_df.T, annot=True, fmt='.3f', cmap='RdYlGn',
               center=0.75, vmin=0.5, vmax=1.0, ax=ax,
               cbar_kws={'label': 'Mean Judge Score'},
               annot_kws={'fontsize': pub_fontsize_tick, 'weight': 'bold'},
               linewidths=1, linecolor='white')
    
    ax.set_title('Judge Scores by Hallucination-Detection Outcome (CLD × Prompt)',
                fontsize=pub_fontsize_title, fontweight='bold', pad=20)
    ax.set_xlabel('', fontsize=pub_fontsize_label, fontweight='bold')
    ax.set_ylabel('Classification Type', fontsize=pub_fontsize_label, fontweight='bold')
    ax.tick_params(axis='both', labelsize=pub_fontsize_tick)
    
    plt.tight_layout()
    output_file = individual_dir / "fig_score_heatmap_cld_prompt.png"
    plt.savefig(output_file, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Score heatmap figure: {output_file.name}")
    plt.close()
    
    print(f"\n✅ Individual high-DPI figures created in: {individual_dir}")
    print(f"   All figures saved at {pub_dpi} DPI for publication quality\n")


def save_results(by_cld_prompt_df: pd.DataFrame, 
                raw_df: pd.DataFrame,
                output_dir: Path,
                assumption_results: dict = None):
    """Save results to Excel."""
    
    excel_path = output_dir / 'rq1a_ground_truth_enhanced_results.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # By CLD and Prompt (main aggregation)
        by_cld_prompt_df.to_excel(writer, sheet_name='By CLD and Prompt', index=False)
        
        # By Prompt only (averaged across CLDs)
        by_prompt = by_cld_prompt_df.groupby('prompt').agg({
            'f1_mean': 'mean',
            'precision_mean': 'mean',
            'recall_mean': 'mean',
            'accuracy_mean': 'mean',
            'point_biserial_r_mean': 'mean',
            'tp_mean_score_mean': 'mean',
            'tn_mean_score_mean': 'mean',
            'fp_mean_score_mean': 'mean',
            'fn_mean_score_mean': 'mean',
            'n_tp_edges_sum': 'sum',
            'n_tn_edges_sum': 'sum',
            'n_fp_edges_sum': 'sum',
            'n_fn_edges_sum': 'sum',
            'n_total_sum': 'sum'
        }).reset_index()
        by_prompt.to_excel(writer, sheet_name='By Prompt', index=False)
        
        # By CLD only (averaged across prompts)
        by_cld = by_cld_prompt_df.groupby('cld').agg({
            'f1_mean': 'mean',
            'precision_mean': 'mean',
            'recall_mean': 'mean',
            'accuracy_mean': 'mean',
            'point_biserial_r_mean': 'mean',
            'tp_mean_score_mean': 'mean',
            'tn_mean_score_mean': 'mean',
            'fp_mean_score_mean': 'mean',
            'fn_mean_score_mean': 'mean',
            'n_tp_edges_sum': 'sum',
            'n_tn_edges_sum': 'sum',
            'n_fp_edges_sum': 'sum',
            'n_fn_edges_sum': 'sum',
            'n_total_sum': 'sum'
        }).reset_index()
        by_cld.to_excel(writer, sheet_name='By CLD', index=False)
        
        # Raw data (all runs)
        raw_df.to_excel(writer, sheet_name='Raw Data', index=False)
        
        # Assumption tests (aligned with corruption-analysis script)
        if assumption_results:
            normality = assumption_results.get('normality_tests', [])
            variance = assumption_results.get('variance_tests', [])
            if normality:
                pd.DataFrame(normality).to_excel(writer, sheet_name='Normality Tests', index=False)
            if variance:
                pd.DataFrame(variance).to_excel(writer, sheet_name='Variance Tests', index=False)
    
    print(f"✓ Results saved to: {excel_path}")
    
    # Generate and save LaTeX tables
    latex_tables = generate_latex_tables(by_cld_prompt_df, raw_df)
    for table_name, content in latex_tables.items():
        latex_path = output_dir / f'table_{table_name}.tex'
        with open(latex_path, 'w') as f:
            f.write(content)
        print(f"✓ LaTeX table saved: {latex_path}")


def generate_latex_tables(by_cld_prompt_df: pd.DataFrame, raw_df: pd.DataFrame) -> dict:
    """Generate LaTeX tables for ground truth validation results."""
    tables = {}
    
    # CLD order for consistency with thesis
    cld_order = ['social_norms', 'depressive', 'emergency_department']
    cld_names = {
        'social_norms': 'Social Norms',
        'depressive': 'Depressive', 
        'emergency_department': 'Emergency Dept'
    }
    
    # Table 1: Main Results (CLD × Prompt)
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{RQ1a: Ground Truth Validation Performance (by CLD and Prompt)}")
    latex_lines.append("\\label{tab:rq1a_gt_main}")
    latex_lines.append("\\begin{tabular}{llcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{CLD} & \\textbf{Prompt} & \\textbf{N} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{AUC} & \\textbf{r$_{pb}$} \\\\")
    latex_lines.append("\\midrule")
    
    for cld in cld_order:
        cld_data = by_cld_prompt_df[by_cld_prompt_df['cld'] == cld].sort_values('prompt')
        if cld_data.empty:
            continue
        for idx, row in cld_data.iterrows():
            cld_label = cld_names.get(cld, cld) if idx == cld_data.index[0] else ''
            p = f"{row['precision_mean']:.3f}"
            r = f"{row['recall_mean']:.3f}"
            f1 = f"{row['f1_mean']:.3f}"
            auc = f"{row.get('roc_auc_mean', 0):.3f}" if pd.notna(row.get('roc_auc_mean')) else "N/A"
            rpb = f"{row['point_biserial_r_mean']:.3f}"
            n = int(row.get('n_total_sum', 0))
            
            latex_lines.append(f"{cld_label} & {row['prompt'].capitalize()} & {n} & {p} & {r} & {f1} & {auc} & {rpb} \\\\")
        latex_lines.append("\\midrule")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    tables['main'] = "\n".join(latex_lines)
    
    # Table 2: Per-Prompt Summary
    by_prompt = by_cld_prompt_df.groupby('prompt').agg({
        'f1_mean': ['mean', 'std', 'min', 'max'],
        'precision_mean': ['mean', 'std', 'min', 'max'],
        'recall_mean': ['mean', 'std', 'min', 'max'],
        'point_biserial_r_mean': ['mean', 'std', 'min', 'max'],
        'n_total_sum': 'sum'
    }).reset_index()
    by_prompt.columns = ['_'.join(col).strip('_') for col in by_prompt.columns.values]
    
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{RQ1a: Ground Truth Validation Performance by Prompt (macro-averaged across CLDs)}")
    latex_lines.append("\\label{tab:rq1a_gt_by_prompt}")
    latex_lines.append("\\begin{tabular}{lccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Prompt} & \\textbf{N} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{r$_{pb}$} \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in by_prompt.iterrows():
        prompt = row['prompt'].capitalize()
        n = int(row['n_total_sum_sum'])
        p = f"${row['precision_mean_mean']:.3f}$ [{row['precision_mean_min']:.3f}, {row['precision_mean_max']:.3f}]"
        r = f"${row['recall_mean_mean']:.3f}$ [{row['recall_mean_min']:.3f}, {row['recall_mean_max']:.3f}]"
        f1 = f"${row['f1_mean_mean']:.3f}$ [{row['f1_mean_min']:.3f}, {row['f1_mean_max']:.3f}]"
        rpb = f"${row['point_biserial_r_mean_mean']:.3f}$ [{row['point_biserial_r_mean_min']:.3f}, {row['point_biserial_r_mean_max']:.3f}]"
        latex_lines.append(f"{prompt} & {n} & {p} & {r} & {f1} & {rpb} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\note{Values shown as mean [min, max] across CLDs per uncertainty reporting rule.}")
    latex_lines.append("\\end{table}")
    tables['by_prompt'] = "\n".join(latex_lines)
    
    # Table 3: Classification Statistics (TP/TN/FP/FN counts)
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{RQ1a: Ground Truth Classification Statistics}")
    latex_lines.append("\\label{tab:rq1a_gt_classification}")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{CLD} & \\textbf{Prompt} & \\textbf{TP} & \\textbf{TN} & \\textbf{FP} & \\textbf{FN} & \\textbf{Total} \\\\")
    latex_lines.append("\\midrule")
    
    for cld in cld_order:
        cld_data = by_cld_prompt_df[by_cld_prompt_df['cld'] == cld].sort_values('prompt')
        if cld_data.empty:
            continue
        for idx, row in cld_data.iterrows():
            cld_label = cld_names.get(cld, cld) if idx == cld_data.index[0] else ''
            tp = int(row.get('n_tp_edges_sum', 0))
            tn = int(row.get('n_tn_edges_sum', 0))
            fp = int(row.get('n_fp_edges_sum', 0))
            fn = int(row.get('n_fn_edges_sum', 0))
            total = tp + tn + fp + fn
            latex_lines.append(f"{cld_label} & {row['prompt'].capitalize()} & {tp} & {tn} & {fp} & {fn} & {total} \\\\")
        latex_lines.append("\\midrule")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\note{TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative.}")
    latex_lines.append("\\end{table}")
    tables['classification'] = "\n".join(latex_lines)
    
    return tables


def main():
    parser = argparse.ArgumentParser(description='RQ1a Ground Truth Enhanced Analysis')
    parser.add_argument('--base_dir', type=str, 
                       default='../final_runs/RQ1a_gt_lit_correctness',
                       help='Base directory containing CLD subdirectories')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--judge-type', type=str, choices=['correctness', 'citation'],
                       default='correctness',
                       help='Judge type: correctness or citation (default: correctness)')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        sys.exit(1)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = base_dir / f'enhanced_analysis_{timestamp}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect results
    results = collect_all_run_results(base_dir)
    
    if not results:
        print("\n❌ No results found")
        sys.exit(1)
    
    # Aggregate
    print("\n" + "="*80)
    print(f"AGGREGATING RESULTS (Judge Type: {args.judge_type})")
    print("="*80)
    by_cld_prompt_df, raw_df = aggregate_results(results, judge_type=args.judge_type)
    
    print(f"✓ Aggregated by CLD and Prompt: {len(by_cld_prompt_df)} rows")
    print(f"✓ Raw data: {len(raw_df)} rows")
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY BY PROMPT (Averaged Across CLDs)")
    print("="*80)
    by_prompt = by_cld_prompt_df.groupby('prompt').agg({
        'roc_auc_mean': 'mean',
        'f1_mean': 'mean',
        'precision_mean': 'mean',
        'recall_mean': 'mean',
        'accuracy_mean': 'mean',
        'point_biserial_r_mean': 'mean'
    }).reset_index()
    print(by_prompt.to_string(index=False))
    
    print("\n" + "="*80)
    print("SUMMARY BY CLD (Averaged Across Prompts)")
    print("="*80)
    by_cld = by_cld_prompt_df.groupby('cld').agg({
        'roc_auc_mean': 'mean',
        'f1_mean': 'mean',
        'precision_mean': 'mean',
        'recall_mean': 'mean',
        'accuracy_mean': 'mean',
        'point_biserial_r_mean': 'mean'
    }).reset_index()
    print(by_cld.to_string(index=False))
    
    # Assumption testing (Shapiro–Wilk + Levene), aligned with corruption-analysis script
    assumption_df = build_assumption_test_df(by_cld_prompt_df)
    assumption_results = perform_assumption_tests(assumption_df)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    save_results(by_cld_prompt_df, raw_df, output_dir, assumption_results=assumption_results)
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_enhanced_visualizations(by_cld_prompt_df, raw_df, output_dir, judge_type=args.judge_type, all_results=results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results directory: {output_dir}")
    print(f"  - rq1a_ground_truth_enhanced_results.xlsx")
    print(f"  - rq1a_ground_truth_enhanced_visualization.png")
    print(f"  - rq1a_ground_truth_score_heatmap.png")
    print(f"  - rq1a_ground_truth_roc_curves.png")
    print()


if __name__ == "__main__":
    main()

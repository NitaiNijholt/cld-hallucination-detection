#!/usr/bin/env python3
"""
RQ2 Single UQ Metric Performance Table Generator

Generates Table: Single UQ Metric Performance for Hallucination Detection (Meta-Analysis)

Uses Mann-Whitney U tests (non-parametric) given evidence of non-normality.
Outputs:
- LaTeX table for thesis (single_metric_table.tex)
- Excel file for reproducibility (single_metric_table.xlsx)
- JSON archive (single_metric_results.json)

Author: Generated for thesis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import warnings
warnings.filterwarnings('ignore')

from rq2_paths import rq2_dirs, repo_root

# Number of single-metric tests for Bonferroni correction (per Methods Section 4.5)
N_METRIC_TESTS = 4  # Perplexity, Min Prob, Max Window Entropy, Cosine Similarity
ALPHA = 0.05
ALPHA_ADJ = ALPHA / N_METRIC_TESTS  # 0.0125


def bonferroni_correction(p_values: dict, alpha: float = 0.05) -> dict:
    """
    Apply Bonferroni correction to a family of p-values.
    
    Per Methods Section 4.5: RQ2 uses 4 single-metric tests → alpha_adj = 0.0125
    
    Args:
        p_values: Dict of {test_name: p_value}
        alpha: Significance level (default 0.05)
    
    Returns:
        Dict with correction results including adjusted p-values
    """
    m = len(p_values)
    if m == 0:
        return {
            'method': 'Bonferroni',
            'n_tests': 0,
            'alpha_original': alpha,
            'alpha_adjusted': np.nan,
            'results': {},
            'n_significant_original': 0,
            'n_significant_adjusted': 0,
        }
    
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
                'significant_adjusted': bool(p_adj < alpha),
                'changed': bool((p < alpha) != (p_adj < alpha))
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


def bootstrap_ci_blocks(data, n_boot=10000, alpha=0.05):
    """Compute bootstrap CI for mean of block-level data."""
    data = np.array(data)
    n = len(data)
    if n < 2:
        return np.nan, np.nan
        
    means = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
        
    return np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])

# CI Metrics to analyze
CI_METRICS = [
    'Gen Perplexity',
    'Gen Min Prob',
    'Gen Max Window Entropy',
    'Gen Cosine Similarity'
]

# Short names for LaTeX table
METRIC_SHORT_NAMES = {
    'Gen Perplexity': 'Gen Perplexity',
    'Gen Min Prob': 'Gen Min Prob',
    'Gen Max Window Entropy': 'Gen Max Window\\\\Entropy',
    'Gen Cosine Similarity': 'Gen Cosine\\\\Similarity'
}


def load_inventory(analyses_dir: Path):
    """Load the deduplicated inventory file."""
    dedup_files = list(analyses_dir.glob("rq2_valid_files_deduplicated_*.json"))
    if dedup_files:
        inventory_path = max(dedup_files, key=lambda p: p.stat().st_mtime)
    else:
        inventory_files = list(analyses_dir.glob("rq2_valid_files_*.json"))
        if not inventory_files:
            raise FileNotFoundError("No inventory file found!")
        inventory_path = max(inventory_files, key=lambda p: p.stat().st_mtime)
    
    with open(inventory_path, 'r') as f:
        return json.load(f), inventory_path


def analyze_file_for_metric(filepath: str, metric: str):
    """
    Analyze a single file for a single metric.
    
    Returns dict with:
    - auc: AUC-ROC
    - correlation_r: Point-biserial correlation
    - mwu_p: Mann-Whitney U p-value
    - mwu_significant: whether p < 0.05
    - rank_biserial: effect size
    - n_edges: number of edges
    - n_halluc: number of hallucinations
    - n_correct: number of correct
    """
    try:
        df = pd.read_excel(filepath, sheet_name='All Edges', engine='openpyxl')
        
        if 'Classification' not in df.columns or metric not in df.columns:
            return None
        
        # Create hallucination label
        df['is_hallucination'] = df['Classification'].isin(['FP', 'FN'])
        
        # Filter valid data
        df_clean = df[df[metric].notna() & np.isfinite(df[metric])].copy()
        
        if len(df_clean) < 10:
            return None
        
        halluc = df_clean[df_clean['is_hallucination']][metric]
        correct = df_clean[~df_clean['is_hallucination']][metric]
        
        if len(halluc) < 2 or len(correct) < 2:
            return None
        
        result = {
            'n_edges': len(df_clean),
            'n_halluc': len(halluc),
            'n_correct': len(correct)
        }
        
        # AUC-ROC
        try:
            result['auc'] = float(roc_auc_score(df_clean['is_hallucination'], df_clean[metric]))
        except:
            result['auc'] = None

        # PR-AUC (Average Precision) — baseline equals positive prevalence under random ranking
        # Useful under class imbalance: focuses on precision/recall trade-off for the positive class.
        try:
            result['ap'] = float(average_precision_score(df_clean['is_hallucination'], df_clean[metric]))
        except:
            result['ap'] = None
        
        # Point-biserial correlation (Pearson with binary)
        try:
            r, p = stats.pearsonr(df_clean[metric], df_clean['is_hallucination'].astype(int))
            result['correlation_r'] = float(r)
            result['correlation_p'] = float(p)
        except:
            result['correlation_r'] = None
            result['correlation_p'] = None
        
        # Mann-Whitney U test (non-parametric)
        try:
            U_stat, p_value = stats.mannwhitneyu(halluc, correct, alternative='two-sided')
            n1, n2 = len(halluc), len(correct)
            rank_biserial = 1 - (2 * U_stat) / (n1 * n2)
            result['mwu_U'] = float(U_stat)
            result['mwu_p'] = float(p_value)
            result['rank_biserial'] = float(rank_biserial)
            result['mwu_significant'] = p_value < 0.05
        except:
            result['mwu_U'] = None
            result['mwu_p'] = None
            result['rank_biserial'] = None
            result['mwu_significant'] = False
        
        return result
        
    except Exception as e:
        return None


def compute_meta_analysis(file_results: list):
    """
    Compute meta-analysis statistics across files using Block-Level Aggregation.
    """
    # Convert to DataFrame for easier grouping
    df = pd.DataFrame(file_results)
    
    # 1. Aggregate to Block Level (CLD, Run)
    # We take the mean AUC/AP/Corr for each block, and sum edges to compute prevalence baselines.
    if 'cld' in df.columns and 'run' in df.columns:
        block_df = (
            df.groupby(['cld', 'run'])
            .agg(
                auc=('auc', 'mean'),
                ap=('ap', 'mean'),
                correlation_r=('correlation_r', 'mean'),
                n_edges=('n_edges', 'sum'),
                n_halluc=('n_halluc', 'sum'),
            )
            .reset_index()
        )
    else:
        # Fallback if no block info (shouldn't happen with new main)
        block_df = df.copy()
    
    block_aucs = block_df['auc'].dropna()
    block_aps = block_df['ap'].dropna()
    block_corrs = block_df['correlation_r'].dropna()
    
    n_blocks = len(block_aucs)
    
    result = {
        'total_edges': df['n_edges'].sum(),
        'total_files': len(df),
        'n_blocks': n_blocks
    }
    
    # --- AUC Meta-Analysis (Block-Level) ---
    if len(block_aucs) >= 2:
        # Mean of blocks
        mean_auc = float(block_aucs.mean())
        std_auc = float(block_aucs.std())
        n = len(block_aucs)
        se = std_auc / np.sqrt(n)
        
        # 95% CI on the mean using t-distribution (small-N uncertainty rule)
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lower_auc = float(mean_auc - t_crit * se)
        ci_upper_auc = float(mean_auc + t_crit * se)
        
        result['mean_auc'] = mean_auc
        result['std_auc'] = std_auc
        result['ci_lower_auc'] = ci_lower_auc
        result['ci_upper_auc'] = ci_upper_auc
        
        # One-sample Wilcoxon signed-rank test vs 0.5 (on blocks): test (AUC - 0.5) vs 0
        deltas_auc = (block_aucs.values - 0.5).astype(float)
        if np.any(np.abs(deltas_auc) > 1e-12):
            w_stat, p_val = stats.wilcoxon(deltas_auc, zero_method="wilcox", alternative="two-sided")
            result['auc_wilcoxon_w'] = float(w_stat)
            result['auc_wilcoxon_p'] = float(p_val)
        else:
            result['auc_wilcoxon_w'] = None
            result['auc_wilcoxon_p'] = 1.0
        result['auc_above_chance'] = (result['auc_wilcoxon_p'] < 0.05) and (mean_auc > 0.5)
        result['auc_below_chance'] = (result['auc_wilcoxon_p'] < 0.05) and (mean_auc < 0.5)

    # --- PR-AUC (Average Precision) Meta-Analysis (Block-Level) ---
    # Baseline under random ranking equals positive prevalence; we test AP - prevalence vs 0 on blocks.
    if len(block_aps) >= 2:
        mean_ap = float(block_aps.mean())
        std_ap = float(block_aps.std())
        n = len(block_aps)
        se = std_ap / np.sqrt(n)

        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lower_ap = float(mean_ap - t_crit * se)
        ci_upper_ap = float(mean_ap + t_crit * se)

        result['mean_ap'] = mean_ap
        result['std_ap'] = std_ap
        result['ci_lower_ap'] = ci_lower_ap
        result['ci_upper_ap'] = ci_upper_ap

        # Prevalence baseline per block (may vary across CLD×run)
        block_prev = (block_df.loc[block_aps.index, 'n_halluc'] / block_df.loc[block_aps.index, 'n_edges']).astype(float)
        result['mean_prevalence'] = float(block_prev.mean())

        # One-sample t-test vs prevalence baseline (on blocks): AP - prevalence
        deltas = block_aps.values - block_prev.values
        t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
        result['ap_delta_ttest_t'] = float(t_stat)
        result['ap_delta_ttest_p'] = float(p_val)
        result['ap_above_baseline'] = (p_val < 0.05) and (float(np.mean(deltas)) > 0.0)

    # --- Correlation Meta-Analysis (Fisher Z on Blocks) ---
    if len(block_corrs) >= 2:
        # Fisher z transform
        zs = np.arctanh(block_corrs)
        z_mean = np.mean(zs)
        z_se = np.std(zs, ddof=1) / np.sqrt(len(zs))
        
        # CI in z-space (t-dist ok here as z is normal-ish)
        t_crit = stats.t.ppf(0.975, len(zs) - 1)
        z_lower = z_mean - t_crit * z_se
        z_upper = z_mean + t_crit * z_se
        
        # Transform back
        mean_corr = float(np.tanh(z_mean))
        ci_lower_corr = float(np.tanh(z_lower))
        ci_upper_corr = float(np.tanh(z_upper))
        
        result['mean_corr'] = mean_corr
        result['ci_lower_corr'] = ci_lower_corr
        result['ci_upper_corr'] = ci_upper_corr
        
        # Test vs 0 using one-sample Wilcoxon signed-rank on block-level correlations
        deltas_corr = block_corrs.values.astype(float)  # null r=0
        if np.any(np.abs(deltas_corr) > 1e-12):
            w_stat, p_val_corr = stats.wilcoxon(deltas_corr, zero_method="wilcox", alternative="two-sided")
            result['corr_wilcoxon_w'] = float(w_stat)
            result['corr_wilcoxon_p'] = float(p_val_corr)
        else:
            result['corr_wilcoxon_w'] = None
            result['corr_wilcoxon_p'] = 1.0
        result['corr_significant'] = result['corr_wilcoxon_p'] < 0.05
    
    # Significant files percentage (Descriptive)
    significant = [r['mwu_significant'] for r in file_results if r.get('mwu_significant') is not None]
    if significant:
        result['significant_files_pct'] = float(sum(significant) / len(significant) * 100)
        result['n_significant_files'] = sum(significant)
    
    return result


def generate_latex_table(metrics_results: dict, bonferroni_results: dict, output_path: Path):
    """Generate LaTeX table for thesis with Bonferroni-adjusted p-values."""
    
    # Sort by AUC descending
    sorted_metrics = sorted(
        metrics_results.items(),
        key=lambda x: x[1].get('mean_auc', 0),
        reverse=True
    )
    
    latex = r"""\begin{table}[H]
\centering
\caption{Single UQ Metric Performance for Hallucination Detection (Meta-Analysis)}
\label{tab:rq2_single_metrics}
\begin{threeparttable}
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{UQ Metric} & \textbf{N Edges} & \textbf{Mean AUC} & \textbf{Mean PR-AUC} & \textbf{Correlation r} & \textbf{Significant} & \textbf{N Files} \\
 & & \textbf{(95\% CI)} & \textbf{(95\% CI)} & \textbf{(95\% CI)} & \textbf{Files (\%)} & \\
\midrule
"""
    
    for metric, meta in sorted_metrics:
        # Format metric name (split long names)
        metric_short = METRIC_SHORT_NAMES.get(metric, metric)
        parts = metric_short.split('\\\\')
        
        # Format AUC with significance (using ADJUSTED p-values from Bonferroni)
        if 'mean_auc' in meta:
            auc_val = meta['mean_auc']
            ci_l, ci_u = meta['ci_lower_auc'], meta['ci_upper_auc']
            ci_hw = (ci_u - ci_l) / 2
            
            # Use Bonferroni-adjusted p-value for significance markers
            bonf_result = bonferroni_results.get('results', {}).get(metric, {})
            p_adj = bonf_result.get('p_adjusted', meta.get('auc_wilcoxon_p', 1.0))
            
            # Significance markers based on adjusted p-value
            if p_adj < 0.001:
                sig = '***'
            elif p_adj < 0.01:
                sig = '**'
            elif p_adj < 0.05:
                sig = '*'
            else:
                sig = '^{ns}'
            
            # Direction marker for below chance (use adjusted significance)
            sig_adj = bonf_result.get('significant_adjusted', False)
            if sig_adj and auc_val < 0.5:
                sig += r'\downarrow'
            
            auc_str = f"{auc_val:.3f}${sig}$"
            auc_ci = f"$\\pm$ {ci_hw:.3f}"
        else:
            auc_str = "N/A"
            auc_ci = ""

        # Format PR-AUC (Average Precision) (descriptive + baseline-aware test stored in JSON/Excel)
        if 'mean_ap' in meta:
            ap_val = meta['mean_ap']
            ci_l, ci_u = meta['ci_lower_ap'], meta['ci_upper_ap']
            ci_hw = (ci_u - ci_l) / 2
            ap_str = f"{ap_val:.3f}"
            ap_ci = f"$\\pm$ {ci_hw:.3f}"
        else:
            ap_str = "N/A"
            ap_ci = ""
        
        # Format correlation (Bonferroni-adjusted across 4 single-metric correlation tests)
        if 'mean_corr' in meta:
            corr_val = meta['mean_corr']
            ci_l, ci_u = meta['ci_lower_corr'], meta['ci_upper_corr']
            ci_hw = (ci_u - ci_l) / 2
            
            p_val = meta.get('corr_wilcoxon_p_adj', meta.get('corr_wilcoxon_p', 1.0))
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = '^{ns}'
            
            corr_str = f"{corr_val:+.3f}${sig}$"
            corr_ci = f"$\\pm$ {ci_hw:.3f}"
        else:
            corr_str = "N/A"
            corr_ci = ""
        
        # Format significant files
        sig_pct = meta.get('significant_files_pct', 0)
        
        # N edges and files
        n_edges = f"{meta['total_edges']:,}"
        n_files = meta['total_files']
        
        # First row
        if len(parts) == 1:
            latex += f"{parts[0]} & {n_edges} & {auc_str} & {ap_str} & {corr_str} & {sig_pct:.1f}\\% & {n_files} \\\\\n"
            latex += f" & & {auc_ci} & {ap_ci} & {corr_ci} & & \\\\\n"
        else:
            latex += f"{parts[0]} & {n_edges} & {auc_str} & {ap_str} & {corr_str} & {sig_pct:.1f}\\% & {n_files} \\\\\n"
            latex += f"{parts[1]} & & {auc_ci} & {ap_ci} & {corr_ci} & & \\\\\n"
        
        latex += r"\midrule" + "\n"
    
    # Remove last \midrule
    latex = latex.rstrip("\n").rstrip(r"\midrule")
    
    # Get Bonferroni correction info for footnote
    n_tests = bonferroni_results.get('n_tests', N_METRIC_TESTS)
    alpha_adj = bonferroni_results.get('alpha_adjusted', ALPHA_ADJ)
    
    latex += r"""\bottomrule
\end{tabular}}
\begin{tablenotes}
\small
\item \textit{Note.} Meta-analysis across experiment files using three logprob-derived generator metrics (perplexity, min prob, max window entropy) and one retrieval-alignment metric (cosine similarity). 
N Edges = total causal edges analyzed; N Files = experiment files containing metric. 
Gen Cosine Similarity has fewer observations because it requires retrieved citations (citation-judging runs only); correctness-judging runs lack retrieved text. One file excluded due to $<$2 hallucinations.
\textbf{Mean AUC} is aggregated at the block level (CLD $\times$ Run, $N=9$ blocks); 95\% CIs computed via t-distribution over blocks.
\textbf{Mean PR-AUC} is the block-level mean of Average Precision (area under the precision--recall curve). Under class imbalance, a random ranking baseline yields PR-AUC equal to the positive prevalence; PR-AUC is therefore reported descriptively alongside AUC.
\textbf{Correlation r} computed via Fisher z-transform aggregation across blocks.
\textbf{Significant Files (\%)} = percentage of files where Mann-Whitney U test (halluc vs.\ correct distributions) yields $p < 0.05$; this is a \textit{descriptive} consistency measure.
"""
    
    latex += f"\\item \\textit{{Multiple comparisons:}} Bonferroni correction applied across {n_tests} single-metric AUC tests and {n_tests} single-metric correlation tests ($\\alpha_{{\\text{{adj}}}} = {alpha_adj:.4f}$ per family). Significance markers (*, **, ***) reflect Bonferroni-adjusted $p$-values ($p_{{\\text{{adj}}}} = p \\times {n_tests}$).\n"
    
    latex += r"""\item Significance levels: *** $p_{\text{adj}}<0.001$, ** $p_{\text{adj}}<0.01$, * $p_{\text{adj}}<0.05$, $^{ns}$ = not significant; $\downarrow$ = significantly below chance (one-sample Wilcoxon signed-rank test on $(\mathrm{AUC}-0.5)$ over blocks).
\item \textit{Assumptions:} Edge-level metric distributions are non-normal (requiring Mann-Whitney U for direct comparison, see Table~\ref{tab:rq2_normality_tests}). For block-level inference, we report mean $\pm$ 95\% CI using the $t$-distribution over blocks ($N=9$), and use one-sample Wilcoxon signed-rank tests on $(\mathrm{AUC}-0.5)$ for above-chance hypothesis tests; interpret $N=9$ p-values as approximate. Block-level aggregation handles dependence within each CLD$\times$run generation scenario.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"  ✅ LaTeX table saved: {output_path.name}")


def generate_excel(metrics_results: dict, bonferroni_results: dict, file_level_results: dict, output_path: Path):
    """Generate Excel file with all results including Bonferroni-adjusted values."""
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_rows = []
        for metric, meta in metrics_results.items():
            summary_rows.append({
                'Metric': metric,
                'N_Edges': meta['total_edges'],
                'N_Files': meta['total_files'],
                'Mean_AUC': meta.get('mean_auc'),
                'AUC_CI_Lower': meta.get('ci_lower_auc'),
                'AUC_CI_Upper': meta.get('ci_upper_auc'),
                'AUC_Std': meta.get('std_auc'),
                'AUC_wilcoxon_p': meta.get('auc_wilcoxon_p'),
                'AUC_wilcoxon_p_adj': meta.get('auc_wilcoxon_p_adj'),  # Bonferroni-adjusted
                'AUC_Above_Chance': meta.get('auc_above_chance'),
                'AUC_Above_Chance_Adj': meta.get('auc_above_chance_adj'),  # Bonferroni-adjusted
                'AUC_Below_Chance': meta.get('auc_below_chance'),
                'AUC_Below_Chance_Adj': meta.get('auc_below_chance_adj'),  # Bonferroni-adjusted
                'Mean_PR_AUC': meta.get('mean_ap'),
                'PR_AUC_CI_Lower': meta.get('ci_lower_ap'),
                'PR_AUC_CI_Upper': meta.get('ci_upper_ap'),
                'PR_AUC_Std': meta.get('std_ap'),
                'Mean_Prevalence': meta.get('mean_prevalence'),
                'PR_AUC_Delta_ttest_p': meta.get('ap_delta_ttest_p'),
                'PR_AUC_Above_Baseline': meta.get('ap_above_baseline'),
                'Mean_Corr': meta.get('mean_corr'),
                'Corr_CI_Lower': meta.get('ci_lower_corr'),
                'Corr_CI_Upper': meta.get('ci_upper_corr'),
                'Corr_Std': meta.get('std_corr'),
                'Corr_wilcoxon_p': meta.get('corr_wilcoxon_p'),
                'Significant_Files_Pct': meta.get('significant_files_pct'),
                'N_Significant_Files': meta.get('n_significant_files')
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # File-level results for each metric
        for metric, results_list in file_level_results.items():
            metric_short = metric.replace('Gen ', '').replace(' ', '_')[:20]
            file_df = pd.DataFrame(results_list)
            file_df.to_excel(writer, sheet_name=f'Files_{metric_short}', index=False)
        
        # Metadata
        metadata = pd.DataFrame([{
            'Generated': datetime.now().isoformat(),
            'Script': 'rq2_single_metric_table.py',
            'Statistical_Test': 'Mann-Whitney U (non-parametric)',
            'AUC_Test': 'One-sample Wilcoxon signed-rank test vs 0.5 on block AUCs',
            'Significance_Alpha': ALPHA,
            'Bonferroni_N_Tests': bonferroni_results.get('n_tests', N_METRIC_TESTS),
            'Bonferroni_Alpha_Adj': bonferroni_results.get('alpha_adjusted', ALPHA_ADJ),
            'Note': 'Bonferroni correction applied across 4 single-metric AUC tests per Methods Section 4.5'
        }])
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
    
    print(f"  ✅ Excel file saved: {output_path.name}")


def main():
    print("\n" + "="*80)
    print("RQ2 SINGLE UQ METRIC TABLE GENERATOR")
    print("="*80)
    print("\nUsing Mann-Whitney U tests (non-parametric) for significance testing\n")
    
    analyses_dir, output_dir = rq2_dirs()
    if not analyses_dir.exists():
        print("❌ Analysis directory not found!")
        print(f"   Expected: {analyses_dir}")
        print("   Run: python3 final_runs/RQ2_unified_analysis.py")
        return 1
    
    # Load inventory
    try:
        valid_files, inventory_path = load_inventory(analyses_dir)
        print(f"📋 Loaded inventory: {inventory_path.name}")
        print(f"   Total files: {len(valid_files)}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    
    # Analyze each metric
    metrics_results = {}
    file_level_results = {}
    
    for metric in CI_METRICS:
        print(f"\n📊 Analyzing: {metric}")
        
        metric_file_results = []
        
        root = repo_root()
        for file_info in valid_files:
            filepath = file_info['filepath']
            fp = Path(filepath)
            if not fp.is_absolute():
                fp = root / fp
            result = analyze_file_for_metric(str(fp), metric)
            
            if result is not None:
                result['filepath'] = str(fp)
                result['cld'] = file_info.get('cld', 'unknown')
                result['run'] = file_info.get('run', 'unknown')
                result['experiment_type'] = file_info.get('experiment_type', 'unknown')
                metric_file_results.append(result)
        
        print(f"   Valid files: {len(metric_file_results)}")
        
        if metric_file_results:
            # Compute meta-analysis
            meta = compute_meta_analysis(metric_file_results)
            metrics_results[metric] = meta
            file_level_results[metric] = metric_file_results
            
            print(f"   Total edges: {meta['total_edges']:,}")
            if 'mean_auc' in meta:
                auc_ci_hw = (meta['ci_upper_auc'] - meta['ci_lower_auc']) / 2
                print(f"   Mean AUC: {meta['mean_auc']:.3f} ± {auc_ci_hw:.3f}")
            if 'mean_corr' in meta:
                corr_ci_hw = (meta['ci_upper_corr'] - meta['ci_lower_corr']) / 2
                print(f"   Mean Corr: {meta['mean_corr']:+.3f} ± {corr_ci_hw:.3f}")
            if 'significant_files_pct' in meta:
                print(f"   Significant files: {meta['significant_files_pct']:.1f}%")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply Bonferroni correction across the 4 single-metric AUC tests
    # Per Methods Section 4.5: RQ2 uses 4 tests → alpha_adj = 0.0125
    print("\n" + "-"*40)
    print("Applying Bonferroni correction...")
    
    auc_p_values = {metric: meta.get('auc_wilcoxon_p', 1.0) for metric, meta in metrics_results.items()}
    bonferroni_results = bonferroni_correction(auc_p_values, alpha=ALPHA)
    
    print(f"  Bonferroni family: {bonferroni_results['n_tests']} tests")
    print(f"  α_original = {bonferroni_results['alpha_original']:.3f}")
    print(f"  α_adjusted = {bonferroni_results['alpha_adjusted']:.4f}")
    print(f"  Significant before correction: {bonferroni_results['n_significant_original']}")
    print(f"  Significant after correction:  {bonferroni_results['n_significant_adjusted']}")
    
    # Add adjusted values back to metrics_results
    for metric in metrics_results:
        if metric in bonferroni_results['results']:
            bonf = bonferroni_results['results'][metric]
            metrics_results[metric]['auc_wilcoxon_p_adj'] = bonf['p_adjusted']
            metrics_results[metric]['auc_significant_adj'] = bonf['significant_adjusted']
            metrics_results[metric]['auc_above_chance_adj'] = (
                bonf['significant_adjusted'] and metrics_results[metric].get('mean_auc', 0) > 0.5
            )
            metrics_results[metric]['auc_below_chance_adj'] = (
                bonf['significant_adjusted'] and metrics_results[metric].get('mean_auc', 0) < 0.5
            )

    # Apply Bonferroni correction across the 4 single-metric correlation tests (separate family)
    corr_p_values = {metric: meta.get('corr_wilcoxon_p', 1.0) for metric, meta in metrics_results.items()}
    bonferroni_corr_results = bonferroni_correction(corr_p_values, alpha=ALPHA)

    # Add adjusted correlation values back to metrics_results
    for metric in metrics_results:
        if metric in bonferroni_corr_results.get('results', {}):
            bonf = bonferroni_corr_results['results'][metric]
            metrics_results[metric]['corr_wilcoxon_p_adj'] = bonf['p_adjusted']
            metrics_results[metric]['corr_significant_adj'] = bonf['significant_adjusted']
    
    # Generate outputs
    print("\n" + "-"*40)
    print("Generating outputs...")
    
    # LaTeX table (with Bonferroni-adjusted p-values for AUC and correlation)
    generate_latex_table(metrics_results, bonferroni_results, output_dir / 'single_metric_table.tex')
    
    # Excel file (with Bonferroni-adjusted values)
    generate_excel(metrics_results, bonferroni_results, file_level_results, output_dir / 'single_metric_table.xlsx')
    
    # JSON archive
    json_path = output_dir / 'single_metric_results.json'
    
    # Convert to JSON-serializable
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    with open(json_path, 'w') as f:
        json.dump(convert_numpy({
            'timestamp': datetime.now().isoformat(),
            'bonferroni_correction': bonferroni_results,
            'metrics': metrics_results,
            'file_level': {k: v for k, v in file_level_results.items()}
        }), f, indent=2)
    print(f"  ✅ JSON archive saved: {json_path.name}")
    
    print("\n" + "="*80)
    print("✅ SINGLE METRIC TABLE GENERATION COMPLETE")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📄 LaTeX: single_metric_table.tex")
    print(f"📊 Excel: single_metric_table.xlsx")
    print(f"📋 JSON: single_metric_results.json\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
















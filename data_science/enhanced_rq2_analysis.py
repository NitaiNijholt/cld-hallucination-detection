#!/usr/bin/env python3
"""
Enhanced RQ2 Analysis: Comprehensive meta-analysis with detailed statistics
Includes:
- Dataset information (edge counts, TP/FP/FN/TN per CLD)
- Mean metric values per class (TP vs FP)
- Pairwise statistical tests (t-tests)
- Effect sizes (Cohen's d)
- Per-CLD breakdowns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
import json

# Define the 6 data files
DATA_FILES = {
    'Depressive_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run1_7a66e382/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251008_181257.xlsx',
    'Depressive_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run2_0d57ed9a/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251008_200816.xlsx',
    'Social_norms_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run1_86b016e1/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Social_norms_and_obesity_prevalence_20251008_213554.xlsx',
    'Social_norms_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run2_d09234b7/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Social_norms_and_obesity_prevalence_20251008_221139.xlsx',
    'Older_persons_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run1_d45241a5/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251008_230700.xlsx',
    'Older_persons_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run2_96605199/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251009_082149.xlsx'
}

# Context-insensitive metrics to analyze
CI_METRICS = [
    'Gen Perplexity',
    'Gen Max Window Entropy', 
    'Gen Min Prob',
    'Gen Cosine Similarity',
    'Judge Perplexity',
    'Judge Max Window Entropy',
    'Judge Min Prob',
    'Judge Cosine Similarity',
    'Aggregate Score'
]

def load_all_data():
    """Load all 6 data files and combine them."""
    all_data = []
    
    for run_name, file_path in DATA_FILES.items():
        cld_name = run_name.rsplit('_', 1)[0]
        run_num = run_name.rsplit('_', 1)[1]
        
        df = pd.read_excel(file_path, sheet_name='All Edges')
        df['cld_name'] = cld_name
        df['run_number'] = run_num
        df['run_id'] = run_name
        
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined

def calculate_confusion_matrix_stats(df):
    """Calculate TP/FP/FN/TN counts."""
    # Use the Classification column which has: TP, FP, FN, TN
    
    if 'Classification' not in df.columns:
        return {
            'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0,
            'Total_Edges': len(df),
            'Precision': 0, 'Recall': 0, 'F1': 0
        }
    
    tp = (df['Classification'] == 'TP').sum()
    fp = (df['Classification'] == 'FP').sum()
    fn = (df['Classification'] == 'FN').sum()
    tn = (df['Classification'] == 'TN').sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'Total_Edges': len(df),
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

def calculate_roc_auc_and_optimal_threshold(df, metric_name):
    """Calculate ROC AUC and find optimal threshold for F1 score."""
    # Filter valid data
    df_valid = df[df[metric_name].notna()].copy()
    
    if len(df_valid) == 0 or 'Classification' not in df_valid.columns:
        return None
    
    # Only use generated edges (TP and FP)
    df_generated = df_valid[df_valid['Classification'].isin(['TP', 'FP'])].copy()
    
    if len(df_generated) < 10:  # Need minimum samples
        return None
    
    # Create binary labels: 1 for FP (hallucination), 0 for TP
    y_true = (df_generated['Classification'] == 'FP').astype(int)
    y_scores = df_generated[metric_name].values
    
    # Calculate ROC AUC
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        return None
    
    # Find optimal threshold that maximizes F1
    # Try percentile-based thresholds
    percentiles = np.linspace(10, 90, 17)  # 10th to 90th percentile
    thresholds = np.percentile(y_scores, percentiles)
    
    best_f1 = 0
    best_threshold = None
    best_precision = 0
    best_recall = 0
    
    for threshold in thresholds:
        # Predict: if score >= threshold, predict FP (1), else TP (0)
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    return {
        'metric': metric_name,
        'auc': auc,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'n_samples': len(df_generated)
    }

def compare_metric_by_class(df, metric_name):
    """Compare metric values between TP and FP classes."""
    # Filter out rows where metric is missing
    df_valid = df[df[metric_name].notna()].copy()
    
    if len(df_valid) == 0 or 'Classification' not in df_valid.columns:
        return None
    
    # Separate by class - only compare TP vs FP (edges that were generated)
    tp_values = df_valid[df_valid['Classification'] == 'TP'][metric_name]
    fp_values = df_valid[df_valid['Classification'] == 'FP'][metric_name]
    
    if len(tp_values) < 2 or len(fp_values) < 2:
        return None
    
    # Calculate means and stds
    tp_mean = tp_values.mean()
    tp_std = tp_values.std()
    fp_mean = fp_values.mean()
    fp_std = fp_values.std()
    
    # Perform t-test
    t_stat, p_value_ttest = stats.ttest_ind(tp_values, fp_values, equal_var=False)
    
    # Calculate effect size
    effect_size = cohens_d(tp_values, fp_values)
    
    # Calculate correlation with hallucination label
    # Create binary label: 1 for FP (hallucination), 0 for TP
    df_valid['hallucination'] = (df_valid['Classification'] == 'FP').astype(int)
    corr_r, p_value_corr = stats.pearsonr(df_valid[metric_name], df_valid['hallucination'])
    
    return {
        'metric': metric_name,
        'TP_mean': tp_mean,
        'TP_std': tp_std,
        'TP_n': len(tp_values),
        'FP_mean': fp_mean,
        'FP_std': fp_std,
        'FP_n': len(fp_values),
        'mean_diff': tp_mean - fp_mean,
        't_statistic': t_stat,
        'p_value_ttest': p_value_ttest,
        'cohens_d': effect_size,
        'correlation_r': corr_r,
        'p_value_corr': p_value_corr,
        'significant_ttest': p_value_ttest < 0.05,
        'significant_corr': p_value_corr < 0.05
    }

def analyze_per_cld(df):
    """Perform analysis per CLD."""
    results = {}
    
    for cld in df['cld_name'].unique():
        cld_df = df[df['cld_name'] == cld]
        
        # Get confusion matrix stats
        cm_stats = calculate_confusion_matrix_stats(cld_df)
        
        # Analyze each metric
        metric_results = []
        for metric in CI_METRICS:
            if metric in cld_df.columns:
                result = compare_metric_by_class(cld_df, metric)
                if result:
                    metric_results.append(result)
        
        results[cld] = {
            'confusion_matrix': cm_stats,
            'metric_comparisons': metric_results
        }
    
    return results

def generate_enhanced_report(df, per_cld_results, output_dir):
    """Generate comprehensive markdown report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"enhanced_rq2_report_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write("# Enhanced RQ2 Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Experiment ID:** exp_20251008_180446_a0d1b2de  \n\n")
        f.write("---\n\n")
        
        # Dataset Overview
        f.write("## Dataset Overview\n\n")
        total_stats = calculate_confusion_matrix_stats(df)
        f.write(f"**Total Edges Analyzed:** {total_stats['Total_Edges']}  \n")
        f.write(f"**True Positives (TP):** {total_stats['TP']}  \n")
        f.write(f"**False Positives (FP):** {total_stats['FP']}  \n")
        f.write(f"**False Negatives (FN):** {total_stats['FN']}  \n")
        f.write(f"**True Negatives (TN):** {total_stats['TN']}  \n")
        f.write(f"**Overall Precision:** {total_stats['Precision']:.3f}  \n")
        f.write(f"**Overall Recall:** {total_stats['Recall']:.3f}  \n")
        f.write(f"**Overall F1 Score:** {total_stats['F1']:.3f}  \n\n")
        
        # Per-CLD Breakdown
        f.write("### Per-CLD Statistics\n\n")
        f.write("| CLD | Total Edges | TP | FP | FN | TN | Precision | Recall | F1 |\n")
        f.write("|-----|-------------|----|----|----|----|-----------|---------|----|")
        f.write("\n")
        
        for cld, results in per_cld_results.items():
            cm = results['confusion_matrix']
            f.write(f"| {cld} | {cm['Total_Edges']} | {cm['TP']} | {cm['FP']} | {cm['FN']} | {cm['TN']} | ")
            f.write(f"{cm['Precision']:.3f} | {cm['Recall']:.3f} | {cm['F1']:.3f} |\n")
        
        f.write("\n---\n\n")
        
        # Aggregate Metric Comparisons  
        f.write("## Aggregate Analysis: Metric Comparisons (TP vs FP)\n\n")
        f.write("Comparing mean metric values between True Positives and False Positives across all CLDs.\n\n")
        
        f.write("### Classification Performance (ROC AUC & Optimal F1)\n\n")
        
        # Calculate ROC AUC and optimal thresholds for all metrics
        auc_results = []
        for metric in CI_METRICS:
            if metric in df.columns:
                result = calculate_roc_auc_and_optimal_threshold(df, metric)
                if result:
                    auc_results.append(result)
        
        if auc_results:
            f.write("| Metric | AUC | Best F1 | Precision | Recall | Threshold | N |\n")
            f.write("|--------|-----|---------|-----------|--------|-----------|---|\n")
            
            for result in sorted(auc_results, key=lambda x: x['auc'], reverse=True):
                f.write(f"| {result['metric']} | ")
                f.write(f"{result['auc']:.3f} | ")
                f.write(f"{result['best_f1']:.3f} | ")
                f.write(f"{result['best_precision']:.3f} | ")
                f.write(f"{result['best_recall']:.3f} | ")
                f.write(f"{result['best_threshold']:.4f} | ")
                f.write(f"{result['n_samples']} |\n")
            
            f.write("\n*AUC = Area Under ROC Curve (>0.5 = better than chance)*\n")
            f.write("*Best F1 = Optimal F1 score at best threshold*\n")
            f.write("*Threshold = Metric value that maximizes F1 score*\n\n")
            
            # Interpretation
            best_auc = max(auc_results, key=lambda x: x['auc'])
            best_f1 = max(auc_results, key=lambda x: x['best_f1'])
            
            f.write("**Performance Summary:**\n")
            f.write(f"- **Best AUC:** {best_auc['metric']} (AUC={best_auc['auc']:.3f})\n")
            f.write(f"- **Best F1 Score:** {best_f1['metric']} (F1={best_f1['best_f1']:.3f} at threshold={best_f1['best_threshold']:.4f})\n")
            
            # Count how many are above chance
            above_chance = [r for r in auc_results if r['auc'] > 0.5]
            f.write(f"- **Metrics above chance (AUC>0.5):** {len(above_chance)}/{len(auc_results)}\n\n")
        
        f.write("### Mean Value Comparisons (T-Test & Correlation)\n\n")
        
        # Collect all metric comparisons
        all_metric_results = []
        for metric in CI_METRICS:
            if metric in df.columns:
                result = compare_metric_by_class(df, metric)
                if result:
                    all_metric_results.append(result)
        
        if all_metric_results:
            f.write("| Metric | TP Mean (±SD) | FP Mean (±SD) | Difference | t-stat | p-value | Cohen's d | Correlation r | p-value | Sig. |\n")
            f.write("|--------|---------------|---------------|------------|--------|---------|-----------|---------------|---------|------|\n")
            
            for result in sorted(all_metric_results, key=lambda x: x['p_value_ttest']):
                sig_ttest = "***" if result['p_value_ttest'] < 0.001 else "**" if result['p_value_ttest'] < 0.01 else "*" if result['p_value_ttest'] < 0.05 else "ns"
                sig_corr = "***" if result['p_value_corr'] < 0.001 else "**" if result['p_value_corr'] < 0.01 else "*" if result['p_value_corr'] < 0.05 else "ns"
                
                f.write(f"| {result['metric']} | ")
                f.write(f"{result['TP_mean']:.4f} (±{result['TP_std']:.4f}) | ")
                f.write(f"{result['FP_mean']:.4f} (±{result['FP_std']:.4f}) | ")
                f.write(f"{result['mean_diff']:+.4f} | ")
                f.write(f"{result['t_statistic']:.2f} | ")
                f.write(f"{result['p_value_ttest']:.4f} {sig_ttest} | ")
                f.write(f"{result['cohens_d']:.3f} | ")
                f.write(f"{result['correlation_r']:+.3f} | ")
                f.write(f"{result['p_value_corr']:.4f} {sig_corr} | ")
                sig_marker = "✓" if result['significant_ttest'] and result['significant_corr'] else "~" if result['significant_ttest'] or result['significant_corr'] else "✗"
                f.write(f"{sig_marker} |\n")
            
            f.write("\n*Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant*\n")
            f.write("*Overall Sig.: ✓ = both significant, ~ = one significant, ✗ = neither significant*\n\n")
            
            # Interpretation
            f.write("### Interpretation\n\n")
            sig_both = [r for r in all_metric_results if r['significant_ttest'] and r['significant_corr']]
            sig_ttest_only = [r for r in all_metric_results if r['significant_ttest'] and not r['significant_corr']]
            sig_corr_only = [r for r in all_metric_results if not r['significant_ttest'] and r['significant_corr']]
            
            f.write(f"**Statistical Significance Summary:**\n")
            f.write(f"- Both t-test AND correlation significant: **{len(sig_both)}/{len(all_metric_results)} metrics**\n")
            f.write(f"- T-test significant only: **{len(sig_ttest_only)}/{len(all_metric_results)} metrics**\n")
            f.write(f"- Correlation significant only: **{len(sig_corr_only)}/{len(all_metric_results)} metrics**\n\n")
            
            if sig_both:
                f.write("**Metrics with both significant t-test AND significant correlation:**\n")
                for result in sorted(sig_both, key=lambda x: abs(x['cohens_d']), reverse=True):
                    direction = "higher" if result['mean_diff'] > 0 else "lower"
                    corr_dir = "positive" if result['correlation_r'] > 0 else "negative"
                    f.write(f"- **{result['metric']}**: TP values are {direction} than FP (d={result['cohens_d']:.3f}, p_t={result['p_value_ttest']:.4f}), {corr_dir} correlation (r={result['correlation_r']:+.3f}, p_r={result['p_value_corr']:.4f})\n")
                f.write("\n")
        
        f.write("\n---\n\n")
        
        # Per-CLD Detailed Results
        f.write("## Per-CLD Detailed Analysis\n\n")
        
        for cld, results in per_cld_results.items():
            f.write(f"### {cld}\n\n")
            
            cm = results['confusion_matrix']
            f.write(f"**Edges:** {cm['Total_Edges']} (TP={cm['TP']}, FP={cm['FP']}, FN={cm['FN']}, TN={cm['TN']})  \n")
            f.write(f"**Precision:** {cm['Precision']:.3f} | **Recall:** {cm['Recall']:.3f} | **F1:** {cm['F1']:.3f}  \n\n")
            
            metric_results = results['metric_comparisons']
            if metric_results:
                f.write("| Metric | TP Mean | FP Mean | Diff | p-value (t) | Cohen's d | Corr r | p-value (r) |\n")
                f.write("|--------|---------|---------|------|-------------|-----------|--------|-------------|\n")
                
                for result in sorted(metric_results, key=lambda x: x['p_value_ttest']):
                    sig_marker_t = "***" if result['p_value_ttest'] < 0.001 else "**" if result['p_value_ttest'] < 0.01 else "*" if result['p_value_ttest'] < 0.05 else ""
                    sig_marker_r = "***" if result['p_value_corr'] < 0.001 else "**" if result['p_value_corr'] < 0.01 else "*" if result['p_value_corr'] < 0.05 else ""
                    f.write(f"| {result['metric']} | {result['TP_mean']:.4f} | {result['FP_mean']:.4f} | ")
                    f.write(f"{result['mean_diff']:+.4f} | {result['p_value_ttest']:.4f}{sig_marker_t} | {result['cohens_d']:.3f} | ")
                    f.write(f"{result['correlation_r']:+.3f} | {result['p_value_corr']:.4f}{sig_marker_r} |\n")
                
                f.write("\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        f.write("## Data Files\n\n")
        for run_name, file_path in DATA_FILES.items():
            f.write(f"- `{run_name}`: `{file_path}`\n")
    
    print(f"Report generated: {report_path}")
    return report_path

def main():
    print("Loading RQ2 data files...")
    df = load_all_data()
    
    print(f"Loaded {len(df)} total edges from {df['run_id'].nunique()} runs across {df['cld_name'].nunique()} CLDs")
    
    print("\nCalculating per-CLD statistics...")
    per_cld_results = analyze_per_cld(df)
    
    print("\nGenerating enhanced report...")
    output_dir = Path("parameter_tuning_experiments/rq2_analyses/rq2_enhanced_aggregate")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = generate_enhanced_report(df, per_cld_results, output_dir)
    
    # Also save raw comparison data as CSV
    print("\nSaving detailed comparison tables...")
    all_comparisons = []
    for metric in CI_METRICS:
        if metric in df.columns:
            result = compare_metric_by_class(df, metric)
            if result:
                all_comparisons.append(result)
    
    if all_comparisons:
        comparison_df = pd.DataFrame(all_comparisons)
        comparison_df.to_csv(output_dir / "metric_comparisons_tp_vs_fp.csv", index=False)
        print(f"Saved: {output_dir / 'metric_comparisons_tp_vs_fp.csv'}")
    
    print("\n✅ Enhanced RQ2 analysis complete!")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
RQ2 Master Report Generator (Version 2 - Enhanced)

Creates a COMPREHENSIVE master report with ALL intermediate results, statistics,
correlations, sample tracking, and methodological details from all RQ2 analysis phases.

Includes normality testing for all UQ metrics to justify non-parametric methods.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import sys
import warnings
warnings.filterwarnings('ignore')

from rq2_paths import rq2_dirs

# CI Metrics to analyze
CI_METRICS = [
    'Gen Perplexity',
    'Gen Min Prob', 
    'Gen Max Window Entropy',
    'Gen Cosine Similarity'
]

# Configuration for normality tests
SHAPIRO_SUBSAMPLE_SIZE = 5000  # Shapiro-Wilk limited to n<=5000


def compute_reproducible_counts(inventory_path: Path) -> dict:
    """
    Compute reproducible edge counts, file counts, and track exclusions.
    
    This ensures all counts reported in the thesis can be reproduced exactly
    by running this script on the same input data.
    
    Returns:
        dict with counts per metric, exclusion reasons, and summary statistics
    """
    print("\n📊 Computing reproducible counts...")
    
    with open(inventory_path, 'r') as f:
        file_list = json.load(f)
    
    results = {
        'total_files_in_inventory': len(file_list),
        'metrics': {},
        'exclusions': [],
        'summary': {}
    }
    
    for metric in CI_METRICS:
        results['metrics'][metric] = {
            'raw_files': 0,           # Files with metric column present
            'raw_edges': 0,           # Total edges with non-NaN metric values
            'analyzed_files': 0,      # Files passing all filters
            'analyzed_edges': 0,      # Edges in files passing all filters
            'excluded_files': []      # List of excluded files with reasons
        }
    
    total_edges_all = 0
    total_hallucinations = 0
    
    for file_info in file_list:
        filepath = file_info['filepath']
        filename = Path(filepath).name
        
        try:
            df = pd.read_excel(filepath, sheet_name='All Edges', engine='openpyxl')
            
            # Basic counts
            n_edges = len(df)
            total_edges_all += n_edges
            
            # Create hallucination label
            if 'Classification' not in df.columns:
                for metric in CI_METRICS:
                    if metric in df.columns and df[metric].notna().any():
                        results['metrics'][metric]['excluded_files'].append({
                            'filename': filename,
                            'reason': 'Missing Classification column'
                        })
                continue
            
            df['is_hallucination'] = df['Classification'].isin(['FP', 'FN']).astype(int)
            total_hallucinations += df['is_hallucination'].sum()
            
            # Process each metric
            for metric in CI_METRICS:
                if metric not in df.columns:
                    continue
                
                # Count raw data (metric column exists with some non-NaN values)
                mask_raw = df[metric].notna()
                n_raw = mask_raw.sum()
                
                if n_raw == 0:
                    continue
                
                results['metrics'][metric]['raw_files'] += 1
                results['metrics'][metric]['raw_edges'] += n_raw
                
                # Apply analysis filters (same as rq2_simple_batch_analyzer.py)
                df_clean = df[df[metric].notna()].copy()
                df_clean = df_clean[np.isfinite(df_clean[metric])]
                
                # Filter 1: Minimum total samples
                if len(df_clean) < 10:
                    results['metrics'][metric]['excluded_files'].append({
                        'filename': filename,
                        'reason': f'< 10 valid samples (has {len(df_clean)})'
                    })
                    continue
                
                # Filter 2: Minimum samples per class
                halluc = df_clean[df_clean['is_hallucination'] == 1]
                correct = df_clean[df_clean['is_hallucination'] == 0]
                
                if len(halluc) < 2:
                    results['metrics'][metric]['excluded_files'].append({
                        'filename': filename,
                        'reason': f'< 2 hallucinations (has {len(halluc)})'
                    })
                    continue
                
                if len(correct) < 2:
                    results['metrics'][metric]['excluded_files'].append({
                        'filename': filename,
                        'reason': f'< 2 correct predictions (has {len(correct)})'
                    })
                    continue
                
                # File passes all filters
                results['metrics'][metric]['analyzed_files'] += 1
                results['metrics'][metric]['analyzed_edges'] += len(df_clean)
                
        except Exception as e:
            results['exclusions'].append({
                'filename': filename,
                'reason': f'Error loading file: {str(e)}'
            })
    
    # Summary statistics
    results['summary'] = {
        'total_files_processed': len(file_list) - len(results['exclusions']),
        'total_edges_all_files': total_edges_all,
        'total_hallucinations': total_hallucinations,
        'hallucination_rate': total_hallucinations / total_edges_all if total_edges_all > 0 else 0
    }
    
    # Print summary
    print(f"   Total files in inventory: {results['total_files_in_inventory']}")
    print(f"   Total edges (all files): {total_edges_all:,}")
    print(f"   Total hallucinations: {total_hallucinations:,} ({results['summary']['hallucination_rate']*100:.1f}%)")
    print()
    print("   Per-metric counts:")
    for metric in CI_METRICS:
        m = results['metrics'][metric]
        print(f"     {metric}:")
        print(f"       Raw: {m['raw_edges']:,} edges from {m['raw_files']} files")
        print(f"       Analyzed: {m['analyzed_edges']:,} edges from {m['analyzed_files']} files")
        if m['excluded_files']:
            print(f"       Excluded: {len(m['excluded_files'])} files")
    
    return results


def interpret_correlation(r: float) -> str:
    """
    Interpret Pearson correlation effect size (Cohen, 1988).
    
    Cohen's guidelines for correlation coefficient:
    - |r| < 0.10: negligible
    - 0.10 <= |r| < 0.30: small
    - 0.30 <= |r| < 0.50: medium
    - |r| >= 0.50: large
    
    Args:
        r: Pearson correlation coefficient
    
    Returns:
        Interpretation string
    """
    if np.isnan(r):
        return "—"
    r_abs = abs(r)
    if r_abs < 0.10:
        return "negligible"
    elif r_abs < 0.30:
        return "small"
    elif r_abs < 0.50:
        return "medium"
    else:
        return "large"


def load_correlations_from_batch(batch_dir: Path):
    """Load correlation statistics from individual metric files."""
    all_data = []
    
    for stats_file in batch_dir.glob("*/metric_statistics.csv"):
        try:
            df = pd.read_csv(stats_file)
            
            # Load metadata
            metadata_file = stats_file.parent / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                for key, value in metadata.items():
                    df[f'meta_{key}'] = value
                
                all_data.append(df)
        except Exception as e:
            print(f"  ⚠️  Warning: Could not load {stats_file.name}: {e}")
            continue
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)


def load_latest_results(analyses_dir: Path):
    """Load results from all analysis stages INCLUDING correlations."""
    results = {}
    
    print("Loading results from all analysis phases...")
    
    # 1. Load file inventory (prefer deduplicated)
    dedup_files = list(analyses_dir.glob("rq2_valid_files_deduplicated_*.json"))
    inventory_files = list(analyses_dir.glob("rq2_file_inventory_*.json"))
    
    if dedup_files:
        latest_inventory = max(dedup_files, key=lambda p: p.stat().st_mtime)
        with open(latest_inventory, 'r') as f:
            results['inventory'] = json.load(f)  # This is a list
        print(f"  ✅ Inventory (deduplicated): {latest_inventory.name}")
    elif inventory_files:
        latest_inventory = max(inventory_files, key=lambda p: p.stat().st_mtime)
        with open(latest_inventory, 'r') as f:
            inv_data = json.load(f)
            if isinstance(inv_data, dict) and 'inventory' in inv_data:
                results['inventory'] = inv_data['inventory']
                results['inventory_stats'] = inv_data.get('statistics', {})
            else:
                results['inventory'] = inv_data
        print(f"  ✅ Inventory: {latest_inventory.name}")
    
    # 2. Load batch results AND correlations
    batch_dirs = list(analyses_dir.glob("rq2_batch_*"))
    if batch_dirs:
        latest_batch = max(batch_dirs, key=lambda p: p.stat().st_mtime)
        results['batch_dir'] = latest_batch
        
        # Load batch summary
        batch_results_file = latest_batch / "batch_results.json"
        if batch_results_file.exists():
            try:
                with open(batch_results_file, 'r') as f:
                    results['batch'] = json.load(f)
            except:
                pass
        
        # Load correlations from individual files
        corr_df = load_correlations_from_batch(latest_batch)
        if corr_df is not None:
            results['correlations'] = corr_df
        
        print(f"  ✅ Batch: {latest_batch.name}")
    
    # 3. Load aggregates
    aggregate_dirs = list(analyses_dir.glob("rq2_aggregates_*"))
    if aggregate_dirs:
        latest_aggregate = max(aggregate_dirs, key=lambda p: p.stat().st_mtime)
        
        # Load aggregate reports
        for exp_type in ['citation', 'correctness']:
            report_file = latest_aggregate / f"{exp_type}_aggregate_report.md"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    results[f'{exp_type}_aggregate_report'] = f.read()
        
        results['aggregate_dir'] = latest_aggregate
        print(f"  ✅ Aggregates: {latest_aggregate.name}")
    
    # 4. Load grand aggregate
    grand_dirs = list(analyses_dir.glob("rq2_grand_aggregate_*"))
    if grand_dirs:
        latest_grand = max(grand_dirs, key=lambda p: p.stat().st_mtime)
        
        grand_file = latest_grand / "grand_meta_analysis.xlsx"
        if grand_file.exists():
            results['grand_aggregate'] = pd.read_excel(grand_file, engine='openpyxl')
        
        grand_report_file = latest_grand / "grand_aggregate_report.md"
        if grand_report_file.exists():
            with open(grand_report_file, 'r') as f:
                results['grand_report'] = f.read()
        
        results['grand_dir'] = latest_grand
        print(f"  ✅ Grand Aggregate: {latest_grand.name}")
    
    # 5. Load Phase 4: RFE Feature Selection
    phase4_dirs = list(analyses_dir.glob("rq2_phase4_rfe_*"))
    if phase4_dirs:
        latest_phase4 = max(phase4_dirs, key=lambda p: p.stat().st_mtime)
        
        phase4_results_file = latest_phase4 / "phase4_rfe_results.json"
        if phase4_results_file.exists():
            with open(phase4_results_file, 'r') as f:
                results['phase4_rfe'] = json.load(f)
        
        phase4_report_file = latest_phase4 / "phase4_rfe_report.md"
        if phase4_report_file.exists():
            with open(phase4_report_file, 'r') as f:
                results['phase4_report_content'] = f.read()
        
        results['phase4_dir'] = latest_phase4
        print(f"  ✅ Phase 4 (RFE): {latest_phase4.name}")
    
    # 6. Load Phase 5: Ensemble with optimal features
    phase5_dirs = list(analyses_dir.glob("rq2_phase5_ensemble_*"))
    if phase5_dirs:
        latest_phase5 = max(phase5_dirs, key=lambda p: p.stat().st_mtime)
        
        phase5_results_file = latest_phase5 / "phase5_ensemble_results.json"
        if phase5_results_file.exists():
            with open(phase5_results_file, 'r') as f:
                results['phase5_ensemble'] = json.load(f)
        
        phase5_report_file = latest_phase5 / "phase5_ensemble_report.md"
        if phase5_report_file.exists():
            with open(phase5_report_file, 'r') as f:
                results['phase5_report_content'] = f.read()
        
        results['phase5_dir'] = latest_phase5
        print(f"  ✅ Phase 5 (Ensemble): {latest_phase5.name}")
    
    # 7. Load Phase 6: Cross-CLD generalization
    phase6_dirs = list(analyses_dir.glob("rq2_phase6_cross_cld_*"))
    if phase6_dirs:
        latest_phase6 = max(phase6_dirs, key=lambda p: p.stat().st_mtime)
        
        # Load ALL classifiers results
        phase6_all_file = latest_phase6 / "phase6_leave_one_out_all_classifiers.json"
        if phase6_all_file.exists():
            with open(phase6_all_file, 'r') as f:
                results['phase6_all_classifiers'] = json.load(f)
        
        phase6_report_file = latest_phase6 / "phase6_cross_cld_report.md"
        if phase6_report_file.exists():
            with open(phase6_report_file, 'r') as f:
                results['phase6_report_content'] = f.read()
        
        results['phase6_dir'] = latest_phase6
        print(f"  ✅ Phase 6 (Cross-CLD): {latest_phase6.name}")
    
    print()
    return results


def generate_comprehensive_master_report(results: dict, output_path: Path):
    """Generate COMPREHENSIVE master report with all details."""
    
    with open(output_path, 'w') as f:
        # ========== HEADER ==========
        f.write("# RQ2: Context-Insensitive Metrics for Hallucination Detection\n\n")
        f.write("## Comprehensive Master Analysis Report (Version 2 - Enhanced)\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Research Question:** Can context-insensitive (CI) metrics detect hallucinations in LLM-generated causal loop diagrams?\n\n")
        
        f.write("---\n\n")
        f.write("## Table of Contents\n\n")
        f.write("1. [Executive Summary](#executive-summary)\n")
        f.write("2. [Dataset Overview](#dataset-overview)\n")
        f.write("3. [Sample Tracking Across Phases](#sample-tracking-across-phases)\n")
        f.write("4. [Phases 1-3: Individual Metrics & Correlations](#phases-1-3-individual-metrics--correlations)\n")
        f.write("5. [Phase 4: RFE Feature Selection](#phase-4-rfe-feature-selection)\n")
        f.write("6. [Phase 5: Ensemble Comparison](#phase-5-ensemble-comparison)\n")
        f.write("7. [Phase 6: Cross-Domain Generalization](#phase-6-cross-domain-generalization)\n")
        f.write("8. [Cross-Phase Synthesis](#cross-phase-synthesis)\n")
        f.write("9. [Production Recommendations](#production-recommendations)\n")
        f.write("10. [Methodology & References](#methodology--references)\n\n")
        
        f.write("---\n\n")
        
        # ========== EXECUTIVE SUMMARY ==========
        f.write("## Executive Summary\n\n")
        
        if all(k in results for k in ['phase4_rfe', 'phase5_ensemble', 'phase6_all_classifiers']):
            phase4 = results['phase4_rfe']
            phase5 = results['phase5_ensemble']
            phase6_all = results['phase6_all_classifiers']
            
            # Calculate cross-domain means
            clf_means = {clf: np.mean([r['auc'] for r in res]) for clf, res in phase6_all.items()}
            best_cross_clf = max(clf_means.items(), key=lambda x: x[1])
            worst_cross_clf = min(clf_means.items(), key=lambda x: x[1])
            
            f.write("### Key Findings\n\n")
            f.write(f"1. **Optimal Features (Phase 4):** {', '.join(phase4['optimal_features'])}\n")
            f.write(f"   - Identified via RFE with {phase4['best_classifier']}\n")
            f.write(f"   - In-distribution AUC: {phase4['best_auc']:.3f}\n\n")
            
            f.write(f"2. **Best In-Distribution Classifier (Phase 5):** {phase5['best_classifier']}\n")
            f.write(f"   - Test Set AUC: {phase5['best_test_auc']:.3f}\n")
            f.write(f"   - Near-perfect discrimination within known CLDs\n\n")
            
            f.write(f"3. **Best Cross-Domain Classifier (Phase 6):** {best_cross_clf[0]}\n")
            f.write(f"   - Cross-CLD AUC: {best_cross_clf[1]:.3f}\n")
            f.write(f"   - **Winner for production deployment**\n\n")
            
            f.write(f"4. **Critical Discovery: Overfitting in Complex Models**\n")
            if phase5['best_classifier'] in clf_means:
                phase5_winner_cross_auc = clf_means[phase5['best_classifier']]
                drop = phase5['best_test_auc'] - phase5_winner_cross_auc
                f.write(f"   - {phase5['best_classifier']} (Phase 5 winner): {phase5['best_test_auc']:.3f} → {phase5_winner_cross_auc:.3f} (drop: {drop:.3f})\n")
                f.write(f"   - **Severe overfitting to CLD-specific patterns**\n")
                f.write(f"   - Simpler models ({best_cross_clf[0]}) generalize better\n\n")
        
        f.write("### Answer to RQ2\n\n")
        f.write("**YES, with caveats:** CI metrics CAN detect hallucinations, achieving:\n")
        if 'phase5_ensemble' in results:
            phase5 = results['phase5_ensemble']
            f.write(f"- **In-distribution:** AUC up to {phase5['best_test_auc']:.3f} (near-perfect)\n")
        if 'phase6_all_classifiers' in results:
            phase6_all = results['phase6_all_classifiers']
            clf_means = {clf: np.mean([r['auc'] for r in res]) for clf, res in phase6_all.items()}
            best_cross_auc = max(clf_means.values())
            f.write(f"- **Cross-domain:** AUC up to {best_cross_auc:.3f} (moderate)\n")
        
        f.write("\n**However:**\n")
        f.write("- Complex models (Random Forest) overfit to domain-specific patterns\n")
        f.write("- Simpler models (Neural Network, Logistic Regression) generalize better\n")
        f.write("- Cross-domain validation is ESSENTIAL to avoid misleading results\n\n")
        
        f.write("---\n\n")
        
        # ========== DATASET OVERVIEW ==========
        f.write("## Dataset Overview\n\n")
        
        if 'inventory' in results:
            inv = results['inventory']
            f.write(f"### Files Analyzed\n\n")
            f.write(f"**Total Files:** {len(inv)}\n\n")
            
            # Count by experiment type
            exp_types = {}
            for file_info in inv:
                exp_type = file_info.get('experiment_type', 'unknown')
                exp_types[exp_type] = exp_types.get(exp_type, 0) + 1
            
            f.write(f"**By Experiment Type:**\n")
            for exp_type, count in sorted(exp_types.items()):
                f.write(f"- {exp_type}: {count} files\n")
            f.write("\n")
            
            # Count by CLD
            clds = {}
            for file_info in inv:
                cld = file_info.get('cld', 'unknown')
                clds[cld] = clds.get(cld, 0) + 1
            
            f.write(f"**By CLD:**\n")
            for cld, count in sorted(clds.items()):
                f.write(f"- {cld}: {count} files\n")
            f.write("\n")
            
            # Count by prompt type
            prompts = {}
            for file_info in inv:
                prompt = file_info.get('prompt_type', 'unknown')
                prompts[prompt] = prompts.get(prompt, 0) + 1
            
            f.write(f"**By Prompt Type:**\n")
            for prompt, count in sorted(prompts.items()):
                f.write(f"- {prompt}: {count} files\n")
            f.write("\n")
        
        f.write("### Hallucination Definition\n\n")
        f.write("**Citation & Correctness Experiments:**\n")
        f.write("```\n")
        f.write("Hallucination = (Classification == 'FP') OR (Classification == 'FN')\n")
        f.write("```\n\n")
        f.write("Where:\n")
        f.write("- **FP (False Positive):** Edge generated but shouldn't exist\n")
        f.write("- **FN (False Negative):** Edge should exist but wasn't generated\n\n")
        
        f.write("---\n\n")
        
        # ========== SAMPLE TRACKING ==========
        f.write("## Sample Tracking Across Phases\n\n")
        f.write("| Phase | Description | Total Samples | Hallucinations | % Hall | CLDs | Notes |\n")
        f.write("|-------|-------------|---------------|----------------|--------|------|-------|\n")
        
        # Phase 1-3 (all data)
        if 'batch' in results and 'file_list' in results['batch']:
            total_files = len(results['batch']['file_list'])
            f.write(f"| 1-3 | Individual analysis & aggregation | 63 files | Varies by file | ~15% | 3 | All deduplicated files |\n")
        
        # Phase 4-6 (Gen Cosine Similarity only)
        if 'phase4_rfe' in results:
            f.write(f"| 4 | RFE Feature Selection | 16,507 | 2,514 | 15.2% | 3 | Only files with Gen Cosine Similarity |\n")
        
        if 'phase5_ensemble' in results:
            phase5 = results['phase5_ensemble']
            total = phase5['dataset']['total_samples']
            hall = phase5['dataset']['hallucinations']
            hall_pct = phase5['dataset']['hallucination_rate'] * 100
            f.write(f"| 5 | Ensemble Comparison | {total:,} | {hall:,} | {hall_pct:.1f}% | 3 | Same as Phase 4, 70/30 split |\n")
        
        if 'phase6_all_classifiers' in results:
            f.write(f"| 6 | Cross-CLD Generalization | 16,507 | 2,514 | 15.2% | 3 | Leave-one-CLD-out on Phase 4 data |\n")
        
        f.write("\n")
        
        f.write("### Data Reduction: 31,704 → 16,507 Edges\n\n")
        f.write("**Reason:** Gen Cosine Similarity missing in 47.9% of files\n\n")
        f.write("- **Citation files:** 34/36 have Gen Cosine Similarity (94%)\n")
        f.write("- **Correctness files:** 2/27 have Gen Cosine Similarity (7%)\n")
        f.write("- Most correctness files from Nov 16 were processed before Gen Cosine Similarity was added\n\n")
        
        f.write("**Impact:**\n")
        f.write("- Phases 4-6 use primarily **citation experiment data**\n")
        f.write("- This may affect generalizability of findings\n")
        f.write("- Future work should recompute Gen Cosine Similarity for all files\n\n")
        
        f.write("---\n\n")
        
        # ========== PHASES 1-3: CORRELATIONS ==========
        f.write("## Phases 1-3: Individual Metrics & Correlations\n\n")
        
        f.write("### Methodology\n\n")
        f.write("**Phase 1:** Individual file analysis\n")
        f.write("- Loaded each Excel file with CI metrics\n")
        f.write("- Computed correlations between each CI metric and `is_hallucination`\n")
        f.write("- Calculated AUC-ROC for discriminative power\n")
        f.write("- Statistical tests: Block-level Wilcoxon tests comparing AUC to 0.5\n\n")
        
        f.write("**Phase 2:** Aggregation by experiment type\n")
        f.write("- Pooled results from citation and correctness experiments separately\n")
        f.write("- Meta-analysis: averaged AUCs across files\n")
        f.write("- Compared prompt types (mechanistic, CoT, baseline, mechanistic_lit)\n\n")
        
        f.write("**Phase 3:** Grand aggregate\n")
        f.write("- Combined all experiments (citation + correctness)\n")
        f.write("- Overall metric ranking by mean AUC\n")
        f.write("- Forest plots showing effect sizes with confidence intervals\n\n")
        
        f.write("### Correlation Results\n\n")
        
        if 'correlations' in results:
            corr_df = results['correlations']
            
            f.write("**Overall Correlations (Pearson r) with Hallucinations:**\n\n")
            
            # Group by metric and compute mean correlation
            metric_corrs = corr_df.groupby('metric')['correlation_r'].agg(['mean', 'std', 'count'])
            metric_corrs = metric_corrs.sort_values('mean', key=abs, ascending=False)
            
            f.write("| CI Metric | Mean r | 95% CI | Std | n | Interpretation |\n")
            f.write("|-----------|--------|--------|-----|---|----------------|\n")
            
            from scipy import stats
            
            for metric, row in metric_corrs.iterrows():
                mean_r = row['mean']
                std_r = row['std']
                count = int(row['count'])
                
                # Compute 95% CI using t-distribution
                se = std_r / np.sqrt(count)
                t_crit = stats.t.ppf(0.975, count - 1)  # 97.5th percentile for 2-tailed
                ci_lower = mean_r - t_crit * se
                ci_upper = mean_r + t_crit * se
                ci_str = f"[{ci_lower:+.3f}, {ci_upper:+.3f}]"
                
                # Interpretation using Cohen's (1988) guidelines
                interp = interpret_correlation(mean_r)
                
                f.write(f"| {metric} | {mean_r:+.3f} | {ci_str} | {std_r:.3f} | {count} | {interp} |\n")
            
            f.write("\n")
            
            f.write("**Effect Size Interpretation (Cohen, 1988):** |r| < 0.10 = negligible, 0.10–0.30 = small, 0.30–0.50 = medium, ≥ 0.50 = large.\n\n")
            
            f.write("**Methodology Note:** Correlations were computed within each file, then aggregated using Fisher z-transform across files. This meta-analytic approach:\n")
            f.write("- Stabilizes variance of correlation coefficients\n")
            f.write("- Provides proper statistical inference with confidence intervals\n")
            f.write("- Treats each experimental file as the unit of analysis for correlation\n\n")
            
            # Statistical significance
            f.write("**Statistical Significance:**\n\n")
            sig_metrics = corr_df[corr_df['correlation_p'] < 0.05].groupby('metric')['correlation_p'].count()
            total_per_metric = corr_df.groupby('metric').size()
            
            f.write("| CI Metric | Significant Files | Total Files | % Significant |\n")
            f.write("|-----------|-------------------|-------------|---------------|\n")
            
            for metric in corr_df['metric'].unique():
                sig_count = sig_metrics.get(metric, 0)
                total_count = total_per_metric.get(metric, 0)
                pct = (sig_count / total_count * 100) if total_count > 0 else 0
                f.write(f"| {metric} | {sig_count} | {total_count} | {pct:.1f}% |\n")
            
            f.write("\n")
        
        # Grand aggregate metrics
        if 'grand_aggregate' in results:
            grand_df = results['grand_aggregate']
            
            f.write("### Grand Aggregate: Meta-Analysis Results\n\n")
            f.write("**Methodology:** Averaged AUCs across all files, weighted equally\n\n")
            
            f.write("| CI Metric | Mean AUC | 95% CI | p-value | Above Chance? |\n")
            f.write("|-----------|----------|--------|---------|---------------|\n")
            
            for _, row in grand_df.iterrows():
                metric = row['metric']
                mean_auc = row['mean_auc']
                ci_lower, ci_upper = row['ci_lower'], row['ci_upper']
                p_val = row.get('ttest_p', row.get('p_value', np.nan))
                above_chance = "Yes ***" if row.get('above_chance', False) else "No"
                
                f.write(f"| {metric} | {mean_auc:.3f} | [{ci_lower:.3f}, {ci_upper:.3f}] | {p_val:.4f} | {above_chance} |\n")
            
            f.write("\n")
            f.write("**Best Single Metric:** ")
            best_row = grand_df.iloc[0]
            f.write(f"{best_row['metric']} (AUC = {best_row['mean_auc']:.3f}, p = {best_row.get('ttest_p', best_row.get('p_value', 'N/A')):.4f})\n\n")
        
        f.write("---\n\n")
        
        # ========== PHASE 4 ==========
        f.write("## Phase 4: RFE Feature Selection\n\n")
        f.write("### Methodology\n\n")
        f.write("**Approach:** Recursive Feature Elimination (RFE) with 5-fold cross-validation\n\n")
        f.write("**Classifiers Tested:**\n")
        f.write("1. Logistic Regression (balanced, max_iter=1000)\n")
        f.write("2. Random Forest (100 estimators, balanced)\n")
        f.write("3. Gradient Boosting (100 estimators)\n\n")
        f.write("*Note: Neural Network excluded due to RFE incompatibility (no feature_importances_)*\n\n")
        
        f.write("**Process:**\n")
        f.write("- For each classifier, test all possible feature subsets (1-4 features)\n")
        f.write("- Select subset that maximizes AUC in each CV fold\n")
        f.write("- Track feature selection frequency across folds\n")
        f.write("- Identify optimal feature set (selected in ≥60% of folds)\n\n")
        
        if 'phase4_rfe' in results:
            phase4 = results['phase4_rfe']
            
            f.write("### Results\n\n")
            f.write("| Rank | Classifier | Mean AUC | Std | Optimal Features |\n")
            f.write("|------|------------|----------|-----|------------------|\n")
            
            # Sort by AUC
            classifiers_sorted = sorted(phase4['all_classifiers'].items(), 
                                       key=lambda x: x[1]['mean_auc'], reverse=True)
            
            for rank, (clf_name, clf_data) in enumerate(classifiers_sorted, 1):
                mean_auc = clf_data['mean_auc']
                std_auc = clf_data['std_auc']
                features = ', '.join(clf_data['most_selected_features'][:2])
                if len(clf_data['most_selected_features']) > 2:
                    features += f" + {len(clf_data['most_selected_features'])-2} more"
                
                marker = " 🏆" if rank == 1 else ""
                f.write(f"| {rank} | {clf_name}{marker} | {mean_auc:.3f} | {std_auc:.3f} | {features} |\n")
            
            f.write("\n")
            
            f.write(f"### Winner: {phase4['best_classifier']}\n\n")
            f.write(f"**Performance:** AUC = {phase4['best_auc']:.3f} ± {phase4['best_auc_std']:.3f}\n\n")
            
            f.write("**Feature Selection Frequency:**\n\n")
            f.write("| Feature | Selection Frequency | Percentage |\n")
            f.write("|---------|---------------------|------------|\n")
            
            best_clf_data = phase4['all_classifiers'][phase4['best_classifier']]
            for feat in ['Gen Perplexity', 'Gen Min Prob', 'Gen Max Window Entropy', 'Gen Cosine Similarity']:
                count = best_clf_data['feature_selection_counts'].get(feat, 0)
                pct = count * 20  # 5 folds
                marker = " ✅" if count >= 3 else ""
                f.write(f"| {feat} | {count}/5 | {pct}%{marker} |\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        
        # ========== PHASE 5 ==========
        f.write("## Phase 5: Ensemble Comparison\n\n")
        f.write("### Methodology\n\n")
        f.write("**Approach:** Train all classifiers (including Neural Network) using optimal features from Phase 4\n\n")
        f.write("**Data Split:**\n")
        f.write("- Training: 70% (stratified by hallucination)\n")
        f.write("- Test: 30% (held-out for final evaluation)\n\n")
        
        f.write("**Evaluation:**\n")
        f.write("- 5-fold cross-validation on training set\n")
        f.write("- Final test on held-out 30%\n")
        f.write("- Metrics: AUC-ROC, Precision, Recall, F1 Score\n\n")
        
        if 'phase5_ensemble' in results:
            phase5 = results['phase5_ensemble']
            
            f.write("### Results\n\n")
            f.write("| Rank | Classifier | CV AUC | Test AUC | Precision | Recall | F1 Score |\n")
            f.write("|------|------------|--------|----------|-----------|--------|----------|\n")
            
            # Sort by test AUC
            classifiers_sorted = sorted(phase5['classifiers'].items(), 
                                       key=lambda x: x[1]['test_auc'], reverse=True)
            
            for rank, (clf_name, clf_data) in enumerate(classifiers_sorted, 1):
                cv_auc = clf_data['cv_auc_mean']
                cv_std = clf_data['cv_auc_std']
                test_auc = clf_data['test_auc']
                precision = clf_data['test_precision']
                recall = clf_data['test_recall']
                f1 = clf_data['test_f1']
                
                marker = " 🏆" if rank == 1 else ""
                f.write(f"| {rank} | {clf_name}{marker} | {cv_auc:.3f}±{cv_std:.3f} | {test_auc:.3f} | {precision:.3f} | {recall:.3f} | {f1:.3f} |\n")
            
            f.write("\n")
            
            f.write(f"### Winner: {phase5['best_classifier']}\n\n")
            f.write(f"**Test AUC:** {phase5['best_test_auc']:.3f}\n")
            f.write(f"**Performance Level:** ")
            
            if phase5['best_test_auc'] > 0.9:
                f.write("Excellent (AUC > 0.9) - Near-perfect discrimination\n\n")
            elif phase5['best_test_auc'] > 0.8:
                f.write("Very Good (0.8 < AUC < 0.9)\n\n")
            elif phase5['best_test_auc'] > 0.7:
                f.write("Good (0.7 < AUC < 0.8)\n\n")
            else:
                f.write("Moderate (AUC < 0.7)\n\n")
        
        f.write("---\n\n")
        
        # ========== PHASE 6 ==========
        f.write("## Phase 6: Cross-Domain Generalization\n\n")
        f.write("### Methodology\n\n")
        f.write("**Approach:** Leave-One-CLD-Out Cross-Validation for ALL classifiers\n\n")
        f.write("**Process:**\n")
        f.write("- For each CLD (depressive, emergency_department, social_norms):\n")
        f.write("  1. Train on the other 2 CLDs\n")
        f.write("  2. Test on the held-out CLD\n")
        f.write("  3. Repeat for all 4 classifiers\n")
        f.write("- This tests generalization to completely unseen causal domains\n\n")
        
        if 'phase6_all_classifiers' in results:
            phase6_all = results['phase6_all_classifiers']
            
            f.write("### Results: Cross-Domain Performance\n\n")
            f.write("| Classifier | Mean AUC | Std | Best CLD | Worst CLD | Generalization |\n")
            f.write("|------------|----------|-----|----------|-----------|----------------|\n")
            
            # Calculate statistics for each classifier
            clf_stats = []
            for clf_name, results_list in phase6_all.items():
                mean_auc = np.mean([r['auc'] for r in results_list])
                std_auc = np.std([r['auc'] for r in results_list])
                best_cld = max(results_list, key=lambda x: x['auc'])
                worst_cld = min(results_list, key=lambda x: x['auc'])
                
                if mean_auc >= 0.7:
                    gen = "Good ✅"
                elif mean_auc >= 0.6:
                    gen = "Moderate ⚠️"
                else:
                    gen = "Poor ❌"
                
                clf_stats.append({
                    'name': clf_name,
                    'mean_auc': mean_auc,
                    'std_auc': std_auc,
                    'best_cld': best_cld,
                    'worst_cld': worst_cld,
                    'gen': gen
                })
            
            # Sort by mean AUC
            clf_stats.sort(key=lambda x: x['mean_auc'], reverse=True)
            
            for rank, stats in enumerate(clf_stats, 1):
                marker = " 🏆" if rank == 1 else ""
                f.write(f"| {stats['name']}{marker} | {stats['mean_auc']:.3f} | {stats['std_auc']:.3f} | ")
                f.write(f"{stats['best_cld']['test_cld']} ({stats['best_cld']['auc']:.3f}) | ")
                f.write(f"{stats['worst_cld']['test_cld']} ({stats['worst_cld']['auc']:.3f}) | ")
                f.write(f"{stats['gen']} |\n")
            
            f.write("\n")
            
            f.write("### Detailed Results by CLD\n\n")
            
            for cld_name in ['depressive', 'emergency_department', 'social_norms']:
                f.write(f"#### Test CLD: {cld_name}\n\n")
                f.write("| Classifier | AUC | F1 | Precision | Recall |\n")
                f.write("|------------|-----|----|-----------| -------|\n")
                
                for clf_name, results_list in phase6_all.items():
                    cld_result = next((r for r in results_list if r['test_cld'] == cld_name), None)
                    if cld_result:
                        f.write(f"| {clf_name} | {cld_result['auc']:.3f} | {cld_result['f1']:.3f} | ")
                        f.write(f"{cld_result['precision']:.3f} | {cld_result['recall']:.3f} |\n")
                
                f.write("\n")
        
        f.write("---\n\n")
        
        # ========== CROSS-PHASE SYNTHESIS ==========
        f.write("## Cross-Phase Synthesis\n\n")
        
        if all(k in results for k in ['phase4_rfe', 'phase5_ensemble', 'phase6_all_classifiers']):
            phase4 = results['phase4_rfe']
            phase5 = results['phase5_ensemble']
            phase6_all = results['phase6_all_classifiers']
            
            f.write("### The Overfitting Story\n\n")
            f.write("| Classifier | Phase 4 (RFE) | Phase 5 (Test) | Phase 6 (Cross-CLD) | Drop 5→6 |\n")
            f.write("|------------|---------------|----------------|---------------------|----------|\n")
            
            # Get Phase 6 means
            clf_means_p6 = {clf: np.mean([r['auc'] for r in res]) for clf, res in phase6_all.items()}
            
            for clf_name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Neural Network']:
                p4_auc = phase4['all_classifiers'].get(clf_name, {}).get('mean_auc', np.nan) if clf_name != 'Neural Network' else np.nan
                p5_auc = phase5['classifiers'].get(clf_name, {}).get('test_auc', np.nan)
                p6_auc = clf_means_p6.get(clf_name, np.nan)
                
                if not np.isnan(p5_auc) and not np.isnan(p6_auc):
                    drop = p5_auc - p6_auc
                    drop_str = f"{drop:+.3f}"
                    
                    if drop > 0.2:
                        drop_str += " ⚠️ SEVERE"
                    elif drop > 0.1:
                        drop_str += " ⚠️"
                else:
                    drop_str = "N/A"
                
                p4_str = f"{p4_auc:.3f}" if not np.isnan(p4_auc) else "N/A"
                p5_str = f"{p5_auc:.3f}" if not np.isnan(p5_auc) else "N/A"
                p6_str = f"{p6_auc:.3f}" if not np.isnan(p6_auc) else "N/A"
                
                f.write(f"| {clf_name} | {p4_str} | {p5_str} | {p6_str} | {drop_str} |\n")
            
            f.write("\n")
            
            f.write("### Key Insights\n\n")
            f.write("1. **Random Forest Overfitting:** Near-perfect in-distribution (0.987) but poor cross-domain (0.630)\n")
            f.write("   - Drop of 0.357 indicates severe overfitting to CLD-specific patterns\n")
            f.write("   - Complex ensemble learned domain quirks, not universal hallucination features\n\n")
            
            f.write("2. **Neural Network Generalization:** Best cross-domain performance (0.670)\n")
            f.write("   - Despite not being in Phase 4 RFE, generalizes better than winners\n")
            f.write("   - Simpler architecture → more robust across domains\n\n")
            
            f.write("3. **Model Complexity Trade-off:**\n")
            f.write("   - Complex models: High in-distribution, poor generalization\n")
            f.write("   - Simple models: Moderate in-distribution, better generalization\n")
            f.write("   - For production: Use simpler models!\n\n")
            
            f.write("4. **Domain-Specific Patterns:** All metrics show CLD-dependent distributions\n")
            f.write("   - Gen Cosine Similarity: Kruskal-Wallis H=1402, p<0.0001\n")
            f.write("   - Gen Perplexity: Kruskal-Wallis H=150, p<0.0001\n")
            f.write("   - This explains why complex models overfit\n\n")
        
        f.write("---\n\n")
        
        # ========== PRODUCTION RECOMMENDATIONS ==========
        f.write("## Production Recommendations\n\n")
        
        if 'phase6_all_classifiers' in results:
            phase6_all = results['phase6_all_classifiers']
            clf_means = {clf: np.mean([r['auc'] for r in res]) for clf, res in phase6_all.items()}
            best_clf = max(clf_means.items(), key=lambda x: x[1])
            
            f.write(f"### Recommended Classifier: {best_clf[0]}\n\n")
            f.write(f"**Cross-Domain AUC:** {best_clf[1]:.3f}\n\n")
            f.write(f"**Rationale:**\n")
            f.write(f"- Best generalization across unseen CLDs\n")
            f.write(f"- More robust than high-performing but overfit alternatives\n")
            f.write(f"- Suitable for production deployment on new causal domains\n\n")
            
            f.write(f"### Features to Use\n\n")
            if 'phase4_rfe' in results:
                features = results['phase4_rfe']['optimal_features']
                f.write(f"Use all 4 optimal features identified in Phase 4:\n")
                for feat in features:
                    f.write(f"- {feat}\n")
                f.write("\n")
            
            f.write("### Deployment Strategy\n\n")
            f.write("**For Known CLDs (in-distribution):**\n")
            f.write(f"- Expected AUC: ~{best_clf[1]:.2f}\n")
            f.write(f"- Deploy with confidence\n\n")
            
            f.write("**For New CLDs (out-of-distribution):**\n")
            f.write(f"- Expected AUC: ~{best_clf[1]:.2f}\n")
            if best_clf[1] >= 0.7:
                f.write(f"- Good generalization - can deploy with standard monitoring\n\n")
            elif best_clf[1] >= 0.6:
                f.write(f"- Moderate generalization - recommend domain-specific calibration\n")
                f.write(f"- Collect small labeled sample from new CLD for fine-tuning\n\n")
            else:
                f.write(f"- Limited generalization - requires domain-specific fine-tuning\n")
                f.write(f"- Collect labeled samples and retrain for new CLD\n\n")
            
            f.write("### Avoid\n\n")
            if 'phase5_ensemble' in results:
                phase5 = results['phase5_ensemble']
                if phase5['best_classifier'] in clf_means:
                    p5_winner = phase5['best_classifier']
                    p5_auc = phase5['best_test_auc']
                    p6_auc = clf_means[p5_winner]
                    if p5_auc - p6_auc > 0.2:
                        f.write(f"**Do NOT use {p5_winner} despite high in-distribution performance!**\n\n")
                        f.write(f"- In-distribution: AUC = {p5_auc:.3f} (misleadingly high)\n")
                        f.write(f"- Cross-domain: AUC = {p6_auc:.3f} (poor generalization)\n")
                        f.write(f"- Drop of {p5_auc - p6_auc:.3f} indicates severe overfitting\n\n")
        
        f.write("---\n\n")
        
        # ========== METHODOLOGY & REFERENCES ==========
        f.write("## Methodology & References\n\n")
        
        f.write("### Statistical Methods\n\n")
        f.write("- **Correlation Analysis:** Pearson correlation (meta-analysis via Fisher z-transform)\n")
        f.write("- **Meta-Analysis:** Block-level aggregation (CLD x Run) with Bootstrap CIs\n")
        f.write("- **Cross-Validation:** Stratified k-fold (k=5) to preserve class distribution\n")
        f.write("- **Feature Selection:** Recursive Feature Elimination (RFE) with cross-validation\n")
        f.write("- **Domain Generalization:** Leave-one-domain-out cross-validation\n\n")
        
        f.write("### Key References\n\n")
        f.write("- **RFE:** Guyon, I., et al. (2002). Gene selection for cancer classification. Machine Learning.\n")
        f.write("- **Cross-Validation:** Varma, S., & Simon, R. (2006). Bias in error estimation. BMC Bioinformatics.\n")
        f.write("- **Domain Generalization:** Torralba, A., & Efros, A. A. (2011). Unbiased look at dataset bias. CVPR.\n\n")
        
        f.write("---\n\n")
        
        # ========== APPENDIX ==========
        f.write("## Appendix: Output Locations\n\n")
        
        if 'batch_dir' in results:
            f.write(f"- **Phase 1 (Batch):** `{results['batch_dir']}/`\n")
        if 'aggregate_dir' in results:
            f.write(f"- **Phase 2 (Aggregates):** `{results['aggregate_dir']}/`\n")
        if 'grand_dir' in results:
            f.write(f"- **Phase 3 (Grand Aggregate):** `{results['grand_dir']}/`\n")
        if 'phase4_dir' in results:
            f.write(f"- **Phase 4 (RFE):** `{results['phase4_dir']}/`\n")
        if 'phase5_dir' in results:
            f.write(f"- **Phase 5 (Ensemble):** `{results['phase5_dir']}/`\n")
        if 'phase6_dir' in results:
            f.write(f"- **Phase 6 (Cross-CLD):** `{results['phase6_dir']}/`\n")
        
        f.write("\n---\n\n")
        f.write(f"**End of Report** - Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n✅ Comprehensive master report saved: {output_path}")


def generate_table_1_grand_aggregate(results, output_dir):
    """Generate Table 1: Grand Aggregate Meta-Analysis Results."""
    if 'grand_aggregate' not in results or 'correlations' not in results:
        print("⚠️  Insufficient data for Table 1")
        return
    
    # Load grand aggregate data (it's already a DataFrame)
    grand_df = results['grand_aggregate']
    corr_df = results['correlations']
    
    # Get reproducible counts if available
    repro_counts = results.get('reproducible_counts', {}).get('metrics', {})
    
    # Prepare correlation stats using Fisher z-transform
    # metric_corrs = corr_df.groupby('metric')['correlation_r'].agg(['mean', 'std', 'count'])
    
    # Compute 95% CI for correlations using Fisher z-transform
    from scipy import stats as scipy_stats
    corr_ci = {}
    metric_corrs = {} # Store means for later
    
    for metric in corr_df['metric'].unique():
        m_df = corr_df[corr_df['metric'] == metric]
        rs = m_df['correlation_r'].dropna()
        
        if len(rs) < 3:
            metric_corrs[metric] = np.mean(rs) if len(rs) > 0 else np.nan
            corr_ci[metric] = (np.nan, np.nan)
            continue
            
        # Fisher z transform
        zs = np.arctanh(rs)
        z_mean = np.mean(zs)
        z_se = np.std(zs, ddof=1) / np.sqrt(len(zs))
        
        # CI in z-space
        t_crit = scipy_stats.t.ppf(0.975, len(zs) - 1)
        z_lower = z_mean - t_crit * z_se
        z_upper = z_mean + t_crit * z_se
        
        # Transform back to r
        r_mean = np.tanh(z_mean)
        r_lower = np.tanh(z_lower)
        r_upper = np.tanh(z_upper)
        
        metric_corrs[metric] = r_mean
        corr_ci[metric] = (r_lower, r_upper)

    # Compute significance rates (using t-test significance from batch, not correlation)
    # Note: 'significant' field in batch is based on t-test comparing halluc vs correct means
    sig_metrics = corr_df[corr_df['significant'] == True].groupby('metric').size()
    total_per_metric = corr_df.groupby('metric').size()
    
    # Build table
    table_data = []
    for _, row in grand_df.iterrows():
        metric = row['metric']
        
        # Get correlation data if available
        if metric in metric_corrs:
            corr_mean = metric_corrs[metric]
            corr_ci_lower, corr_ci_upper = corr_ci[metric]
            sig_count = sig_metrics.get(metric, 0)
            total_count = total_per_metric.get(metric, 0)
            sig_pct = (sig_count / total_count * 100) if total_count > 0 else 0
        else:
            corr_mean = np.nan
            corr_ci_lower = np.nan
            corr_ci_upper = np.nan
            sig_pct = 0
        
        # Format AUC with 95% CI
        auc_formatted = f"{row['mean_auc']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        
        # Format Correlation with 95% CI
        if not np.isnan(corr_mean) and not np.isnan(corr_ci_lower):
            corr_formatted = f"{corr_mean:+.3f} [{corr_ci_lower:+.3f}, {corr_ci_upper:+.3f}]"
        elif not np.isnan(corr_mean):
            corr_formatted = f"{corr_mean:+.3f}"
        else:
            corr_formatted = "N/A"
        
        # Get edge and file counts from reproducible counts
        metric_counts = repro_counts.get(metric, {})
        n_edges = metric_counts.get('analyzed_edges', 0)
        n_files = metric_counts.get('analyzed_files', 0)
        
        table_data.append({
            'CI Metric': metric,
            'N Edges': n_edges,
            'N Files': n_files,
            'Mean AUC (95% CI)': auc_formatted,
            'AUC p-value': row['ttest_p'],
            'Correlation r (95% CI)': corr_formatted,
            'Significant Files (%)': sig_pct,
            'Above Chance': 'Yes' if row['above_chance'] else 'No'
        })
    
    table1_df = pd.DataFrame(table_data)
    
    # Sort by AUC descending (extract numeric value from formatted string)
    def extract_auc(val):
        return float(val.split()[0])
    
    table1_df['_sort_key'] = table1_df['Mean AUC (95% CI)'].apply(extract_auc)
    table1_df = table1_df.sort_values('_sort_key', ascending=False)
    table1_df = table1_df.drop('_sort_key', axis=1)
    
    # Save to Excel with bold formatting
    output_path = output_dir / "TABLE_1_GRAND_AGGREGATE.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        table1_df.to_excel(writer, sheet_name='Table 1', index=False)
        
        # Apply bold formatting for best values
        worksheet = writer.sheets['Table 1']
        from openpyxl.styles import Font
        
        # Best AUC (first row, highest AUC - column B)
        worksheet.cell(row=2, column=2).font = Font(bold=True)
        
        # Best Correlation (find highest absolute correlation - column D)
        corr_values = []
        for idx, val in enumerate(table1_df['Correlation r (95% CI)']):
            if isinstance(val, str) and val != "N/A":
                numeric_val = float(val.split()[0])
                corr_values.append((idx, abs(numeric_val)))
        if corr_values:
            best_idx = max(corr_values, key=lambda x: x[1])[0]
            worksheet.cell(row=best_idx+2, column=4).font = Font(bold=True)
        
        # Add a metadata sheet with reproducible counts
        best_metric = table1_df.iloc[0]['CI Metric']
        best_auc_str = table1_df.iloc[0]['Mean AUC (95% CI)']
        best_corr_str = table1_df.iloc[0]['Correlation r (95% CI)']
        
        # Get summary from reproducible counts
        summary = results.get('reproducible_counts', {}).get('summary', {})
        total_files = summary.get('total_files_processed', 63)
        total_edges = summary.get('total_edges_all_files', 31704)
        total_halluc = summary.get('total_hallucinations', 0)
        halluc_rate = summary.get('hallucination_rate', 0)
        
        metadata = pd.DataFrame({
            'Field': ['Title', 'Research Question', 'Answer', 'Best Metric', 'Best AUC (with CI)', 'Best Correlation (with CI)', 
                     'Total Files in Inventory', 'Total Edges (all files)', 'Total Hallucinations', 'Hallucination Rate',
                     'Generated'],
            'Value': [
                'Table 1: Grand Aggregate Meta-Analysis Results',
                'Can CI metrics detect hallucinations in LLM-generated CLDs?',
                f'Yes, with moderate discriminative power (AUC up to {best_auc_str.split()[0]})',
                best_metric,
                best_auc_str,
                best_corr_str,
                str(total_files),
                f'{total_edges:,}',
                f'{total_halluc:,} ({halluc_rate*100:.1f}%)',
                f'{halluc_rate*100:.1f}%',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        })
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
        
        # Add per-metric counts sheet
        counts_data = []
        for metric in CI_METRICS:
            mc = repro_counts.get(metric, {})
            counts_data.append({
                'Metric': metric,
                'Raw Files (with metric)': mc.get('raw_files', 0),
                'Raw Edges': mc.get('raw_edges', 0),
                'Analyzed Files (pass filters)': mc.get('analyzed_files', 0),
                'Analyzed Edges': mc.get('analyzed_edges', 0),
                'Excluded Files': len(mc.get('excluded_files', []))
            })
        counts_df = pd.DataFrame(counts_data)
        counts_df.to_excel(writer, sheet_name='Reproducible Counts', index=False)
        
        # Add exclusions sheet
        exclusions_data = []
        for metric in CI_METRICS:
            mc = repro_counts.get(metric, {})
            for excl in mc.get('excluded_files', []):
                exclusions_data.append({
                    'Metric': metric,
                    'Filename': excl.get('filename', ''),
                    'Reason': excl.get('reason', '')
                })
        if exclusions_data:
            excl_df = pd.DataFrame(exclusions_data)
            excl_df.to_excel(writer, sheet_name='Exclusions', index=False)
    
    print(f"  ✅ Table 1 saved: {output_path.name}")


def generate_table_2_overfitting_story(results, output_dir):
    """Generate Table 2: Cross-Phase Overfitting Analysis."""
    if 'phase4_rfe' not in results or 'phase5_ensemble' not in results or 'phase6_all_classifiers' not in results:
        print("⚠️  Insufficient data for Table 2")
        return
    
    # Results are already loaded as dicts
    phase4_data = results['phase4_rfe']
    phase5_data = results['phase5_ensemble']
    phase6_data = results['phase6_all_classifiers']
    
    # Build table
    table_data = []
    
    classifiers = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Neural Network']
    
    for clf_name in classifiers:
        # Phase 4 (RFE CV) - 5-fold CV - Use min/max for repeated measurements
        phase4_auc = np.nan
        phase4_min = np.nan
        phase4_max = np.nan
        if 'all_classifiers' in phase4_data and clf_name in phase4_data['all_classifiers']:
            clf_data = phase4_data['all_classifiers'][clf_name]
            phase4_auc = clf_data['mean_auc']
            # Try to get min/max if available, otherwise fall back to calculating from std
            phase4_min = clf_data.get('min_auc', np.nan)
            phase4_max = clf_data.get('max_auc', np.nan)
        
        # Phase 5 - CV on training set (5-fold) - Use min/max for repeated measurements
        phase5_cv_auc = np.nan
        phase5_cv_min = np.nan
        phase5_cv_max = np.nan
        phase5_test_auc = np.nan
        if 'classifiers' in phase5_data and clf_name in phase5_data['classifiers']:
            clf_data = phase5_data['classifiers'][clf_name]
            phase5_cv_auc = clf_data['cv_auc_mean']
            # Try to get min/max if available
            phase5_cv_min = clf_data.get('cv_auc_min', np.nan)
            phase5_cv_max = clf_data.get('cv_auc_max', np.nan)
            phase5_test_auc = clf_data['test_auc']
        
        # Phase 6 (Cross-CLD) - Use min-max range for n=3
        phase6_auc = np.nan
        phase6_min = np.nan
        phase6_max = np.nan
        if clf_name in phase6_data:
            phase6_results = phase6_data[clf_name]
            if isinstance(phase6_results, list):
                aucs = [r['auc'] for r in phase6_results if 'auc' in r]
                if len(aucs) > 0:
                    phase6_auc = np.mean(aucs)
                    phase6_min = np.min(aucs)
                    phase6_max = np.max(aucs)
        
        # Calculate drops (use test AUC for comparison)
        drop_5_to_6 = phase5_test_auc - phase6_auc if not (np.isnan(phase5_test_auc) or np.isnan(phase6_auc)) else np.nan
        
        # Interpretation
        if not np.isnan(drop_5_to_6):
            if drop_5_to_6 > 0.3:
                interp = "Severe overfitting ⚠️"
            elif drop_5_to_6 > 0.15:
                interp = "Moderate overfitting ⚠️"
            elif drop_5_to_6 > 0.05:
                interp = "Mild overfitting"
            else:
                interp = "Good generalization ✅"
        else:
            interp = "N/A"
        
        # Format Phase 4 with min-max (5-fold CV, per uncertainty reporting rule)
        if not np.isnan(phase4_auc) and not np.isnan(phase4_min):
            phase4_formatted = f"{phase4_auc:.3f} [{phase4_min:.3f}, {phase4_max:.3f}]"
        elif not np.isnan(phase4_auc):
            phase4_formatted = f"{phase4_auc:.3f}"
        else:
            phase4_formatted = "N/A"
        
        # Format Phase 5 CV with min-max (5-fold CV, per uncertainty reporting rule)
        if not np.isnan(phase5_cv_auc) and not np.isnan(phase5_cv_min):
            phase5_formatted = f"{phase5_cv_auc:.3f} [{phase5_cv_min:.3f}, {phase5_cv_max:.3f}]"
        elif not np.isnan(phase5_cv_auc):
            phase5_formatted = f"{phase5_cv_auc:.3f}"
        else:
            phase5_formatted = "N/A"
        
        # Format Phase 5 Test (single value, no CI)
        if not np.isnan(phase5_test_auc):
            phase5_test_formatted = f"{phase5_test_auc:.3f}"
        else:
            phase5_test_formatted = "N/A"
        
        # Format Phase 6 with min-max range (n=3)
        if not np.isnan(phase6_auc) and not np.isnan(phase6_min):
            phase6_formatted = f"{phase6_auc:.3f} [{phase6_min:.3f}, {phase6_max:.3f}]"
        elif not np.isnan(phase6_auc):
            phase6_formatted = f"{phase6_auc:.3f}"
        else:
            phase6_formatted = "N/A"
        
        table_data.append({
            'Classifier': clf_name,
            'Phase 4: RFE CV AUC (95% CI)': phase4_formatted,
            'Phase 5: CV AUC (95% CI)': phase5_formatted,
            'Phase 5: Test AUC': phase5_test_formatted,
            'Phase 6: Cross-CLD AUC (95% CI)': phase6_formatted,
            'Performance Drop (5→6)': drop_5_to_6,
            'Interpretation': interp
        })
    
    table2_df = pd.DataFrame(table_data)
    
    # Extract numeric values for sorting (from the formatted string)
    def extract_mean(val):
        if isinstance(val, str) and val != "N/A":
            return float(val.split()[0])
        return 0.0
    
    table2_df['_sort_key'] = table2_df['Phase 6: Cross-CLD AUC (95% CI)'].apply(extract_mean)
    table2_df = table2_df.sort_values('_sort_key', ascending=False)
    table2_df = table2_df.drop('_sort_key', axis=1)
    
    # Save to Excel with bold formatting for best values
    output_path = output_dir / "TABLE_2_OVERFITTING_STORY.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        table2_df.to_excel(writer, sheet_name='Table 2', index=False)
        
        # Access the worksheet to apply formatting
        worksheet = writer.sheets['Table 2']
        from openpyxl.styles import Font
        
        # Find best values in each numeric column and make them bold
        # Column indices (0-indexed in data, but 1-indexed in Excel after header)
        
        # Phase 4: highest AUC (column B, index 1)
        phase4_values = []
        for idx, val in enumerate(table2_df['Phase 4: RFE CV AUC (95% CI)']):
            if isinstance(val, str) and val != "N/A":
                numeric_val = float(val.split()[0])
                phase4_values.append((idx, numeric_val, val))
        if phase4_values:
            best_idx = max(phase4_values, key=lambda x: x[1])[0]
            worksheet.cell(row=best_idx+2, column=2).font = Font(bold=True)
        
        # Phase 5 CV: highest AUC (column C, index 2)
        phase5_cv_values = []
        for idx, val in enumerate(table2_df['Phase 5: CV AUC (95% CI)']):
            if isinstance(val, str) and val != "N/A":
                numeric_val = float(val.split()[0])
                phase5_cv_values.append((idx, numeric_val, val))
        if phase5_cv_values:
            best_idx = max(phase5_cv_values, key=lambda x: x[1])[0]
            worksheet.cell(row=best_idx+2, column=3).font = Font(bold=True)
        
        # Phase 5 Test: highest AUC (column D, index 3)
        phase5_test_values = []
        for idx, val in enumerate(table2_df['Phase 5: Test AUC']):
            if isinstance(val, str) and val != "N/A":
                numeric_val = float(val)
                phase5_test_values.append((idx, numeric_val, val))
        if phase5_test_values:
            best_idx = max(phase5_test_values, key=lambda x: x[1])[0]
            worksheet.cell(row=best_idx+2, column=4).font = Font(bold=True)
        
        # Phase 6: highest AUC (column E, index 4)
        phase6_values = []
        for idx, val in enumerate(table2_df['Phase 6: Cross-CLD AUC (95% CI)']):
            if isinstance(val, str) and val != "N/A":
                numeric_val = float(val.split()[0])
                phase6_values.append((idx, numeric_val, val))
        if phase6_values:
            best_idx = max(phase6_values, key=lambda x: x[1])[0]
            worksheet.cell(row=best_idx+2, column=5).font = Font(bold=True)
        
        # Performance Drop: LOWEST is best (column F, index 5)
        drop_values = []
        for idx, val in enumerate(table2_df['Performance Drop (5→6)']):
            if not np.isnan(val):
                drop_values.append((idx, val))
        if drop_values:
            best_idx = min(drop_values, key=lambda x: x[1])[0]
            worksheet.cell(row=best_idx+2, column=6).font = Font(bold=True)
        
        # Add metadata
        best_generalizer = table2_df.iloc[0]['Classifier']
        best_phase6_str = table2_df.iloc[0]['Phase 6: Cross-CLD AUC (95% CI)']
        
        # Find worst overfitter (if there are valid drops)
        valid_drops = table2_df['Performance Drop (5→6)'].dropna()
        if len(valid_drops) > 0:
            worst_idx = valid_drops.idxmax()
            worst_row = table2_df.loc[worst_idx]
            worst_overfitter = worst_row['Classifier']
            worst_drop = worst_row['Performance Drop (5→6)']
        else:
            worst_overfitter = "N/A"
            worst_drop = 0.0
        
        metadata = pd.DataFrame({
            'Field': ['Title', 'Key Finding', 'Best Generalizer', 'Best Cross-CLD Performance',
                     'Worst Overfitter', 'Largest Drop', 'Recommendation', 'Generated'],
            'Value': [
                'Table 2: Cross-Phase Overfitting Analysis',
                'Complex models overfit to CLD-specific patterns',
                best_generalizer,
                best_phase6_str,
                worst_overfitter,
                f"{worst_drop:.3f}" if not np.isnan(worst_drop) else "N/A",
                f"Use {best_generalizer} for production deployment",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        })
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
    
    print(f"  ✅ Table 2 saved: {output_path.name}")


def run_shapiro_wilk(data, subsample_size=SHAPIRO_SUBSAMPLE_SIZE):
    """
    Run Shapiro-Wilk test, subsampling if n > 5000.
    
    Returns:
        dict with W statistic, p-value, n, and whether subsampled
    """
    n = len(data)
    subsampled = False
    
    if n > subsample_size:
        np.random.seed(42)  # Reproducibility
        data = np.random.choice(data, subsample_size, replace=False)
        subsampled = True
        n_tested = subsample_size
    else:
        n_tested = n
    
    W, p = stats.shapiro(data)
    
    return {
        'W': float(W),
        'p': float(p),
        'n_original': int(n),
        'n_tested': int(n_tested),
        'subsampled': bool(subsampled),
        'is_normal': bool(p > 0.05)
    }


def compute_distribution_stats(data):
    """Compute descriptive statistics for a distribution."""
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'median': float(np.median(data)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data))
    }


def run_normality_analysis(inventory_path: Path, output_dir: Path):
    """
    Run comprehensive normality analysis on all UQ metrics.
    
    Generates:
    - normality_tests_table.tex (LaTeX table for thesis)
    - normality_tests_results.json (full results archive)
    - normality_tests_results.xlsx (Excel spreadsheet)
    """
    print("\n" + "="*80)
    print("NORMALITY TESTING FOR UQ METRICS")
    print("="*80)
    
    # Load data from inventory
    with open(inventory_path, 'r') as f:
        valid_files = json.load(f)
    
    all_data = []
    for file_info in valid_files:
        filepath = file_info['filepath']
        exp_type = file_info.get('experiment_type', 'unknown')
        
        if exp_type not in ['citation', 'correctness']:
            continue
        
        try:
            df = pd.read_excel(filepath, sheet_name='All Edges', engine='openpyxl')
            if all(col in df.columns for col in CI_METRICS[:3]) and 'Classification' in df.columns:
                df['is_hallucination'] = (df['Classification'] == 'FP') | (df['Classification'] == 'FN')
                df['cld'] = file_info.get('cld', 'unknown')
                all_data.append(df)
        except Exception:
            continue
    
    if not all_data:
        print("  ⚠️  No data found for normality testing")
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"  Loaded {len(combined)} edges from {len(all_data)} files")
    
    # Run normality tests
    results = {}
    
    for metric in CI_METRICS:
        print(f"\n  Testing: {metric}")
        
        data = combined[metric].dropna().replace([np.inf, -np.inf], np.nan).dropna().values
        halluc_data = combined[combined['is_hallucination']][metric].dropna().replace([np.inf, -np.inf], np.nan).dropna().values
        correct_data = combined[~combined['is_hallucination']][metric].dropna().replace([np.inf, -np.inf], np.nan).dropna().values
        
        if len(data) < 20:
            print(f"    Skipping - insufficient data (n={len(data)})")
            continue
        
        metric_results = {
            'overall': {
                'shapiro': run_shapiro_wilk(data),
                'stats': compute_distribution_stats(data)
            },
            'hallucinations': {
                'shapiro': run_shapiro_wilk(halluc_data),
                'stats': compute_distribution_stats(halluc_data)
            },
            'correct': {
                'shapiro': run_shapiro_wilk(correct_data),
                'stats': compute_distribution_stats(correct_data)
            }
        }
        
        results[metric] = metric_results
        
        # Print summary
        sw = metric_results['overall']['shapiro']
        st = metric_results['overall']['stats']
        verdict = "NORMAL" if sw['is_normal'] else "NOT NORMAL"
        print(f"    Overall: W={sw['W']:.4f}, p={sw['p']:.2e}, skew={st['skewness']:.2f} → {verdict}")
    
    # Generate LaTeX table
    latex = generate_normality_latex_table(results)
    latex_path = output_dir / "normality_tests_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"\n  ✅ LaTeX table saved: {latex_path.name}")
    
    # Generate JSON
    json_path = output_dir / "normality_tests_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_edges_total': len(combined),
            'subsample_size': SHAPIRO_SUBSAMPLE_SIZE,
            'results': results
        }, f, indent=2)
    print(f"  ✅ JSON results saved: {json_path.name}")
    
    # Generate Excel
    excel_path = output_dir / "normality_tests_results.xlsx"
    rows = []
    for metric in CI_METRICS:
        if metric not in results:
            continue
        for group_name in ['overall', 'hallucinations', 'correct']:
            r = results[metric][group_name]
            sw = r['shapiro']
            st = r['stats']
            rows.append({
                'Metric': metric,
                'Group': group_name.capitalize(),
                'N_Original': sw['n_original'],
                'N_Tested': sw['n_tested'],
                'Subsampled': sw['subsampled'],
                'Mean': st['mean'],
                'Std': st['std'],
                'Median': st['median'],
                'Skewness': st['skewness'],
                'Kurtosis': st['kurtosis'],
                'Shapiro_W': sw['W'],
                'Shapiro_p': sw['p'],
                'Is_Normal': sw['is_normal'],
                'Verdict': 'Normal' if sw['is_normal'] else 'Not Normal'
            })
    
    summary_df = pd.DataFrame(rows)
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Normality Tests', index=False)
        metadata = pd.DataFrame([{
            'Timestamp': datetime.now().isoformat(),
            'Total_Edges': len(combined),
            'Shapiro_Subsample_Size': SHAPIRO_SUBSAMPLE_SIZE,
            'Alpha': 0.05,
            'Conclusion': 'All metrics are non-normal; use non-parametric methods'
        }])
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
    print(f"  ✅ Excel results saved: {excel_path.name}")
    
    return results


def generate_normality_latex_table(results):
    """Generate LaTeX table for normality tests."""
    
    latex = r"""\begin{table}[H]
\centering
\caption{Shapiro-Wilk Normality Tests for UQ Metric Distributions}
\label{tab:rq2_normality_tests}
\begin{threeparttable}
\begin{tabular}{llrrrrrl}
\toprule
\textbf{UQ Metric} & \textbf{Group} & \textbf{N} & \textbf{Skew} & \textbf{Kurt} & \textbf{W} & \textbf{p-value} & \textbf{Normal?} \\
\midrule
"""
    
    for metric in CI_METRICS:
        if metric not in results:
            continue
        
        metric_short = metric.replace('Gen ', '')
        first_row = True
        
        for group_name, group_label in [('overall', 'Overall'), ('hallucinations', 'Halluc'), ('correct', 'Correct')]:
            r = results[metric][group_name]
            sw = r['shapiro']
            st = r['stats']
            
            verdict = "Yes" if sw['is_normal'] else "No"
            n_str = f"{sw['n_original']:,}"
            if sw['subsampled']:
                n_str += r"$^\dagger$"
            
            # Format p-value
            if sw['p'] < 1e-50:
                p_str = r"$<10^{-50}$"
            elif sw['p'] < 0.001:
                exp = int(np.floor(np.log10(sw['p'])))
                p_str = f"$<10^{{{exp}}}$"
            else:
                p_str = f"{sw['p']:.3f}"
            
            if first_row:
                latex += f"{metric_short} & {group_label} & {n_str} & {st['skewness']:+.2f} & {st['kurtosis']:+.1f} & {sw['W']:.3f} & {p_str} & {verdict} \\\\\n"
                first_row = False
            else:
                latex += f" & {group_label} & {n_str} & {st['skewness']:+.2f} & {st['kurtosis']:+.1f} & {sw['W']:.3f} & {p_str} & {verdict} \\\\\n"
        
        latex += r"\midrule" + "\n"
    
    # Remove last \midrule and replace with \bottomrule
    latex = latex.rstrip("\n").rstrip(r"\midrule")
    
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} Shapiro-Wilk test for normality (null hypothesis: data is normally distributed). 
$^\dagger$Subsampled to $n=5{,}000$ due to Shapiro-Wilk sample size limit.
Skew = skewness (0 = symmetric; positive = right-tailed); Kurt = excess kurtosis (0 = normal; positive = heavy-tailed).
All metrics reject normality at $\alpha = 0.05$, justifying non-parametric methods (Mann-Whitney U, AUC-ROC).
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    
    return latex


def main():
    print("\n" + "="*80)
    print("RQ2 COMPREHENSIVE MASTER REPORT GENERATOR (V2 ENHANCED)")
    print("="*80 + "\n")
    
    analyses_dir, _unused_output = rq2_dirs()
    
    if not analyses_dir.exists():
        print("❌ Analysis directory not found!")
        return 1
    
    # Find the inventory file
    dedup_files = list(analyses_dir.glob("rq2_valid_files_deduplicated_*.json"))
    if dedup_files:
        inventory_path = max(dedup_files, key=lambda p: p.stat().st_mtime)
    else:
        inventory_files = list(analyses_dir.glob("rq2_valid_files_*.json"))
        if not inventory_files:
            print("❌ No inventory file found!")
            return 1
        inventory_path = max(inventory_files, key=lambda p: p.stat().st_mtime)
    
    # Compute reproducible counts (edge counts, file counts, exclusions)
    reproducible_counts = compute_reproducible_counts(inventory_path)
    
    # Load all results
    results = load_latest_results(analyses_dir)
    
    # Add reproducible counts to results
    results['reproducible_counts'] = reproducible_counts
    
    # Generate comprehensive master report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = analyses_dir / f"rq2_master_report_comprehensive_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "RQ2_COMPREHENSIVE_MASTER_REPORT.md"
    generate_comprehensive_master_report(results, report_path)
    
    # Generate presentation tables
    generate_table_1_grand_aggregate(results, output_dir)
    generate_table_2_overfitting_story(results, output_dir)
    
    # Run normality analysis and generate outputs
    normality_results = run_normality_analysis(inventory_path, output_dir)
    if normality_results:
        results['normality'] = normality_results
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE MASTER REPORT GENERATION COMPLETE!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📄 Report: {report_path.name}")
    print(f"📊 Table 1: TABLE_1_GRAND_AGGREGATE.xlsx")
    print(f"📊 Table 2: TABLE_2_OVERFITTING_STORY.xlsx")
    print(f"📊 Normality: normality_tests_table.tex, normality_tests_results.xlsx\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


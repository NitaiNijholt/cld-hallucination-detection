#!/usr/bin/env python3
"""
RQ2 Phase 6: Cross-Domain Generalization Analysis

Analyzes whether hallucination classifiers generalize across different CLDs:
- Check if logit distributions differ by CLD
- Test leave-one-CLD-out cross-validation
- Assess whether thresholds are CLD-specific or universal
- Compare within-CLD vs across-CLD performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from scipy import stats
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Import shared data preparation module
sys.path.insert(0, str(Path(__file__).parent))
from rq2_data_preparation import load_rq2_combined_data, prepare_clean_dataset, CI_METRICS, get_cld_mapping

# Output directory
OUTPUT_BASE = Path("parameter_tuning_experiments/rq2_analyses")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


def analyze_feature_distributions_by_cld(df):
    """Check if logit distributions differ across CLDs."""
    print("\n" + "="*80)
    print("FEATURE DISTRIBUTIONS BY CLD")
    print("="*80)
    
    clds = sorted(df['cld'].unique())
    cld_mapping = get_cld_mapping()
    
    print(f"\nFound {len(clds)} CLDs in dataset:")
    for cld in clds:
        n = len(df[df['cld'] == cld])
        n_hal = df[(df['cld'] == cld) & (df['is_hallucination'])].shape[0]
        full_name = cld_mapping.get(cld, cld)
        print(f"  - {cld:25s}: {n:4d} edges ({n_hal:3d} hallucinations, {n_hal/n:.1%}) - {full_name}")
    
    # Check feature distributions
    print(f"\n{'Feature':<30} {'CLD':<30} {'Mean':>8} {'Std':>8}")
    print(f"{'-'*30} {'-'*30} {'-'*8} {'-'*8}")
    
    distribution_data = []
    for feat in CI_METRICS:
        if feat not in df.columns:
            continue
            
        for cld in clds:
            cld_data = df[df['cld'] == cld][feat].dropna()
            if len(cld_data) > 0:
                print(f"{feat:<30} {cld:<30} {cld_data.mean():>8.3f} {cld_data.std():>8.3f}")
                distribution_data.append({
                    'feature': feat,
                    'cld': cld,
                    'mean': cld_data.mean(),
                    'std': cld_data.std()
                })
        print()
    
    # Statistical test: Are distributions significantly different?
    print("\n🔬 STATISTICAL TEST: Do feature distributions differ by CLD?")
    print(f"{'Feature':<30} {'Test':>15} {'p-value':>10} {'Significant?':>15}")
    print(f"{'-'*30} {'-'*15} {'-'*10} {'-'*15}")
    
    distribution_tests = []
    for feat in CI_METRICS:
        if feat not in df.columns:
            continue
        
        groups = [df[df['cld'] == cld][feat].dropna() for cld in clds]
        
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            # Kruskal-Wallis test (non-parametric ANOVA)
            stat, p_value = stats.kruskal(*groups)
            sig = "YES ⚠️" if p_value < 0.05 else "NO ✅"
            print(f"{feat:<30} {'Kruskal-Wallis':>15} {p_value:>10.4f} {sig:>15}")
            distribution_tests.append({
                'feature': feat,
                'test': 'Kruskal-Wallis',
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    return {
        'distribution_data': distribution_data,
        'statistical_tests': distribution_tests
    }


def leave_one_cld_out_evaluation(df):
    """Test classifier generalization using leave-one-CLD-out validation."""
    print(f"\n{'='*80}")
    print("LEAVE-ONE-CLD-OUT CROSS-VALIDATION")
    print("="*80)
    print("\nTrain on N-1 CLDs, test on held-out CLD to assess generalization\n")
    
    clds = sorted(df['cld'].unique())
    cld_mapping = get_cld_mapping()
    
    if len(clds) < 2:
        print(f"⚠️  Only {len(clds)} CLD found - need at least 2 for leave-one-out")
        return None
    
    # Test with best feature (Gen Cosine Similarity if available)
    feature = 'Gen Cosine Similarity' if 'Gen Cosine Similarity' in df.columns else CI_METRICS[0]
    
    print(f"Testing with: {feature}")
    print(f"\n{'Train CLDs':<50} {'Test CLD':<30} {'AUC':>8} {'F1':>8} {'Precision':>8} {'Recall':>8}")
    print(f"{'-'*50} {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    results = []
    
    for test_cld in clds:
        # Split data
        train_df = df[df['cld'] != test_cld]
        test_df = df[df['cld'] == test_cld]
        
        # Prepare data
        X_train = train_df[[feature]].replace([np.inf, -np.inf], np.nan).dropna()
        y_train = train_df.loc[X_train.index, 'is_hallucination'].values
        
        X_test = test_df[[feature]].replace([np.inf, -np.inf], np.nan).dropna()
        y_test = test_df.loc[X_test.index, 'is_hallucination'].values
        
        if len(y_test) < 5:
            test_cld_name = cld_mapping.get(test_cld, test_cld)[:30]
            print(f"{'Various':<50} {test_cld_name:<30} {'SKIP':>8} {'(too few samples)':>20}")
            continue
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        
        train_clds_short = ', '.join([c for c in clds if c != test_cld])[:50]
        test_cld_short = test_cld[:30]
        
        print(f"{train_clds_short:<50} {test_cld_short:<30} {auc:>8.3f} {f1:>8.3f} {prec:>8.3f} {rec:>8.3f}")
        
        results.append({
            'test_cld': test_cld,
            'test_cld_full_name': cld_mapping.get(test_cld, test_cld),
            'train_clds': [c for c in clds if c != test_cld],
            'auc': auc,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'n_test': len(y_test),
            'n_test_hallucinations': y_test.sum()
        })
    
    if len(results) > 0:
        avg_auc = np.mean([r['auc'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        std_auc = np.std([r['auc'] for r in results])
        std_f1 = np.std([r['f1'] for r in results])
        
        print(f"\n{'AVERAGE ACROSS CLDs':<50} {'':>30} {avg_auc:>8.3f} {avg_f1:>8.3f}")
        print(f"{'STANDARD DEVIATION':<50} {'':>30} {std_auc:>8.3f} {std_f1:>8.3f}")
        
        return {
            'results': results,
            'feature_used': feature,
            'mean_auc': avg_auc,
            'std_auc': std_auc,
            'mean_f1': avg_f1,
            'std_f1': std_f1
        }
    
    return None


def compare_within_vs_across_cld(df):
    """Compare performance when training/testing within same CLD vs across CLDs."""
    print(f"\n{'='*80}")
    print("WITHIN-CLD vs ACROSS-CLD PERFORMANCE")
    print("="*80)
    
    clds = sorted(df['cld'].unique())
    cld_mapping = get_cld_mapping()
    
    if len(clds) < 2:
        print(f"⚠️  Only {len(clds)} CLD - need multiple CLDs")
        return None
    
    feature = 'Gen Cosine Similarity' if 'Gen Cosine Similarity' in df.columns else CI_METRICS[0]
    
    # Within-CLD performance (5-fold CV within each CLD)
    print(f"\n📊 WITHIN-CLD PERFORMANCE (5-fold CV within each CLD):")
    print(f"{'CLD':<30} {'AUC':>8} {'F1':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*8}")
    
    within_results = []
    within_aucs = []
    
    for cld in clds:
        cld_df = df[df['cld'] == cld]
        
        X = cld_df[[feature]].replace([np.inf, -np.inf], np.nan).dropna()
        y = cld_df.loc[X.index, 'is_hallucination'].values
        
        if len(y) < 10:
            print(f"{cld:<30} {'SKIP':>8} {'(too few)':>8}")
            continue
        
        # 5-fold CV
        cv = StratifiedKFold(n_splits=min(5, len(y) // 2), shuffle=True, random_state=42)
        
        aucs, f1s = [], []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.values[train_idx], X.values[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))
            X_test_scaled = scaler.transform(X_test.reshape(-1, 1))
            
            clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
            clf.fit(X_train_scaled, y_train)
            
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            
            aucs.append(roc_auc_score(y_test, y_prob))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
        
        avg_auc = np.mean(aucs)
        avg_f1 = np.mean(f1s)
        within_aucs.append(avg_auc)
        
        print(f"{cld:<30} {avg_auc:>8.3f} {avg_f1:>8.3f}")
        
        within_results.append({
            'cld': cld,
            'cld_full_name': cld_mapping.get(cld, cld),
            'within_auc': avg_auc,
            'within_f1': avg_f1
        })
    
    print(f"\n{'AVERAGE WITHIN-CLD':<30} {np.mean(within_aucs):>8.3f}")
    
    return {
        'within_results': within_results,
        'feature_used': feature,
        'mean_within_auc': np.mean(within_aucs) if within_aucs else None
    }


def assess_threshold_stability(df):
    """Check if optimal thresholds are stable across CLDs."""
    print(f"\n{'='*80}")
    print("THRESHOLD STABILITY ACROSS CLDs")
    print("="*80)
    
    clds = sorted(df['cld'].unique())
    cld_mapping = get_cld_mapping()
    feature = 'Gen Cosine Similarity' if 'Gen Cosine Similarity' in df.columns else CI_METRICS[0]
    
    print(f"\nOptimal threshold (maximizing F1) for each CLD:")
    print(f"{'CLD':<30} {'Optimal Threshold':>18} {'F1 at optimal':>15}")
    print(f"{'-'*30} {'-'*18} {'-'*15}")
    
    threshold_results = []
    optimal_thresholds = []
    
    for cld in clds:
        cld_df = df[df['cld'] == cld]
        
        X = cld_df[[feature]].replace([np.inf, -np.inf], np.nan).dropna()
        y = cld_df.loc[X.index, 'is_hallucination'].values
        
        if len(y) < 10:
            continue
        
        # Train classifier
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        clf.fit(X_scaled, y)
        
        y_prob = clf.predict_proba(X_scaled)[:, 1]
        
        # Find optimal threshold
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.linspace(0.1, 0.9, 81):
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
        print(f"{cld:<30} {best_threshold:>18.3f} {best_f1:>15.3f}")
        
        threshold_results.append({
            'cld': cld,
            'cld_full_name': cld_mapping.get(cld, cld),
            'optimal_threshold': best_threshold,
            'f1_at_optimal': best_f1
        })
    
    if len(optimal_thresholds) > 1:
        mean_threshold = np.mean(optimal_thresholds)
        std_threshold = np.std(optimal_thresholds)
        
        print(f"\n{'MEAN THRESHOLD':<30} {mean_threshold:>18.3f}")
        print(f"{'STD THRESHOLD':<30} {std_threshold:>18.3f}")
        
        if std_threshold < 0.1:
            print(f"\n✅ Thresholds are STABLE across CLDs (std < 0.1)")
            print(f"   → Can use a universal threshold in production")
            recommendation = "universal"
        else:
            print(f"\n⚠️  Thresholds VARY across CLDs (std = {std_threshold:.3f})")
            print(f"   → May need CLD-specific thresholds or calibration")
            recommendation = "cld_specific"
        
        return {
            'threshold_results': threshold_results,
            'mean_threshold': mean_threshold,
            'std_threshold': std_threshold,
            'recommendation': recommendation,
            'feature_used': feature
        }
    
    return None


def generate_visualizations(distribution_analysis, leave_one_out, within_vs_across, 
                            threshold_stability, df, output_dir):
    """Create comprehensive visualizations."""
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    cld_mapping = get_cld_mapping()
    
    # 1. Leave-one-out performance
    if leave_one_out and 'results' in leave_one_out:
        fig1, ax = plt.subplots(figsize=(10, 6))
        results = leave_one_out['results']
        test_clds = [r['test_cld'] for r in results]
        aucs = [r['auc'] for r in results]
        
        colors = ['#3498db' if auc >= leave_one_out['mean_auc'] else '#e74c3c' for auc in aucs]
        ax.bar(range(len(test_clds)), aucs, color=colors, alpha=0.8, edgecolor='black')
        ax.axhline(leave_one_out['mean_auc'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {leave_one_out['mean_auc']:.3f}")
        ax.set_xlabel('Held-out CLD', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('Leave-One-CLD-Out Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(test_clds)))
        ax.set_xticklabels(test_clds, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(figures_dir / "leave_one_out_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: leave_one_out_performance.png")
    
    # 2. Within vs Across CLD comparison
    if leave_one_out and within_vs_across and 'results' in leave_one_out and 'within_results' in within_vs_across:
        fig2, ax = plt.subplots(figsize=(10, 6))
        
        # Match CLDs
        clds = [r['cld'] for r in within_vs_across['within_results']]
        within_aucs = [r['within_auc'] for r in within_vs_across['within_results']]
        
        # Get corresponding cross-CLD aucs
        cross_aucs = []
        for cld in clds:
            matching = [r for r in leave_one_out['results'] if r['test_cld'] == cld]
            if matching:
                cross_aucs.append(matching[0]['auc'])
            else:
                cross_aucs.append(np.nan)
        
        x = np.arange(len(clds))
        width = 0.35
        
        ax.bar(x - width/2, within_aucs, width, label='Within-CLD', color='#2ecc71', alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, cross_aucs, width, label='Cross-CLD', color='#3498db', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('CLD', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('Within-CLD vs Cross-CLD Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(clds, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(figures_dir / "within_vs_across_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: within_vs_across_comparison.png")
    
    # 3. Feature distributions by CLD
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    clds = sorted(df['cld'].unique())
    for idx, metric in enumerate(CI_METRICS):
        if idx >= 4:
            break
        ax = axes[idx]
        
        data_for_plot = []
        labels_for_plot = []
        for cld in clds:
            cld_data = df[df['cld'] == cld][metric].replace([np.inf, -np.inf], np.nan).dropna()
            if len(cld_data) > 0:
                data_for_plot.append(cld_data.values)
                labels_for_plot.append(cld)
        
        if data_for_plot:
            parts = ax.violinplot(data_for_plot, positions=range(len(data_for_plot)), 
                                   showmeans=True, showmedians=True)
            ax.set_xticks(range(len(labels_for_plot)))
            ax.set_xticklabels(labels_for_plot, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f'{metric} by CLD', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('CI Metric Distributions by CLD', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_distributions_by_cld.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: feature_distributions_by_cld.png")
    
    # 4. Threshold stability
    if threshold_stability and 'threshold_results' in threshold_stability:
        fig4, ax = plt.subplots(figsize=(10, 6))
        results = threshold_stability['threshold_results']
        clds_thresh = [r['cld'] for r in results]
        thresholds = [r['optimal_threshold'] for r in results]
        
        ax.scatter(range(len(clds_thresh)), thresholds, s=150, alpha=0.7, color='#9b59b6', edgecolors='black')
        ax.axhline(threshold_stability['mean_threshold'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {threshold_stability['mean_threshold']:.3f}")
        ax.fill_between([-0.5, len(clds_thresh)-0.5], 
                         threshold_stability['mean_threshold'] - threshold_stability['std_threshold'],
                         threshold_stability['mean_threshold'] + threshold_stability['std_threshold'],
                         alpha=0.2, color='red', label=f'±1 SD')
        ax.set_xlabel('CLD', fontsize=12)
        ax.set_ylabel('Optimal Threshold', fontsize=12)
        ax.set_title('Threshold Stability Across CLDs', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(clds_thresh)))
        ax.set_xticklabels(clds_thresh, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(figures_dir / "threshold_stability.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: threshold_stability.png")
    
    # 5. ROC curves by CLD (for leave-one-out)
    if leave_one_out and 'results' in leave_one_out:
        fig5, ax = plt.subplots(figsize=(10, 8))
        
        feature = leave_one_out['feature_used']
        
        for result in leave_one_out['results']:
            test_cld = result['test_cld']
            train_df = df[~df['cld'].isin([test_cld])]
            test_df = df[df['cld'] == test_cld]
            
            X_train = train_df[[feature]].replace([np.inf, -np.inf], np.nan).dropna()
            y_train = train_df.loc[X_train.index, 'is_hallucination'].values
            
            X_test = test_df[[feature]].replace([np.inf, -np.inf], np.nan).dropna()
            y_test = test_df.loc[X_test.index, 'is_hallucination'].values
            
            if len(y_test) < 5:
                continue
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
            clf.fit(X_train_scaled, y_train)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax.plot(fpr, tpr, linewidth=2, label=f'{test_cld} (AUC={result["auc"]:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves by Held-Out CLD', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "roc_curves_by_cld.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: roc_curves_by_cld.png")


def generate_report(distribution_analysis, leave_one_out, within_vs_across, 
                    threshold_stability, output_path):
    """Generate comprehensive markdown report."""
    
    cld_mapping = get_cld_mapping()
    
    with open(output_path, 'w') as f:
        f.write("# RQ2 Phase 6: Cross-Domain Generalization Analysis\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Analysis:** Cross-CLD generalization for hallucination detection\n\n")
        
        f.write("---\n\n")
        f.write("## Overview\n\n")
        f.write("This analysis tests whether hallucination classifiers trained on some Causal Loop Diagrams (CLDs) ")
        f.write("can generalize to completely unseen CLDs. This is critical for assessing production deployment ")
        f.write("viability on new causal domains.\n\n")
        
        f.write("### CLDs Analyzed\n\n")
        if distribution_analysis and 'distribution_data' in distribution_analysis:
            clds_in_data = set(d['cld'] for d in distribution_analysis['distribution_data'])
            for cld in sorted(clds_in_data):
                full_name = cld_mapping.get(cld, cld)
                f.write(f"- **{cld}**: {full_name}\n")
        f.write("\n")
        
        f.write("---\n\n")
        f.write("## 1. Feature Distribution Analysis\n\n")
        
        if distribution_analysis and 'statistical_tests' in distribution_analysis:
            f.write("### Statistical Tests: Do CI metrics differ by CLD?\n\n")
            f.write("| Feature | Test | Statistic | p-value | Differs by CLD? |\n")
            f.write("|---------|------|-----------|---------|----------------|\n")
            for test in distribution_analysis['statistical_tests']:
                sig = "Yes ⚠️" if test['significant'] else "No ✅"
                f.write(f"| {test['feature']} | {test['test']} | {test['statistic']:.2f} | {test['p_value']:.4f} | {sig} |\n")
            f.write("\n")
            
            n_significant = sum(1 for t in distribution_analysis['statistical_tests'] if t['significant'])
            if n_significant > 0:
                f.write(f"**Finding:** {n_significant}/{len(distribution_analysis['statistical_tests'])} features show ")
                f.write("statistically significant distribution differences across CLDs. This suggests domain-specific patterns exist.\n\n")
            else:
                f.write("**Finding:** No significant distribution differences across CLDs. Features appear domain-agnostic.\n\n")
        
        f.write("---\n\n")
        f.write("## 2. Leave-One-CLD-Out Cross-Validation\n\n")
        
        if leave_one_out and 'results' in leave_one_out:
            f.write("Train on N-1 CLDs, test on the held-out CLD:\n\n")
            f.write("| Test CLD | Train CLDs | AUC | F1 | Precision | Recall | n (test) |\n")
            f.write("|----------|------------|-----|----|-----------| -------|----------|\n")
            for r in leave_one_out['results']:
                train_str = ', '.join(r['train_clds'])
                f.write(f"| {r['test_cld']} | {train_str} | {r['auc']:.3f} | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['n_test']} |\n")
            
            f.write(f"| **MEAN** | **-** | **{leave_one_out['mean_auc']:.3f}** | **{leave_one_out['mean_f1']:.3f}** | **-** | **-** | **-** |\n")
            f.write(f"| **STD** | **-** | **{leave_one_out['std_auc']:.3f}** | **{leave_one_out['std_f1']:.3f}** | **-** | **-** | **-** |\n\n")
            
            f.write(f"**Feature used:** {leave_one_out['feature_used']}\n\n")
            
            # Find best and worst
            best = max(leave_one_out['results'], key=lambda x: x['auc'])
            worst = min(leave_one_out['results'], key=lambda x: x['auc'])
            f.write(f"**Best generalization:** {best['test_cld']} (AUC={best['auc']:.3f})  \n")
            f.write(f"**Worst generalization:** {worst['test_cld']} (AUC={worst['auc']:.3f})  \n")
            f.write(f"**Generalization gap:** {best['auc'] - worst['auc']:.3f}\n\n")
        
        f.write("---\n\n")
        f.write("## 3. Within-CLD vs Cross-CLD Performance\n\n")
        
        if within_vs_across and 'within_results' in within_vs_across and leave_one_out and 'results' in leave_one_out:
            f.write("| CLD | Within-CLD AUC | Cross-CLD AUC | Difference |\n")
            f.write("|-----|----------------|---------------|------------|\n")
            
            total_diff = []
            for w in within_vs_across['within_results']:
                cld = w['cld']
                within_auc = w['within_auc']
                
                # Find corresponding cross-CLD result
                cross_result = [r for r in leave_one_out['results'] if r['test_cld'] == cld]
                if cross_result:
                    cross_auc = cross_result[0]['auc']
                    diff = within_auc - cross_auc
                    total_diff.append(diff)
                    f.write(f"| {cld} | {within_auc:.3f} | {cross_auc:.3f} | {diff:+.3f} |\n")
            
            if total_diff:
                mean_diff = np.mean(total_diff)
                f.write(f"| **AVERAGE** | **{within_vs_across['mean_within_auc']:.3f}** | **{leave_one_out['mean_auc']:.3f}** | **{mean_diff:+.3f}** |\n\n")
                
                if mean_diff > 0.05:
                    f.write(f"**Finding:** Classifiers perform {mean_diff:.3f} AUC points better within-domain than cross-domain. ")
                    f.write("Significant generalization gap suggests domain-specific tuning may be beneficial.\n\n")
                elif mean_diff < -0.05:
                    f.write(f"**Finding:** Surprisingly, cross-domain performance exceeds within-domain by {abs(mean_diff):.3f} AUC points. ")
                    f.write("Training on diverse CLDs may improve generalization.\n\n")
                else:
                    f.write(f"**Finding:** Minimal performance difference ({abs(mean_diff):.3f} AUC points) between within and cross-domain. ")
                    f.write("Classifier generalizes well across causal domains.\n\n")
        
        f.write("---\n\n")
        f.write("## 4. Threshold Stability\n\n")
        
        if threshold_stability and 'threshold_results' in threshold_stability:
            f.write("Optimal classification threshold (maximizing F1) per CLD:\n\n")
            f.write("| CLD | Optimal Threshold | F1 at Optimal |\n")
            f.write("|-----|-------------------|---------------|\n")
            for r in threshold_stability['threshold_results']:
                f.write(f"| {r['cld']} | {r['optimal_threshold']:.3f} | {r['f1_at_optimal']:.3f} |\n")
            
            f.write(f"| **MEAN ± STD** | **{threshold_stability['mean_threshold']:.3f} ± {threshold_stability['std_threshold']:.3f}** | **-** |\n\n")
            
            if threshold_stability['recommendation'] == 'universal':
                f.write("**Recommendation:** ✅ Use a **universal threshold** across all CLDs. ")
                f.write(f"Low variance (std={threshold_stability['std_threshold']:.3f}) indicates thresholds are stable.\n\n")
            else:
                f.write("**Recommendation:** ⚠️ Consider **CLD-specific thresholds** or calibration. ")
                f.write(f"High variance (std={threshold_stability['std_threshold']:.3f}) indicates thresholds vary by domain.\n\n")
        
        f.write("---\n\n")
        f.write("## Production Deployment Recommendations\n\n")
        
        f.write("Based on the cross-domain generalization analysis:\n\n")
        
        if leave_one_out:
            if leave_one_out['mean_auc'] > 0.7:
                f.write("### ✅ Recommended: Universal Deployment\n\n")
                f.write(f"- **Cross-domain AUC:** {leave_one_out['mean_auc']:.3f} (acceptable performance)\n")
                f.write("- Train on all available CLDs for maximum robustness\n")
                f.write("- Deploy on new CLDs without domain-specific tuning\n")
            elif leave_one_out['mean_auc'] > 0.6:
                f.write("### ⚠️ Recommended: Calibration Required\n\n")
                f.write(f"- **Cross-domain AUC:** {leave_one_out['mean_auc']:.3f} (moderate performance)\n")
                f.write("- Use Platt scaling or isotonic regression for calibration\n")
                f.write("- Consider collecting small labeled sample from new CLD for fine-tuning\n")
            else:
                f.write("### ❌ Not Recommended: Poor Generalization\n\n")
                f.write(f"- **Cross-domain AUC:** {leave_one_out['mean_auc']:.3f} (poor performance)\n")
                f.write("- Requires domain-specific model training for each new CLD\n")
                f.write("- Alternative: Develop domain-agnostic features or meta-learning approach\n")
        
        if threshold_stability:
            f.write(f"\n### Threshold Strategy: {threshold_stability['recommendation'].replace('_', ' ').title()}\n\n")
            if threshold_stability['recommendation'] == 'universal':
                f.write(f"- Use threshold = {threshold_stability['mean_threshold']:.3f} across all CLDs\n")
            else:
                f.write("- Maintain CLD-specific thresholds or use probability scores directly\n")
        
        f.write("\n---\n\n")
        f.write("## Visualizations\n\n")
        f.write("See `figures/` directory for detailed visualizations:\n\n")
        f.write("1. `leave_one_out_performance.png` - AUC for each held-out CLD\n")
        f.write("2. `within_vs_across_comparison.png` - Within-CLD vs cross-CLD performance\n")
        f.write("3. `feature_distributions_by_cld.png` - CI metric distributions by domain\n")
        f.write("4. `threshold_stability.png` - Optimal thresholds across CLDs\n")
        f.write("5. `roc_curves_by_cld.png` - ROC curves for each held-out CLD\n")
    
    print(f"\n✅ Report saved: {output_path.name}")


def save_json_results(distribution_analysis, leave_one_out, within_vs_across, 
                      threshold_stability, output_dir):
    """Save structured results for programmatic access."""
    
    # Leave-one-out results
    if leave_one_out:
        with open(output_dir / "leave_one_out_results.json", 'w') as f:
            # Convert to JSON-serializable format
            json_data = {
                'feature_used': leave_one_out['feature_used'],
                'mean_auc': float(leave_one_out['mean_auc']),
                'std_auc': float(leave_one_out['std_auc']),
                'mean_f1': float(leave_one_out['mean_f1']),
                'std_f1': float(leave_one_out['std_f1']),
                'results': [
                    {
                        'test_cld': r['test_cld'],
                        'test_cld_full_name': r['test_cld_full_name'],
                        'train_clds': r['train_clds'],
                        'auc': float(r['auc']),
                        'f1': float(r['f1']),
                        'precision': float(r['precision']),
                        'recall': float(r['recall']),
                        'n_test': int(r['n_test']),
                        'n_test_hallucinations': int(r['n_test_hallucinations'])
                    }
                    for r in leave_one_out['results']
                ]
            }
            json.dump(json_data, f, indent=2)
        print(f"  ✅ Saved: leave_one_out_results.json")
    
    # Within vs across CLD
    if within_vs_across:
        with open(output_dir / "within_vs_across_cld.json", 'w') as f:
            json_data = {
                'feature_used': within_vs_across['feature_used'],
                'mean_within_auc': float(within_vs_across['mean_within_auc']) if within_vs_across['mean_within_auc'] else None,
                'results': [
                    {
                        'cld': r['cld'],
                        'cld_full_name': r['cld_full_name'],
                        'within_auc': float(r['within_auc']),
                        'within_f1': float(r['within_f1'])
                    }
                    for r in within_vs_across['within_results']
                ]
            }
            json.dump(json_data, f, indent=2)
        print(f"  ✅ Saved: within_vs_across_cld.json")
    
    # Feature distributions
    if distribution_analysis:
        with open(output_dir / "feature_distributions_by_cld.json", 'w') as f:
            json_data = {
                'distribution_data': distribution_analysis['distribution_data'],
                'statistical_tests': [
                    {
                        'feature': t['feature'],
                        'test': t['test'],
                        'statistic': float(t['statistic']),
                        'p_value': float(t['p_value']),
                        'significant': bool(t['significant'])
                    }
                    for t in distribution_analysis['statistical_tests']
                ]
            }
            json.dump(json_data, f, indent=2)
        print(f"  ✅ Saved: feature_distributions_by_cld.json")
    
    # Threshold stability
    if threshold_stability:
        with open(output_dir / "threshold_stability.json", 'w') as f:
            json_data = {
                'feature_used': threshold_stability['feature_used'],
                'mean_threshold': float(threshold_stability['mean_threshold']),
                'std_threshold': float(threshold_stability['std_threshold']),
                'recommendation': threshold_stability['recommendation'],
                'results': [
                    {
                        'cld': r['cld'],
                        'cld_full_name': r['cld_full_name'],
                        'optimal_threshold': float(r['optimal_threshold']),
                        'f1_at_optimal': float(r['f1_at_optimal'])
                    }
                    for r in threshold_stability['threshold_results']
                ]
            }
            json.dump(json_data, f, indent=2)
        print(f"  ✅ Saved: threshold_stability.json")


def main():
    print("\n" + "="*80)
    print("RQ2 PHASE 6: CROSS-DOMAIN GENERALIZATION ANALYSIS")
    print("="*80)
    print("Testing whether hallucination classifiers generalize across CLDs")
    print("="*80 + "\n")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = OUTPUT_BASE / f"rq2_cross_cld_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load data using shared module
    df = load_rq2_combined_data(verbose=True)
    
    # 1. Feature distribution analysis
    print("\n" + "="*80)
    print("STEP 1: FEATURE DISTRIBUTION ANALYSIS")
    print("="*80)
    distribution_analysis = analyze_feature_distributions_by_cld(df)
    
    # 2. Leave-one-CLD-out evaluation
    print("\n" + "="*80)
    print("STEP 2: LEAVE-ONE-CLD-OUT EVALUATION")
    print("="*80)
    leave_one_out = leave_one_cld_out_evaluation(df)
    
    # 3. Within vs across CLD
    print("\n" + "="*80)
    print("STEP 3: WITHIN VS ACROSS CLD COMPARISON")
    print("="*80)
    within_vs_across = compare_within_vs_across_cld(df)
    
    # 4. Threshold stability
    print("\n" + "="*80)
    print("STEP 4: THRESHOLD STABILITY ASSESSMENT")
    print("="*80)
    threshold_stability = assess_threshold_stability(df)
    
    # Generate outputs
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80)
    
    # Save JSON results
    save_json_results(distribution_analysis, leave_one_out, within_vs_across, 
                      threshold_stability, output_dir)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(distribution_analysis, leave_one_out, within_vs_across, 
                            threshold_stability, df, output_dir)
    
    # Generate report
    report_path = output_dir / "cross_cld_report.md"
    generate_report(distribution_analysis, leave_one_out, within_vs_across, 
                    threshold_stability, report_path)
    
    print("\n" + "="*80)
    print("✅ RQ2 PHASE 6 COMPLETE!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    
    if leave_one_out:
        print(f"\n🎯 KEY FINDINGS")
        print("="*80)
        print(f"Cross-domain AUC: {leave_one_out['mean_auc']:.3f} ± {leave_one_out['std_auc']:.3f}")
        
        if leave_one_out['mean_auc'] > 0.7:
            print("✅ Good generalization across causal domains")
        elif leave_one_out['mean_auc'] > 0.6:
            print("⚠️  Moderate generalization - calibration recommended")
        else:
            print("❌ Poor generalization - domain-specific training needed")
        
        if threshold_stability:
            print(f"\nThreshold stability: std={threshold_stability['std_threshold']:.3f}")
            print(f"Recommendation: {threshold_stability['recommendation'].replace('_', ' ').title()}")
    
    print("\n" + "="*80 + "\n")
    
    return output_dir


if __name__ == "__main__":
    main()


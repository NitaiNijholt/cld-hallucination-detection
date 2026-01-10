#!/usr/bin/env python3
"""
RQ2 Phase 6: Cross-CLD Generalization with Best Classifier

Uses the winning classifier and features from Phase 5 to test
cross-domain generalization via leave-one-CLD-out validation.
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, average_precision_score
from scipy import stats
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Centralized path overrides (optional)
from rq2_paths import rq2_dirs

# Import shared data preparation module
sys.path.insert(0, str(Path(__file__).parent))
from rq2_data_preparation import load_rq2_combined_data, CI_METRICS

# Output directory
OUTPUT_BASE, _unused_output = rq2_dirs()


def get_classifier_by_name(clf_name):
    """Return classifier instance by name (matching Phase 5)."""
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    }
    return classifiers.get(clf_name)


def load_phase5_results():
    """Load best classifier and features from Phase 5."""
    phase5_dirs = list(OUTPUT_BASE.glob("rq2_phase5_ensemble_*"))
    if not phase5_dirs:
        print("⚠️  No Phase 5 results found. Using defaults.")
        return None
    
    latest_phase5 = max(phase5_dirs, key=lambda p: p.stat().st_mtime)
    results_file = latest_phase5 / "phase5_ensemble_results.json"
    
    if not results_file.exists():
        print("⚠️  Phase 5 results file not found. Using defaults.")
        return None
    
    with open(results_file, 'r') as f:
        phase5_results = json.load(f)
    
    print(f"✅ Loaded Phase 5 results from: {latest_phase5.name}")
    print(f"   Best classifier: {phase5_results['best_classifier']} (AUC={phase5_results['best_test_auc']:.3f})")
    print(f"   Features: {', '.join(phase5_results['features_used'])}\n")
    
    return phase5_results


def analyze_feature_distributions_by_cld(df, features, output_dir):
    """Test if CI metric distributions differ by CLD and by hallucination status."""
    print(f"\n{'='*80}")
    print("1. FEATURE DISTRIBUTION ANALYSIS BY CLD")
    print(f"{'='*80}\n")
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for feat in features:
        groups = [df[df['cld'] == cld][feat].dropna().values for cld in df['cld'].unique()]
        stat, p_value = stats.kruskal(*groups)
        
        results[feat] = {
            'statistic': float(stat),
            'p_value': float(p_value),
            'differs': p_value < 0.05
        }
        
        print(f"{feat:30s}: Kruskal-Wallis H={stat:.2f}, p={p_value:.4f} {'⚠️ Differs' if p_value < 0.05 else '✓ Similar'}")
    
    # Visualization 1: By CLD (overall)
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 4*n_features))
    if n_features == 1:
        axes = [axes]
    
    for ax, feat in zip(axes, features):
        for cld in df['cld'].unique():
            data = df[df['cld'] == cld][feat].dropna()
            # Use percentile-based bins to handle outliers
            if len(data) > 0:
                q99 = data.quantile(0.99)
                data_clipped = data[data <= q99]
                ax.hist(data_clipped, alpha=0.5, bins=30, label=f'{cld} (n={len(data)})', density=True)
        
        ax.set_xlabel(feat, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{feat} Distribution by CLD (99th percentile, p={results[feat]["p_value"]:.4f})', 
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "phase6_feature_distributions_by_cld.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: phase6_feature_distributions_by_cld.png")
    
    # Visualization 2: Correct vs Hallucinations (separate subplots for each feature)
    fig, axes = plt.subplots(n_features, 3, figsize=(18, 4*n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    clds = sorted(df['cld'].unique())
    
    for feat_idx, feat in enumerate(features):
        for cld_idx, cld in enumerate(clds):
            ax = axes[feat_idx, cld_idx]
            
            cld_df = df[df['cld'] == cld]
            correct_data = cld_df[cld_df['is_hallucination'] == False][feat].dropna()
            hallu_data = cld_df[cld_df['is_hallucination'] == True][feat].dropna()
            
            if len(correct_data) > 0 and len(hallu_data) > 0:
                # Clip to 99th percentile of combined data to handle outliers
                combined_data = pd.concat([correct_data, hallu_data])
                q99 = combined_data.quantile(0.99)
                
                correct_clipped = correct_data[correct_data <= q99]
                hallu_clipped = hallu_data[hallu_data <= q99]
                
                # Use same bins for both
                bins = np.histogram_bin_edges(combined_data[combined_data <= q99], bins=30)
                
                ax.hist(correct_clipped, bins=bins, alpha=0.6, label=f'Correct (n={len(correct_data)})', 
                       color='green', density=True, edgecolor='black', linewidth=0.5)
                ax.hist(hallu_clipped, bins=bins, alpha=0.6, label=f'Hallucination (n={len(hallu_data)})', 
                       color='red', density=True, edgecolor='black', linewidth=0.5)
                
                # Statistical test
                stat, p_val = stats.mannwhitneyu(correct_data, hallu_data, alternative='two-sided')
                
                ax.set_title(f'{cld}\n{feat}\np={p_val:.4f}', fontsize=10, fontweight='bold')
                ax.set_xlabel(feat, fontsize=9)
                ax.set_ylabel('Density', fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{cld} - {feat}', fontsize=10)
    
    plt.suptitle('Feature Distributions: Correct vs Hallucinations by CLD (99th percentile)', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(figures_dir / "phase6_correct_vs_hallucination_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: phase6_correct_vs_hallucination_distributions.png")
    
    return results


def leave_one_cld_out_evaluation_all_classifiers(df, features, output_dir):
    """Train on N-1 CLDs, test on held-out CLD for ALL classifiers."""
    print(f"\n{'='*80}")
    print("2. LEAVE-ONE-CLD-OUT CROSS-VALIDATION (ALL CLASSIFIERS)")
    print(f"{'='*80}\n")
    
    print(f"Features: {', '.join(features)}\n")
    
    clds = df['cld'].unique()
    all_results = {}
    
    # Test each classifier
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    }
    
    for clf_name, clf_template in classifiers.items():
        print(f"\n{'─'*80}")
        print(f"TESTING: {clf_name}")
        print(f"{'─'*80}")
        
        results = []
    
        for test_cld in clds:
            # Split data
            train_df = df[df['cld'] != test_cld]
            test_df = df[df['cld'] == test_cld]
            
            X_train = train_df[features].values
            y_train = train_df['is_hallucination'].values
            X_test = test_df[features].values
            y_test = test_df['is_hallucination'].values
            
            train_clds = train_df['cld'].unique()
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            clf = get_classifier_by_name(clf_name)  # Fresh instance
            clf.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            # Deployment-aligned thresholding (fixed policy): predict hallucination if p >= 0.5
            # NOTE: We make this explicit (instead of clf.predict) to guarantee consistency across estimators.
            threshold = 0.5
            y_pred = (y_prob >= threshold).astype(int)
            
            auc = roc_auc_score(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            print(f"  {test_cld:25s}: AUC={auc:.3f}, PR-AUC={ap:.3f}, F1@0.5={f1:.3f}")
            
            results.append({
                'test_cld': test_cld,
                'train_clds': list(train_clds),
                'auc': float(auc),
                'ap': float(ap),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'threshold': float(threshold),
                'threshold_rule': 'p(hallucination) >= 0.5',
                'n_test': int(len(test_df)),
                'y_prob': y_prob,
                'y_true': y_test
            })
        
        # Summary for this classifier
        mean_auc = np.mean([r['auc'] for r in results])
        std_auc = np.std([r['auc'] for r in results])
        mean_ap = np.mean([r['ap'] for r in results])
        std_ap = np.std([r['ap'] for r in results])
        
        print(f"\n  Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        print(f"  Mean PR-AUC: {mean_ap:.3f} ± {std_ap:.3f}")
        print(f"  Range: [{min(r['auc'] for r in results):.3f}, {max(r['auc'] for r in results):.3f}]")
        
        all_results[clf_name] = results
    
    # Save all results
    results_to_save = {
        clf_name: [{k: v for k, v in r.items() if k not in ['y_prob', 'y_true']} for r in results]
        for clf_name, results in all_results.items()
    }
    with open(output_dir / "phase6_leave_one_out_all_classifiers.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\n✅ Saved: phase6_leave_one_out_all_classifiers.json")
    
    # Print comparison table with all uncertainty metrics
    print(f"\n{'='*80}")
    print("CLASSIFIER COMPARISON: CROSS-DOMAIN GENERALIZATION")
    print(f"{'='*80}\n")
    
    # Compute and display all uncertainty metrics for each classifier
    summary_stats = {}
    for clf_name, results in all_results.items():
        aucs = [r['auc'] for r in results]
        n = len(aucs)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs, ddof=1)  # sample std (ddof=1)
        min_auc = np.min(aucs)
        max_auc = np.max(aucs)
        se = std_auc / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)  # t-critical for 95% CI
        ci_lower = mean_auc - t_crit * se
        ci_upper = mean_auc + t_crit * se
        best_cld = max(results, key=lambda x: x['auc'])['test_cld']
        worst_cld = min(results, key=lambda x: x['auc'])['test_cld']
        
        summary_stats[clf_name] = {
            'n': n,
            'mean': float(mean_auc),
            'std': float(std_auc),
            'min': float(min_auc),
            'max': float(max_auc),
            'se': float(se),
            't_crit': float(t_crit),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'best_cld': best_cld,
            'worst_cld': worst_cld,
            'per_cld': {r['test_cld']: float(r['auc']) for r in results}
        }
    
    # Print summary table
    print(f"{'Classifier':<25} {'Mean':<8} {'±1 SD':<16} {'95% CI (t-dist)':<20} {'[min, max]':<16}")
    print("─"*90)
    
    for clf_name, s in summary_stats.items():
        sd_str = f"{s['mean']:.3f} ± {s['std']:.3f}"
        ci_str = f"[{s['ci_lower']:.3f}, {s['ci_upper']:.3f}]"
        range_str = f"[{s['min']:.3f}, {s['max']:.3f}]"
        print(f"{clf_name:<25} {s['mean']:<8.3f} {sd_str:<16} {ci_str:<20} {range_str:<16}")
    
    # Explain the metrics
    n_obs = list(summary_stats.values())[0]['n']
    t_crit_val = list(summary_stats.values())[0]['t_crit']
    print(f"\n{'─'*90}")
    print(f"Note: n={n_obs} (leave-one-CLD-out), df={n_obs-1}")
    print(f"  - ±1 SD: Describes observed variability (no distributional assumptions)")
    print(f"  - 95% CI: Uses t-distribution (t-crit={t_crit_val:.3f} for df={n_obs-1})")
    print(f"  - [min, max]: Actual observed range")
    print(f"  ⚠️  For n=3, [min, max] or ±1 SD is recommended over 95% CI (assumptions unverifiable)")
    
    # Save summary stats to JSON
    with open(output_dir / "phase6_summary_stats.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"\n✅ Saved: phase6_summary_stats.json")
    
    return all_results


def compare_within_vs_across_cld(df, features, clf_name, output_dir):
    """Compare within-CLD vs cross-CLD performance."""
    print(f"\n{'='*80}")
    print("3. WITHIN-CLD VS CROSS-CLD PERFORMANCE")
    print(f"{'='*80}\n")
    
    clf_template = get_classifier_by_name(clf_name)
    if clf_template is None:
        print(f"❌ Classifier '{clf_name}' not found!")
        return {}
    
    clds = df['cld'].unique()
    results = []
    
    for cld in clds:
        print(f"\nCLD: {cld}")
        
        cld_df = df[df['cld'] == cld]
        X = cld_df[features].values
        y = cld_df['is_hallucination'].values
        
        # Within-CLD: 5-fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        within_aucs = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = get_classifier_by_name(clf_name)
            clf.fit(X_train_scaled, y_train)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            
            within_aucs.append(roc_auc_score(y_test, y_prob))
        
        within_auc = np.mean(within_aucs)
        
        # Cross-CLD: Train on other CLDs, test on this one
        train_df = df[df['cld'] != cld]
        test_df = df[df['cld'] == cld]
        
        X_train = train_df[features].values
        y_train = train_df['is_hallucination'].values
        X_test = test_df[features].values
        y_test = test_df['is_hallucination'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = get_classifier_by_name(clf_name)
        clf.fit(X_train_scaled, y_train)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        cross_auc = roc_auc_score(y_test, y_prob)
        
        diff = within_auc - cross_auc
        print(f"  Within-CLD AUC: {within_auc:.3f}")
        print(f"  Cross-CLD AUC:  {cross_auc:.3f}")
        print(f"  Difference:     {diff:+.3f}")
        
        results.append({
            'cld': cld,
            'within_cld_auc': float(within_auc),
            'cross_cld_auc': float(cross_auc),
            'difference': float(diff)
        })
    
    # Save
    with open(output_dir / "phase6_within_vs_across.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: phase6_within_vs_across.json")
    
    return results


def assess_threshold_stability(df, features, clf_name, output_dir):
    """Check if optimal thresholds vary by CLD."""
    print(f"\n{'='*80}")
    print("4. THRESHOLD STABILITY ANALYSIS")
    print(f"{'='*80}\n")
    
    clf_template = get_classifier_by_name(clf_name)
    if clf_template is None:
        print(f"❌ Classifier '{clf_name}' not found!")
        return {}
    
    clds = df['cld'].unique()
    thresholds = []
    
    for cld in clds:
        cld_df = df[df['cld'] == cld]
        X = cld_df[features].values
        y = cld_df['is_hallucination'].values
        
        # Train on this CLD
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = get_classifier_by_name(clf_name)
        clf.fit(X_scaled, y)
        y_prob = clf.predict_proba(X_scaled)[:, 1]
        
        # Find optimal threshold
        fpr, tpr, thresh = roc_curve(y, y_prob)
        f1_scores = []
        for t in thresh:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y, y_pred, zero_division=0)
            f1_scores.append(f1)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresh[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"{cld:30s}: Optimal threshold = {best_threshold:.3f} (F1={best_f1:.3f})")
        thresholds.append(best_threshold)
    
    mean_threshold = np.mean(thresholds)
    std_threshold = np.std(thresholds)
    
    print(f"\n{'='*60}")
    print(f"Mean threshold: {mean_threshold:.3f} ± {std_threshold:.3f}")
    if std_threshold > 0.1:
        print("⚠️ High variance - consider CLD-specific thresholds")
    else:
        print("✅ Low variance - universal threshold may work")
    
    results = {
        'mean_threshold': float(mean_threshold),
        'std_threshold': float(std_threshold),
        'thresholds_by_cld': {cld: float(thresh) for cld, thresh in zip(clds, thresholds)}
    }
    
    with open(output_dir / "phase6_threshold_stability.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: phase6_threshold_stability.json")
    
    return results


def generate_visualizations(leave_one_out_results, within_vs_across, features, clf_name, output_dir):
    """Create visualizations."""
    figures_dir = output_dir / "figures"
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 1. Leave-one-out performance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    clds = [r['test_cld'] for r in leave_one_out_results]
    aucs = [r['auc'] for r in leave_one_out_results]
    
    colors = ['#2ecc71' if auc == max(aucs) else '#e74c3c' if auc == min(aucs) else '#3498db' for auc in aucs]
    ax.bar(range(len(clds)), aucs, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(clds)))
    ax.set_xticklabels(clds, rotation=45, ha='right')
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'Leave-One-CLD-Out Performance\n{clf_name} with {", ".join(features)}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Random')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "phase6_leave_one_out_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: phase6_leave_one_out_performance.png")
    
    # 2. Within vs Across
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cldsva = [r['cld'] for r in within_vs_across]
    within_aucs = [r['within_cld_auc'] for r in within_vs_across]
    cross_aucs = [r['cross_cld_auc'] for r in within_vs_across]
    
    x = np.arange(len(cldsva))
    width = 0.35
    
    ax.bar(x - width/2, within_aucs, width, label='Within-CLD', color='#3498db', alpha=0.8, edgecolor='black')
    ax.bar(x + width/2, cross_aucs, width, label='Cross-CLD', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('CLD', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Within-CLD vs Cross-CLD Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cldsva, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "phase6_within_vs_across.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: phase6_within_vs_across.png")
    
    # 3. ROC curves by CLD
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for result in leave_one_out_results:
        fpr, tpr, _ = roc_curve(result['y_true'], result['y_prob'])
        ax.plot(fpr, tpr, linewidth=2, label=f"{result['test_cld']} (AUC={result['auc']:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves by CLD - Phase 6\n{clf_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "phase6_roc_curves_by_cld.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: phase6_roc_curves_by_cld.png")


def generate_report(phase5_results, feat_dist_results, leave_one_out_results, within_vs_across, 
                    threshold_stability, output_path):
    """Generate markdown report."""
    clf_name = phase5_results['best_classifier']
    features = phase5_results['features_used']
    
    with open(output_path, 'w') as f:
        f.write("# RQ2 Phase 6: Cross-Domain Generalization with Best Classifier\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Purpose:** Test generalization across CLDs using Phase 5 winner\n\n")
        
        f.write("---\n\n")
        f.write("## Setup\n\n")
        f.write(f"**Classifier:** {clf_name} (from Phase 5)\n")
        f.write(f"**Features:** {', '.join(features)} (from Phase 4 RFE)\n")
        f.write(f"**Phase 5 Performance:** AUC = {phase5_results['best_test_auc']:.3f}\n\n")
        
        f.write("---\n\n")
        f.write("## 1. Feature Distribution Analysis\n\n")
        f.write("| Feature | Kruskal-Wallis H | p-value | Differs by CLD? |\n")
        f.write("|---------|------------------|---------|----------------|\n")
        for feat, result in feat_dist_results.items():
            f.write(f"| {feat} | {result['statistic']:.2f} | {result['p_value']:.4f} | ")
            f.write(f"{'Yes ⚠️' if result['differs'] else 'No ✓'} |\n")
        
        n_differ = sum(1 for r in feat_dist_results.values() if r['differs'])
        f.write(f"\n**Finding:** {n_differ}/{len(feat_dist_results)} features show significant distribution differences across CLDs.\n\n")
        
        f.write("---\n\n")
        f.write("## 2. Leave-One-CLD-Out Performance\n\n")
        f.write("| Test CLD | Train CLDs | AUC | F1 | Precision | Recall | n (test) |\n")
        f.write("|----------|------------|-----|----|-----------| -------|----------|\n")
        
        for result in leave_one_out_results:
            train_str = ', '.join(result['train_clds'])
            f.write(f"| {result['test_cld']} | {train_str} | {result['auc']:.3f} | {result['f1']:.3f} | ")
            f.write(f"{result['precision']:.3f} | {result['recall']:.3f} | {result['n_test']} |\n")
        
        mean_auc = np.mean([r['auc'] for r in leave_one_out_results])
        std_auc = np.std([r['auc'] for r in leave_one_out_results])
        
        f.write(f"\n**Mean Cross-CLD AUC:** {mean_auc:.3f} ± {std_auc:.3f}\n\n")
        
        best_gen = max(leave_one_out_results, key=lambda x: x['auc'])
        worst_gen = min(leave_one_out_results, key=lambda x: x['auc'])
        
        f.write(f"**Best generalization:** {best_gen['test_cld']} (AUC={best_gen['auc']:.3f})  \n")
        f.write(f"**Worst generalization:** {worst_gen['test_cld']} (AUC={worst_gen['auc']:.3f})  \n")
        f.write(f"**Generalization gap:** {best_gen['auc'] - worst_gen['auc']:.3f}\n\n")
        
        f.write("---\n\n")
        f.write("## 3. Within-CLD vs Cross-CLD Comparison\n\n")
        f.write("| CLD | Within-CLD AUC | Cross-CLD AUC | Difference |\n")
        f.write("|-----|----------------|---------------|------------|\n")
        
        for result in within_vs_across:
            f.write(f"| {result['cld']} | {result['within_cld_auc']:.3f} | {result['cross_cld_auc']:.3f} | {result['difference']:+.3f} |\n")
        
        avg_diff = np.mean([r['difference'] for r in within_vs_across])
        f.write(f"\n**Average difference:** {avg_diff:+.3f}\n\n")
        
        if abs(avg_diff) < 0.05:
            f.write("✅ Minimal performance difference - good generalization\n\n")
        else:
            f.write("⚠️ Noticeable performance difference - domain-specific patterns exist\n\n")
        
        f.write("---\n\n")
        f.write("## 4. Threshold Stability\n\n")
        f.write("| CLD | Optimal Threshold | F1 at Optimal |\n")
        f.write("|-----|-------------------|---------------|\n")
        
        # We don't have F1 scores stored, but we can show thresholds
        for cld, thresh in threshold_stability['thresholds_by_cld'].items():
            f.write(f"| {cld} | {thresh:.3f} | - |\n")
        
        f.write(f"\n**Mean ± STD:** {threshold_stability['mean_threshold']:.3f} ± {threshold_stability['std_threshold']:.3f}\n\n")
        
        if threshold_stability['std_threshold'] > 0.1:
            f.write("**Recommendation:** ⚠️ Consider **CLD-specific thresholds**\n\n")
        else:
            f.write("**Recommendation:** ✅ **Universal threshold** may work\n\n")
        
        f.write("---\n\n")
        f.write("## Production Deployment Recommendations\n\n")
        
        if mean_auc >= 0.8:
            f.write("### ✅ Recommended: Direct Deployment\n\n")
            f.write(f"- Cross-domain AUC is strong ({mean_auc:.3f})\n")
        elif mean_auc >= 0.7:
            f.write("### ⚠️ Recommended: Calibration\n\n")
            f.write(f"- Cross-domain AUC is moderate ({mean_auc:.3f})\n")
            f.write("- Use Platt scaling or isotonic regression\n")
        else:
            f.write("### ⚠️ Recommended: Fine-tuning Required\n\n")
            f.write(f"- Cross-domain AUC is low ({mean_auc:.3f})\n")
            f.write("- Collect labeled samples from new CLDs\n")
        
        f.write("\n---\n\n")
        f.write("## Visualizations\n\n")
        f.write("See `figures/` directory for:\n")
        f.write("1. `phase6_leave_one_out_performance.png` - AUC for each held-out CLD\n")
        f.write("2. `phase6_within_vs_across.png` - Within-CLD vs cross-CLD performance\n")
        f.write("3. `phase6_feature_distributions_by_cld.png` - CI metric distributions\n")
        f.write("4. `phase6_roc_curves_by_cld.png` - ROC curves for each CLD\n")
    
    print(f"\n✅ Report saved: {output_path.name}")


def generate_combined_visualizations(all_clf_results, features, output_dir):
    """Create combined visualizations showing all classifiers."""
    figures_dir = output_dir / "figures"
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Get CLDs from first classifier's results
    clds = sorted(set([r['test_cld'] for r in list(all_clf_results.values())[0]]))
    
    # 1. Leave-one-out performance comparison (all classifiers)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(clds))
    width = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (clf_name, results) in enumerate(all_clf_results.items()):
        aucs = [r['auc'] for r in sorted(results, key=lambda x: x['test_cld'])]
        ax.bar(x + i*width, aucs, width, label=clf_name, color=colors[i], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Test CLD', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title(f'Leave-One-CLD-Out Performance: All Classifiers\nFeatures: {", ".join(features)}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(clds, rotation=45, ha='right')
    ax.set_ylim([0.4, 1.0])
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Random')
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "phase6_leave_one_out_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: phase6_leave_one_out_performance.png")
    
    # 2. ROC curves by CLD (all classifiers, subplots for each CLD)
    from sklearn.metrics import roc_curve
    
    n_clds = len(clds)
    fig, axes = plt.subplots(1, n_clds, figsize=(6*n_clds, 5))
    if n_clds == 1:
        axes = [axes]
    
    for cld_idx, cld in enumerate(clds):
        ax = axes[cld_idx]
        
        for clf_name, results in all_clf_results.items():
            # Find results for this CLD
            cld_result = next((r for r in results if r['test_cld'] == cld), None)
            if cld_result and 'y_true' in cld_result and 'y_prob' in cld_result:
                # Compute ROC curve from stored predictions
                fpr, tpr, _ = roc_curve(cld_result['y_true'], cld_result['y_prob'])
                ax.plot(fpr, tpr, 
                       label=f"{clf_name} (AUC={cld_result['auc']:.3f})", linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'Test CLD: {cld}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "phase6_roc_curves_by_cld.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: phase6_roc_curves_by_cld.png")
    
    # 3. Mean AUC comparison (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    clf_names = list(all_clf_results.keys())
    mean_aucs = [np.mean([r['auc'] for r in results]) for results in all_clf_results.values()]
    std_aucs = [np.std([r['auc'] for r in results]) for results in all_clf_results.values()]
    
    colors_bar = ['#2ecc71' if auc == max(mean_aucs) else '#e74c3c' if auc == min(mean_aucs) else '#3498db' 
                  for auc in mean_aucs]
    
    bars = ax.bar(range(len(clf_names)), mean_aucs, color=colors_bar, alpha=0.8, edgecolor='black')
    ax.errorbar(range(len(clf_names)), mean_aucs, yerr=std_aucs, fmt='none', 
                ecolor='black', capsize=5, capthick=2)
    
    ax.set_xticks(range(len(clf_names)))
    ax.set_xticklabels(clf_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Cross-Domain AUC', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Domain Generalization: Classifier Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0.5, 0.75])
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Random')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, auc, std) in enumerate(zip(bars, mean_aucs, std_aucs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{auc:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "phase6_within_vs_across.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: phase6_within_vs_across.png")
    
    print(f"\n✅ All visualizations saved to {figures_dir}/\n")


def main():
    print("\n" + "="*80)
    print("RQ2 PHASE 6: CROSS-CLD GENERALIZATION (ALL CLASSIFIERS)")
    print("="*80)
    print("Testing generalization across CLDs for all classifiers")
    print("="*80 + "\n")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = OUTPUT_BASE / f"rq2_phase6_cross_cld_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load Phase 5 results
    phase5_results = load_phase5_results()
    
    if not phase5_results:
        print("⚠️  No Phase 5 results found, using default features")
        features = CI_METRICS  # Use module-level CI_METRICS
    else:
        features = phase5_results['features_used']
    
    # Load data
    df = load_rq2_combined_data(verbose=True)
    
    # Ensure we have CLD column
    if 'cld' not in df.columns:
        print("❌ CLD column not found in data!")
        return 1
    
    print(f"\nCLDs in dataset: {', '.join(df['cld'].unique())}")
    
    # Clean data: remove rows with missing CI metrics
    print(f"\n{'='*80}")
    print("CLEANING DATA")
    print(f"{'='*80}")
    
    before = len(df)
    # For distribution analysis, we want ALL generator metrics available
    # For classifier analysis, we only need the RFE-selected features
    # Use the union of both sets for cleaning
    required_cols = list(set(features + CI_METRICS + ['is_hallucination', 'cld']))
    df_clean = df.dropna(subset=required_cols)
    after = len(df_clean)
    
    if before > after:
        print(f"  Removed {before - after} rows with missing values ({(before-after)/before*100:.1f}%)")
        print(f"  Remaining: {after} samples")
    else:
        print(f"  No missing values found. All {after} samples are clean.")
    
    print(f"\nCLDs in clean dataset:")
    for cld in sorted(df_clean['cld'].unique()):
        n = len(df_clean[df_clean['cld'] == cld])
        print(f"  {cld:25s}: {n:5d} edges")
    
    # Run analyses - TEST ALL CLASSIFIERS
    # Use ALL generator metrics for distribution analysis (not just RFE-selected features)
    feat_dist_results = analyze_feature_distributions_by_cld(df_clean, CI_METRICS, output_dir)
    all_clf_results = leave_one_cld_out_evaluation_all_classifiers(df_clean, features, output_dir)
    
    # Generate combined visualizations for all classifiers
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    generate_combined_visualizations(all_clf_results, features, output_dir)
    
    # Generate simple summary report
    report_path = output_dir / "phase6_cross_cld_report.md"
    with open(report_path, 'w') as f:
        f.write("# RQ2 Phase 6: Cross-Domain Generalization (All Classifiers)\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
        
        f.write("## Cross-CLD Generalization Results\n\n")
        f.write(f"**Features:** {', '.join(features)}\n\n")
        
        f.write("### Leave-One-CLD-Out Performance\n\n")
        f.write("| Classifier | Mean AUC | Std | Best CLD | Worst CLD | Generalization |\n")
        f.write("|------------|----------|-----|----------|-----------|----------------|\n")
        
        for clf_name, results in all_clf_results.items():
            mean_auc = np.mean([r['auc'] for r in results])
            std_auc = np.std([r['auc'] for r in results])
            best = max(results, key=lambda x: x['auc'])
            worst = min(results, key=lambda x: x['auc'])
            
            if mean_auc >= 0.7:
                gen = "Good ✅"
            elif mean_auc >= 0.6:
                gen = "Moderate ⚠️"
            else:
                gen = "Poor ❌"
            
            f.write(f"| {clf_name} | {mean_auc:.3f} | {std_auc:.3f} | {best['test_cld']} ({best['auc']:.3f}) | {worst['test_cld']} ({worst['auc']:.3f}) | {gen} |\n")
        
        f.write("\n### Key Findings\n\n")
        
        # Find best and worst generalizing classifiers
        clf_means = {clf: np.mean([r['auc'] for r in res]) for clf, res in all_clf_results.items()}
        best_clf = max(clf_means.items(), key=lambda x: x[1])
        worst_clf = min(clf_means.items(), key=lambda x: x[1])
        
        f.write(f"**Best Generalization:** {best_clf[0]} (AUC={best_clf[1]:.3f})\n")
        f.write(f"**Worst Generalization:** {worst_clf[0]} (AUC={worst_clf[1]:.3f})\n")
        f.write(f"**Generalization Gap:** {best_clf[1] - worst_clf[1]:.3f}\n\n")
        
        if phase5_results and 'best_classifier' in phase5_results:
            phase5_winner = phase5_results['best_classifier']
            phase5_auc = phase5_results['best_test_auc']
            phase6_auc = clf_means.get(phase5_winner, 0)
            
            f.write(f"### Phase 5 Winner Performance\n\n")
            f.write(f"**Classifier:** {phase5_winner}\n")
            f.write(f"**Phase 5 (in-distribution):** AUC = {phase5_auc:.3f}\n")
            f.write(f"**Phase 6 (cross-domain):** AUC = {phase6_auc:.3f}\n")
            f.write(f"**Performance Drop:** {phase5_auc - phase6_auc:.3f}\n\n")
            
            if phase5_auc - phase6_auc > 0.2:
                f.write("⚠️ **Large generalization gap indicates overfitting to CLD-specific patterns**\n\n")
            elif phase5_auc - phase6_auc > 0.1:
                f.write("⚠️ **Moderate generalization gap - domain-specific calibration may help**\n\n")
            else:
                f.write("✅ **Good generalization - model is robust across domains**\n\n")
    
    print(f"\n✅ Report saved: {report_path.name}")
    
    # Summary
    print("\n" + "="*80)
    print("✅ PHASE 6 COMPLETE!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n🎯 CROSS-DOMAIN GENERALIZATION COMPARISON")
    print("="*80)
    
    for clf_name, results in all_clf_results.items():
        mean_auc = np.mean([r['auc'] for r in results])
        print(f"{clf_name:<25}: AUC = {mean_auc:.3f}")
    
    print("\n" + "="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


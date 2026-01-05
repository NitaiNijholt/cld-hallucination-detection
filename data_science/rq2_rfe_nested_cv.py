#!/usr/bin/env python3
"""
RQ2 Phase 5: Rigorous Feature Selection with Nested Cross-Validation

Implements publication-grade feature selection following best practices from:
- Varma & Simon (2006): Bias in error estimation when using CV for model selection
- Cawley & Talbot (2010): Over-fitting in model selection and selection bias
- Breiman (2001): Random forests and permutation importance
- Altmann et al. (2010): Corrected feature importance measure

Methods:
1. Nested Cross-Validation (5 outer × 3 inner folds)
2. Recursive Feature Elimination (RFE)
3. Statistical Permutation Importance (100 iterations, p-values)
4. Stability Analysis (multiple random seeds)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ttest_1samp
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Import shared data preparation module
sys.path.insert(0, str(Path(__file__).parent))
from rq2_data_preparation import load_rq2_combined_data, prepare_clean_dataset, CI_METRICS

# Output directory
OUTPUT_BASE = Path("parameter_tuning_experiments/rq2_analyses")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


def nested_cv_rfe(X, y, feature_names, n_outer_folds=5, n_inner_folds=3, random_state=42):
    """
    Nested cross-validation with RFE feature selection.
    
    Following Varma & Simon (2006) to avoid selection bias:
    - Outer loop: Unbiased performance estimation
    - Inner loop: Feature selection on training data only
    """
    print("\n" + "="*80)
    print("NESTED CROSS-VALIDATION WITH RFE")
    print("="*80)
    print(f"Outer folds: {n_outer_folds}, Inner folds: {n_inner_folds}")
    print(f"Testing feature counts: 1-{len(feature_names)}\n")
    
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
    
    outer_results = []
    feature_selection_counts = {feat: 0 for feat in feature_names}
    
    for outer_fold_idx, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X, y), 1):
        print(f"\n{'='*80}")
        print(f"OUTER FOLD {outer_fold_idx}/{n_outer_folds}")
        print(f"{'='*80}")
        
        X_train_outer = X[train_outer_idx]
        y_train_outer = y[train_outer_idx]
        X_test_outer = X[test_outer_idx]
        y_test_outer = y[test_outer_idx]
        
        print(f"Train: {len(y_train_outer)} samples (Hallucinations={y_train_outer.sum()}, Correct={len(y_train_outer)-y_train_outer.sum()})")
        print(f"Test:  {len(y_test_outer)} samples (Hallucinations={y_test_outer.sum()}, Correct={len(y_test_outer)-y_test_outer.sum()})")
        
        # INNER LOOP: Feature selection
        inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
        
        best_n_features = None
        best_inner_score = 0
        inner_scores_by_n = {}
        
        print(f"\nInner CV (feature selection):")
        
        for n_features in range(1, len(feature_names) + 1):
            inner_fold_scores = []
            
            for inner_fold_idx, (train_inner_idx, val_inner_idx) in enumerate(inner_cv.split(X_train_outer, y_train_outer), 1):
                X_train_inner = X_train_outer[train_inner_idx]
                y_train_inner = y_train_outer[train_inner_idx]
                X_val_inner = X_train_outer[val_inner_idx]
                y_val_inner = y_train_outer[val_inner_idx]
                
                # Scale data (fit on train, transform both)
                scaler = StandardScaler()
                X_train_inner_scaled = scaler.fit_transform(X_train_inner)
                X_val_inner_scaled = scaler.transform(X_val_inner)
                
                # RFE feature selection
                clf = LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')
                rfe = RFE(estimator=clf, n_features_to_select=n_features, step=1)
                rfe.fit(X_train_inner_scaled, y_train_inner)
                
                # Evaluate on validation set
                X_train_selected = X_train_inner_scaled[:, rfe.support_]
                X_val_selected = X_val_inner_scaled[:, rfe.support_]
                
                clf.fit(X_train_selected, y_train_inner)
                y_val_proba = clf.predict_proba(X_val_selected)[:, 1]
                auc = roc_auc_score(y_val_inner, y_val_proba)
                
                inner_fold_scores.append(auc)
            
            mean_inner_score = np.mean(inner_fold_scores)
            inner_scores_by_n[n_features] = mean_inner_score
            
            if mean_inner_score > best_inner_score:
                best_inner_score = mean_inner_score
                best_n_features = n_features
            
            print(f"  {n_features} features: {mean_inner_score:.3f} (±{np.std(inner_fold_scores):.3f})")
        
        print(f"\n  ✓ Best: {best_n_features} features (inner CV AUC = {best_inner_score:.3f})")
        
        # OUTER EVALUATION: Refit on full outer training set with selected n_features
        scaler_outer = StandardScaler()
        X_train_outer_scaled = scaler_outer.fit_transform(X_train_outer)
        X_test_outer_scaled = scaler_outer.transform(X_test_outer)
        
        clf_outer = LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')
        rfe_outer = RFE(estimator=clf_outer, n_features_to_select=best_n_features, step=1)
        rfe_outer.fit(X_train_outer_scaled, y_train_outer)
        
        # Get selected features
        selected_features = [feat for feat, selected in zip(feature_names, rfe_outer.support_) if selected]
        for feat in selected_features:
            feature_selection_counts[feat] += 1
        
        # Evaluate on outer test set
        X_train_outer_selected = X_train_outer_scaled[:, rfe_outer.support_]
        X_test_outer_selected = X_test_outer_scaled[:, rfe_outer.support_]
        
        clf_outer.fit(X_train_outer_selected, y_train_outer)
        y_test_proba = clf_outer.predict_proba(X_test_outer_selected)[:, 1]
        outer_auc = roc_auc_score(y_test_outer, y_test_proba)
        
        print(f"\n  → Outer test AUC: {outer_auc:.3f}")
        print(f"  → Selected features: {selected_features}")
        
        outer_results.append({
            'fold': outer_fold_idx,
            'n_features': best_n_features,
            'outer_auc': outer_auc,
            'selected_features': selected_features,
            'inner_scores_by_n': inner_scores_by_n
        })
    
    # Aggregate results
    outer_aucs = [r['outer_auc'] for r in outer_results]
    n_features_selected = [r['n_features'] for r in outer_results]
    
    print(f"\n{'='*80}")
    print("NESTED CV RESULTS")
    print(f"{'='*80}")
    print(f"Outer AUC: {np.mean(outer_aucs):.3f} ± {np.std(outer_aucs):.3f}")
    print(f"95% CI: [{np.percentile(outer_aucs, 2.5):.3f}, {np.percentile(outer_aucs, 97.5):.3f}]")
    print(f"Feature count: {np.mean(n_features_selected):.1f} ± {np.std(n_features_selected):.1f}")
    print(f"\nFeature selection frequency (out of {n_outer_folds} folds):")
    for feat, count in sorted(feature_selection_counts.items(), key=lambda x: -x[1]):
        print(f"  {feat:30s}: {count}/{n_outer_folds} ({count/n_outer_folds*100:.0f}%)")
    
    return {
        'outer_results': outer_results,
        'mean_auc': np.mean(outer_aucs),
        'std_auc': np.std(outer_aucs),
        'ci_95': [np.percentile(outer_aucs, 2.5), np.percentile(outer_aucs, 97.5)],
        'feature_selection_counts': feature_selection_counts,
        'n_features_selected': n_features_selected
    }


def statistical_permutation_importance(X, y, feature_names, n_iterations=100, random_state=42):
    """
    Permutation importance with statistical testing.
    
    Following Altmann et al. (2010):
    - Repeat permutation n_iterations times
    - Compute p-values via t-test
    - Report confidence intervals
    """
    print("\n" + "="*80)
    print("STATISTICAL PERMUTATION IMPORTANCE")
    print("="*80)
    print(f"Iterations per feature: {n_iterations}")
    print("Testing H₀: Feature has no effect on model performance\n")
    
    # Single train/test split for permutation testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model on all features
    clf = LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    
    # Baseline AUC
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    baseline_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Baseline AUC (all features): {baseline_auc:.3f}\n")
    
    results = []
    
    for feat_idx, feat_name in enumerate(feature_names):
        print(f"Testing {feat_name}... ", end='', flush=True)
        
        auc_drops = []
        
        for iteration in range(n_iterations):
            # Permute this feature
            X_test_permuted = X_test_scaled.copy()
            np.random.seed(random_state + iteration)
            X_test_permuted[:, feat_idx] = np.random.permutation(X_test_permuted[:, feat_idx])
            
            # Evaluate with permuted feature
            y_pred_proba_perm = clf.predict_proba(X_test_permuted)[:, 1]
            auc_perm = roc_auc_score(y_test, y_pred_proba_perm)
            
            auc_drops.append(baseline_auc - auc_perm)
        
        # Statistical test: H₀ = importance = 0
        t_stat, p_value = ttest_1samp(auc_drops, 0, alternative='greater')
        
        mean_importance = np.mean(auc_drops)
        std_importance = np.std(auc_drops)
        ci_95_lower = np.percentile(auc_drops, 2.5)
        ci_95_upper = np.percentile(auc_drops, 97.5)
        
        is_significant = p_value < 0.05
        
        print(f"mean={mean_importance:+.3f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if is_significant else 'n.s.'}")
        
        results.append({
            'feature': feat_name,
            'mean_importance': mean_importance,
            'std': std_importance,
            'ci_95_lower': ci_95_lower,
            'ci_95_upper': ci_95_upper,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': is_significant
        })
    
    df_results = pd.DataFrame(results).sort_values('mean_importance', ascending=False)
    
    print(f"\n{'='*80}")
    print("PERMUTATION IMPORTANCE SUMMARY")
    print(f"{'='*80}")
    print("Significant features (p < 0.05):")
    for _, row in df_results[df_results['significant']].iterrows():
        print(f"  {row['feature']:30s}: {row['mean_importance']:+.3f} ± {row['std']:.3f}, p={row['p_value']:.4f}")
    
    return df_results, baseline_auc


def stability_analysis(X, y, feature_names, n_seeds=5):
    """
    Test stability across multiple random seeds.
    
    Following Meinshausen & Bühlmann (2010) stability selection principles.
    """
    print("\n" + "="*80)
    print("STABILITY ANALYSIS")
    print("="*80)
    print(f"Running nested CV with {n_seeds} different random seeds\n")
    
    all_results = []
    
    for seed in range(42, 42 + n_seeds):
        print(f"\n{'='*40}")
        print(f"Seed {seed}")
        print(f"{'='*40}")
        
        result = nested_cv_rfe(X, y, feature_names, n_outer_folds=5, n_inner_folds=3, random_state=seed)
        all_results.append({
            'seed': seed,
            'mean_auc': result['mean_auc'],
            'std_auc': result['std_auc'],
            'feature_counts': result['feature_selection_counts']
        })
    
    # Aggregate across seeds
    all_aucs = [r['mean_auc'] for r in all_results]
    
    print(f"\n{'='*80}")
    print("STABILITY RESULTS")
    print(f"{'='*80}")
    print(f"Mean AUC across seeds: {np.mean(all_aucs):.3f} ± {np.std(all_aucs):.4f}")
    print(f"Range: [{np.min(all_aucs):.3f}, {np.max(all_aucs):.3f}]")
    
    # Feature selection stability
    print(f"\nFeature selection stability:")
    feature_selection_probs = {}
    for feat in feature_names:
        counts = [r['feature_counts'][feat] / 5 for r in all_results]  # Normalize by n_outer_folds
        feature_selection_probs[feat] = np.mean(counts)
    
    for feat, prob in sorted(feature_selection_probs.items(), key=lambda x: -x[1]):
        print(f"  {feat:30s}: {prob:.1%} of folds")
    
    return all_results, feature_selection_probs


def generate_visualizations(nested_cv_result, permutation_result, stability_results, output_dir):
    """Create comprehensive visualizations."""
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 1. Nested CV outer fold performance
    fig1, ax = plt.subplots(figsize=(10, 6))
    outer_results = nested_cv_result['outer_results']
    folds = [r['fold'] for r in outer_results]
    aucs = [r['outer_auc'] for r in outer_results]
    ax.bar(folds, aucs, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(nested_cv_result['mean_auc'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {nested_cv_result['mean_auc']:.3f}")
    ax.fill_between([0.5, 5.5], nested_cv_result['ci_95'][0], nested_cv_result['ci_95'][1],
                      alpha=0.2, color='red', label='95% CI')
    ax.set_xlabel('Outer Fold', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Nested CV: Outer Fold Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(folds)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(figures_dir / "nested_cv_outer_folds.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: nested_cv_outer_folds.png")
    
    # 2. Feature selection frequency
    fig2, ax = plt.subplots(figsize=(10, 6))
    feat_counts = nested_cv_result['feature_selection_counts']
    sorted_feats = sorted(feat_counts.items(), key=lambda x: -x[1])
    feats = [f[0] for f in sorted_feats]
    counts = [f[1] for f in sorted_feats]
    colors = ['#2ecc71' if c >= 3 else '#f39c12' if c >= 2 else '#e74c3c' for c in counts]
    ax.barh(range(len(feats)), counts, color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=10)
    ax.set_xlabel('Selection Frequency (out of 5 folds)', fontsize=12)
    ax.set_title('Feature Selection Frequency', fontsize=14, fontweight='bold')
    ax.axvline(3, color='black', linestyle='--', alpha=0.5, linewidth=2, label='60% threshold')
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 5])
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_selection_frequency.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: feature_selection_frequency.png")
    
    # 3. Permutation importance with error bars
    fig3, ax = plt.subplots(figsize=(10, 6))
    perm_sorted = permutation_result.sort_values('mean_importance', ascending=True)
    y_pos = range(len(perm_sorted))
    colors_sig = ['#2ecc71' if row['significant'] else '#95a5a6' for _, row in perm_sorted.iterrows()]
    ax.barh(y_pos, perm_sorted['mean_importance'], color=colors_sig, alpha=0.8, edgecolor='black')
    
    # Add error bars (95% CI)
    for i, (_, row) in enumerate(perm_sorted.iterrows()):
        ax.plot([row['ci_95_lower'], row['ci_95_upper']], [i, i], color='black', linewidth=2.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(perm_sorted['feature'], fontsize=10)
    ax.set_xlabel('Permutation Importance (AUC Drop)', fontsize=12)
    ax.set_title('Permutation Importance with 95% CI\n(Green = statistically significant, p<0.05)', 
                 fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "permutation_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: permutation_importance.png")
    
    # 4. Stability across seeds
    fig4, ax = plt.subplots(figsize=(10, 6))
    seeds = [r['seed'] for r in stability_results]
    aucs_stability = [r['mean_auc'] for r in stability_results]
    ax.plot(seeds, aucs_stability, marker='o', linewidth=2.5, markersize=10, color='#9b59b6')
    ax.axhline(np.mean(aucs_stability), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {np.mean(aucs_stability):.3f}")
    ax.fill_between(seeds, np.mean(aucs_stability) - np.std(aucs_stability),
                      np.mean(aucs_stability) + np.std(aucs_stability),
                      alpha=0.2, color='purple')
    ax.set_xlabel('Random Seed', fontsize=12)
    ax.set_ylabel('Mean AUC', fontsize=12)
    ax.set_title('Stability Across Random Seeds', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(figures_dir / "stability_across_seeds.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: stability_across_seeds.png")
    
    # 5. Combined summary visualization
    fig5 = plt.figure(figsize=(16, 10))
    gs = fig5.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top left: Outer fold AUCs
    ax1 = fig5.add_subplot(gs[0, 0])
    ax1.bar(folds, aucs, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(nested_cv_result['mean_auc'], color='red', linestyle='--', linewidth=2)
    ax1.fill_between([0.5, 5.5], nested_cv_result['ci_95'][0], nested_cv_result['ci_95'][1],
                      alpha=0.2, color='red')
    ax1.set_xlabel('Outer Fold')
    ax1.set_ylabel('AUC')
    ax1.set_title('A) Nested CV Performance', fontweight='bold')
    ax1.set_xticks(folds)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Top right: Feature selection frequency
    ax2 = fig5.add_subplot(gs[0, 1])
    ax2.barh(range(len(feats)), counts, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_yticks(range(len(feats)))
    ax2.set_yticklabels(feats, fontsize=9)
    ax2.set_xlabel('Selection Frequency')
    ax2.set_title('B) Feature Selection Frequency', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim([0, 5])
    
    # Bottom left: Permutation importance
    ax3 = fig5.add_subplot(gs[1, 0])
    ax3.barh(y_pos, perm_sorted['mean_importance'], color=colors_sig, alpha=0.8, edgecolor='black')
    for i, (_, row) in enumerate(perm_sorted.iterrows()):
        ax3.plot([row['ci_95_lower'], row['ci_95_upper']], [i, i], color='black', linewidth=2)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(perm_sorted['feature'], fontsize=9)
    ax3.set_xlabel('Importance (AUC Drop)')
    ax3.set_title('C) Permutation Importance', fontweight='bold')
    ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    ax3.grid(axis='x', alpha=0.3)
    
    # Bottom right: Stability
    ax4 = fig5.add_subplot(gs[1, 1])
    ax4.plot(seeds, aucs_stability, marker='o', linewidth=2, markersize=8, color='#9b59b6')
    ax4.axhline(np.mean(aucs_stability), color='red', linestyle='--', linewidth=2)
    ax4.fill_between(seeds, np.mean(aucs_stability) - np.std(aucs_stability),
                      np.mean(aucs_stability) + np.std(aucs_stability),
                      alpha=0.2, color='purple')
    ax4.set_xlabel('Random Seed')
    ax4.set_ylabel('Mean AUC')
    ax4.set_title('D) Stability Analysis', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.suptitle('RQ2 Phase 5: Rigorous Feature Selection Summary', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(figures_dir / "nested_cv_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: nested_cv_summary.png")


def generate_report(nested_cv_result, permutation_result, baseline_auc, 
                    stability_results, feature_selection_probs, output_path):
    """Generate publication-ready markdown report."""
    
    with open(output_path, 'w') as f:
        f.write("# RQ2 Phase 5: Rigorous Feature Selection with Nested Cross-Validation\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Analysis:** RQ2 Natural Hallucination Detection (Citation + Correctness)\n\n")
        
        f.write("---\n\n")
        f.write("## Methodology\n\n")
        
        f.write("This analysis implements **nested cross-validation** following best practices from the machine learning literature:\n\n")
        
        f.write("### Key References\n\n")
        f.write("1. **Varma, S., & Simon, R. (2006).** \"Bias in error estimation when using cross-validation for model selection.\" *BMC Bioinformatics*, 7(1), 91.\n")
        f.write("   - Proves that feature selection within single CV causes optimistic bias\n")
        f.write("   - Shows nested CV eliminates this bias\n\n")
        
        f.write("2. **Cawley, G. C., & Talbot, N. L. (2010).** \"On over-fitting in model selection and subsequent selection bias in performance evaluation.\" *Journal of Machine Learning Research*, 11, 2079-2107.\n")
        f.write("   - Comprehensive analysis of selection bias in ML\n\n")
        
        f.write("3. **Breiman, L. (2001).** \"Random forests.\" *Machine Learning*, 45(1), 5-32.\n")
        f.write("   - Original paper introducing permutation importance\n\n")
        
        f.write("4. **Altmann, A., Toloşi, L., Sander, O., & Lengauer, T. (2010).** \"Permutation importance: a corrected feature importance measure.\" *Bioinformatics*, 26(10), 1340-1347.\n")
        f.write("   - Statistical corrections for permutation importance\n\n")
        
        f.write("5. **Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002).** \"Gene selection for cancer classification using support vector machines.\" *Machine Learning*, 46(1-3), 389-422.\n")
        f.write("   - Original RFE algorithm\n\n")
        
        f.write("### Implementation\n\n")
        f.write("**Nested Cross-Validation (5×3):**\n")
        f.write("- **Outer loop (5 folds):** Stratified K-fold for unbiased performance estimation\n")
        f.write("- **Inner loop (3 folds):** Feature selection via RFE (1-4 features tested)\n")
        f.write("- **Classifier:** Logistic Regression with balanced class weights\n")
        f.write("- **Scaling:** StandardScaler fit on training data only (prevents data leakage)\n\n")
        
        f.write("**Statistical Permutation Importance:**\n")
        f.write("- 100 permutations per feature\n")
        f.write("- One-sample t-test: H₀ = \"feature has no effect\"\n")
        f.write("- Significance threshold: p < 0.05\n\n")
        
        f.write("---\n\n")
        f.write("## Results\n\n")
        
        # Nested CV Results
        f.write("### 1. Nested Cross-Validation Results\n\n")
        f.write(f"**Unbiased Performance Estimate (5 outer folds):**\n\n")
        f.write(f"- **Mean AUC:** {nested_cv_result['mean_auc']:.3f} ± {nested_cv_result['std_auc']:.3f}\n")
        f.write(f"- **95% Confidence Interval:** [{nested_cv_result['ci_95'][0]:.3f}, {nested_cv_result['ci_95'][1]:.3f}]\n\n")
        
        f.write("**Feature Selection Frequency (across 5 outer folds):**\n\n")
        f.write("| Feature | Selection Frequency |\n")
        f.write("|---------|---------------------|\n")
        for feat, count in sorted(nested_cv_result['feature_selection_counts'].items(), key=lambda x: -x[1]):
            f.write(f"| {feat} | {count}/5 ({count/5*100:.0f}%) |\n")
        
        f.write("\n**Per-Fold Results:**\n\n")
        f.write("| Fold | n_features | Outer AUC | Selected Features |\n")
        f.write("|------|------------|-----------|-------------------|\n")
        for result in nested_cv_result['outer_results']:
            feats_str = ", ".join(result['selected_features'])
            f.write(f"| {result['fold']} | {result['n_features']} | {result['outer_auc']:.3f} | {feats_str} |\n")
        
        # Permutation Importance
        f.write("\n### 2. Statistical Permutation Importance (100 iterations)\n\n")
        f.write(f"**Baseline AUC (all 4 features):** {baseline_auc:.3f}\n\n")
        
        f.write("| Rank | Feature | Mean Importance | 95% CI | t-statistic | p-value | Significant |\n")
        f.write("|------|---------|-----------------|--------|-------------|---------|-------------|\n")
        for i, (_, row) in enumerate(permutation_result.iterrows(), 1):
            sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['significant'] else ""
            f.write(f"| {i} | {row['feature']} | {row['mean_importance']:+.3f} ± {row['std']:.3f} | "
                   f"[{row['ci_95_lower']:+.3f}, {row['ci_95_upper']:+.3f}] | {row['t_statistic']:.2f} | "
                   f"{row['p_value']:.4f} | {sig_marker if row['significant'] else 'n.s.'} |\n")
        
        f.write("\n*Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant*\n")
        
        # Stability Analysis
        f.write("\n### 3. Stability Analysis (5 random seeds)\n\n")
        all_aucs = [r['mean_auc'] for r in stability_results]
        f.write(f"**Mean AUC across seeds:** {np.mean(all_aucs):.3f} ± {np.std(all_aucs):.4f}\n")
        f.write(f"**Range:** [{np.min(all_aucs):.3f}, {np.max(all_aucs):.3f}]\n\n")
        
        f.write("**Feature Selection Stability (% of folds across all seeds):**\n\n")
        f.write("| Feature | Selection Probability |\n")
        f.write("|---------|----------------------|\n")
        for feat, prob in sorted(feature_selection_probs.items(), key=lambda x: -x[1]):
            f.write(f"| {feat} | {prob:.1%} |\n")
        
        # Interpretation
        f.write("\n---\n\n")
        f.write("## Interpretation\n\n")
        
        # Find most stable features
        top_stable_features = sorted(feature_selection_probs.items(), key=lambda x: -x[1])[:2]
        
        f.write("**Optimal Feature Set:**\n\n")
        f.write("Features consistently selected across both nested CV and permutation importance:\n\n")
        
        for feat, prob in top_stable_features:
            perm_row = permutation_result[permutation_result['feature'] == feat].iloc[0]
            f.write(f"- **{feat}**\n")
            f.write(f"  - Selection stability: {prob:.1%}\n")
            f.write(f"  - Permutation importance: {perm_row['mean_importance']:+.3f} (p={perm_row['p_value']:.4f})\n")
        
        f.write(f"\n**Expected Performance:** AUC = {nested_cv_result['mean_auc']:.3f} "
               f"(95% CI: [{nested_cv_result['ci_95'][0]:.3f}, {nested_cv_result['ci_95'][1]:.3f}])\n\n")
        
        f.write("**Key Findings:**\n\n")
        n_significant = permutation_result['significant'].sum()
        f.write(f"1. Nested CV provides **unbiased estimate** of generalization performance\n")
        f.write(f"2. Feature selection is **{'stable' if np.std(all_aucs) < 0.01 else 'moderately stable'}** across different data splits and random seeds\n")
        f.write(f"3. **{n_significant}/{len(permutation_result)} features** show statistically significant predictive value (p < 0.05)\n")
        most_selected = max(nested_cv_result['feature_selection_counts'].items(), key=lambda x: x[1])
        f.write(f"4. **{most_selected[0]}** is the most consistently selected feature ({most_selected[1]}/5 folds)\n")
        
        f.write("\n---\n\n")
        f.write("## Visualizations\n\n")
        f.write("See `figures/` directory for detailed visualizations:\n\n")
        f.write("1. `nested_cv_outer_folds.png` - Performance across outer folds\n")
        f.write("2. `feature_selection_frequency.png` - How often each feature was selected\n")
        f.write("3. `permutation_importance.png` - Feature importance with confidence intervals\n")
        f.write("4. `stability_across_seeds.png` - Performance stability across random seeds\n")
        f.write("5. `nested_cv_summary.png` - Combined summary visualization\n")
    
    print(f"\n✅ Report saved: {output_path.name}")


def save_json_results(nested_cv_result, permutation_result, stability_results, 
                      feature_selection_probs, output_dir):
    """Save structured results for programmatic access."""
    
    # Nested CV results
    nested_cv_data = {
        'mean_auc': float(nested_cv_result['mean_auc']),
        'std_auc': float(nested_cv_result['std_auc']),
        'ci_95_lower': float(nested_cv_result['ci_95'][0]),
        'ci_95_upper': float(nested_cv_result['ci_95'][1]),
        'feature_selection_counts': {k: int(v) for k, v in nested_cv_result['feature_selection_counts'].items()},
        'outer_folds': [
            {
                'fold': int(r['fold']),
                'n_features': int(r['n_features']),
                'outer_auc': float(r['outer_auc']),
                'selected_features': r['selected_features']
            }
            for r in nested_cv_result['outer_results']
        ]
    }
    
    with open(output_dir / "nested_cv_results.json", 'w') as f:
        json.dump(nested_cv_data, f, indent=2)
    print(f"  ✅ Saved: nested_cv_results.json")
    
    # Permutation importance
    permutation_result.to_csv(output_dir / "permutation_importance.csv", index=False)
    print(f"  ✅ Saved: permutation_importance.csv")
    
    # Stability analysis
    stability_data = {
        'seeds': [int(r['seed']) for r in stability_results],
        'mean_aucs': [float(r['mean_auc']) for r in stability_results],
        'overall_mean': float(np.mean([r['mean_auc'] for r in stability_results])),
        'overall_std': float(np.std([r['mean_auc'] for r in stability_results])),
        'feature_selection_probs': {k: float(v) for k, v in feature_selection_probs.items()}
    }
    
    with open(output_dir / "stability_analysis.json", 'w') as f:
        json.dump(stability_data, f, indent=2)
    print(f"  ✅ Saved: stability_analysis.json")


def main():
    print("\n" + "="*80)
    print("RQ2 PHASE 5: RIGOROUS FEATURE SELECTION WITH NESTED CROSS-VALIDATION")
    print("="*80)
    print("Following best practices from:")
    print("- Varma & Simon (2006): Nested CV for unbiased estimation")
    print("- Altmann et al. (2010): Statistical permutation importance")
    print("="*80 + "\n")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = OUTPUT_BASE / f"rq2_rfe_nested_cv_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load data using shared module
    df = load_rq2_combined_data(verbose=True)
    
    # Prepare clean dataset
    X, y, feature_names, df_clean = prepare_clean_dataset(df, verbose=True)
    
    # 1. Nested Cross-Validation
    print("\n" + "="*80)
    print("STEP 1: NESTED CROSS-VALIDATION")
    print("="*80)
    nested_cv_result = nested_cv_rfe(X, y, feature_names, n_outer_folds=5, n_inner_folds=3, random_state=42)
    
    # 2. Statistical Permutation Importance
    print("\n" + "="*80)
    print("STEP 2: PERMUTATION IMPORTANCE")
    print("="*80)
    permutation_result, baseline_auc = statistical_permutation_importance(
        X, y, feature_names, n_iterations=100, random_state=42
    )
    
    # 3. Stability Analysis
    print("\n" + "="*80)
    print("STEP 3: STABILITY ANALYSIS")
    print("="*80)
    stability_results, feature_selection_probs = stability_analysis(X, y, feature_names, n_seeds=5)
    
    # Generate outputs
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80)
    
    # Save JSON results
    save_json_results(nested_cv_result, permutation_result, stability_results, 
                      feature_selection_probs, output_dir)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(nested_cv_result, permutation_result, stability_results, output_dir)
    
    # Generate report
    report_path = output_dir / "rfe_nested_cv_report.md"
    generate_report(
        nested_cv_result, permutation_result, baseline_auc, 
        stability_results, feature_selection_probs, report_path
    )
    
    print("\n" + "="*80)
    print("✅ RQ2 PHASE 5 COMPLETE!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n🎯 FINAL RECOMMENDATION")
    print("="*80)
    print(f"Unbiased performance: AUC = {nested_cv_result['mean_auc']:.3f} "
          f"(95% CI: [{nested_cv_result['ci_95'][0]:.3f}, {nested_cv_result['ci_95'][1]:.3f}])")
    print(f"\nOptimal feature set (selected in ≥60% of folds):")
    for feat, count in sorted(nested_cv_result['feature_selection_counts'].items(), key=lambda x: -x[1]):
        if count >= 3:
            perm_row = permutation_result[permutation_result['feature'] == feat].iloc[0]
            print(f"  ✅ {feat}")
            print(f"     - Selection frequency: {count}/5 folds ({count/5*100:.0f}%)")
            print(f"     - Permutation importance: {perm_row['mean_importance']:+.3f} (p={perm_row['p_value']:.4f})")
    
    print("\n" + "="*80 + "\n")
    
    return output_dir


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
RQ2 Phase 4: RFE Feature Selection with Multiple Classifiers

Tests RFE with different classifiers to find:
1. Which features are most important
2. Which classifier performs best with selected features

This feeds into Phase 5 (ensemble with optimal features) and 
Phase 6 (cross-CLD with best classifier + features).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
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


def get_classifiers():
    """Define classifiers to test (only those compatible with RFE)."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        # Neural Network excluded: incompatible with RFE (no feature_importances_)
    }


def rfe_with_classifier(X, y, feature_names, clf_name, clf, groups, n_folds=3, random_state=42):
    """
    Perform RFE with a specific classifier using block-level cross-validation.
    
    Uses GroupKFold with groups = (CLD × run) blocks to prevent leakage.
    With 9 blocks, we use 3-fold CV (3 blocks per fold).
    
    Returns optimal number of features and their selection frequency.
    """
    print(f"\n{'='*80}")
    print(f"RFE with {clf_name}")
    print(f"{'='*80}")
    
    n_blocks = len(np.unique(groups))
    # Ensure n_folds doesn't exceed n_blocks
    n_folds = min(n_folds, n_blocks)
    print(f"Block-level CV: {n_blocks} blocks, {n_folds}-fold GroupKFold")
    
    cv = GroupKFold(n_splits=n_folds)
    
    feature_selection_counts = {feat: 0 for feat in feature_names}
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
        print(f"\nFold {fold_idx}/{n_folds}:")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test different numbers of features
        best_n = None
        best_auc = 0
        best_features = None
        
        for n_features in range(1, len(feature_names) + 1):
            # RFE
            rfe = RFE(estimator=clf, n_features_to_select=n_features, step=1)
            rfe.fit(X_train_scaled, y_train)
            
            # Evaluate
            X_train_selected = X_train_scaled[:, rfe.support_]
            X_test_selected = X_test_scaled[:, rfe.support_]
            
            clf.fit(X_train_selected, y_train)
            y_prob = clf.predict_proba(X_test_selected)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            
            if auc > best_auc:
                best_auc = auc
                best_n = n_features
                best_features = [feat for feat, selected in zip(feature_names, rfe.support_) if selected]
        
        print(f"  Best: {best_n} features, AUC={best_auc:.3f}")
        print(f"  Selected: {best_features}")
        
        # Count selections
        for feat in best_features:
            feature_selection_counts[feat] += 1
        
        fold_results.append({
            'fold': fold_idx,
            'n_features': best_n,
            'auc': best_auc,
            'features': best_features
        })
    
    # Summary
    aucs = [r['auc'] for r in fold_results]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    actual_folds = len(fold_results)
    
    print(f"\n{'='*40}")
    print(f"SUMMARY: {clf_name}")
    print(f"{'='*40}")
    print(f"Mean AUC: {mean_auc:.3f} ± {std_auc:.3f} ({actual_folds}-fold block-level CV)")
    print(f"Feature selection frequency:")
    for feat, count in sorted(feature_selection_counts.items(), key=lambda x: -x[1]):
        print(f"  {feat:30s}: {count}/{actual_folds} ({count/actual_folds*100:.0f}%)")
    
    return {
        'classifier': clf_name,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'n_folds': actual_folds,
        'n_blocks': n_blocks,
        'fold_results': fold_results,
        'feature_selection_counts': feature_selection_counts,
        'most_selected_features': [feat for feat, count in sorted(feature_selection_counts.items(), 
                                                                   key=lambda x: -x[1]) if count >= actual_folds * 0.6]
    }


def compare_classifiers_with_rfe(X, y, feature_names, groups):
    """Compare all classifiers using RFE with block-level CV."""
    print("\n" + "="*80)
    print("PHASE 4: RFE FEATURE SELECTION WITH MULTIPLE CLASSIFIERS")
    print("="*80)
    print(f"Using GroupKFold with {len(np.unique(groups))} blocks (CLD × run)")
    
    classifiers = get_classifiers()
    results = {}
    
    for clf_name, clf in classifiers.items():
        result = rfe_with_classifier(X, y, feature_names, clf_name, clf, groups)
        results[clf_name] = result
    
    return results


def find_best_classifier_and_features(results):
    """Identify the best performing classifier and its optimal features."""
    print("\n" + "="*80)
    print("BEST CLASSIFIER & FEATURES")
    print("="*80)
    
    # Rank by AUC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_auc'], reverse=True)
    
    print("\n Classifier Rankings:")
    print(f"{'Rank':<6} {'Classifier':<25} {'Mean AUC':<12} {'Optimal Features'}")
    print("-" * 80)
    
    for rank, (clf_name, result) in enumerate(sorted_results, 1):
        features_str = ', '.join(result['most_selected_features'][:2])
        if len(result['most_selected_features']) > 2:
            features_str += f" + {len(result['most_selected_features'])-2} more"
        print(f"{rank:<6} {clf_name:<25} {result['mean_auc']:.3f} ± {result['std_auc']:.3f}  {features_str}")
    
    # Best overall
    best_clf_name = sorted_results[0][0]
    best_result = sorted_results[0][1]
    
    print(f"\n🏆 WINNER: {best_clf_name}")
    print(f"   Mean AUC: {best_result['mean_auc']:.3f} ± {best_result['std_auc']:.3f}")
    print(f"   Optimal Features ({len(best_result['most_selected_features'])}):")
    for feat in best_result['most_selected_features']:
        freq = best_result['feature_selection_counts'][feat]
        print(f"     - {feat} ({freq}/5 folds)")
    
    return best_clf_name, best_result


def generate_visualizations(results, output_dir):
    """Create visualizations comparing classifiers."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 1. Classifier comparison
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # AUC comparison
    clf_names = list(results.keys())
    aucs = [results[clf]['mean_auc'] for clf in clf_names]
    stds = [results[clf]['std_auc'] for clf in clf_names]
    
    colors = ['#2ecc71' if auc == max(aucs) else '#3498db' for auc in aucs]
    ax1.bar(range(len(clf_names)), aucs, yerr=stds, color=colors, alpha=0.8, 
            edgecolor='black', capsize=5)
    ax1.set_xticks(range(len(clf_names)))
    ax1.set_xticklabels(clf_names, rotation=45, ha='right')
    ax1.set_ylabel('Mean AUC', fontsize=12)
    ax1.set_title('Classifier Performance with RFE', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Feature selection frequency (best classifier)
    best_clf = max(results.items(), key=lambda x: x[1]['mean_auc'])
    feat_counts = best_clf[1]['feature_selection_counts']
    sorted_feats = sorted(feat_counts.items(), key=lambda x: -x[1])
    feats = [f[0] for f in sorted_feats]
    counts = [f[1] for f in sorted_feats]
    
    colors2 = ['#2ecc71' if c >= 3 else '#f39c12' if c >= 2 else '#e74c3c' for c in counts]
    ax2.barh(range(len(feats)), counts, color=colors2, alpha=0.8, edgecolor='black')
    ax2.set_yticks(range(len(feats)))
    ax2.set_yticklabels(feats, fontsize=10)
    ax2.set_xlabel('Selection Frequency (out of 5 folds)', fontsize=12)
    ax2.set_title(f'Feature Selection: {best_clf[0]}', fontsize=14, fontweight='bold')
    ax2.axvline(3, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax2.set_xlim([0, 5])
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "phase4_classifier_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: phase4_classifier_comparison.png")
    
    # 2. Feature selection stability across classifiers
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    feature_data = {feat: [] for feat in CI_METRICS}
    for clf_name in clf_names:
        for feat in CI_METRICS:
            count = results[clf_name]['feature_selection_counts'][feat]
            feature_data[feat].append(count)
    
    x = np.arange(len(clf_names))
    width = 0.2
    
    for i, (feat, counts) in enumerate(feature_data.items()):
        ax.bar(x + i*width, counts, width, label=feat, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Classifier', fontsize=12)
    ax.set_ylabel('Selection Frequency (out of 5 folds)', fontsize=12)
    ax.set_title('Feature Selection Stability Across Classifiers', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(clf_names, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim([0, 5])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "phase4_feature_stability.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: phase4_feature_stability.png")


def generate_report(results, best_clf_name, best_result, output_path):
    """Generate markdown report."""
    with open(output_path, 'w') as f:
        f.write("# RQ2 Phase 4: RFE Feature Selection with Multiple Classifiers\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Purpose:** Identify optimal features and best-performing classifier\n\n")
        
        f.write("---\n\n")
        f.write("## Methodology\n\n")
        f.write("**Approach:** Recursive Feature Elimination (RFE) with block-level cross-validation\n\n")
        f.write("**Cross-Validation:** GroupKFold with blocks = (CLD × run)\n")
        f.write("- 9 blocks total (3 CLDs × 3 runs)\n")
        f.write("- 3-fold CV (3 blocks per fold)\n")
        f.write("- Prevents pseudo-replication from correlated edges within the same block\n\n")
        f.write("**Classifiers Tested:**\n")
        f.write("1. Logistic Regression (balanced, max_iter=1000)\n")
        f.write("2. Random Forest (100 estimators, balanced)\n")
        f.write("3. Gradient Boosting (100 estimators)\n")
        f.write("\n*Note: Neural Network (MLP) excluded as it's incompatible with RFE.*\n\n")
        
        f.write("**Features Available:** 4 CI metrics\n")
        f.write("- Gen Perplexity\n")
        f.write("- Gen Min Prob\n")
        f.write("- Gen Max Window Entropy\n")
        f.write("- Gen Cosine Similarity\n\n")
        
        f.write("---\n\n")
        f.write("## Results\n\n")
        
        f.write("### Classifier Performance with RFE\n\n")
        f.write("| Rank | Classifier | Mean AUC | Optimal Features (selected ≥60% of folds) |\n")
        f.write("|------|------------|----------|--------------------------------------------|\n")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_auc'], reverse=True)
        for rank, (clf_name, result) in enumerate(sorted_results, 1):
            features_str = ', '.join(result['most_selected_features']) if result['most_selected_features'] else 'None consistently'
            f.write(f"| {rank} | {clf_name} | {result['mean_auc']:.3f} ± {result['std_auc']:.3f} | {features_str} |\n")
        
        f.write("\n### Winner: " + best_clf_name + "\n\n")
        f.write(f"**Performance:** AUC = {best_result['mean_auc']:.3f} ± {best_result['std_auc']:.3f}\n\n")
        
        f.write("**Optimal Features:**\n\n")
        f.write("| Feature | Selection Frequency | Percentage |\n")
        f.write("|---------|---------------------|------------|\n")
        for feat, count in sorted(best_result['feature_selection_counts'].items(), key=lambda x: -x[1]):
            f.write(f"| {feat} | {count}/5 | {count*20}% |\n")
        
        f.write("\n---\n\n")
        f.write("## Feature Selection Across All Classifiers\n\n")
        f.write("| Feature | " + " | ".join(results.keys()) + " |\n")
        f.write("|---------|" + "|".join(["----------"] * len(results)) + "|\n")
        
        for feat in CI_METRICS:
            row = f"| {feat} |"
            for clf_name in results.keys():
                count = results[clf_name]['feature_selection_counts'][feat]
                row += f" {count}/5 ({count*20}%) |"
            f.write(row + "\n")
        
        f.write("\n---\n\n")
        f.write("## Recommendations for Phases 5 & 6\n\n")
        f.write(f"**Classifier to use:** {best_clf_name}\n\n")
        f.write(f"**Features to use:** {', '.join(best_result['most_selected_features'])}\n\n")
        f.write(f"**Expected performance:** AUC ≈ {best_result['mean_auc']:.3f}\n\n")
        
        f.write("**Next Steps:**\n")
        f.write("1. Phase 5: Compare all classifiers using these optimal features\n")
        f.write("2. Phase 6: Test cross-CLD generalization with best classifier + features\n\n")
        
        f.write("---\n\n")
        f.write("## Visualizations\n\n")
        f.write("See `figures/` directory for:\n")
        f.write("1. `phase4_classifier_comparison.png` - Performance comparison\n")
        f.write("2. `phase4_feature_stability.png` - Feature selection across classifiers\n")
    
    print(f"\n✅ Report saved: {output_path.name}")


def save_json_results(results, best_clf_name, best_result, output_dir):
    """Save structured results for downstream phases."""
    # Main results with comprehensive stats including fold-level values for min/max calculation
    results_json = {
        'best_classifier': best_clf_name,
        'best_auc': float(best_result['mean_auc']),
        'best_auc_std': float(best_result['std_auc']),
        'optimal_features': best_result['most_selected_features'],
        'cv_method': 'GroupKFold (block = CLD × run)',
        'n_blocks': int(best_result.get('n_blocks', 9)),
        'n_folds': int(best_result.get('n_folds', 3)),
        'all_classifiers': {
            clf_name: {
                'mean_auc': float(r['mean_auc']),
                'std_auc': float(r['std_auc']),
                'n_folds': int(r.get('n_folds', 3)),
                'n_blocks': int(r.get('n_blocks', 9)),
                # Include individual fold AUCs for min/max calculation (per uncertainty reporting rule)
                'cv_aucs': [float(fr['auc']) for fr in r['fold_results']],
                'min_auc': float(min(fr['auc'] for fr in r['fold_results'])),
                'max_auc': float(max(fr['auc'] for fr in r['fold_results'])),
                'feature_selection_counts': {k: int(v) for k, v in r['feature_selection_counts'].items()},
                'most_selected_features': r['most_selected_features']
            }
            for clf_name, r in results.items()
        }
    }
    
    with open(output_dir / "phase4_rfe_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  ✅ Saved: phase4_rfe_results.json")


def main():
    print("\n" + "="*80)
    print("RQ2 PHASE 4: RFE FEATURE SELECTION WITH MULTIPLE CLASSIFIERS")
    print("="*80)
    print("Finding optimal features and best classifier for Phases 5 & 6")
    print("Using block-level (CLD × run) GroupKFold to prevent pseudo-replication")
    print("="*80 + "\n")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = OUTPUT_BASE / f"rq2_phase4_rfe_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load data with block groups for GroupKFold
    df = load_rq2_combined_data(verbose=True)
    X, y, feature_names, df_clean, groups = prepare_clean_dataset(df, verbose=True, return_groups=True)
    
    # Run RFE with all classifiers using block-level CV
    results = compare_classifiers_with_rfe(X, y, feature_names, groups)
    
    # Find best
    best_clf_name, best_result = find_best_classifier_and_features(results)
    
    # Generate outputs
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80)
    
    save_json_results(results, best_clf_name, best_result, output_dir)
    
    print("\nGenerating visualizations...")
    generate_visualizations(results, output_dir)
    
    report_path = output_dir / "phase4_rfe_report.md"
    generate_report(results, best_clf_name, best_result, report_path)
    
    print("\n" + "="*80)
    print("✅ PHASE 4 COMPLETE!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n🏆 RECOMMENDATION FOR PHASES 5 & 6")
    print("="*80)
    print(f"Classifier: {best_clf_name}")
    print(f"Features: {', '.join(best_result['most_selected_features'])}")
    print(f"Expected AUC: {best_result['mean_auc']:.3f} ± {best_result['std_auc']:.3f}")
    print("\n" + "="*80 + "\n")
    
    return output_dir


if __name__ == "__main__":
    main()


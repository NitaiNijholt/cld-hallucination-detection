#!/usr/bin/env python3
"""
RQ2 Phase 5: Ensemble Comparison with RFE-Selected Features

Uses the optimal features identified in Phase 4 to compare all classifiers
on a held-out test set. This provides the final performance estimate for
each classifier with the optimized feature set.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, 
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Import shared data preparation module
sys.path.insert(0, str(Path(__file__).parent))
from rq2_data_preparation import load_rq2_combined_data, prepare_clean_dataset

# Output directory
OUTPUT_BASE = Path("parameter_tuning_experiments/rq2_analyses")


def get_classifiers():
    """Define all classifiers to test (including Neural Network for generalization comparison)."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    }


def load_phase4_results():
    """Load optimal features from Phase 4."""
    phase4_dirs = list(OUTPUT_BASE.glob("rq2_phase4_rfe_*"))
    if not phase4_dirs:
        print("⚠️  No Phase 4 results found. Using all 4 features as fallback.")
        return None
    
    latest_phase4 = max(phase4_dirs, key=lambda p: p.stat().st_mtime)
    results_file = latest_phase4 / "phase4_rfe_results.json"
    
    if not results_file.exists():
        print("⚠️  Phase 4 results file not found. Using all 4 features as fallback.")
        return None
    
    with open(results_file, 'r') as f:
        phase4_results = json.load(f)
    
    print(f"✅ Loaded Phase 4 results from: {latest_phase4.name}")
    print(f"   Optimal features: {', '.join(phase4_results['optimal_features'])}")
    print(f"   Best classifier from Phase 4: {phase4_results['best_classifier']} (AUC={phase4_results['best_auc']:.3f})\n")
    
    return phase4_results


def train_and_evaluate_classifiers(X, y, feature_names, groups):
    """Train all classifiers and evaluate on held-out test blocks.
    
    Uses GroupShuffleSplit for train/test split and GroupKFold for CV,
    ensuring entire blocks (CLD × run) stay together in train OR test.
    """
    print(f"\n{'='*80}")
    print("TRAINING ENSEMBLE CLASSIFIERS WITH OPTIMAL FEATURES")
    print(f"{'='*80}")
    print(f"Features used: {', '.join(feature_names)}")
    
    n_blocks = len(np.unique(groups))
    print(f"Block-level CV: {n_blocks} blocks (CLD × run)")
    
    # Block-level train/test split using GroupShuffleSplit
    # Hold out ~3 blocks (1/3 of 9) for testing
    gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    
    train_blocks = np.unique(groups[train_idx])
    test_blocks = np.unique(groups[test_idx])
    
    print(f"\nBlock-level data split:")
    print(f"  Training blocks: {list(train_blocks)} ({len(X_train_raw)} edges)")
    print(f"  Test blocks: {list(test_blocks)} ({len(X_test_raw)} edges)")
    print(f"  Training hallucination rate: {y_train.mean():.1%}")
    print(f"  Test hallucination rate: {y_test.mean():.1%}\n")
    
    # Standardize (fit on training blocks only to avoid leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    classifiers = get_classifiers()
    results = {}
    
    # Determine CV folds (max = n_train_blocks)
    n_train_blocks = len(train_blocks)
    n_cv_folds = min(3, n_train_blocks)
    
    for clf_name, clf in classifiers.items():
        print(f"{'='*80}")
        print(f"Training: {clf_name}")
        print(f"{'='*80}")
        
        # Block-level cross-validation on training set
        cv = GroupKFold(n_splits=n_cv_folds)
        cv_scores = []
        for train_cv_idx, val_cv_idx in cv.split(X_train, y_train, groups_train):
            clf_cv = get_classifiers()[clf_name]  # Fresh instance
            # Fit scaler inside CV fold to avoid leakage between CV-train and CV-val
            fold_scaler = StandardScaler()
            X_tr = fold_scaler.fit_transform(X_train_raw[train_cv_idx])
            X_val = fold_scaler.transform(X_train_raw[val_cv_idx])
            clf_cv.fit(X_tr, y_train[train_cv_idx])
            y_prob_cv = clf_cv.predict_proba(X_val)[:, 1]
            cv_scores.append(roc_auc_score(y_train[val_cv_idx], y_prob_cv))
        
        cv_scores = np.array(cv_scores)
        cv_auc_mean = cv_scores.mean()
        cv_auc_std = cv_scores.std()
        
        print(f"{n_cv_folds}-Fold Block-Level CV AUC: {cv_auc_mean:.4f} (±{cv_auc_std:.4f})")
        
        # Train on full training blocks (scaled with train-fitted scaler)
        clf.fit(X_train, y_train)
        
        # Evaluate on held-out test blocks
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        test_auc = roc_auc_score(y_test, y_prob)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        
        print(f"Test AUC (held-out blocks): {test_auc:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
        
        results[clf_name] = {
            'cv_auc_mean': cv_auc_mean,
            'cv_auc_std': cv_auc_std,
            'cv_aucs': cv_scores.tolist(),  # Individual fold AUCs for min/max
            'cv_auc_min': float(cv_scores.min()),
            'cv_auc_max': float(cv_scores.max()),
            'n_cv_folds': n_cv_folds,
            'n_train_blocks': n_train_blocks,
            'n_test_blocks': len(test_blocks),
            'test_auc': test_auc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'y_prob': y_prob,
            'y_true': y_test,
            'train_blocks': list(train_blocks),
            'test_blocks': list(test_blocks)
        }
    
    return results, scaler


def generate_visualizations(results, feature_names, output_dir):
    """Create visualizations."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for clf_name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_true'], result['y_prob'])
        ax.plot(fpr, tpr, linewidth=2, label=f"{clf_name} (AUC={result['test_auc']:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves - Phase 5 Ensemble\n(Features: {", ".join(feature_names)})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "phase5_ensemble_roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: phase5_ensemble_roc_curves.png")
    
    # Performance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    clf_names = list(results.keys())
    test_aucs = [results[clf]['test_auc'] for clf in clf_names]
    cv_aucs = [results[clf]['cv_auc_mean'] for clf in clf_names]
    cv_stds = [results[clf]['cv_auc_std'] for clf in clf_names]
    
    # Test AUC
    colors = ['#2ecc71' if auc == max(test_aucs) else '#3498db' for auc in test_aucs]
    ax1.bar(range(len(clf_names)), test_aucs, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(clf_names)))
    ax1.set_xticklabels(clf_names, rotation=45, ha='right')
    ax1.set_ylabel('Test AUC', fontsize=12)
    ax1.set_title('Test Set Performance', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # CV AUC with error bars
    ax2.bar(range(len(clf_names)), cv_aucs, yerr=cv_stds, color=colors, alpha=0.8, 
            edgecolor='black', capsize=5)
    ax2.set_xticks(range(len(clf_names)))
    ax2.set_xticklabels(clf_names, rotation=45, ha='right')
    ax2.set_ylabel('CV AUC', fontsize=12)
    ax2.set_title('Cross-Validation Performance', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "phase5_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: phase5_performance_comparison.png")


def generate_report(results, feature_names, phase4_results, output_path, n_samples, n_hallucinations):
    """Generate markdown report."""
    with open(output_path, 'w') as f:
        f.write("# RQ2 Phase 5: Ensemble Comparison with RFE-Selected Features\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Purpose:** Compare all classifiers using optimal features from Phase 4\n\n")
        
        f.write("---\n\n")
        f.write("## Setup\n\n")
        
        if phase4_results:
            f.write(f"**Features Used:** {', '.join(feature_names)} (from Phase 4 RFE)\n")
            f.write(f"**Phase 4 Winner:** {phase4_results['best_classifier']} (AUC={phase4_results['best_auc']:.3f})\n")
            f.write(f"**Note:** Neural Network included here for comparison even though it wasn't in Phase 4 RFE.\n\n")
        else:
            f.write(f"**Features Used:** {', '.join(feature_names)} (all available)\n\n")
        
        f.write(f"**Dataset:** {n_samples:,} samples ({n_hallucinations:,} hallucinations, {n_hallucinations/n_samples*100:.1f}%)\n")
        f.write(f"**Cross-Validation:** Block-level GroupKFold (block = CLD × run)\n")
        f.write(f"**Split:** GroupShuffleSplit (~67% train blocks, ~33% test blocks)\n")
        f.write(f"**Evaluation:** Block-level CV on train, final test on held-out blocks\n\n")
        
        f.write("---\n\n")
        f.write("## Results\n\n")
        
        f.write("### Classifier Performance\n\n")
        f.write("| Classifier | CV AUC | Test AUC | Precision | Recall | F1 Score |\n")
        f.write("|------------|--------|----------|-----------|--------|----------|\n")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['test_auc'], reverse=True)
        for clf_name, result in sorted_results:
            f.write(f"| {clf_name} | {result['cv_auc_mean']:.4f} (±{result['cv_auc_std']:.4f}) | ")
            f.write(f"{result['test_auc']:.4f} | {result['test_precision']:.4f} | ")
            f.write(f"{result['test_recall']:.4f} | {result['test_f1']:.4f} |\n")
        
        best_clf = sorted_results[0]
        f.write(f"\n**Winner:** {best_clf[0]} with Test AUC = {best_clf[1]['test_auc']:.4f}\n\n")
        
        if best_clf[1]['test_auc'] > 0.9:
            f.write("**Performance Level:** Excellent (AUC > 0.9) - Near-perfect discrimination\n")
        elif best_clf[1]['test_auc'] > 0.8:
            f.write("**Performance Level:** Very Good (0.8 < AUC < 0.9)\n")
        elif best_clf[1]['test_auc'] > 0.7:
            f.write("**Performance Level:** Good (0.7 < AUC < 0.8)\n")
        else:
            f.write("**Performance Level:** Moderate (AUC < 0.7)\n")
        
        f.write("\n---\n\n")
        f.write("## Comparison with Phase 4\n\n")
        
        if phase4_results:
            phase4_winner = phase4_results['best_classifier']
            phase4_auc = phase4_results['best_auc']
            
            if phase4_winner in results:
                phase5_auc = results[phase4_winner]['test_auc']
                diff = phase5_auc - phase4_auc
                f.write(f"**Phase 4 Winner ({phase4_winner}):**\n")
                f.write(f"- Phase 4 (5-fold CV): AUC = {phase4_auc:.3f}\n")
                f.write(f"- Phase 5 (held-out test): AUC = {phase5_auc:.3f}\n")
                f.write(f"- Difference: {diff:+.3f}\n\n")
                
                if abs(diff) < 0.02:
                    f.write("✅ Performance is consistent between phases.\n\n")
                elif diff > 0:
                    f.write("⚠️ Higher test performance - may indicate lucky split.\n\n")
                else:
                    f.write("⚠️ Lower test performance - may indicate overfitting in Phase 4 CV.\n\n")
        
        f.write("---\n\n")
        f.write("## Recommendation for Phase 6\n\n")
        f.write(f"**Classifier:** {best_clf[0]}\n")
        f.write(f"**Features:** {', '.join(feature_names)}\n")
        f.write(f"**Expected Performance:** AUC ≈ {best_clf[1]['test_auc']:.3f}\n\n")
        f.write("This configuration will be used for cross-CLD generalization testing in Phase 6.\n\n")
        
        f.write("---\n\n")
        f.write("## Visualizations\n\n")
        f.write("See `figures/` directory for:\n")
        f.write("1. `phase5_ensemble_roc_curves.png` - ROC curves for all classifiers\n")
        f.write("2. `phase5_performance_comparison.png` - Test vs CV performance\n")
    
    print(f"\n✅ Report saved: {output_path.name}")


def save_json_results(results, feature_names, phase4_results, n_samples, n_hallucinations, output_dir):
    """Save structured results for Phase 6."""
    best_clf = max(results.items(), key=lambda x: x[1]['test_auc'])
    
    # Get block info from best classifier result
    best_result = best_clf[1]
    
    results_json = {
        'features_used': feature_names,
        'n_features': len(feature_names),
        'cv_method': 'GroupKFold (block = CLD × run)',
        'split_method': 'GroupShuffleSplit (block-level)',
        'dataset': {
            'total_samples': int(n_samples),
            'hallucinations': int(n_hallucinations),
            'hallucination_rate': float(n_hallucinations / n_samples)
        },
        'best_classifier': best_clf[0],
        'best_test_auc': float(best_result['test_auc']),
        'best_cv_auc': float(best_result['cv_auc_mean']),
        'train_blocks': best_result.get('train_blocks', []),
        'test_blocks': best_result.get('test_blocks', []),
        'classifiers': {
            clf_name: {
                'cv_auc_mean': float(r['cv_auc_mean']),
                'cv_auc_std': float(r['cv_auc_std']),
                'n_cv_folds': int(r.get('n_cv_folds', 3)),
                'n_train_blocks': int(r.get('n_train_blocks', 6)),
                'n_test_blocks': int(r.get('n_test_blocks', 3)),
                # Include individual fold AUCs for min/max calculation (per uncertainty reporting rule)
                'cv_aucs': [float(v) for v in r['cv_aucs']],
                'cv_auc_min': float(r['cv_auc_min']),
                'cv_auc_max': float(r['cv_auc_max']),
                'test_auc': float(r['test_auc']),
                'test_precision': float(r['test_precision']),
                'test_recall': float(r['test_recall']),
                'test_f1': float(r['test_f1']),
                'train_blocks': r.get('train_blocks', []),
                'test_blocks': r.get('test_blocks', [])
            }
            for clf_name, r in results.items()
        },
        'phase4_reference': phase4_results if phase4_results else None
    }
    
    with open(output_dir / "phase5_ensemble_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  ✅ Saved: phase5_ensemble_results.json")


def main():
    print("\n" + "="*80)
    print("RQ2 PHASE 5: ENSEMBLE COMPARISON WITH RFE-SELECTED FEATURES")
    print("="*80)
    print("Using optimal features from Phase 4 to find best classifier")
    print("Using block-level (CLD × run) GroupKFold to prevent pseudo-replication")
    print("="*80 + "\n")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = OUTPUT_BASE / f"rq2_phase5_ensemble_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load Phase 4 results
    phase4_results = load_phase4_results()
    
    # Load data with block groups
    df = load_rq2_combined_data(verbose=True)
    
    # Use Phase 4 optimal features if available
    if phase4_results and phase4_results['optimal_features']:
        optimal_features = phase4_results['optimal_features']
    else:
        from rq2_data_preparation import CI_METRICS
        optimal_features = CI_METRICS
    
    X, y, _, df_clean, groups = prepare_clean_dataset(df, features=optimal_features, verbose=True, return_groups=True)
    
    # Train and evaluate with block-level CV
    results, scaler = train_and_evaluate_classifiers(X, y, optimal_features, groups)
    
    # Generate outputs
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80)
    
    save_json_results(results, optimal_features, phase4_results, len(y), y.sum(), output_dir)
    
    print("\nGenerating visualizations...")
    generate_visualizations(results, optimal_features, output_dir)
    
    report_path = output_dir / "phase5_ensemble_report.md"
    generate_report(results, optimal_features, phase4_results, report_path, len(y), y.sum())
    
    # Find winner
    best_clf = max(results.items(), key=lambda x: x[1]['test_auc'])
    
    print("\n" + "="*80)
    print("✅ PHASE 5 COMPLETE!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n🏆 WINNER FOR PHASE 6")
    print("="*80)
    print(f"Classifier: {best_clf[0]}")
    print(f"Features: {', '.join(optimal_features)}")
    print(f"Test AUC: {best_clf[1]['test_auc']:.3f}")
    print(f"Precision: {best_clf[1]['test_precision']:.3f}, Recall: {best_clf[1]['test_recall']:.3f}")
    print("\n" + "="*80 + "\n")
    
    return output_dir


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
RQ2 Ensemble Classifier Training
Trains ML classifiers using CI metrics to predict hallucinations.
Evaluates performance and compares different algorithms.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, 
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys

from rq2_paths import rq2_dirs

# Import shared data preparation module
sys.path.insert(0, str(Path(__file__).parent))
from rq2_data_preparation import load_rq2_combined_data, prepare_clean_dataset, CI_METRICS

# Note: Data loading now handled by shared module (rq2_data_preparation.py)

def train_classifiers(X, y, feature_names):
    """Train ensemble classifiers."""
    print(f"\n{'='*80}")
    print("TRAINING ENSEMBLE CLASSIFIERS")
    print(f"{'='*80}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples ({y_train.mean():.1%} hallucinations)")
    print(f"  Test: {len(X_test)} samples ({y_test.mean():.1%} hallucinations)")
    
    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    }
    
    results = {}
    
    for clf_name, clf in classifiers.items():
        print(f"\n{'─'*80}")
        print(f"Training: {clf_name}")
        print(f"{'─'*80}")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        print(f"  5-Fold CV AUC: {cv_auc_scores.mean():.4f} (±{cv_auc_scores.std():.4f})")
        
        # Train on full training set
        clf.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        print(f"  Test AUC: {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {cm[0,0]:3d}  FP: {cm[0,1]:3d}")
        print(f"    FN: {cm[1,0]:3d}  TP: {cm[1,1]:3d}")
        
        results[clf_name] = {
            'cv_auc_mean': cv_auc_scores.mean(),
            'cv_auc_std': cv_auc_scores.std(),
            'test_auc': auc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'model': clf
        }
    
    return results, scaler, X_test, y_test

def plot_roc_curves(results, output_dir):
    """Plot ROC curves for all classifiers."""
    plt.figure(figsize=(10, 8))
    
    for clf_name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
        auc = result['test_auc']
        plt.plot(fpr, tpr, label=f"{clf_name} (AUC={auc:.3f})", linewidth=2)
    
    # Baseline
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC=0.500)', linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Ensemble Classifiers for Hallucination Detection', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'figures' / 'ensemble_roc_curves.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved ROC curves: {output_path}")

def generate_report(results, df_stats, output_dir):
    """Generate markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    best_clf = max(results.items(), key=lambda x: x[1]['test_auc'])
    
    report_path = output_dir / "ensemble_classifier_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# RQ2 Ensemble Classifier Report\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write("---\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"**Best Classifier:** {best_clf[0]}\n")
        f.write(f"**Test AUC:** {best_clf[1]['test_auc']:.4f}\n")
        f.write(f"**Cross-Validation AUC:** {best_clf[1]['cv_auc_mean']:.4f} (±{best_clf[1]['cv_auc_std']:.4f})\n\n")
        
        f.write("---\n\n")
        
        f.write("## Dataset\n\n")
        f.write(f"**Total Samples:** {len(df_stats)}\n")
        f.write(f"**Hallucinations:** {df_stats['is_hallucination'].sum()} ({df_stats['is_hallucination'].sum()/len(df_stats)*100:.1f}%)\n")
        f.write(f"**Correct:** {(~df_stats['is_hallucination']).sum()} ({(~df_stats['is_hallucination']).sum()/len(df_stats)*100:.1f}%)\n\n")
        
        # Breakdown by experiment type
        f.write("**By Experiment Type:**\n\n")
        for exp_type in df_stats['experiment_type'].unique():
            exp_df = df_stats[df_stats['experiment_type'] == exp_type]
            f.write(f"- {exp_type}: {len(exp_df)} samples "
                   f"({exp_df['is_hallucination'].sum()} hallucinations, {exp_df['is_hallucination'].sum()/len(exp_df)*100:.1f}%)\n")
        
        f.write("\n---\n\n")
        
        f.write("## Classifier Comparison\n\n")
        f.write("| Classifier | CV AUC | Test AUC | Precision | Recall | F1 Score |\n")
        f.write("|------------|--------|----------|-----------|--------|----------|\n")
        
        for clf_name, result in results.items():
            f.write(f"| {clf_name} | {result['cv_auc_mean']:.4f} (±{result['cv_auc_std']:.4f}) | "
                   f"{result['test_auc']:.4f} | {result['test_precision']:.4f} | "
                   f"{result['test_recall']:.4f} | {result['test_f1']:.4f} |\n")
        
        f.write("\n---\n\n")
        
        f.write("## Interpretation\n\n")
        f.write(f"**Best performing classifier:** {best_clf[0]} with AUC = {best_clf[1]['test_auc']:.4f}\n\n")
        
        if best_clf[1]['test_auc'] > 0.7:
            f.write("**Performance level:** Good (AUC > 0.7)\n\n")
        elif best_clf[1]['test_auc'] > 0.6:
            f.write("**Performance level:** Moderate (0.6 < AUC < 0.7)\n\n")
        else:
            f.write("**Performance level:** Weak (AUC < 0.6)\n\n")
        
        f.write("**Key Findings:**\n\n")
        f.write(f"1. Ensemble classifiers can predict hallucinations with AUC = {best_clf[1]['test_auc']:.4f}\n")
        f.write(f"2. Precision-Recall tradeoff: {best_clf[1]['test_precision']:.1%} precision, {best_clf[1]['test_recall']:.1%} recall\n")
        f.write(f"3. Cross-validation shows {('stable' if best_clf[1]['cv_auc_std'] < 0.05 else 'moderate' if best_clf[1]['cv_auc_std'] < 0.1 else 'high')} variance\n")
    
    print(f"✅ Saved report: {report_path}")

def main():
    print("\n" + "="*80)
    print("RQ2 ENSEMBLE CLASSIFIER TRAINING")
    print("="*80 + "\n")
    
    # Create output directory
    analyses_dir, _unused_output = rq2_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = analyses_dir / f"rq2_ensemble_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}\n")
    
    # Load data using shared module
    df = load_rq2_combined_data(verbose=True)
    
    # Prepare features using shared module
    X, y, feature_names, df_clean = prepare_clean_dataset(df, verbose=True)
    
    print(f"\n📁 Output directory: {output_dir.name}")
    
    # Train classifiers
    results, scaler, X_test, y_test = train_classifiers(X, y, feature_names)
    
    # Plot ROC curves
    plot_roc_curves(results, output_dir)
    
    # Generate report
    generate_report(results, df_clean, output_dir)
    
    # Save results summary
    results_summary = {
        'timestamp': timestamp,
        'dataset': {
            'total_samples': int(len(df_clean)),
            'hallucinations': int(df_clean['is_hallucination'].sum()),
            'hallucination_rate': float(df_clean['is_hallucination'].mean())
        },
        'classifiers': {
            name: {
                'cv_auc_mean': float(r['cv_auc_mean']),
                'cv_auc_std': float(r['cv_auc_std']),
                'test_auc': float(r['test_auc']),
                'test_precision': float(r['test_precision']),
                'test_recall': float(r['test_recall']),
                'test_f1': float(r['test_f1'])
            }
            for name, r in results.items()
        }
    }
    
    with open(output_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ ENSEMBLE CLASSIFIER TRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



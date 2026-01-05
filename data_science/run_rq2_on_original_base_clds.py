#!/usr/bin/env python3
"""
Run RQ2 Analysis and Ensemble Classifier on Original Base CLDs
Uses the ORIGINAL generation files that already have all CI metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
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
from datetime import datetime
import shutil

# Input files (ORIGINAL base generation CLDs with CI metrics)
INPUT_DIR = Path("parameter_tuning_experiments/results/rq1_base_generation_original_20251012_200603")
INPUT_FILES = {
    'Social_norms': INPUT_DIR / 'result_excel_path_20251012_200705.xlsx',
    'Depressive_symptoms': INPUT_DIR / 'result_excel_path_20251012_212700.xlsx',
    'Older_persons': INPUT_DIR / 'result_excel_path_20251012_201943.xlsx'
}

# Context-insensitive metrics (9 Gen metrics available in base generation files)
# Judge metrics AND Aggregate Score are NULL until judging happens
CI_METRICS = [
    'Gen Perplexity',
    'Gen Min Prob',
    'Gen Max Window Entropy',
    'Gen Mean Token Prob',
    'Gen Prob Variance',
    'Gen Prob Std',
    'Gen Max Prob Diff',
    'Gen Token Prob Slope',
    'Gen Cosine Similarity'
]

def load_and_prepare_data():
    """Load all base CLDs and prepare for classification."""
    print("=" * 80)
    print("LOADING ORIGINAL BASE CLDs (WITH CI METRICS)")
    print("=" * 80)
    
    all_data = []
    
    for cld_name, file_path in INPUT_FILES.items():
        print(f"\nLoading: {cld_name}")
        print(f"  File: {file_path.name}")
        
        # Load the "All Edges" sheet
        df = pd.read_excel(file_path, sheet_name='All Edges')
        df['cld_name'] = cld_name
        
        print(f"  Total edges: {len(df)}")
        
        # Show classification breakdown
        if 'Classification' in df.columns:
            class_counts = df['Classification'].value_counts()
            for cls, count in class_counts.items():
                print(f"    {cls}: {count}")
        
        # Check CI metrics availability
        ci_available = sum(1 for metric in CI_METRICS if metric in df.columns)
        print(f"  CI metrics: {ci_available}/{len(CI_METRICS)} columns present")
        
        all_data.append(df)
    
    # Combine all CLDs
    combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\n{'='*80}")
    print(f"COMBINED DATA")
    print(f"{'='*80}")
    print(f"Total edges across all CLDs: {len(combined)}")
    
    # Filter to only TP and FP (edges that were generated)
    df_generated = combined[combined['Classification'].isin(['TP', 'FP'])].copy()
    
    # Create binary label: 1 = hallucination (FP), 0 = correct (TP)
    df_generated['is_hallucination'] = (df_generated['Classification'] == 'FP').astype(int)
    
    print(f"\nGenerated edges (TP + FP): {len(df_generated)}")
    print(f"  ✅ True Positives (correct): {(df_generated['is_hallucination'] == 0).sum()}")
    print(f"  ❌ False Positives (hallucinations): {(df_generated['is_hallucination'] == 1).sum()}")
    print(f"  📊 Class balance: {df_generated['is_hallucination'].mean():.1%} hallucinations")
    
    return df_generated

def prepare_features(df):
    """Prepare feature matrix X and labels y."""
    print(f"\n{'='*80}")
    print("PREPARING FEATURES")
    print(f"{'='*80}")
    
    # Select only rows with all metrics available
    feature_cols = [col for col in CI_METRICS if col in df.columns]
    print(f"Available CI metrics: {len(feature_cols)}/{len(CI_METRICS)}")
    
    if len(feature_cols) < len(CI_METRICS):
        missing = set(CI_METRICS) - set(feature_cols)
        print(f"  ⚠️  Missing metrics: {missing}")
    
    df_clean = df[feature_cols + ['is_hallucination', 'cld_name']].copy()
    
    # Replace inf values with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with extreme values (Judge Perplexity has overflow issues)
    for col in feature_cols:
        if 'Perplexity' in col:
            before = len(df_clean)
            df_clean = df_clean[df_clean[col] < 10]
            removed = before - len(df_clean)
            if removed > 0:
                print(f"  Removed {removed} rows with {col} > 10")
    
    # Remove any remaining rows with NaN
    before_dropna = len(df_clean)
    df_clean = df_clean.dropna()
    removed_na = before_dropna - len(df_clean)
    if removed_na > 0:
        print(f"  Removed {removed_na} rows with missing values")
    
    print(f"\nFinal samples for training: {len(df_clean)}")
    
    X = df_clean[feature_cols].values
    y = df_clean['is_hallucination'].values
    cld_names = df_clean['cld_name'].values
    
    # Show feature statistics
    print(f"\nFeature ranges:")
    for i, col in enumerate(feature_cols):
        print(f"  {col:30s}: [{X[:, i].min():.4f}, {X[:, i].max():.4f}] (mean={X[:, i].mean():.4f})")
    
    return X, y, feature_cols, cld_names, df_clean

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
        'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
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
        
        # Feature importance (if available)
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            print(f"\n  Top 3 features:")
            for idx in np.argsort(importances)[::-1][:3]:
                print(f"    {feature_names[idx]}: {importances[idx]:.4f}")
        elif hasattr(clf, 'coef_'):
            coefs = np.abs(clf.coef_[0])
            print(f"\n  Top 3 features (by |coefficient|):")
            for idx in np.argsort(coefs)[::-1][:3]:
                print(f"    {feature_names[idx]}: {coefs[idx]:.4f}")
        
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
    
    return results, scaler

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
    plt.title('ROC Curves: Ensemble Classifiers on Original Base CLDs', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'ensemble_roc_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved ROC curves: {output_path}")

def generate_report(results, output_dir, df_stats, num_metrics):
    """Generate markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Find best classifier
    best_clf = max(results.items(), key=lambda x: x[1]['test_auc'])
    
    report = f"""# Ensemble Classifier Report: Original Base CLDs

**Generated:** {timestamp}  
**Data Source:** Original base generation CLDs (Oct 12, 2025)  
**CI Metrics Used:** {num_metrics}/9

## Executive Summary

**Best Classifier:** {best_clf[0]}  
**Test AUC:** {best_clf[1]['test_auc']:.4f}  
**Cross-Validation AUC:** {best_clf[1]['cv_auc_mean']:.4f} (±{best_clf[1]['cv_auc_std']:.4f})  

---

## Dataset

**CLDs Analyzed:**
- Social Norms & Obesity Prevalence
- Depressive Symptoms in Response to a Stressor
- Older Persons Emergency Department Visits

**Samples:**
- Total generated edges (TP + FP): {len(df_stats)}
- True Positives (correct): {(df_stats['is_hallucination'] == 0).sum()}
- False Positives (hallucinations): {(df_stats['is_hallucination'] == 1).sum()}
- Class balance: {df_stats['is_hallucination'].mean():.1%} hallucinations

---

## Classifier Comparison

| Classifier | CV AUC | Test AUC | Precision | Recall | F1 Score |
|------------|--------|----------|-----------|--------|----------|
"""
    
    for clf_name, result in results.items():
        report += f"| {clf_name} | {result['cv_auc_mean']:.4f} (±{result['cv_auc_std']:.4f}) | {result['test_auc']:.4f} | {result['test_precision']:.4f} | {result['test_recall']:.4f} | {result['test_f1']:.4f} |\n"
    
    report += f"""
---

## Interpretation

### Key Findings:

1. **Best performing classifier:** {best_clf[0]} with AUC = {best_clf[1]['test_auc']:.4f}
2. **Discrimination ability:** {"EXCELLENT" if best_clf[1]['test_auc'] > 0.80 else "Good" if best_clf[1]['test_auc'] > 0.70 else "Moderate" if best_clf[1]['test_auc'] > 0.60 else "Weak"}
3. **Precision-Recall tradeoff:** {best_clf[1]['test_precision']:.1%} precision, {best_clf[1]['test_recall']:.1%} recall

### Comparison with Previous Results:

**Previous ensemble classifier** (from 6-run experiment):
- Best AUC: 0.7250 (Logistic Regression)
- Dataset: 6 runs across 3 CLDs

**Current results** (original base CLDs):
- Best AUC: {best_clf[1]['test_auc']:.4f} ({best_clf[0]})
- Dataset: Single generation per CLD (3 CLDs)
- **All {num_metrics} CI metrics used!**

### Significance:

This analysis uses the **ORIGINAL base generation files** which already contained all CI metrics from the generation phase. No separate judging was needed for CI metric calculation.

---

## Files

- `ensemble_roc_curves.png` - ROC curve comparison
- Input Excel files copied to this directory

---

*Report generated by run_rq2_on_original_base_clds.py*
"""
    
    report_path = output_dir / 'ensemble_classifier_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved report: {report_path}")

def main():
    """Run the analysis pipeline."""
    print("\n" + "="*80)
    print("RQ2 ANALYSIS: ORIGINAL BASE CLDs (WITH CI METRICS)")
    print("="*80 + "\n")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"parameter_tuning_experiments/rq2_analyses/rq2_original_base_clds_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}\n")
    
    # Load data
    df = load_and_prepare_data()
    
    # Prepare features
    X, y, feature_names, cld_names, df_clean = prepare_features(df)
    
    # Train classifiers
    results, scaler = train_classifiers(X, y, feature_names)
    
    # Plot ROC curves
    plot_roc_curves(results, output_dir)
    
    # Generate report
    generate_report(results, output_dir, df_clean, len(feature_names))
    
    # Copy input Excel files to output directory
    print(f"\nCopying input files to output directory...")
    excel_dir = output_dir / 'input_excel_files'
    excel_dir.mkdir(exist_ok=True)
    
    for cld_name, file_path in INPUT_FILES.items():
        dest = excel_dir / file_path.name
        shutil.copy2(file_path, dest)
        print(f"  ✓ Copied: {file_path.name}")
    
    # Save results summary as JSON
    results_summary = {
        'timestamp': timestamp,
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
        },
        'dataset': {
            'total_samples': int(len(df_clean)),
            'tp_count': int((df_clean['is_hallucination'] == 0).sum()),
            'fp_count': int((df_clean['is_hallucination'] == 1).sum()),
            'hallucination_rate': float(df_clean['is_hallucination'].mean())
        },
        'ci_metrics_used': len(feature_names)
    }
    
    import json
    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nFiles generated:")
    print(f"  • ensemble_classifier_report.md")
    print(f"  • ensemble_roc_curves.png")
    print(f"  • results_summary.json")
    print(f"  • input_excel_files/ (3 Excel files)")

if __name__ == "__main__":
    main()





















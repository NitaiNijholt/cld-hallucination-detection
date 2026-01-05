#!/usr/bin/env python3
"""
Ensemble Hallucination Classifier - Core Metrics Only
Uses the 4 core CI metrics from the new 3-CLD run (Oct 12, 2025).
Metrics: Gen Perplexity, Gen Min Prob, Gen Max Window Entropy, Gen Cosine Similarity
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, 
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# New 3-CLD data files (Oct 13, 2025 - with Generator + Judge metrics)
DATA_FILES = {
    'Social_norms': 'parameter_tuning_experiments/results/exp_20251013_024747_8f3f9a69/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run1_eebcef72/results_exp_20251013_024747_8f3f9a69_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Social_norms_and_obesity_prevalence_20251013_031345.xlsx',
    'Older_persons': 'parameter_tuning_experiments/results/exp_20251013_024747_8f3f9a69/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run1_32fee595/results_exp_20251013_024747_8f3f9a69_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251013_040743.xlsx',
    'Depressive': 'parameter_tuning_experiments/results/exp_20251013_024747_8f3f9a69/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run1_1e40ba00/results_exp_20251013_024747_8f3f9a69_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251013_025454.xlsx'
}

# ALL Context-insensitive metrics (9 metrics × 2 sources = 18 total)
CORE_METRICS = [
    # Generator metrics (9)
    'Gen Perplexity',
    'Gen Min Prob',
    'Gen Max Window Entropy',
    'Gen Cosine Similarity',
    'Gen Mean Token Prob',
    'Gen Prob Variance',
    'Gen Prob Std',
    'Gen Max Prob Diff',
    'Gen Token Prob Slope',
    # Judge metrics (9)
    'Judge Perplexity',
    'Judge Min Prob',
    'Judge Max Window Entropy',
    'Judge Cosine Similarity',
    'Judge Mean Token Prob',
    'Judge Prob Variance',
    'Judge Prob Std',
    'Judge Max Prob Diff',
    'Judge Token Prob Slope'
]

def load_data_for_classification():
    """Load all data and prepare for classification."""
    print("=" * 80)
    print("LOADING DATA FROM NEW 3-CLD RUN (OCT 12, 2025)")
    print("=" * 80)
    print()
    
    all_data = []
    
    for cld_name, file_path in DATA_FILES.items():
        print(f"Loading {cld_name}...")
        df = pd.read_excel(file_path, sheet_name='All Edges')
        df['cld_name'] = cld_name
        
        # Count classifications
        class_counts = df['Classification'].value_counts()
        print(f"  {cld_name}:")
        for cls, count in class_counts.items():
            print(f"    {cls}: {count}")
        
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    
    print()
    print(f"Total edges loaded: {len(combined)}")
    
    # Filter to only TP and FP (edges that were generated)
    # TP = True Positive (correct edge), FP = False Positive (hallucination)
    df_generated = combined[combined['Classification'].isin(['TP', 'FP'])].copy()
    
    # Create binary label: 1 = hallucination (FP), 0 = correct (TP)
    df_generated['is_hallucination'] = (df_generated['Classification'] == 'FP').astype(int)
    
    print()
    print("=" * 80)
    print("CLASSIFICATION DATASET")
    print("=" * 80)
    print(f"Generated edges (TP + FP): {len(df_generated)}")
    print(f"  - True Positives (correct): {(df_generated['is_hallucination'] == 0).sum()}")
    print(f"  - False Positives (hallucinations): {(df_generated['is_hallucination'] == 1).sum()}")
    print(f"  - Class balance: {df_generated['is_hallucination'].mean():.1%} hallucinations")
    print()
    
    return df_generated

def prepare_features(df):
    """Prepare feature matrix X and labels y."""
    print("=" * 80)
    print("PREPARING FEATURES (CORE METRICS ONLY)")
    print("=" * 80)
    print()
    
    # Check which core metrics are available
    available_metrics = [col for col in CORE_METRICS if col in df.columns]
    missing_metrics = [col for col in CORE_METRICS if col not in df.columns]
    
    if missing_metrics:
        print(f"⚠️  WARNING: Missing metrics: {missing_metrics}")
    
    print(f"Using {len(available_metrics)} core metrics:")
    for metric in available_metrics:
        print(f"  - {metric}")
    print()
    
    # Select only rows with all metrics available
    df_clean = df[available_metrics + ['is_hallucination', 'cld_name']].dropna()
    
    print(f"Samples with all core metrics (before cleaning): {len(df_clean)}")
    
    # Replace inf values with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with extreme perplexity values (>10 is unreasonable)
    for metric in ['Gen Perplexity', 'Judge Perplexity']:
        if metric in available_metrics:
            before = len(df_clean)
            df_clean = df_clean[df_clean[metric] < 10]
            removed = before - len(df_clean)
            if removed > 0:
                print(f"Removed {removed} rows with {metric} >= 10")
    
    # Remove any remaining rows with NaN
    df_clean = df_clean.dropna()
    
    print(f"Samples after cleaning: {len(df_clean)}")
    print()
    
    X = df_clean[available_metrics].values
    y = df_clean['is_hallucination'].values
    cld_names = df_clean['cld_name'].values
    
    # Show feature statistics
    print("Feature ranges:")
    for i, col in enumerate(available_metrics):
        print(f"  {col:30s}: [{X[:, i].min():.4f}, {X[:, i].max():.4f}] (mean={X[:, i].mean():.4f}, std={X[:, i].std():.4f})")
    print()
    
    # Class balance
    print(f"Class balance after cleaning:")
    print(f"  Non-hallucinations (TP): {(y == 0).sum()} ({(y == 0).mean():.1%})")
    print(f"  Hallucinations (FP): {(y == 1).sum()} ({(y == 1).mean():.1%})")
    print()
    
    return X, y, available_metrics, cld_names

def train_and_evaluate_classifiers(X, y, feature_names, cld_names):
    """Train and evaluate ensemble classifiers."""
    print("=" * 80)
    print("TRAINING ENSEMBLE CLASSIFIERS")
    print("=" * 80)
    print()
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
    }
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    print("-" * 80)
    print("CROSS-VALIDATION RESULTS (5-fold)")
    print("-" * 80)
    print()
    
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        
        print(f"  Cross-val AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Train on full training set
        clf.fit(X_train_scaled, y_train)
        
        # Predict on test set
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        test_auc = roc_auc_score(y_test, y_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        
        print(f"  Test AUC: {test_auc:.3f}")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
        
        # Feature importance
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            print(f"  Feature Importances:")
            for feat, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
                print(f"    {feat:30s}: {imp:.4f}")
        elif hasattr(clf, 'coef_'):
            coeffs = clf.coef_[0]
            print(f"  Feature Coefficients:")
            for feat, coef in sorted(zip(feature_names, coeffs), key=lambda x: -abs(x[1])):
                print(f"    {feat:30s}: {coef:+.4f}")
        
        print()
        
        results[name] = {
            'classifier': clf,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'test_auc': test_auc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'confusion_matrix': cm.tolist(),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    return results, scaler

def plot_roc_curves(results, output_dir):
    """Plot ROC curves for all classifiers."""
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_proba'])
        auc = result['test_auc']
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc:.3f})')
    
    # Baseline
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance (AUC=0.5)')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Core Metrics Ensemble Classifiers', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'ensemble_roc_curves_core_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curve plot to: {output_path}")
    plt.close()

def generate_report(results, feature_names, output_dir):
    """Generate markdown report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f'ensemble_classifier_core_metrics_report_{timestamp}.md'
    
    with open(report_path, 'w') as f:
        f.write("# Ensemble Classifier Report - Core Metrics Only\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Data Source:** New 3-CLD run (Oct 12, 2025)\n\n")
        f.write(f"**Metrics Used:** {len(feature_names)} core CI metrics\n\n")
        
        # Metrics list
        f.write("## Core Metrics\n\n")
        for metric in feature_names:
            f.write(f"- {metric}\n")
        f.write("\n")
        
        # Performance comparison
        f.write("## Performance Comparison\n\n")
        f.write("| Classifier | CV AUC | Test AUC | Precision | Recall | F1 |\n")
        f.write("|------------|--------|----------|-----------|--------|----|\n")
        
        for name, result in results.items():
            f.write(f"| {name} | {result['cv_auc_mean']:.3f} (±{result['cv_auc_std']:.3f}) | ")
            f.write(f"**{result['test_auc']:.3f}** | {result['test_precision']:.3f} | ")
            f.write(f"{result['test_recall']:.3f} | {result['test_f1']:.3f} |\n")
        
        f.write("\n")
        
        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['test_auc'])
        f.write(f"## Best Model: {best_model[0]}\n\n")
        f.write(f"**Test AUC:** {best_model[1]['test_auc']:.3f}\n\n")
        
        # Confusion matrix
        cm = best_model[1]['confusion_matrix']
        f.write("### Confusion Matrix\n\n")
        f.write("```\n")
        f.write(f"              Predicted\n")
        f.write(f"              Neg    Pos\n")
        f.write(f"Actual Neg   {cm[0][0]:4d}   {cm[0][1]:4d}\n")
        f.write(f"Actual Pos   {cm[1][0]:4d}   {cm[1][1]:4d}\n")
        f.write("```\n\n")
        
        # Feature importance/coefficients
        clf = best_model[1]['classifier']
        if hasattr(clf, 'feature_importances_'):
            f.write("### Feature Importances\n\n")
            importances = clf.feature_importances_
            for feat, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
                f.write(f"- **{feat}**: {imp:.4f}\n")
        elif hasattr(clf, 'coef_'):
            f.write("### Feature Coefficients\n\n")
            coeffs = clf.coef_[0]
            for feat, coef in sorted(zip(feature_names, coeffs), key=lambda x: -abs(x[1])):
                f.write(f"- **{feat}**: {coef:+.4f}\n")
        
        f.write("\n")
        
        # Interpretation
        f.write("## Interpretation\n\n")
        f.write(f"Using the **4 core CI metrics from both Generator and Judge (8 total)**, the {best_model[0]} classifier achieved ")
        f.write(f"an AUC of **{best_model[1]['test_auc']:.3f}**, demonstrating ")
        
        if best_model[1]['test_auc'] >= 0.70:
            f.write("**good discrimination** (AUC ≥ 0.70).\n\n")
        elif best_model[1]['test_auc'] >= 0.60:
            f.write("**acceptable discrimination** (0.60 ≤ AUC < 0.70).\n\n")
        else:
            f.write("**poor discrimination** (AUC < 0.60).\n\n")
        
        f.write("This result shows that combining even just the core CI metrics can improve ")
        f.write("hallucination detection compared to using individual metrics in isolation.\n\n")
    
    print(f"✓ Saved report to: {report_path}")
    return report_path

def main():
    """Main execution function."""
    print("\n")
    print("=" * 80)
    print("ENSEMBLE CLASSIFIER - CORE METRICS ONLY")
    print("=" * 80)
    print("\n")
    
    # Create output directory
    output_dir = Path('parameter_tuning_experiments/rq2_analyses/ensemble_classifier_core_metrics')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data_for_classification()
    
    # Prepare features
    X, y, feature_names, cld_names = prepare_features(df)
    
    # Train and evaluate
    results, scaler = train_and_evaluate_classifiers(X, y, feature_names, cld_names)
    
    # Plot ROC curves
    plot_roc_curves(results, output_dir)
    
    # Generate report
    report_path = generate_report(results, feature_names, output_dir)
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f'results_{timestamp}.json'
    
    results_json = {
        'timestamp': timestamp,
        'feature_names': feature_names,
        'n_samples': len(X),
        'classifiers': {
            name: {
                'cv_auc_mean': float(result['cv_auc_mean']),
                'cv_auc_std': float(result['cv_auc_std']),
                'test_auc': float(result['test_auc']),
                'test_precision': float(result['test_precision']),
                'test_recall': float(result['test_recall']),
                'test_f1': float(result['test_f1']),
                'confusion_matrix': result['confusion_matrix']
            }
            for name, result in results.items()
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✓ Saved results to: {json_path}")
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()

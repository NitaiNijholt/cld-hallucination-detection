#!/usr/bin/env python3
"""
Ensemble Classifier: TP vs FP Only (Reproduce 0.7250 AUC)

Goal: Run the same classification task as the original 0.7250 AUC experiment:
- Use ONLY generated edges (TP vs FP)
- Exclude False Negatives (FN) and True Negatives (TN)
- Question: "Is this generated edge a hallucination?"

This should allow us to reproduce the 0.7250 AUC on the recent dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Data source
BASE_DIR = Path("parameter_tuning_experiments/results/rq1_base_judging_correctness_20251012_225925")

JUDGED_FILES = [
    BASE_DIR / "judged_Social_norms_and_obesity_prevalence_citations_20251012_230158.xlsx",
    BASE_DIR / "judged_older_persons_emergency_department_visits_citations_20251012_233401.xlsx",
    BASE_DIR / "judged_Depressive_symptoms_in_response_to_a_stressor_citations_20251013_001809.xlsx",
]

# Context-Insensitive Metrics (same 9 as original 0.7250 experiment)
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

def load_and_combine_data():
    """Load and combine all judged Excel files."""
    all_dfs = []
    for file_path in JUDGED_FILES:
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        print(f"Loading: {file_path.name}")
        df = pd.read_excel(file_path, sheet_name="All Edges")
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n✓ Loaded {len(combined)} total edges from {len(all_dfs)} files")
    return combined

def main():
    print("=" * 80)
    print("ENSEMBLE CLASSIFIER: TP vs FP ONLY (Reproduce 0.7250 AUC)")
    print("=" * 80)
    print("Task: Classify generated edges only (TP vs FP)")
    print("Question: 'Is this generated edge a hallucination?'\n")
    
    # Load data
    print("Loading data files...")
    combined = load_and_combine_data()
    
    # Filter to ONLY generated edges (TP + FP)
    # Exclude False Negatives (FN) and True Negatives (TN)
    print("\n" + "=" * 80)
    print("FILTERING TO GENERATED EDGES ONLY")
    print("=" * 80)
    
    print(f"\nOriginal dataset breakdown:")
    print(combined['Classification'].value_counts().sort_index())
    
    # Keep only TP and FP (edges that were actually generated)
    df_generated = combined[combined['Classification'].isin(['TP', 'FP'])].copy()
    
    print(f"\n✓ Filtered to {len(df_generated)} generated edges (TP + FP)")
    print(f"  - True Positives (correct): {(df_generated['Classification'] == 'TP').sum()}")
    print(f"  - False Positives (hallucinations): {(df_generated['Classification'] == 'FP').sum()}")
    
    # Create binary label: 1 = Hallucination (FP), 0 = Correct (TP)
    df_generated['is_hallucination'] = (df_generated['Classification'] == 'FP').astype(int)
    # Class 0: TP (True Positive) = CORRECT edge (should be in CLD)
    # Class 1: FP (False Positive) = HALLUCINATION (LLM made it up)
    
    hallucination_rate = df_generated['is_hallucination'].mean()
    print(f"  - Class balance: {hallucination_rate*100:.1f}% hallucinations")
    
    # Prepare features
    print("\nPreparing features...")
    print(f"Using {len(CI_METRICS)} CI metrics (same as original 0.7250 experiment)")
    
    # Check which metrics are available
    available_metrics = [m for m in CI_METRICS if m in df_generated.columns]
    missing_metrics = [m for m in CI_METRICS if m not in df_generated.columns]
    
    if missing_metrics:
        print(f"⚠️  Missing metrics: {missing_metrics}")
        print(f"✓ Available metrics: {len(available_metrics)}")
        CI_METRICS_FINAL = available_metrics
    else:
        print(f"✓ All {len(CI_METRICS)} metrics available")
        CI_METRICS_FINAL = CI_METRICS
    
    # Filter to samples with all metrics
    X = df_generated[CI_METRICS_FINAL].copy()
    y = df_generated['is_hallucination'].copy()
    
    # Check for missing values
    print(f"Samples with all metrics (before cleaning): {len(X) - X.isnull().any(axis=1).sum()}")
    
    # Remove rows with any NaN values
    mask_complete = ~X.isnull().any(axis=1)
    X = X[mask_complete]
    y = y[mask_complete]
    
    # Remove inf values
    mask_finite = np.isfinite(X).all(axis=1)
    X = X[mask_finite]
    y = y[mask_finite]
    
    # Remove extreme perplexity values (same as original)
    perplexity_cols = [col for col in X.columns if 'Perplexity' in col]
    for col in perplexity_cols:
        mask_reasonable = X[col] < 10  # Same threshold as original
        X = X[mask_reasonable]
        y = y[mask_reasonable]
    
    print(f"Samples after cleaning inf/extreme values: {len(X)}")
    
    if len(X) < 50:
        print("\n❌ ERROR: Not enough samples after cleaning. Need at least 50.")
        return
    
    # Print feature ranges
    print("\nFeature ranges (after cleaning):")
    for col in X.columns:
        print(f"  {col:30s}: [{X[col].min():.4f}, {X[col].max():.4f}]")
    
    # Print class distribution
    print(f"\nFinal class distribution:")
    print(f"  Class 0 (TP - Correct): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"  Class 1 (FP - Hallucination): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
    print(f"  Imbalance ratio: {(y == 1).sum() / (y == 0).sum():.2f}:1")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train classifiers
    print("\n" + "=" * 80)
    print("TRAINING ENSEMBLE CLASSIFIERS")
    print("=" * 80)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    all_fpr = {}
    all_tpr = {}
    all_auc = {}
    
    for name, model in models.items():
        print(f"\n{'-'*80}")
        print(f"Training: {name}")
        print(f"{'-'*80}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"Cross-validation AUC scores: {cv_scores}")
        print(f"Mean CV AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Train on full training set
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\nTest Set Performance:")
        print(f"  AUC:       {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:3d}  FP: {cm[0,1]:3d}")
        print(f"  FN: {cm[1,0]:3d}  TP: {cm[1,1]:3d}")
        
        # Feature importance/coefficients
        if hasattr(model, 'coef_'):
            print(f"\nFeature Coefficients (magnitude = importance):")
            coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
            for feat, coef in coefs.items():
                sign = '+' if coef >= 0 else ''
                print(f"  {feat:30s}: {sign}{coef:.4f}")
        elif hasattr(model, 'feature_importances_'):
            print(f"\nFeature Importance:")
            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            for feat, imp in importances.items():
                print(f"  {feat:30s}: {imp:.4f}")
        
        # Store results
        results[name] = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        # Store ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        all_fpr[name] = fpr
        all_tpr[name] = tpr
        all_auc[name] = auc
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for name in models.keys():
        plt.plot(all_fpr[name], all_tpr[name], linewidth=2, 
                label=f'{name} (AUC = {all_auc[name]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Hallucination Detection (TP vs FP Only)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save results
    output_dir = BASE_DIR / "ensemble_analysis_tp_vs_fp"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.savefig(output_dir / "roc_curves_tp_vs_fp.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved ROC curves: {output_dir / 'roc_curves_tp_vs_fp.png'}")
    
    # Save JSON results
    results_json = {
        'experiment': 'TP vs FP Only (Reproduce 0.7250 AUC)',
        'timestamp': timestamp,
        'dataset': {
            'total_generated_edges': len(df_generated),
            'clean_samples': len(X),
            'num_features': len(CI_METRICS_FINAL),
            'features_used': CI_METRICS_FINAL,
            'class_distribution': {
                'TP (correct)': int((y == 0).sum()),
                'FP (hallucination)': int((y == 1).sum()),
                'hallucination_rate': float(y.mean())
            },
            'train_size': len(X_train),
            'test_size': len(X_test)
        },
        'models': results
    }
    
    with open(output_dir / f"ensemble_results_tp_vs_fp_{timestamp}.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✓ Saved results: {output_dir / f'ensemble_results_tp_vs_fp_{timestamp}.json'}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("✅ ENSEMBLE CLASSIFIER TRAINING COMPLETE!")
    print("=" * 80)
    
    best_model = max(results.items(), key=lambda x: x[1]['auc'])
    print(f"\n🏆 Best Model: {best_model[0]}")
    print(f"   Test AUC: {best_model[1]['auc']:.4f}")
    print(f"   Precision: {best_model[1]['precision']:.4f}")
    print(f"   Recall: {best_model[1]['recall']:.4f}")
    print(f"   F1 Score: {best_model[1]['f1']:.4f}")
    
    print(f"\n📊 Results saved to: {output_dir}/")
    print(f"   - ROC Curves: roc_curves_tp_vs_fp.png")
    print(f"   - Results: ensemble_results_tp_vs_fp_{timestamp}.json")
    
    # Compare to original 0.7250 result
    print("\n" + "=" * 80)
    print("COMPARISON TO ORIGINAL 0.7250 AUC EXPERIMENT")
    print("=" * 80)
    print("\nOriginal (6 CLDs, 136 samples):")
    print("  - Logistic Regression AUC: 0.7250")
    print("  - Precision: 87.5%, Recall: 70%, F1: 77.8%")
    print("  - Class balance: 78.4% hallucinations")
    
    print(f"\nCurrent (3 CLDs, {len(X)} samples):")
    print(f"  - Logistic Regression AUC: {results['Logistic Regression']['auc']:.4f}")
    print(f"  - Precision: {results['Logistic Regression']['precision']*100:.1f}%, " + 
          f"Recall: {results['Logistic Regression']['recall']*100:.1f}%, " +
          f"F1: {results['Logistic Regression']['f1']*100:.1f}%")
    print(f"  - Class balance: {y.mean()*100:.1f}% hallucinations")
    
    auc_diff = results['Logistic Regression']['auc'] - 0.7250
    if abs(auc_diff) < 0.05:
        print(f"\n✅ REPRODUCTION SUCCESSFUL! AUC difference: {auc_diff:+.4f}")
    else:
        print(f"\n⚠️  AUC difference: {auc_diff:+.4f} (>0.05)")

if __name__ == "__main__":
    main()

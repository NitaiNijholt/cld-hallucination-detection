#!/usr/bin/env python3
"""
Train Classifier on Run2 Only (3 CLDs)

Test if using only run2 samples (which had better Judge metric coverage)
can reproduce the 0.7250 AUC result.

From data loss analysis:
- Depressive run1: 0% retention (all Judge metrics missing)
- Depressive run2: 91.8% retention ✅
- Social Norms run1: 69.2% retention
- Social Norms run2: 92.9% retention ✅
- Older Persons run1: 0.7% retention (all Judge metrics missing)
- Older Persons run2: 45.6% retention ✅

Using only run2 = better data quality
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

# Run2 only (3 CLDs with better Judge metric coverage)
RUN2_FILES = {
    'Depressive_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run2_0d57ed9a/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251008_200816.xlsx',
    'Social_norms_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run2_d09234b7/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Social_norms_and_obesity_prevalence_20251008_221139.xlsx',
    'Older_persons_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run2_96605199/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251009_082149.xlsx'
}

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

def load_run2_data():
    """Load run2 data only."""
    all_dfs = []
    for name, filepath in RUN2_FILES.items():
        try:
            df = pd.read_excel(filepath, sheet_name="All Edges")
            all_dfs.append(df)
            print(f"  ✓ Loaded {name}")
        except Exception as e:
            print(f"  ⚠️  Could not load {name}: {e}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    df_generated = combined[combined['Classification'].isin(['TP', 'FP'])].copy()
    df_generated['is_hallucination'] = (df_generated['Classification'] == 'FP').astype(int)
    return df_generated

def clean_data(df):
    """Apply same cleaning as original classifier."""
    X = df[CI_METRICS].copy()
    y = df['is_hallucination'].copy()
    
    initial = len(X)
    
    # Remove rows with NaN
    mask_complete = ~X.isnull().any(axis=1)
    X = X[mask_complete]
    y = y[mask_complete]
    after_nan = len(X)
    
    # Remove inf values
    mask_finite = np.isfinite(X).all(axis=1)
    X = X[mask_finite]
    y = y[mask_finite]
    after_inf = len(X)
    
    # Remove extreme perplexity values
    perplexity_cols = [col for col in X.columns if 'Perplexity' in col]
    for col in perplexity_cols:
        mask_reasonable = X[col] < 10
        X = X[mask_reasonable]
        y = y[mask_reasonable]
    
    after_perplexity = len(X)
    
    print(f"\n  Data cleaning:")
    print(f"    Initial: {initial}")
    print(f"    After NaN removal: {after_nan} (lost {initial - after_nan})")
    print(f"    After inf removal: {after_inf} (lost {after_nan - after_inf})")
    print(f"    After extreme perplexity: {after_perplexity} (lost {after_inf - after_perplexity})")
    print(f"    Retention: {after_perplexity}/{initial} = {after_perplexity/initial*100:.1f}%")
    
    return X, y

def main():
    print("=" * 80)
    print("CLASSIFIER TRAINING: RUN2 ONLY (3 CLDs)")
    print("=" * 80)
    print("Testing if better Judge metric coverage improves AUC\n")
    
    # Load data
    print("Loading run2 data (3 CLDs)...")
    df = load_run2_data()
    print(f"✓ Loaded {len(df)} generated edges (TP + FP)")
    
    print("\nCleaning data (9 features: Gen + Judge + Aggregate)...")
    X, y = clean_data(df)
    
    if len(X) < 50:
        print(f"\n❌ Only {len(X)} samples - too few for reliable training!")
        return
    
    print(f"\n✓ Clean samples: {len(X)}")
    print(f"  - TP (correct): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"  - FP (hallucination): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
    
    # Split data
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'-'*80}")
        print(f"Training: {name}")
        print(f"{'-'*80}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"Cross-validation AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Test
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
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
        
        # Feature importance
        if hasattr(model, 'coef_'):
            print(f"\nFeature Coefficients:")
            coefs = pd.Series(model.coef_[0], index=CI_METRICS).sort_values(key=abs, ascending=False)
            for feat, coef in coefs.items():
                sign = '+' if coef >= 0 else ''
                print(f"  {feat:30s}: {sign}{coef:.4f}")
        
        results[name] = {
            'cv_auc': cv_scores.mean(),
            'test_auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
    
    # Save results
    output_dir = Path("parameter_tuning_experiments/rq2_analyses/run2_only")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_dir / f"run2_only_results_{timestamp}.json", 'w') as f:
        json.dump({
            'experiment': 'Run2 Only (3 CLDs)',
            'timestamp': timestamp,
            'dataset': {
                'initial_edges': len(df),
                'clean_samples': len(X),
                'retention': f"{len(X)/len(df)*100:.1f}%"
            },
            'results': results
        }, f, indent=2)
    
    # Comparison
    print("\n" + "=" * 80)
    print("📊 COMPARISON: RUN2-ONLY vs ORIGINAL (6 CLDs)")
    print("=" * 80)
    
    print("\nOriginal (6 CLDs: run1 + run2):")
    print("  - Total edges: 407")
    print("  - Clean samples: 136 (33.4% retention)")
    print("  - Logistic Regression AUC: 0.7250")
    print("  - Issue: run1 had missing Judge metrics")
    
    best_model = max(results.items(), key=lambda x: x[1]['test_auc'])
    print(f"\nRun2 Only (3 CLDs: run2 only):")
    print(f"  - Total edges: {len(df)}")
    print(f"  - Clean samples: {len(X)} ({len(X)/len(df)*100:.1f}% retention)")
    print(f"  - Best Model: {best_model[0]}")
    print(f"  - Test AUC: {best_model[1]['test_auc']:.4f}")
    print(f"  - Precision: {best_model[1]['precision']:.1%}, Recall: {best_model[1]['recall']:.1%}")
    
    auc_diff = best_model[1]['test_auc'] - 0.7250
    retention_diff = (len(X)/len(df)*100) - 33.4
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    
    if abs(auc_diff) < 0.05:
        print(f"\n✅ AUC COMPARABLE! (diff: {auc_diff:+.4f})")
        print("   Run2-only data has similar performance to full 6 CLDs")
    elif auc_diff > 0:
        print(f"\n🎉 AUC BETTER! (diff: {auc_diff:+.4f})")
        print("   Cleaner data (run2 only) improves performance")
    else:
        print(f"\n⚠️  AUC LOWER (diff: {auc_diff:+.4f})")
        print("   Less diversity (3 runs vs 6) may hurt generalization")
    
    if retention_diff > 10:
        print(f"✅ Better retention (+{retention_diff:.1f}%)")
    
    print(f"\nKey insight: Using only run2 (better Judge coverage) shows whether")
    print(f"the 0.7250 AUC was due to data quality or just having more runs.")

if __name__ == "__main__":
    main()

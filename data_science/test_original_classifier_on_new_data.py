#!/usr/bin/env python3
"""
Cross-Experiment Generalization Test

Train classifier on ORIGINAL dataset (6 CLDs, 0.7250 AUC)
Test on RECENT dataset (3 CLDs, different generation parameters)

This tests whether the classifier generalizes across different experimental setups.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Original data (TRAIN)
ORIGINAL_FILES = {
    'Depressive_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run1_7a66e382/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251008_181257.xlsx',
    'Depressive_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run2_0d57ed9a/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251008_200816.xlsx',
    'Social_norms_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run1_86b016e1/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Social_norms_and_obesity_prevalence_20251008_213554.xlsx',
    'Social_norms_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run2_d09234b7/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Social_norms_and_obesity_prevalence_20251008_221139.xlsx',
    'Older_persons_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run1_d45241a5/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251008_230700.xlsx',
    'Older_persons_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run2_96605199/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251009_082149.xlsx'
}

# Recent data (TEST)
RECENT_DIR = Path("parameter_tuning_experiments/results/rq1_base_judging_correctness_20251012_225925")
RECENT_FILES = [
    RECENT_DIR / "judged_Social_norms_and_obesity_prevalence_citations_20251012_230158.xlsx",
    RECENT_DIR / "judged_older_persons_emergency_department_visits_citations_20251012_233401.xlsx",
    RECENT_DIR / "judged_Depressive_symptoms_in_response_to_a_stressor_citations_20251013_001809.xlsx",
]

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

def load_original_data():
    """Load original data (TRAIN)."""
    all_dfs = []
    for name, filepath in ORIGINAL_FILES.items():
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

def load_recent_data():
    """Load recent data (TEST)."""
    all_dfs = []
    for filepath in RECENT_FILES:
        df = pd.read_excel(filepath, sheet_name="All Edges")
        all_dfs.append(df)
        print(f"  ✓ Loaded {filepath.name}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    df_generated = combined[combined['Classification'].isin(['TP', 'FP'])].copy()
    df_generated['is_hallucination'] = (df_generated['Classification'] == 'FP').astype(int)
    return df_generated

def clean_data(df):
    """Apply same cleaning as original classifier."""
    X = df[CI_METRICS].copy()
    y = df['is_hallucination'].copy()
    
    # Remove rows with NaN
    mask_complete = ~X.isnull().any(axis=1)
    X = X[mask_complete]
    y = y[mask_complete]
    
    # Remove inf values
    mask_finite = np.isfinite(X).all(axis=1)
    X = X[mask_finite]
    y = y[mask_finite]
    
    # Remove extreme perplexity values
    perplexity_cols = [col for col in X.columns if 'Perplexity' in col]
    for col in perplexity_cols:
        mask_reasonable = X[col] < 10
        X = X[mask_reasonable]
        y = y[mask_reasonable]
    
    return X, y

def main():
    print("=" * 80)
    print("CROSS-EXPERIMENT GENERALIZATION TEST")
    print("=" * 80)
    print("Train: Original 6 CLDs (exp_20251008_180446_a0d1b2de)")
    print("Test:  Recent 3 CLDs (rq1_base_judging_correctness_20251012_225925)")
    print()
    
    # Load training data
    print("Loading TRAINING data (original 6 CLDs)...")
    df_train = load_original_data()
    print(f"✓ Loaded {len(df_train)} generated edges")
    
    print("\nCleaning training data...")
    X_train, y_train = clean_data(df_train)
    print(f"✓ Clean samples: {len(X_train)}")
    print(f"  - TP (correct): {(y_train == 0).sum()}")
    print(f"  - FP (hallucination): {(y_train == 1).sum()}")
    print(f"  - Hallucination rate: {y_train.mean()*100:.1f}%")
    
    # Load test data
    print("\nLoading TEST data (recent 3 CLDs)...")
    df_test = load_recent_data()
    print(f"✓ Loaded {len(df_test)} generated edges")
    
    print("\nCleaning test data...")
    X_test, y_test = clean_data(df_test)
    print(f"✓ Clean samples: {len(X_test)}")
    print(f"  - TP (correct): {(y_test == 0).sum()}")
    print(f"  - FP (hallucination): {(y_test == 1).sum()}")
    print(f"  - Hallucination rate: {y_test.mean()*100:.1f}%")
    
    # Standardize features
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION ON ORIGINAL DATA")
    print("=" * 80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model (same as original 0.7250 result)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    print("\nModel trained successfully!")
    print("\nFeature Coefficients:")
    coefs = pd.Series(model.coef_[0], index=CI_METRICS).sort_values(key=abs, ascending=False)
    for feat, coef in coefs.items():
        sign = '+' if coef >= 0 else ''
        print(f"  {feat:30s}: {sign}{coef:.4f}")
    
    # Test on original data (sanity check)
    print("\n" + "=" * 80)
    print("SANITY CHECK: Performance on Training Data")
    print("=" * 80)
    
    y_train_pred = model.predict(X_train_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_train_proba)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    train_cm = confusion_matrix(y_train, y_train_pred)
    
    print(f"\nTraining Set Performance:")
    print(f"  AUC:       {train_auc:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    print(f"  F1 Score:  {train_f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {train_cm[0,0]:3d}  FP: {train_cm[0,1]:3d}")
    print(f"  FN: {train_cm[1,0]:3d}  TP: {train_cm[1,1]:3d}")
    
    # Test on new data
    print("\n" + "=" * 80)
    print("🎯 MAIN TEST: Performance on NEW Data (Recent 3 CLDs)")
    print("=" * 80)
    
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    print(f"\nTest Set Performance:")
    print(f"  AUC:       {test_auc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {test_cm[0,0]:3d}  FP: {test_cm[0,1]:3d}")
    print(f"  FN: {test_cm[1,0]:3d}  TP: {test_cm[1,1]:3d}")
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # Training ROC
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    plt.plot(fpr_train, tpr_train, linewidth=2, 
            label=f'Training (6 CLDs) - AUC = {train_auc:.4f}', color='blue')
    
    # Test ROC
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    plt.plot(fpr_test, tpr_test, linewidth=2, 
            label=f'Test (3 CLDs, new data) - AUC = {test_auc:.4f}', color='red')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Cross-Experiment Generalization: Train on Original, Test on Recent', 
             fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save results
    output_dir = RECENT_DIR / "cross_experiment_test"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.savefig(output_dir / "cross_experiment_roc.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved ROC curve: {output_dir / 'cross_experiment_roc.png'}")
    
    # Save JSON results
    results = {
        'experiment': 'Cross-Experiment Generalization Test',
        'timestamp': timestamp,
        'training_data': {
            'source': 'exp_20251008_180446_a0d1b2de (6 CLDs)',
            'samples': len(X_train),
            'hallucination_rate': float(y_train.mean()),
            'performance': {
                'auc': train_auc,
                'precision': train_precision,
                'recall': train_recall,
                'f1': train_f1,
                'confusion_matrix': train_cm.tolist()
            }
        },
        'test_data': {
            'source': 'rq1_base_judging_correctness_20251012_225925 (3 CLDs)',
            'samples': len(X_test),
            'hallucination_rate': float(y_test.mean()),
            'performance': {
                'auc': test_auc,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1,
                'confusion_matrix': test_cm.tolist()
            }
        },
        'generalization_gap': {
            'auc_drop': float(train_auc - test_auc),
            'auc_drop_pct': float((train_auc - test_auc) / train_auc * 100)
        }
    }
    
    with open(output_dir / f"cross_experiment_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results: {output_dir / f'cross_experiment_results_{timestamp}.json'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    print(f"\nTraining (Original 6 CLDs):")
    print(f"  - Samples: {len(X_train)}")
    print(f"  - AUC: {train_auc:.4f}")
    print(f"  - Precision: {train_precision:.1%}, Recall: {train_recall:.1%}, F1: {train_f1:.1%}")
    
    print(f"\nTest (Recent 3 CLDs):")
    print(f"  - Samples: {len(X_test)}")
    print(f"  - AUC: {test_auc:.4f}")
    print(f"  - Precision: {test_precision:.1%}, Recall: {test_recall:.1%}, F1: {test_f1:.1%}")
    
    auc_drop = train_auc - test_auc
    auc_drop_pct = (auc_drop / train_auc * 100)
    
    print(f"\nGeneralization Gap:")
    print(f"  - AUC drop: {auc_drop:.4f} ({auc_drop_pct:.1f}%)")
    
    if auc_drop < 0.05:
        print(f"\n✅ EXCELLENT GENERALIZATION! (drop < 0.05)")
    elif auc_drop < 0.10:
        print(f"\n✓ Good generalization (drop < 0.10)")
    elif auc_drop < 0.15:
        print(f"\n⚠️  Moderate generalization gap (drop < 0.15)")
    else:
        print(f"\n❌ POOR GENERALIZATION (drop > 0.15)")
        print(f"   The classifier does not transfer well to new experimental conditions.")

if __name__ == "__main__":
    main()

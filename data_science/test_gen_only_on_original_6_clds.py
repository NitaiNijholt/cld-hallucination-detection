#!/usr/bin/env python3
"""
Gen-Only Classifier on Original 6 CLDs

Test if using ONLY generator features allows us to recover more samples
from the original dataset (since 63% were missing Judge metrics).

Compare:
- Full 9 features (Gen + Judge): 407 → 136 (33% retention)
- Gen-only 9 features: 407 → ??? (expected higher retention)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Original data (6 CLDs)
ORIGINAL_FILES = {
    'Depressive_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run1_7a66e382/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251008_181257.xlsx',
    'Depressive_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run2_0d57ed9a/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251008_200816.xlsx',
    'Social_norms_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run1_86b016e1/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Social_norms_and_obesity_prevalence_20251008_213554.xlsx',
    'Social_norms_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run2_d09234b7/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Social_norms_and_obesity_prevalence_20251008_221139.xlsx',
    'Older_persons_run1': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run1_d45241a5/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251008_230700.xlsx',
    'Older_persons_run2': 'parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run2_96605199/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251009_082149.xlsx'
}

# Generator-only metrics (no Judge metrics needed!)
GEN_METRICS = [
    'Gen Perplexity',
    'Gen Max Window Entropy',
    'Gen Min Prob',
    'Gen Cosine Similarity',
    'Gen Mean Token Prob',
    'Gen Prob Variance',
    'Gen Prob Std',
    'Gen Max Prob Diff',
    'Gen Token Prob Slope'
]

def load_original_data():
    """Load original data (6 CLDs)."""
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

def clean_data_gen_only(df):
    """Clean data using ONLY Gen metrics."""
    # Check which Gen metrics exist
    available_metrics = [m for m in GEN_METRICS if m in df.columns]
    print(f"\n  Available Gen metrics: {len(available_metrics)}/{len(GEN_METRICS)}")
    for m in GEN_METRICS:
        if m not in df.columns:
            print(f"    ⚠️  Missing: {m}")
    
    if not available_metrics:
        print("  ❌ No Gen metrics found!")
        return None, None
    
    X = df[available_metrics].copy()
    y = df['is_hallucination'].copy()
    
    initial = len(X)
    print(f"\n  Starting with: {initial} samples")
    
    # Remove rows with NaN
    mask_complete = ~X.isnull().any(axis=1)
    X = X[mask_complete]
    y = y[mask_complete]
    print(f"  After removing NaN: {len(X)} samples (lost {initial - len(X)})")
    
    if len(X) == 0:
        return None, None
    
    # Remove inf values
    after_nan = len(X)
    mask_finite = np.isfinite(X).all(axis=1)
    X = X[mask_finite]
    y = y[mask_finite]
    print(f"  After removing inf: {len(X)} samples (lost {after_nan - len(X)})")
    
    if len(X) == 0:
        return None, None
    
    # Remove extreme Gen Perplexity values
    after_inf = len(X)
    if 'Gen Perplexity' in X.columns:
        mask_reasonable = X['Gen Perplexity'] < 10
        X = X[mask_reasonable]
        y = y[mask_reasonable]
        print(f"  After removing extreme Gen Perplexity: {len(X)} samples (lost {after_inf - len(X)})")
    
    print(f"\n  ✅ Final retention: {len(X)}/{initial} = {len(X)/initial*100:.1f}%")
    
    return X, y, available_metrics

def main():
    print("=" * 80)
    print("GEN-ONLY CLASSIFIER ON ORIGINAL 6 CLDs")
    print("=" * 80)
    print("Goal: Recover lost samples by dropping Judge metric requirement\n")
    
    # Load data
    print("Loading original 6 CLDs...")
    df = load_original_data()
    print(f"✓ Loaded {len(df)} generated edges (TP + FP)")
    print(f"  - TP (correct): {(df['is_hallucination'] == 0).sum()}")
    print(f"  - FP (hallucination): {(df['is_hallucination'] == 1).sum()}")
    
    # Clean using Gen-only metrics
    print("\n" + "=" * 80)
    print("CLEANING WITH GEN-ONLY METRICS")
    print("=" * 80)
    
    result = clean_data_gen_only(df)
    if result[0] is None:
        print("\n❌ No usable samples after cleaning!")
        return
    
    X, y, available_metrics = result
    
    if len(X) < 50:
        print(f"\n⚠️  Only {len(X)} samples - too few for reliable training!")
    
    # Print class distribution
    print(f"\nClass distribution:")
    print(f"  - TP (correct): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"  - FP (hallucination): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
    
    # Split data (80/20)
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION (GEN-ONLY)")
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
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"\nCross-validation AUC scores: {cv_scores}")
    print(f"Mean CV AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Train on full training set
    model.fit(X_train_scaled, y_train)
    
    print("\n✓ Model trained successfully!")
    print("\nFeature Coefficients:")
    coefs = pd.Series(model.coef_[0], index=available_metrics).sort_values(key=abs, ascending=False)
    for feat, coef in coefs.items():
        sign = '+' if coef >= 0 else ''
        print(f"  {feat:30s}: {sign}{coef:.4f}")
    
    # Test performance
    print("\n" + "=" * 80)
    print("TEST SET PERFORMANCE")
    print("=" * 80)
    
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
    
    # Save results
    output_dir = Path("parameter_tuning_experiments/rq2_analyses/gen_only_original")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, linewidth=2, label=f'Gen-Only Classifier (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Gen-Only Classifier on Original 6 CLDs', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / "gen_only_roc.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved ROC curve: {output_dir / 'gen_only_roc.png'}")
    
    # Save JSON results
    results = {
        'experiment': 'Gen-Only Classifier on Original 6 CLDs',
        'timestamp': timestamp,
        'dataset': {
            'initial_edges': len(df),
            'clean_samples': len(X),
            'retention_rate': float(len(X) / len(df) * 100),
            'num_features': len(available_metrics),
            'features_used': available_metrics,
            'class_distribution': {
                'TP (correct)': int((y == 0).sum()),
                'FP (hallucination)': int((y == 1).sum()),
                'hallucination_rate': float(y.mean())
            },
            'train_size': len(X_train),
            'test_size': len(X_test)
        },
        'performance': {
            'cv_auc_mean': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
            'test_auc': auc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'confusion_matrix': cm.tolist()
        }
    }
    
    with open(output_dir / f"gen_only_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results: {output_dir / f'gen_only_results_{timestamp}.json'}")
    
    # Final comparison
    print("\n" + "=" * 80)
    print("📊 COMPARISON: GEN-ONLY vs FULL FEATURES")
    print("=" * 80)
    
    print("\nOriginal (9 features: Gen + Judge + Aggregate):")
    print("  - Samples: 407 → 136 (33.4% retention)")
    print("  - Test AUC: 0.7250 (from previous run)")
    print("  - Issue: 63% lost due to missing Judge metrics")
    
    print(f"\nGen-Only ({len(available_metrics)} features: Gen metrics only):")
    print(f"  - Samples: {len(df)} → {len(X)} ({len(X)/len(df)*100:.1f}% retention)")
    print(f"  - Test AUC: {auc:.4f}")
    print(f"  - Recovered: {len(X) - 136} additional samples!")
    
    retention_gain = (len(X)/len(df)*100) - 33.4
    sample_gain = len(X) - 136
    
    if sample_gain > 0:
        print(f"\n✅ RECOVERED {sample_gain} SAMPLES ({retention_gain:+.1f}% retention gain)")
    else:
        print(f"\n⚠️  No samples recovered (same retention)")
    
    auc_diff = auc - 0.7250
    if abs(auc_diff) < 0.05:
        print(f"✅ AUC comparable (diff: {auc_diff:+.4f})")
    elif auc_diff < 0:
        print(f"⚠️  AUC lower (diff: {auc_diff:+.4f})")
    else:
        print(f"🎉 AUC BETTER (diff: {auc_diff:+.4f})")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if sample_gain > 50 and abs(auc_diff) < 0.10:
        print("\n✅ Gen-only approach is VIABLE!")
        print("   - Recovers significantly more samples")
        print("   - Maintains comparable AUC")
        print("   - Simpler, faster (no judging needed)")
    elif sample_gain > 0 and auc_diff < -0.10:
        print("\n⚠️  Trade-off: More samples but lower AUC")
        print("   - Consider if sample size is more important than accuracy")
    else:
        print("\n📊 Results are comparable to full-feature approach")

if __name__ == "__main__":
    main()

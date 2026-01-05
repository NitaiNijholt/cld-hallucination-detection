#!/usr/bin/env python3
"""
Explain what calibration means with concrete examples
Show why classifiers can be mis-calibrated and how to fix it
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/combined_3_recent_experiments.xlsx")

def load_data():
    """Load data."""
    df = pd.read_excel(DATA_FILE, sheet_name=0)
    
    # Filter to generated edges only (TP + FP)
    df_clean = df[df['Gen Perplexity'] <= 100].copy()
    df_generated = df_clean[df_clean['Classification'].isin(['TP', 'FP'])].copy()
    
    # Create hallucination label
    df_generated['is_hallucination'] = (df_generated['Classification'] == 'FP').astype(int)
    
    return df_generated

def demonstrate_miscalibration():
    """Show concrete example of miscalibration."""
    print("="*80)
    print("WHAT IS CALIBRATION? (Concrete Example)")
    print("="*80)
    
    df = load_data()
    
    # Use Cosine Similarity
    feature = 'Gen Cosine Similarity'
    
    df_clean = df[[feature, 'is_hallucination']].dropna()
    X = df_clean[[feature]].values
    y = df_clean['is_hallucination'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train uncalibrated classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    
    # Get predictions
    y_prob_uncalibrated = clf.predict_proba(X_test_scaled)[:, 1]
    
    print("\n📊 EXAMPLE: What does '70% probability' mean?\n")
    print("Imagine your classifier predicts these probabilities for 10 edges:")
    print("  Edge 1: 72% hallucination")
    print("  Edge 2: 68% hallucination")
    print("  Edge 3: 71% hallucination")
    print("  ... (7 more edges, all around 70%)")
    print()
    print("If the classifier is WELL-CALIBRATED:")
    print("  → Out of these 10 edges, ~7 should actually be hallucinations")
    print("  → 70% probability means 'in the long run, 70% of edges with this score are hallucinations'")
    print()
    print("If the classifier is MIS-CALIBRATED:")
    print("  → Maybe only 4 out of 10 are actually hallucinations (overconfident!)")
    print("  → Or maybe 9 out of 10 are hallucinations (underconfident!)")
    print()
    
    # Check actual calibration
    print("="*80)
    print("CHECKING YOUR CLASSIFIER'S CALIBRATION")
    print("="*80)
    
    # Bin predictions and check actual frequencies
    prob_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print("\nUNCALIBRATED Classifier:")
    print(f"{'Predicted Prob Range':<25} {'# Edges':>10} {'Actual % Halluc':>20} {'Calibration Error':>20}")
    print(f"{'-'*25} {'-'*10} {'-'*20} {'-'*20}")
    
    for i in range(len(prob_bins) - 1):
        lower, upper = prob_bins[i], prob_bins[i+1]
        mask = (y_prob_uncalibrated >= lower) & (y_prob_uncalibrated < upper)
        
        if mask.sum() == 0:
            continue
        
        n_edges = mask.sum()
        actual_rate = y_test[mask].mean()
        predicted_avg = y_prob_uncalibrated[mask].mean()
        error = abs(actual_rate - predicted_avg)
        
        print(f"{lower:.1f} - {upper:.1f} ({predicted_avg:.2f} avg) {n_edges:>10} {actual_rate:>19.1%} {error:>19.3f}")
    
    # Calibrate classifier
    print("\n🔧 CALIBRATING the classifier...\n")
    
    # Use isotonic regression for calibration
    calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
    
    # Need a separate calibration set (use part of training data)
    X_train_cal, X_val_cal, y_train_cal, y_val_cal = train_test_split(
        X_train_scaled, y_train, test_size=0.3, random_state=42, stratify=y_train
    )
    
    # Refit base classifier on reduced training set
    clf_for_calib = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    clf_for_calib.fit(X_train_cal, y_train_cal)
    
    # Calibrate on validation set
    calibrated_clf = CalibratedClassifierCV(clf_for_calib, method='isotonic', cv='prefit')
    calibrated_clf.fit(X_val_cal.reshape(-1, 1), y_val_cal)
    
    # Get calibrated predictions
    y_prob_calibrated = calibrated_clf.predict_proba(X_test_scaled)[:, 1]
    
    print("CALIBRATED Classifier:")
    print(f"{'Predicted Prob Range':<25} {'# Edges':>10} {'Actual % Halluc':>20} {'Calibration Error':>20}")
    print(f"{'-'*25} {'-'*10} {'-'*20} {'-'*20}")
    
    for i in range(len(prob_bins) - 1):
        lower, upper = prob_bins[i], prob_bins[i+1]
        mask = (y_prob_calibrated >= lower) & (y_prob_calibrated < upper)
        
        if mask.sum() == 0:
            continue
        
        n_edges = mask.sum()
        actual_rate = y_test[mask].mean()
        predicted_avg = y_prob_calibrated[mask].mean()
        error = abs(actual_rate - predicted_avg)
        
        print(f"{lower:.1f} - {upper:.1f} ({predicted_avg:.2f} avg) {n_edges:>10} {actual_rate:>19.1%} {error:>19.3f}")
    
    # Calculate calibration metrics
    print("\n📈 CALIBRATION METRICS:")
    
    brier_uncal = brier_score_loss(y_test, y_prob_uncalibrated)
    brier_cal = brier_score_loss(y_test, y_prob_calibrated)
    
    print(f"  Brier Score (lower is better):")
    print(f"    Uncalibrated: {brier_uncal:.4f}")
    print(f"    Calibrated:   {brier_cal:.4f}")
    
    if brier_cal < brier_uncal:
        improvement = ((brier_uncal - brier_cal) / brier_uncal) * 100
        print(f"    ✅ Calibration improved by {improvement:.1f}%")
    
    # Plot calibration curves
    plot_calibration_curves(y_test, y_prob_uncalibrated, y_prob_calibrated)
    
    return y_test, y_prob_uncalibrated, y_prob_calibrated

def plot_calibration_curves(y_true, y_prob_uncal, y_prob_cal):
    """Plot calibration curves."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Uncalibrated
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, y_prob_uncal, n_bins=5)
    
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    axes[0].plot(prob_pred_uncal, prob_true_uncal, 'o-', linewidth=2, markersize=8, 
                 label='Uncalibrated', color='#e74c3c')
    axes[0].set_xlabel('Predicted Probability', fontsize=11)
    axes[0].set_ylabel('Actual Fraction of Hallucinations', fontsize=11)
    axes[0].set_title('UNCALIBRATED: Predictions Don\'t Match Reality', fontweight='bold', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # Calibrated
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob_cal, n_bins=5)
    
    axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    axes[1].plot(prob_pred_cal, prob_true_cal, 'o-', linewidth=2, markersize=8,
                 label='Calibrated', color='#27ae60')
    axes[1].set_xlabel('Predicted Probability', fontsize=11)
    axes[1].set_ylabel('Actual Fraction of Hallucinations', fontsize=11)
    axes[1].set_title('CALIBRATED: Predictions Match Reality', fontweight='bold', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    output_path = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/rq2_analyses/calibration_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Calibration curves saved: {output_path}")
    plt.close()

def explain_how_calibration_works():
    """Explain the calibration process step by step."""
    print("\n" + "="*80)
    print("HOW CALIBRATION WORKS (Step by Step)")
    print("="*80)
    
    print("\n1️⃣  Train your base classifier (Logistic Regression):")
    print("    X_train (perplexity, cosine sim, etc.) → y_train (is_hallucination)")
    print("    ↓")
    print("    clf.fit(X_train, y_train)")
    print()
    
    print("2️⃣  Classifier outputs RAW probabilities:")
    print("    clf.predict_proba(X_test) → [0.72, 0.45, 0.89, ...]")
    print("    BUT: These probabilities might be systematically wrong!")
    print()
    
    print("3️⃣  Calibration learns a CORRECTION FUNCTION:")
    print("    It takes a separate validation set and asks:")
    print("      'When the classifier says 70%, what's the ACTUAL rate of hallucinations?'")
    print()
    print("    Example mapping it learns:")
    print("      Raw prob 0.20 → Calibrated prob 0.35 (classifier was underconfident)")
    print("      Raw prob 0.50 → Calibrated prob 0.50 (pretty good!)")
    print("      Raw prob 0.80 → Calibrated prob 0.72 (classifier was overconfident)")
    print()
    
    print("4️⃣  In production, apply BOTH steps:")
    print("    new_edge → clf.predict_proba() → raw_prob")
    print("               ↓")
    print("            calibration_function(raw_prob) → calibrated_prob")
    print()
    
    print("🎯 KEY INSIGHT:")
    print("   - Calibration does NOT change your FEATURES (perplexity stays the same!)")
    print("   - It does NOT renormalize raw data")
    print("   - It only adjusts the FINAL probability output to be more accurate")
    print("   - Think of it as 'correcting the classifier's confidence'")

def what_calibration_is_not():
    """Clarify what calibration is NOT."""
    print("\n" + "="*80)
    print("WHAT CALIBRATION IS NOT")
    print("="*80)
    
    print("\n❌ NOT renormalizing perplexity after CLD generation:")
    print("   - Raw features (perplexity, cosine sim) stay exactly the same")
    print("   - No changes to your data preprocessing")
    print()
    
    print("❌ NOT retraining the model:")
    print("   - The base classifier's weights don't change")
    print("   - It's an additional post-processing step")
    print()
    
    print("❌ NOT feature scaling/normalization:")
    print("   - Feature scaling happens BEFORE training (StandardScaler)")
    print("   - Calibration happens AFTER prediction")
    print()
    
    print("✅ WHAT IT IS:")
    print("   - A correction applied to the classifier's OUTPUT probabilities")
    print("   - Makes 'predicted probability' match 'actual frequency'")
    print("   - Like recalibrating a thermometer that's consistently 5° off")

def main():
    print("\n" + "="*80)
    print("CALIBRATION EXPLAINED: What Are 'Calibrated Probabilities'?")
    print("="*80 + "\n")
    
    # Step 1: Show concrete miscalibration
    y_test, y_prob_uncal, y_prob_cal = demonstrate_miscalibration()
    
    # Step 2: Explain how it works
    explain_how_calibration_works()
    
    # Step 3: Clarify what it's NOT
    what_calibration_is_not()
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY: Why Calibration Helps with Your CLD Problem")
    print("="*80)
    
    print("\n🎯 YOUR PROBLEM:")
    print("   Different CLDs have different distributions")
    print("   → A threshold of 0.5 works on CLD A but not CLD B")
    print()
    
    print("✅ HOW CALIBRATION HELPS:")
    print("   1. Return calibrated probabilities instead of hard 0/1 decisions")
    print("   2. These probabilities have real meaning across CLDs:")
    print("      '80% hallucination' = 'truly 80% chance, not just classifier confidence'")
    print("   3. Users can set their own threshold based on risk tolerance")
    print("   4. More robust when CLD distributions shift")
    print()
    
    print("💡 IN PRODUCTION:")
    print("   # Train once")
    print("   clf = LogisticRegression(...)")
    print("   calibrated_clf = CalibratedClassifierCV(clf, method='isotonic')")
    print("   calibrated_clf.fit(X_train, y_train)")
    print()
    print("   # Use on new CLD")
    print("   prob = calibrated_clf.predict_proba(new_edge_features)[0][1]")
    print("   return {'hallucination_probability': prob}  # Meaningful across CLDs!")
    print()

if __name__ == "__main__":
    main()

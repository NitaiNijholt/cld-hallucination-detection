#!/usr/bin/env python3
"""
Compare performance of different feature sets:
1. Just Gen Cosine Similarity (100% selection)
2. + Gen Perplexity (44% selection, high permutation importance)
3. + Gen Max Prob Diff (44% selection, low permutation importance)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load combined data
DATA_FILE = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/combined_3_recent_experiments.xlsx")

def load_data():
    """Load and prepare data."""
    df = pd.read_excel(DATA_FILE, sheet_name=0)
    
    # Filter to generated edges only (TP + FP)
    df_clean = df[df['Gen Perplexity'] <= 100].copy()
    df_generated = df_clean[df_clean['Classification'].isin(['TP', 'FP'])].copy()
    
    # Create hallucination label
    df_generated['is_hallucination'] = (df_generated['Classification'] == 'FP').astype(int)
    
    return df_generated

def evaluate_feature_set(df, feature_names, n_folds=5, n_seeds=3):
    """Evaluate a specific set of features using cross-validation."""
    # Prepare data
    df_clean = df[feature_names + ['is_hallucination']].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df_clean[feature_names].values
    y = df_clean['is_hallucination'].values
    
    results = []
    
    for seed in range(42, 42 + n_seeds):
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        fold_scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            clf = LogisticRegression(random_state=seed, max_iter=1000, class_weight='balanced')
            clf.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            
            fold_scores.append({
                'auc': roc_auc_score(y_test, y_prob),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            })
        
        # Average across folds for this seed
        results.append({
            'seed': seed,
            'auc': np.mean([s['auc'] for s in fold_scores]),
            'precision': np.mean([s['precision'] for s in fold_scores]),
            'recall': np.mean([s['recall'] for s in fold_scores]),
            'f1': np.mean([s['f1'] for s in fold_scores])
        })
    
    return results

def main():
    print("="*80)
    print("FEATURE SET COMPARISON")
    print("="*80)
    print("Comparing performance of different feature combinations\n")
    
    # Load data
    df = load_data()
    print(f"Dataset: {len(df)} samples\n")
    
    # Define feature sets to compare
    feature_sets = {
        '1. Cosine Sim Only': ['Gen Cosine Similarity'],
        '2. + Perplexity': ['Gen Cosine Similarity', 'Gen Perplexity'],
        '3. + Max Prob Diff': ['Gen Cosine Similarity', 'Gen Perplexity', 'Gen Max Prob Diff']
    }
    
    all_results = {}
    
    for name, features in feature_sets.items():
        print(f"\n{'='*80}")
        print(f"{name}")
        print(f"Features: {features}")
        print(f"{'='*80}")
        
        results = evaluate_feature_set(df, features, n_folds=5, n_seeds=3)
        all_results[name] = results
        
        # Print results
        aucs = [r['auc'] for r in results]
        f1s = [r['f1'] for r in results]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        
        print(f"\nResults across 3 seeds:")
        print(f"  AUC:       {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        print(f"  F1:        {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        print(f"  Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
        print(f"  Recall:    {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Feature Set':<30} {'AUC':>12} {'F1':>12} {'Improvement':>15}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*15}")
    
    baseline_auc = np.mean([r['auc'] for r in all_results['1. Cosine Sim Only']])
    
    for name, results in all_results.items():
        auc = np.mean([r['auc'] for r in results])
        f1 = np.mean([r['f1'] for r in results])
        improvement = ((auc - baseline_auc) / baseline_auc) * 100 if baseline_auc > 0 else 0
        
        print(f"{name:<30} {auc:>12.3f} {f1:>12.3f} {improvement:>13.1f}%")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}\n")
    
    # Compare feature set 1 vs 2
    auc_1 = np.mean([r['auc'] for r in all_results['1. Cosine Sim Only']])
    auc_2 = np.mean([r['auc'] for r in all_results['2. + Perplexity']])
    auc_3 = np.mean([r['auc'] for r in all_results['3. + Max Prob Diff']])
    
    improvement_2 = ((auc_2 - auc_1) / auc_1) * 100
    improvement_3 = ((auc_3 - auc_2) / auc_2) * 100
    
    if improvement_2 > 2:
        print(f"✅ Adding Gen Perplexity improves AUC by {improvement_2:.1f}% - RECOMMENDED")
    else:
        print(f"⚠️  Adding Gen Perplexity only improves AUC by {improvement_2:.1f}% - marginal benefit")
    
    if improvement_3 > 2:
        print(f"✅ Adding Gen Max Prob Diff improves AUC by {improvement_3:.1f}% - RECOMMENDED")
    else:
        print(f"⚠️  Adding Gen Max Prob Diff only improves AUC by {improvement_3:.1f}% - marginal benefit")
    
    print(f"\nBest feature set: ", end="")
    if auc_3 > auc_2 + 0.01 and auc_3 > auc_1 + 0.01:
        print("All 3 features (Cosine Sim + Perplexity + Max Prob Diff)")
    elif auc_2 > auc_1 + 0.01:
        print("Cosine Sim + Perplexity")
    else:
        print("Cosine Sim only (simplest)")
    
    print()

if __name__ == "__main__":
    main()

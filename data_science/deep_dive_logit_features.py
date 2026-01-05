#!/usr/bin/env python3
"""
Deep dive into logit-based features only (no Cosine Similarity)
- Test individual logit features
- Test combinations of logit features
- Compare different logit-only classifiers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Load combined data
DATA_FILE = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/combined_3_recent_experiments.xlsx")

# Logit-based features (everything except Cosine Similarity)
LOGIT_FEATURES = [
    'Gen Perplexity',
    'Gen Min Prob',
    'Gen Max Window Entropy',
    'Gen Mean Token Prob',
    'Gen Prob Variance',
    'Gen Prob Std',
    'Gen Max Prob Diff',
    'Gen Token Prob Slope'
]

def load_data():
    """Load and prepare data."""
    df = pd.read_excel(DATA_FILE, sheet_name=0)
    
    # Filter to generated edges only (TP + FP)
    df_clean = df[df['Gen Perplexity'] <= 100].copy()
    df_generated = df_clean[df_clean['Classification'].isin(['TP', 'FP'])].copy()
    
    # Create hallucination label
    df_generated['is_hallucination'] = (df_generated['Classification'] == 'FP').astype(int)
    
    # Clean features
    df_final = df_generated[LOGIT_FEATURES + ['is_hallucination']].copy()
    df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df_final

def evaluate_feature_set(df, feature_names, n_folds=5, n_seeds=5):
    """Evaluate a specific set of features using cross-validation."""
    # Prepare data
    df_clean = df[feature_names + ['is_hallucination']].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df_clean[feature_names].values
    y = df_clean['is_hallucination'].values
    
    all_aucs = []
    all_f1s = []
    all_precisions = []
    all_recalls = []
    
    for seed in range(42, 42 + n_seeds):
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
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
            
            all_aucs.append(roc_auc_score(y_test, y_prob))
            all_f1s.append(f1_score(y_test, y_pred, zero_division=0))
            all_precisions.append(precision_score(y_test, y_pred, zero_division=0))
            all_recalls.append(recall_score(y_test, y_pred, zero_division=0))
    
    return {
        'auc': np.mean(all_aucs),
        'auc_std': np.std(all_aucs),
        'f1': np.mean(all_f1s),
        'f1_std': np.std(all_f1s),
        'precision': np.mean(all_precisions),
        'recall': np.mean(all_recalls)
    }

def test_individual_features(df):
    """Test each logit feature individually."""
    print("="*80)
    print("INDIVIDUAL LOGIT FEATURE PERFORMANCE")
    print("="*80)
    print(f"\n{'Feature':<35} {'AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    results = []
    
    for feat in LOGIT_FEATURES:
        result = evaluate_feature_set(df, [feat], n_folds=5, n_seeds=5)
        results.append({
            'feature': feat,
            **result
        })
        print(f"{feat:<35} {result['auc']:>10.3f} {result['f1']:>10.3f} "
              f"{result['precision']:>10.3f} {result['recall']:>10.3f}")
    
    return sorted(results, key=lambda x: x['auc'], reverse=True)

def test_feature_pairs(df, top_features):
    """Test pairs of top logit features."""
    print(f"\n{'='*80}")
    print("TOP LOGIT FEATURE PAIRS")
    print("="*80)
    
    # Test all pairs of top 5 features
    top_5 = [f['feature'] for f in top_features[:5]]
    
    pair_results = []
    
    for feat1, feat2 in combinations(top_5, 2):
        result = evaluate_feature_set(df, [feat1, feat2], n_folds=5, n_seeds=3)
        pair_results.append({
            'features': f"{feat1} + {feat2}",
            **result
        })
    
    # Sort by AUC
    pair_results = sorted(pair_results, key=lambda x: x['auc'], reverse=True)
    
    print(f"\n{'Feature Pair':<70} {'AUC':>10} {'F1':>10}")
    print(f"{'-'*70} {'-'*10} {'-'*10}")
    
    for result in pair_results[:10]:  # Show top 10 pairs
        print(f"{result['features']:<70} {result['auc']:>10.3f} {result['f1']:>10.3f}")
    
    return pair_results

def test_progressive_addition(df, individual_results):
    """Progressively add features by AUC rank."""
    print(f"\n{'='*80}")
    print("PROGRESSIVE FEATURE ADDITION (by AUC rank)")
    print("="*80)
    
    sorted_features = [f['feature'] for f in individual_results]
    
    print(f"\n{'Features Used':<45} {'#':>3} {'AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print(f"{'-'*45} {'-'*3} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    progressive_results = []
    
    for i in range(1, len(sorted_features) + 1):
        features = sorted_features[:i]
        result = evaluate_feature_set(df, features, n_folds=5, n_seeds=3)
        
        feat_str = features[0] if i == 1 else f"{features[0]} + ... + {features[-1]}"
        
        progressive_results.append({
            'n_features': i,
            'features': features,
            **result
        })
        
        print(f"{feat_str:<45} {i:>3} {result['auc']:>10.3f} {result['f1']:>10.3f} "
              f"{result['precision']:>10.3f} {result['recall']:>10.3f}")
    
    return progressive_results

def test_different_classifiers(df, best_features):
    """Test different classifiers on best logit feature set."""
    print(f"\n{'='*80}")
    print("DIFFERENT CLASSIFIERS ON BEST LOGIT FEATURES")
    print("="*80)
    print(f"Features: {best_features}\n")
    
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Prepare data
    df_clean = df[best_features + ['is_hallucination']].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df_clean[best_features].values
    y = df_clean['is_hallucination'].values
    
    print(f"{'Classifier':<25} {'AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for clf_name, clf in classifiers.items():
        aucs, f1s, precisions, recalls = [], [], [], []
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            clf.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            
            aucs.append(roc_auc_score(y_test, y_prob))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred, zero_division=0))
        
        print(f"{clf_name:<25} {np.mean(aucs):>10.3f} {np.mean(f1s):>10.3f} "
              f"{np.mean(precisions):>10.3f} {np.mean(recalls):>10.3f}")

def main():
    print("\n" + "="*80)
    print("DEEP DIVE: LOGIT-BASED FEATURES ONLY")
    print("="*80 + "\n")
    
    # Load data
    df = load_data()
    print(f"Dataset: {len(df)} samples")
    print(f"Hallucination rate: {df['is_hallucination'].mean():.1%}\n")
    
    # 1. Test individual features
    individual_results = test_individual_features(df)
    
    # 2. Test feature pairs
    pair_results = test_feature_pairs(df, individual_results)
    
    # 3. Progressive addition
    progressive_results = test_progressive_addition(df, individual_results)
    
    # Find best progressive result
    best_progressive = max(progressive_results, key=lambda x: x['auc'])
    best_features = best_progressive['features']
    
    # 4. Test different classifiers on best feature set
    test_different_classifiers(df, best_features)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n📊 BEST INDIVIDUAL LOGIT FEATURE:")
    best_single = individual_results[0]
    print(f"  {best_single['feature']}")
    print(f"    AUC: {best_single['auc']:.3f}, F1: {best_single['f1']:.3f}")
    
    print(f"\n📊 BEST LOGIT FEATURE PAIR:")
    best_pair = pair_results[0]
    print(f"  {best_pair['features']}")
    print(f"    AUC: {best_pair['auc']:.3f}, F1: {best_pair['f1']:.3f}")
    
    print(f"\n📊 BEST LOGIT FEATURE SET (progressive):")
    print(f"  {best_progressive['n_features']} features: {', '.join(best_progressive['features'])}")
    print(f"    AUC: {best_progressive['auc']:.3f}, F1: {best_progressive['f1']:.3f}")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"  Best logit-only AUC: {best_progressive['auc']:.3f}")
    print(f"  For comparison, Cosine Similarity alone: 0.685")
    print(f"  For comparison, Cosine + Perplexity: 0.709")
    
    if best_progressive['auc'] < 0.65:
        print(f"\n  ⚠️  Logit features alone are INSUFFICIENT for good hallucination detection")
        print(f"  ✅  Embedding-based features (Cosine Similarity) are CRITICAL")
    else:
        print(f"\n  ✅  Logit features can provide reasonable performance")
        print(f"  💡  But combining with Cosine Similarity is still better!")
    
    print()

if __name__ == "__main__":
    main()

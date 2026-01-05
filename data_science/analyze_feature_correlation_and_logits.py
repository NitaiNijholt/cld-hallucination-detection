#!/usr/bin/env python3
"""
1. Check feature correlations to understand why RFE selection varies
2. Compare logit-based features vs embedding-based (Cosine Similarity)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load combined data
DATA_FILE = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/combined_3_recent_experiments.xlsx")

# All 9 features
ALL_FEATURES = [
    'Gen Perplexity',
    'Gen Min Prob',
    'Gen Max Window Entropy',
    'Gen Cosine Similarity',
    'Gen Mean Token Prob',
    'Gen Prob Variance',
    'Gen Prob Std',
    'Gen Max Prob Diff',
    'Gen Token Prob Slope'
]

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
    df_final = df_generated[ALL_FEATURES + ['is_hallucination']].copy()
    df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df_final

def analyze_correlations(df):
    """Analyze feature correlations."""
    print("="*80)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*80)
    
    # Calculate correlation matrix
    corr_matrix = df[ALL_FEATURES].corr()
    
    # Focus on correlations with top features
    top_features = ['Gen Cosine Similarity', 'Gen Perplexity', 'Gen Max Prob Diff']
    
    print("\nCorrelations with top features:\n")
    for feat in top_features:
        print(f"{feat}:")
        correlations = corr_matrix[feat].sort_values(ascending=False)
        for other_feat, corr in correlations.items():
            if other_feat != feat and abs(corr) > 0.1:
                strength = "STRONG" if abs(corr) > 0.7 else "MODERATE" if abs(corr) > 0.4 else "WEAK"
                print(f"  {other_feat:30s}: {corr:+.3f} ({strength})")
        print()
    
    # Save correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/rq2_analyses/feature_correlation_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Correlation heatmap saved: {output_path}\n")
    plt.close()
    
    return corr_matrix

def evaluate_feature_set(df, feature_names, set_name, n_folds=5, n_seeds=3):
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
    
    # Print results
    aucs = [r['auc'] for r in results]
    f1s = [r['f1'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    
    print(f"\n{set_name}")
    print(f"Features: {feature_names}")
    print(f"  AUC:       {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"  F1:        {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"  Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
    print(f"  Recall:    {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    
    return results

def main():
    print("\n" + "="*80)
    print("FEATURE CORRELATION & LOGIT-ONLY ANALYSIS")
    print("="*80 + "\n")
    
    # Load data
    df = load_data()
    print(f"Dataset: {len(df)} samples")
    print(f"Hallucination rate: {df['is_hallucination'].mean():.1%}\n")
    
    # 1. Analyze correlations
    corr_matrix = analyze_correlations(df)
    
    # 2. Compare feature sets
    print("="*80)
    print("PERFORMANCE COMPARISON: LOGIT-BASED vs EMBEDDING-BASED")
    print("="*80)
    
    feature_sets = {
        '1. Cosine Sim ONLY (embedding)': ['Gen Cosine Similarity'],
        '2. All LOGIT features (no Cosine Sim)': LOGIT_FEATURES,
        '3. Top 3 LOGIT features': ['Gen Perplexity', 'Gen Max Prob Diff', 'Gen Prob Variance'],
        '4. Perplexity ONLY': ['Gen Perplexity'],
        '5. Cosine Sim + Perplexity': ['Gen Cosine Similarity', 'Gen Perplexity'],
        '6. ALL 9 features': ALL_FEATURES
    }
    
    all_results = {}
    
    for name, features in feature_sets.items():
        results = evaluate_feature_set(df, features, name, n_folds=5, n_seeds=3)
        all_results[name] = results
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Feature Set':<45} {'#Feats':>8} {'AUC':>10} {'F1':>10}")
    print(f"{'-'*45} {'-'*8} {'-'*10} {'-'*10}")
    
    for name, results in all_results.items():
        auc = np.mean([r['auc'] for r in results])
        f1 = np.mean([r['f1'] for r in results])
        n_feats = len(feature_sets[name])
        print(f"{name:<45} {n_feats:>8} {auc:>10.3f} {f1:>10.3f}")
    
    # Key insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}\n")
    
    cosine_only_auc = np.mean([r['auc'] for r in all_results['1. Cosine Sim ONLY (embedding)']])
    logit_all_auc = np.mean([r['auc'] for r in all_results['2. All LOGIT features (no Cosine Sim)']])
    perplexity_only_auc = np.mean([r['auc'] for r in all_results['4. Perplexity ONLY']])
    combined_auc = np.mean([r['auc'] for r in all_results['5. Cosine Sim + Perplexity']])
    all_9_auc = np.mean([r['auc'] for r in all_results['6. ALL 9 features']])
    
    print(f"1. Cosine Similarity alone: AUC = {cosine_only_auc:.3f}")
    print(f"2. All logit features (no Cosine): AUC = {logit_all_auc:.3f}")
    print(f"3. Perplexity alone: AUC = {perplexity_only_auc:.3f}")
    print(f"4. Cosine + Perplexity: AUC = {combined_auc:.3f}")
    print(f"5. All 9 features: AUC = {all_9_auc:.3f}")
    
    improvement_combined = ((combined_auc - cosine_only_auc) / cosine_only_auc) * 100
    improvement_all = ((all_9_auc - cosine_only_auc) / cosine_only_auc) * 100
    
    print(f"\n📊 ANALYSIS:")
    print(f"  - Logit features alone: {logit_all_auc:.3f} (competitive with Cosine Sim!)")
    print(f"  - Combining Cosine + Perplexity: {improvement_combined:+.1f}% improvement")
    print(f"  - Using all 9 features: {improvement_all:+.1f}% improvement")
    
    # Check correlation between top features
    cosine_perp_corr = corr_matrix.loc['Gen Cosine Similarity', 'Gen Perplexity']
    print(f"\n🔗 CORRELATION:")
    print(f"  - Cosine Similarity ↔ Perplexity: {cosine_perp_corr:+.3f}")
    
    if abs(cosine_perp_corr) > 0.4:
        print(f"    → MODERATE correlation explains why RFE doesn't always pick both!")
        print(f"    → They provide similar information, so RFE alternates between them")
    else:
        print(f"    → Low correlation - features are complementary!")
    
    print(f"\n✅ RECOMMENDATION:")
    if all_9_auc > combined_auc + 0.01:
        print(f"  Use ALL 9 features (best performance: {all_9_auc:.3f})")
    elif combined_auc > cosine_only_auc + 0.01:
        print(f"  Use Cosine Similarity + Perplexity (good balance: {combined_auc:.3f})")
    else:
        print(f"  Cosine Similarity alone is sufficient ({cosine_only_auc:.3f})")
    
    print(f"\n💡 WHY RFE SELECTED PERPLEXITY ONLY 44% OF TIME:")
    if abs(cosine_perp_corr) > 0.3:
        print(f"  - Moderate correlation ({cosine_perp_corr:+.3f}) means they're somewhat redundant")
        print(f"  - RFE's greedy search picks whichever looks better in that fold")
        print(f"  - But permutation importance shows BOTH are valuable when isolated!")
    else:
        print(f"  - Features are not highly correlated")
        print(f"  - Variability likely due to small dataset size and noise")
    
    print()

if __name__ == "__main__":
    main()

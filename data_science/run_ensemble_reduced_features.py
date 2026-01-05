#!/usr/bin/env python3
"""
Reduced Feature Ensemble Classifier
Selects top 10 uncorrelated features to simplify model while maintaining AUC.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Judged data directory
JUDGED_DIR = Path("parameter_tuning_experiments/results/rq1_base_judging_correctness_20251012_225925")

# All 19 Context-insensitive metrics
ALL_CI_METRICS = [
    'Gen Perplexity', 'Gen Max Window Entropy', 'Gen Min Prob', 'Gen Cosine Similarity',
    'Gen Mean Token Prob', 'Gen Prob Variance', 'Gen Prob Std', 'Gen Max Prob Diff', 'Gen Token Prob Slope',
    'Judge Perplexity', 'Judge Max Window Entropy', 'Judge Min Prob', 'Judge Cosine Similarity',
    'Judge Mean Token Prob', 'Judge Prob Variance', 'Judge Prob Std', 'Judge Max Prob Diff', 'Judge Token Prob Slope',
    'Aggregate Score'
]

def load_and_prepare_data():
    """Load data and prepare features."""
    print("Loading data...")
    all_data = []
    
    for excel_file in JUDGED_DIR.glob("judged_*.xlsx"):
        cld_name = excel_file.stem.replace("judged_", "").split("_citations_")[0]
        df = pd.read_excel(excel_file, sheet_name='All Edges')
        df['cld_name'] = cld_name
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    combined['in_ground_truth'] = combined['Classification'].isin(['TP', 'FN']).astype(int)
    
    # Prepare features
    feature_cols = [col for col in ALL_CI_METRICS if col in combined.columns]
    df_clean = combined[feature_cols + ['in_ground_truth']].copy()
    df_clean[feature_cols] = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # Filter extreme perplexities
    for col in feature_cols:
        if 'Perplexity' in col:
            df_clean = df_clean[df_clean[col] < 1e10]
    
    df_clean = df_clean.dropna(subset=feature_cols)
    
    X_full = df_clean[feature_cols].values
    y = df_clean['in_ground_truth'].values
    
    print(f"Loaded {len(X_full)} samples, {sum(y)} positive class")
    
    return X_full, y, feature_cols

def select_uncorrelated_features(X, feature_names, target_count=10):
    """
    Intelligently select uncorrelated features based on:
    1. Feature importance from prior analysis
    2. Low correlation with already-selected features
    """
    print("\n" + "="*80)
    print("FEATURE SELECTION: TOP 10 UNCORRELATED FEATURES")
    print("="*80)
    
    # Feature importance rankings from prior analysis (Gradient Boosting)
    importance_ranking = [
        'Gen Cosine Similarity',      # 30.0%
        'Aggregate Score',             # 14.7%
        'Gen Max Prob Diff',           # 7.4%
        'Judge Mean Token Prob',       # 4.5%
        'Gen Token Prob Slope',        # 4.4%
        'Gen Min Prob',                # 4.0%
        'Judge Token Prob Slope',      # 4.0%
        'Judge Max Prob Diff',         # 4.0%
        'Judge Cosine Similarity',     # 3.8%
        'Judge Prob Variance',         # 3.2%
        'Gen Mean Token Prob',         # remaining
        'Judge Perplexity',
        'Gen Max Window Entropy',
        'Gen Prob Variance',
        'Judge Max Window Entropy',
        'Judge Min Prob',
        'Judge Prob Std',
        'Gen Perplexity',
        'Gen Prob Std',
    ]
    
    # Compute correlation matrix
    df_features = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df_features.corr()
    
    selected_features = []
    selected_indices = []
    
    print("\nSelecting features with correlation threshold |r| < 0.80:")
    
    for feature in importance_ranking:
        if feature not in feature_names:
            continue
            
        if len(selected_features) >= target_count:
            break
        
        feature_idx = feature_names.index(feature)
        
        # Check correlation with already-selected features
        if selected_indices:
            max_corr = max(abs(corr_matrix.iloc[feature_idx, idx]) for idx in selected_indices)
        else:
            max_corr = 0.0
        
        if max_corr < 0.80:
            selected_features.append(feature)
            selected_indices.append(feature_idx)
            print(f"  ✓ {len(selected_features):2d}. {feature:35s} (max corr with selected: {max_corr:.3f})")
        else:
            print(f"  ✗     {feature:35s} (max corr with selected: {max_corr:.3f} - REJECTED)")
    
    print(f"\nSelected {len(selected_features)} features")
    
    # Extract selected features
    X_reduced = df_features[selected_features].values
    
    return X_reduced, selected_features

def train_and_compare(X_full, X_reduced, y, full_features, reduced_features):
    """Train models on both full and reduced feature sets and compare."""
    print("\n" + "="*80)
    print("TRAINING MODELS: FULL vs REDUCED FEATURES")
    print("="*80)
    
    results = {'full': {}, 'reduced': {}}
    
    for feature_set_name, X, feature_list in [
        ('full', X_full, full_features),
        ('reduced', X_reduced, reduced_features)
    ]:
        print(f"\n{'='*80}")
        print(f"Feature Set: {feature_set_name.upper()} ({len(feature_list)} features)")
        print(f"{'='*80}")
        
        # Standardize and split
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train == 1).sum() - 1))
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=1000)
        }
        
        for model_name, model in models.items():
            # Train
            model.fit(X_train_resampled, y_train_resampled)
            
            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            auc = roc_auc_score(y_test, y_proba)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=0
            )
            
            results[feature_set_name][model_name] = {
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_test': y_test,
                'y_proba': y_proba
            }
            
            print(f"\n{model_name}:")
            print(f"  AUC: {auc:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    
    return results

def plot_comparison(results, output_dir):
    """Plot comparison of full vs reduced features."""
    
    # ROC curves comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, feature_set, title in [
        (ax1, 'full', 'Full Feature Set (19 features)'),
        (ax2, 'reduced', 'Reduced Feature Set (10 features)')
    ]:
        for model_name, result in results[feature_set].items():
            fpr, tpr, _ = roc_curve(result['y_test'], result['y_proba'])
            auc = result['auc']
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})", linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'roc_comparison_full_vs_reduced.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 ROC comparison saved to: {output_path}")
    plt.close()
    
    # Performance comparison bar chart
    models = list(results['full'].keys())
    metrics = ['auc', 'precision', 'recall', 'f1']
    metric_labels = ['ROC AUC', 'Precision', 'Recall', 'F1 Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        full_scores = [results['full'][m][metric] for m in models]
        reduced_scores = [results['reduced'][m][metric] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, full_scores, width, label='Full (19 features)', color='steelblue')
        ax.bar(x + width/2, reduced_scores, width, label='Reduced (10 features)', color='coral')
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for i, (f, r) in enumerate(zip(full_scores, reduced_scores)):
            ax.text(i - width/2, f + 0.02, f'{f:.3f}', ha='center', fontsize=8)
            ax.text(i + width/2, r + 0.02, f'{r:.3f}', ha='center', fontsize=8)
    
    plt.suptitle('Performance Comparison: Full vs Reduced Feature Sets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'performance_comparison_bars.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Performance comparison saved to: {output_path}")
    plt.close()

def save_comparison_summary(results, full_features, reduced_features, output_dir):
    """Save comparison summary."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'full_feature_set': {
            'count': len(full_features),
            'features': full_features,
            'results': {model: {k: float(v) for k, v in res.items() if k not in ['y_test', 'y_proba']}
                       for model, res in results['full'].items()}
        },
        'reduced_feature_set': {
            'count': len(reduced_features),
            'features': reduced_features,
            'results': {model: {k: float(v) for k, v in res.items() if k not in ['y_test', 'y_proba']}
                       for model, res in results['reduced'].items()}
        },
        'comparison': {}
    }
    
    # Compute differences
    for model in results['full'].keys():
        summary['comparison'][model] = {
            'auc_diff': float(results['reduced'][model]['auc'] - results['full'][model]['auc']),
            'auc_change_pct': float((results['reduced'][model]['auc'] - results['full'][model]['auc']) / results['full'][model]['auc'] * 100)
        }
    
    output_path = output_dir / 'feature_reduction_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Comparison summary saved to: {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY: FULL vs REDUCED")
    print("="*80)
    print(f"\n{'Model':<25s} {'Full AUC':>10s} {'Reduced AUC':>12s} {'Δ AUC':>10s} {'% Change':>10s}")
    print("-" * 80)
    
    for model in results['full'].keys():
        full_auc = results['full'][model]['auc']
        reduced_auc = results['reduced'][model]['auc']
        diff = reduced_auc - full_auc
        pct = (diff / full_auc * 100) if full_auc > 0 else 0
        
        print(f"{model:<25s} {full_auc:>10.4f} {reduced_auc:>12.4f} {diff:>+10.4f} {pct:>+9.2f}%")

def main():
    print("="*80)
    print("REDUCED FEATURE ENSEMBLE CLASSIFIER")
    print("="*80)
    
    # Create output directory
    output_dir = JUDGED_DIR / "ensemble_analysis_reduced"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data
    X_full, y, full_features = load_and_prepare_data()
    
    # Select uncorrelated features
    X_reduced, reduced_features = select_uncorrelated_features(X_full, full_features, target_count=10)
    
    # Train and compare
    results = train_and_compare(X_full, X_reduced, y, full_features, reduced_features)
    
    # Plot comparison
    plot_comparison(results, output_dir)
    
    # Save summary
    save_comparison_summary(results, full_features, reduced_features, output_dir)
    
    print("\n" + "="*80)
    print("✅ FEATURE REDUCTION ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

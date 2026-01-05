#!/usr/bin/env python3
"""
Compare Gen-only vs Judge-only vs Combined Features
Tests if generator features alone are sufficient for hallucination detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Judged data directory
JUDGED_DIR = Path("parameter_tuning_experiments/results/rq1_base_judging_correctness_20251012_225925")

# Feature sets
GEN_METRICS = [
    'Gen Perplexity', 'Gen Max Window Entropy', 'Gen Min Prob', 'Gen Cosine Similarity',
    'Gen Mean Token Prob', 'Gen Prob Variance', 'Gen Prob Std', 'Gen Max Prob Diff', 'Gen Token Prob Slope'
]

JUDGE_METRICS = [
    'Judge Perplexity', 'Judge Max Window Entropy', 'Judge Min Prob', 'Judge Cosine Similarity',
    'Judge Mean Token Prob', 'Judge Prob Variance', 'Judge Prob Std', 'Judge Max Prob Diff', 'Judge Token Prob Slope',
    'Aggregate Score'
]

ALL_METRICS = GEN_METRICS + JUDGE_METRICS

def load_and_prepare_data():
    """Load data and prepare all feature sets."""
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
    feature_cols = [col for col in ALL_METRICS if col in combined.columns]
    df_clean = combined[feature_cols + ['in_ground_truth']].copy()
    df_clean[feature_cols] = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # Filter extreme perplexities
    for col in feature_cols:
        if 'Perplexity' in col:
            df_clean = df_clean[df_clean[col] < 1e10]
    
    df_clean = df_clean.dropna(subset=feature_cols)
    
    # Create feature sets
    X_gen = df_clean[GEN_METRICS].values
    X_judge = df_clean[JUDGE_METRICS].values
    X_all = df_clean[feature_cols].values
    y = df_clean['in_ground_truth'].values
    
    print(f"Loaded {len(X_all)} samples, {sum(y)} positive class")
    print(f"  Gen features: {len(GEN_METRICS)}")
    print(f"  Judge features: {len(JUDGE_METRICS)}")
    print(f"  Total features: {len(feature_cols)}")
    
    return X_gen, X_judge, X_all, y

def train_and_evaluate_feature_sets(X_gen, X_judge, X_all, y):
    """Train models on different feature sets."""
    print("\n" + "="*80)
    print("COMPARING FEATURE SETS: GEN vs JUDGE vs ALL")
    print("="*80)
    
    feature_sets = {
        'Gen Only (9)': X_gen,
        'Judge Only (10)': X_judge,
        'All Features (19)': X_all
    }
    
    results = {}
    
    # Use only Gradient Boosting and Logistic Regression (best performers)
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    for set_name, X in feature_sets.items():
        print(f"\n{'='*80}")
        print(f"Feature Set: {set_name}")
        print(f"{'='*80}")
        
        results[set_name] = {}
        
        # Standardize and split
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train == 1).sum() - 1))
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
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
            
            results[set_name][model_name] = {
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_test': y_test,
                'y_proba': y_proba
            }
            
            print(f"\n{model_name}:")
            print(f"  AUC: {auc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
    
    return results

def plot_comparison(results, output_dir):
    """Plot comprehensive comparison."""
    
    # 1. ROC curves for each feature set
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (set_name, set_results) in zip(axes, results.items()):
        for model_name, result in set_results.items():
            fpr, tpr, _ = roc_curve(result['y_test'], result['y_proba'])
            auc = result['auc']
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})", linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(set_name, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'roc_gen_vs_judge_vs_all.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 ROC comparison saved to: {output_path}")
    plt.close()
    
    # 2. Performance comparison bar chart
    feature_sets_list = list(results.keys())
    models = list(results[feature_sets_list[0]].keys())
    metrics = ['auc', 'precision', 'recall', 'f1']
    metric_labels = ['ROC AUC', 'Precision', 'Recall', 'F1 Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['steelblue', 'coral', 'lightgreen']
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        x = np.arange(len(models))
        width = 0.25
        
        for i, set_name in enumerate(feature_sets_list):
            scores = [results[set_name][m][metric] for m in models]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, scores, width, label=set_name, color=colors[i])
            
            # Add value labels
            for j, score in enumerate(scores):
                ax.text(j + offset, score + 0.02, f'{score:.3f}', 
                       ha='center', fontsize=7, rotation=0)
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
    
    plt.suptitle('Performance Comparison: Gen vs Judge vs All Features', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'performance_gen_vs_judge_vs_all.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Performance bars saved to: {output_path}")
    plt.close()

def print_summary_table(results):
    """Print detailed summary table."""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY: GEN vs JUDGE vs ALL")
    print("="*80)
    
    # Get all models
    models = list(results[list(results.keys())[0]].keys())
    
    for model in models:
        print(f"\n{model}:")
        print(f"{'Feature Set':<20s} {'AUC':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
        print("-" * 60)
        
        for set_name, set_results in results.items():
            r = set_results[model]
            print(f"{set_name:<20s} {r['auc']:>8.4f} {r['precision']:>10.4f} "
                  f"{r['recall']:>8.4f} {r['f1']:>8.4f}")
    
    # AUC difference analysis
    print("\n" + "="*80)
    print("AUC DELTA ANALYSIS (vs All Features)")
    print("="*80)
    
    for model in models:
        print(f"\n{model}:")
        all_auc = results['All Features (19)'][model]['auc']
        
        for set_name in ['Gen Only (9)', 'Judge Only (10)']:
            set_auc = results[set_name][model]['auc']
            delta = set_auc - all_auc
            pct = (delta / all_auc * 100) if all_auc > 0 else 0
            
            print(f"  {set_name:<20s}: Δ AUC = {delta:+.4f} ({pct:+.2f}%)")

def save_summary(results, output_dir):
    """Save results summary."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'feature_sets': {
            'gen_only': {'count': 9, 'features': GEN_METRICS},
            'judge_only': {'count': 10, 'features': JUDGE_METRICS},
            'all': {'count': 19, 'features': GEN_METRICS + JUDGE_METRICS}
        },
        'results': {}
    }
    
    for set_name, set_results in results.items():
        summary['results'][set_name] = {}
        for model, res in set_results.items():
            summary['results'][set_name][model] = {
                k: float(v) for k, v in res.items() if k not in ['y_test', 'y_proba']
            }
    
    output_path = output_dir / 'gen_vs_judge_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Results saved to: {output_path}")

def main():
    print("="*80)
    print("GEN vs JUDGE vs ALL FEATURES COMPARISON")
    print("="*80)
    
    # Create output directory
    output_dir = JUDGED_DIR / "ensemble_analysis_gen_vs_judge"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data
    X_gen, X_judge, X_all, y = load_and_prepare_data()
    
    # Train and evaluate
    results = train_and_evaluate_feature_sets(X_gen, X_judge, X_all, y)
    
    # Plot comparison
    plot_comparison(results, output_dir)
    
    # Print summary
    print_summary_table(results)
    
    # Save results
    save_summary(results, output_dir)
    
    print("\n" + "="*80)
    print("✅ GEN vs JUDGE COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

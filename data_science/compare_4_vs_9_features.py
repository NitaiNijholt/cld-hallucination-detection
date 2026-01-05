#!/usr/bin/env python3
"""
Compare classifier performance: 4 Core Features vs All 9 Gen Features

Oct 14, 2025 data - Generator metrics only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, confusion_matrix

# Data paths
DATA_DIR = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/rq1_base_generation_original_20251014_004654")
DATA_FILES = {
    'Social_Norms': DATA_DIR / "result_excel_path_20251014_004801.xlsx",
    'Older_Persons': DATA_DIR / "result_excel_path_20251014_010206.xlsx",
    'Depressive_Symptoms': DATA_DIR / "result_excel_path_20251014_020108.xlsx",
}

# Feature sets
CORE_4_FEATURES = [
    'Gen Perplexity',
    'Gen Min Prob',
    'Gen Max Window Entropy',
    'Gen Cosine Similarity'
]

ALL_9_FEATURES = [
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

OUTPUT_DIR = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/rq2_analyses/feature_comparison_4vs9")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean_data():
    """Load all 3 CLDs, filter extreme perplexity, combine TP+FP edges."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    all_data = []
    for cld_name, excel_path in DATA_FILES.items():
        print(f"  Loading {cld_name}...")
        df = pd.read_excel(excel_path, sheet_name=0)
        
        # Filter extreme perplexity
        df_clean = df[df['Gen Perplexity'] <= 100].copy()
        
        # Filter to TP and FP only
        df_generated = df_clean[df_clean['Classification'].isin(['TP', 'FP'])].copy()
        df_generated['cld_name'] = cld_name
        df_generated['is_hallucination'] = (df_generated['Classification'] == 'FP').astype(int)
        
        all_data.append(df_generated)
    
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal edges: {len(df_combined)} (TP={np.sum(df_combined['is_hallucination']==0)}, FP={np.sum(df_combined['is_hallucination']==1)})")
    
    return df_combined


def prepare_features(df, feature_list):
    """Prepare feature matrix for given feature list."""
    df_clean = df[feature_list + ['is_hallucination']].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    
    X = df_clean[feature_list].values
    y = df_clean['is_hallucination'].values
    
    return X, y, feature_list


def train_and_evaluate(X, y, feature_names, feature_set_name):
    """Train ensemble classifiers and return results."""
    print(f"\n{'='*80}")
    print(f"TRAINING ON: {feature_set_name}")
    print(f"Features: {len(feature_names)}")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\n  {name}:")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        
        # Train and test
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        test_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"    CV AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        print(f"    Test AUC: {test_auc:.3f}")
        print(f"    Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
        
        # Feature importance
        if hasattr(clf, 'coef_'):
            coefficients = clf.coef_[0]
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients,
                'abs_coef': np.abs(coefficients)
            }).sort_values('abs_coef', ascending=False)
            
            print(f"    Top 3 Features:")
            for idx, row in feature_importance.head(3).iterrows():
                print(f"      {row['feature']:30s}: {row['coefficient']:+.4f}")
        
        results[name] = {
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'test_auc': test_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }
    
    return results


def plot_comparison(results_4, results_9, output_path):
    """Plot ROC curves comparing 4 vs 9 features."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 4 features
    ax1.set_title('4 Core Features', fontsize=14, fontweight='bold')
    for name, res in results_4.items():
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_pred_proba'])
        ax1.plot(fpr, tpr, label=f'{name} (AUC={res["test_auc"]:.3f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', label='Chance', linewidth=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 9 features
    ax2.set_title('All 9 Generator Features', fontsize=14, fontweight='bold')
    for name, res in results_9.items():
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_pred_proba'])
        ax2.plot(fpr, tpr, label=f'{name} (AUC={res["test_auc"]:.3f})', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', label='Chance', linewidth=1)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved: {output_path}")
    plt.close()


def generate_comparison_report(results_4, results_9, output_path):
    """Generate comparison report."""
    with open(output_path, 'w') as f:
        f.write("# Feature Set Comparison: 4 Core vs 9 Generator Features\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Data:** Oct 14, 2025 (3 base CLDs, 219 edges)\n\n")
        
        f.write("---\n\n")
        f.write("## Feature Sets\n\n")
        
        f.write("### 4 Core Features (Original)\n")
        for feat in CORE_4_FEATURES:
            f.write(f"- {feat}\n")
        
        f.write("\n### All 9 Generator Features (+5 New from Paper)\n")
        for feat in ALL_9_FEATURES:
            marker = " ✨" if feat not in CORE_4_FEATURES else ""
            f.write(f"- {feat}{marker}\n")
        
        f.write("\n---\n\n")
        f.write("## Performance Comparison\n\n")
        
        # Performance table
        f.write("| Classifier | 4 Features (CV AUC) | 4 Features (Test AUC) | 9 Features (CV AUC) | 9 Features (Test AUC) | Δ AUC |\n")
        f.write("|------------|---------------------|----------------------|---------------------|----------------------|--------|\n")
        
        for clf_name in results_4.keys():
            res4 = results_4[clf_name]
            res9 = results_9[clf_name]
            delta = res9['test_auc'] - res4['test_auc']
            delta_str = f"{delta:+.3f}"
            
            f.write(f"| {clf_name} | {res4['cv_auc_mean']:.3f} (±{res4['cv_auc_std']:.3f}) | "
                   f"**{res4['test_auc']:.3f}** | {res9['cv_auc_mean']:.3f} (±{res9['cv_auc_std']:.3f}) | "
                   f"**{res9['test_auc']:.3f}** | {delta_str} |\n")
        
        f.write("\n**Δ AUC:** Change from 4 features to 9 features (positive = improvement)\n\n")
        
        # Best models
        best_4 = max(results_4.keys(), key=lambda k: results_4[k]['test_auc'])
        best_9 = max(results_9.keys(), key=lambda k: results_9[k]['test_auc'])
        
        f.write("---\n\n")
        f.write("## Key Findings\n\n")
        
        auc_4_best = results_4[best_4]['test_auc']
        auc_9_best = results_9[best_9]['test_auc']
        improvement = ((auc_9_best - auc_4_best) / auc_4_best) * 100
        
        f.write(f"1. **Best 4-Feature Model:** {best_4} (AUC={auc_4_best:.3f})\n")
        f.write(f"2. **Best 9-Feature Model:** {best_9} (AUC={auc_9_best:.3f})\n")
        f.write(f"3. **Improvement:** {improvement:+.1f}% ({auc_9_best - auc_4_best:+.3f} AUC points)\n\n")
        
        if improvement > 0:
            f.write("✅ **Conclusion:** The 5 new features from the paper provide measurable improvement!\n\n")
        elif improvement > -2:
            f.write("⚖️ **Conclusion:** The 5 new features provide minimal additional value (~equivalent performance).\n\n")
        else:
            f.write("❌ **Conclusion:** The 4 core features perform better (possible feature bloat from 5 new features).\n\n")
        
        f.write("---\n\n")
        f.write("## Interpretation\n\n")
        f.write("The **4 core features** (Perplexity, Min Prob, Max Window Entropy, Cosine Similarity) "
               "were identified as the most important in hallucination detection literature.\n\n")
        f.write("The **5 new features** (Mean Token Prob, Prob Variance, Prob Std, Max Prob Diff, Token Prob Slope) "
               "were added based on recent papers (arXiv:2405.19648v1, CHAIR arXiv:2501.02518v2).\n\n")
        f.write("This comparison helps determine if the additional features justify the added complexity.\n\n")
    
    print(f"✅ Report saved: {output_path}")


def main():
    print("\n" + "="*80)
    print("FEATURE COMPARISON: 4 CORE vs 9 GENERATOR FEATURES")
    print("="*80)
    
    # Load data
    df = load_and_clean_data()
    
    # Prepare feature sets
    print("\n" + "="*80)
    print("PREPARING FEATURE SETS")
    print("="*80)
    
    X_4, y_4, features_4 = prepare_features(df, CORE_4_FEATURES)
    print(f"  4 Core Features: {len(X_4)} samples prepared")
    
    X_9, y_9, features_9 = prepare_features(df, ALL_9_FEATURES)
    print(f"  9 Gen Features:  {len(X_9)} samples prepared")
    
    # Train and evaluate both
    results_4 = train_and_evaluate(X_4, y_4, features_4, "4 Core Features")
    results_9 = train_and_evaluate(X_9, y_9, features_9, "All 9 Features")
    
    # Generate comparison plot
    plot_path = OUTPUT_DIR / "feature_comparison_4vs9_roc.png"
    plot_comparison(results_4, results_9, plot_path)
    
    # Generate comparison report
    report_path = OUTPUT_DIR / f"feature_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    generate_comparison_report(results_4, results_9, report_path)
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE!")
    print("="*80)
    print(f"\n📁 Output: {OUTPUT_DIR}")
    print(f"📊 Plot: {plot_path.name}")
    print(f"📄 Report: {report_path.name}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Systematic Feature Selection Analysis

Tests which features contribute most to hallucination detection using:
1. Recursive Feature Elimination (RFE)
2. Sequential Forward Selection
3. Permutation Importance
4. Individual Feature AUC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

# Data paths
DATA_DIR = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/rq1_base_generation_original_20251014_004654")
DATA_FILES = {
    'Social_Norms': DATA_DIR / "result_excel_path_20251014_004801.xlsx",
    'Older_Persons': DATA_DIR / "result_excel_path_20251014_010206.xlsx",
    'Depressive_Symptoms': DATA_DIR / "result_excel_path_20251014_020108.xlsx",
}

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

OUTPUT_DIR = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/rq2_analyses/feature_selection_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load and prepare data."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    all_data = []
    for cld_name, excel_path in DATA_FILES.items():
        df = pd.read_excel(excel_path, sheet_name=0)
        df_clean = df[df['Gen Perplexity'] <= 100].copy()
        df_generated = df_clean[df_clean['Classification'].isin(['TP', 'FP'])].copy()
        df_generated['is_hallucination'] = (df_generated['Classification'] == 'FP').astype(int)
        all_data.append(df_generated)
    
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Prepare features
    df_clean = df_combined[ALL_9_FEATURES + ['is_hallucination']].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df_clean[ALL_9_FEATURES].values
    y = df_clean['is_hallucination'].values
    
    print(f"Total samples: {len(X)} (FP={y.sum()}, TP={len(y)-y.sum()})")
    return X, y


def individual_feature_auc(X, y, feature_names):
    """Test each feature individually."""
    print("\n" + "="*80)
    print("METHOD 1: INDIVIDUAL FEATURE AUC")
    print("="*80)
    print("Testing each feature alone to see its standalone predictive power...")
    
    results = []
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    for i, feat in enumerate(feature_names):
        # Use single feature
        X_train_single = X_train[:, i].reshape(-1, 1)
        X_test_single = X_test[:, i].reshape(-1, 1)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_single)
        X_test_scaled = scaler.transform(X_test_single)
        
        # Train
        clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({'feature': feat, 'auc': auc})
        print(f"  {feat:30s}: AUC = {auc:.3f}")
    
    df_results = pd.DataFrame(results).sort_values('auc', ascending=False)
    return df_results


def recursive_feature_elimination(X, y, feature_names):
    """Use RFE to rank features."""
    print("\n" + "="*80)
    print("METHOD 2: RECURSIVE FEATURE ELIMINATION (RFE)")
    print("="*80)
    print("Removing features one by one (worst first) to find optimal subset...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    
    # Test different numbers of features
    results = []
    
    for n_features in range(1, len(feature_names) + 1):
        rfe = RFE(estimator=clf, n_features_to_select=n_features, step=1)
        rfe.fit(X_train_scaled, y_train)
        
        # Get selected features
        selected_features = [feat for feat, selected in zip(feature_names, rfe.support_) if selected]
        
        # Train on selected features
        X_train_selected = X_train_scaled[:, rfe.support_]
        X_test_selected = X_test_scaled[:, rfe.support_]
        
        clf.fit(X_train_selected, y_train)
        y_pred_proba = clf.predict_proba(X_test_selected)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'n_features': n_features,
            'auc': auc,
            'features': selected_features
        })
        
        print(f"\n  Top {n_features} features: AUC = {auc:.3f}")
        for feat in selected_features:
            print(f"    - {feat}")
    
    df_results = pd.DataFrame(results)
    
    # Find optimal number
    best_idx = df_results['auc'].idxmax()
    best_n = df_results.loc[best_idx, 'n_features']
    best_auc = df_results.loc[best_idx, 'auc']
    
    print(f"\n  🏆 Optimal: {best_n} features (AUC = {best_auc:.3f})")
    
    return df_results


def sequential_forward_selection(X, y, feature_names):
    """Add features one by one (best first)."""
    print("\n" + "="*80)
    print("METHOD 3: SEQUENTIAL FORWARD SELECTION")
    print("="*80)
    print("Adding features one by one (best first) until no improvement...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    
    results = []
    
    for n_features in range(1, len(feature_names) + 1):
        # SFS requires n_features_to_select < n_features, so handle the full case separately
        if n_features == len(feature_names):
            # Use all features
            selected_features = feature_names
            support = [True] * len(feature_names)
            
            clf.fit(X_train_scaled, y_train)
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            sfs = SequentialFeatureSelector(
                clf, n_features_to_select=n_features, direction='forward',
                scoring='roc_auc', cv=3, n_jobs=-1
            )
            sfs.fit(X_train_scaled, y_train)
            
            # Get selected features
            selected_features = [feat for feat, selected in zip(feature_names, sfs.support_) if selected]
            
            # Train on selected features
            X_train_selected = X_train_scaled[:, sfs.support_]
            X_test_selected = X_test_scaled[:, sfs.support_]
            
            clf.fit(X_train_selected, y_train)
            y_pred_proba = clf.predict_proba(X_test_selected)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'n_features': n_features,
            'auc': auc,
            'features': selected_features
        })
        
        print(f"\n  {n_features} features: AUC = {auc:.3f}")
        for feat in selected_features:
            print(f"    - {feat}")
    
    df_results = pd.DataFrame(results)
    
    # Find optimal
    best_idx = df_results['auc'].idxmax()
    best_n = df_results.loc[best_idx, 'n_features']
    best_auc = df_results.loc[best_idx, 'auc']
    
    print(f"\n  🏆 Optimal: {best_n} features (AUC = {best_auc:.3f})")
    
    return df_results


def permutation_importance_analysis(X, y, feature_names):
    """Measure importance by shuffling each feature."""
    print("\n" + "="*80)
    print("METHOD 4: PERMUTATION IMPORTANCE")
    print("="*80)
    print("Measuring AUC drop when each feature is randomly shuffled...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    
    # Permutation importance
    perm_importance = permutation_importance(
        clf, X_test_scaled, y_test, n_repeats=10, random_state=42, scoring='roc_auc'
    )
    
    results = []
    for i, feat in enumerate(feature_names):
        results.append({
            'feature': feat,
            'importance': perm_importance.importances_mean[i],
            'std': perm_importance.importances_std[i]
        })
        print(f"  {feat:30s}: {perm_importance.importances_mean[i]:+.3f} (±{perm_importance.importances_std[i]:.3f})")
    
    df_results = pd.DataFrame(results).sort_values('importance', ascending=False)
    
    return df_results


def plot_results(individual_auc, rfe_results, sfs_results, perm_importance, output_path):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Individual Feature AUC
    ax = axes[0, 0]
    individual_auc_sorted = individual_auc.sort_values('auc')
    ax.barh(range(len(individual_auc_sorted)), individual_auc_sorted['auc'])
    ax.set_yticks(range(len(individual_auc_sorted)))
    ax.set_yticklabels(individual_auc_sorted['feature'], fontsize=9)
    ax.set_xlabel('AUC (Single Feature)')
    ax.set_title('Individual Feature Performance', fontweight='bold')
    ax.axvline(0.5, color='red', linestyle='--', label='Chance')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    
    # 2. RFE: AUC vs Number of Features
    ax = axes[0, 1]
    ax.plot(rfe_results['n_features'], rfe_results['auc'], marker='o', linewidth=2)
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('AUC')
    ax.set_title('Recursive Feature Elimination', fontweight='bold')
    ax.grid(True, alpha=0.3)
    best_idx = rfe_results['auc'].idxmax()
    ax.axvline(rfe_results.loc[best_idx, 'n_features'], color='red', linestyle='--', 
               label=f"Best: {rfe_results.loc[best_idx, 'n_features']} features")
    ax.legend()
    
    # 3. Sequential Forward Selection
    ax = axes[1, 0]
    ax.plot(sfs_results['n_features'], sfs_results['auc'], marker='s', linewidth=2, color='green')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('AUC')
    ax.set_title('Sequential Forward Selection', fontweight='bold')
    ax.grid(True, alpha=0.3)
    best_idx = sfs_results['auc'].idxmax()
    ax.axvline(sfs_results.loc[best_idx, 'n_features'], color='red', linestyle='--',
               label=f"Best: {sfs_results.loc[best_idx, 'n_features']} features")
    ax.legend()
    
    # 4. Permutation Importance
    ax = axes[1, 1]
    perm_sorted = perm_importance.sort_values('importance')
    colors = ['green' if x > 0 else 'red' for x in perm_sorted['importance']]
    ax.barh(range(len(perm_sorted)), perm_sorted['importance'], color=colors)
    ax.set_yticks(range(len(perm_sorted)))
    ax.set_yticklabels(perm_sorted['feature'], fontsize=9)
    ax.set_xlabel('Permutation Importance (AUC Drop)')
    ax.set_title('Permutation Importance', fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_path}")
    plt.close()


def generate_report(individual_auc, rfe_results, sfs_results, perm_importance, output_path):
    """Generate comprehensive report."""
    with open(output_path, 'w') as f:
        f.write("# Systematic Feature Selection Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Data:** Oct 14, 2025 (3 CLDs, 219 edges, 9 Gen features)\n\n")
        
        f.write("---\n\n")
        f.write("## Summary of All Methods\n\n")
        
        # Method 1: Individual AUC
        f.write("### Method 1: Individual Feature AUC\n\n")
        f.write("| Rank | Feature | Standalone AUC |\n")
        f.write("|------|---------|----------------|\n")
        for i, (_, row) in enumerate(individual_auc.iterrows(), 1):
            f.write(f"| {i} | {row['feature']} | {row['auc']:.3f} |\n")
        
        best_single = individual_auc.iloc[0]
        f.write(f"\n**Best single feature:** {best_single['feature']} (AUC={best_single['auc']:.3f})\n\n")
        
        # Method 2: RFE
        f.write("### Method 2: Recursive Feature Elimination (RFE)\n\n")
        best_rfe_idx = rfe_results['auc'].idxmax()
        best_rfe = rfe_results.loc[best_rfe_idx]
        f.write(f"**Optimal:** {best_rfe['n_features']} features (AUC={best_rfe['auc']:.3f})\n\n")
        f.write("**Selected features:**\n")
        for feat in best_rfe['features']:
            f.write(f"- {feat}\n")
        
        # Method 3: SFS
        f.write("\n### Method 3: Sequential Forward Selection (SFS)\n\n")
        best_sfs_idx = sfs_results['auc'].idxmax()
        best_sfs = sfs_results.loc[best_sfs_idx]
        f.write(f"**Optimal:** {best_sfs['n_features']} features (AUC={best_sfs['auc']:.3f})\n\n")
        f.write("**Selected features:**\n")
        for feat in best_sfs['features']:
            f.write(f"- {feat}\n")
        
        # Method 4: Permutation Importance
        f.write("\n### Method 4: Permutation Importance\n\n")
        f.write("| Rank | Feature | Importance (AUC Drop) |\n")
        f.write("|------|---------|----------------------|\n")
        for i, (_, row) in enumerate(perm_importance.iterrows(), 1):
            f.write(f"| {i} | {row['feature']} | {row['importance']:+.3f} (±{row['std']:.3f}) |\n")
        
        f.write("\n---\n\n")
        f.write("## 🎯 Consensus Recommendation\n\n")
        
        # Find consensus
        top3_individual = set(individual_auc.head(3)['feature'].tolist())
        top3_perm = set(perm_importance.head(3)['feature'].tolist())
        consensus = top3_individual & top3_perm
        
        f.write(f"**Features appearing in top 3 of multiple methods:**\n\n")
        for feat in consensus:
            f.write(f"- ✅ **{feat}**\n")
        
        f.write(f"\n**Optimal feature count:** {best_rfe['n_features']}-{best_sfs['n_features']} features\n")
        f.write(f"**Expected AUC:** {max(best_rfe['auc'], best_sfs['auc']):.3f}\n\n")
        
        f.write("---\n\n")
        f.write("## Interpretation\n\n")
        f.write("- **Individual AUC:** Shows standalone predictive power of each feature\n")
        f.write("- **RFE:** Removes weakest features iteratively (finds minimal sufficient set)\n")
        f.write("- **SFS:** Adds strongest features iteratively (finds optimal additive set)\n")
        f.write("- **Permutation Importance:** Measures feature contribution in ensemble context\n\n")
    
    print(f"✅ Report saved: {output_path}")


def main():
    print("\n" + "="*80)
    print("SYSTEMATIC FEATURE SELECTION ANALYSIS")
    print("="*80)
    print()
    
    # Load data
    X, y = load_data()
    
    # Run all methods
    individual_auc = individual_feature_auc(X, y, ALL_9_FEATURES)
    rfe_results = recursive_feature_elimination(X, y, ALL_9_FEATURES)
    sfs_results = sequential_forward_selection(X, y, ALL_9_FEATURES)
    perm_importance = permutation_importance_analysis(X, y, ALL_9_FEATURES)
    
    # Generate outputs
    plot_path = OUTPUT_DIR / "feature_selection_analysis.png"
    plot_results(individual_auc, rfe_results, sfs_results, perm_importance, plot_path)
    
    report_path = OUTPUT_DIR / f"feature_selection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    generate_report(individual_auc, rfe_results, sfs_results, perm_importance, report_path)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 Output: {OUTPUT_DIR}")
    print(f"📊 Plot: {plot_path.name}")
    print(f"📄 Report: {report_path.name}\n")


if __name__ == "__main__":
    main()

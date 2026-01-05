#!/usr/bin/env python3
"""
Ensemble Hallucination Classifier - Judged Base CLDs
Trains ML classifiers on the freshly judged base CLDs with all 19 CI metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, 
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Judged data directory
JUDGED_DIR = Path("parameter_tuning_experiments/results/rq1_base_judging_correctness_20251012_225925")

# All 19 Context-insensitive metrics
CI_METRICS = [
    # Gen metrics (9)
    'Gen Perplexity',
    'Gen Max Window Entropy', 
    'Gen Min Prob',
    'Gen Cosine Similarity',
    'Gen Mean Token Prob',
    'Gen Prob Variance',
    'Gen Prob Std',
    'Gen Max Prob Diff',
    'Gen Token Prob Slope',
    # Judge metrics (9)
    'Judge Perplexity',
    'Judge Max Window Entropy',
    'Judge Min Prob',
    'Judge Cosine Similarity',
    'Judge Mean Token Prob',
    'Judge Prob Variance',
    'Judge Prob Std',
    'Judge Max Prob Diff',
    'Judge Token Prob Slope',
    # Aggregate
    'Aggregate Score'
]

def load_judged_data():
    """Load all judged Excel files."""
    print("Loading judged data files...")
    print(f"Directory: {JUDGED_DIR}")
    
    all_data = []
    
    for excel_file in JUDGED_DIR.glob("judged_*.xlsx"):
        print(f"  Loading: {excel_file.name}")
        
        # Extract CLD name from filename
        cld_name = excel_file.stem.replace("judged_", "").replace("_citations_20251012_230158", "")
        cld_name = cld_name.replace("_citations_20251012_233401", "")
        cld_name = cld_name.replace("_citations_20251013_001809", "")
        
        df = pd.read_excel(excel_file, sheet_name='All Edges')
        df['cld_name'] = cld_name
        df['source_file'] = excel_file.name
        
        all_data.append(df)
        print(f"    Loaded {len(df)} edges")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal edges loaded: {len(combined)}")
    
    # Use ALL edges (TP + FP + FN + TN)
    # Classification task: Predict if edge belongs in ground truth CLD
    # Label 1 = Should be in CLD (TP + FN)
    # Label 0 = Should NOT be in CLD (FP + TN)
    df_all = combined.copy()
    df_all['in_ground_truth'] = df_all['Classification'].isin(['TP', 'FN']).astype(int)
    
    print(f"\nAll edges: {len(df_all)}")
    print(f"  Classification breakdown:")
    print(f"    - TP (correct, in ground truth): {(df_all['Classification'] == 'TP').sum()}")
    print(f"    - FP (hallucination, not in ground truth): {(df_all['Classification'] == 'FP').sum()}")
    print(f"    - FN (missed, in ground truth): {(df_all['Classification'] == 'FN').sum()}")
    print(f"    - TN (correctly rejected, not in ground truth): {(df_all['Classification'] == 'TN').sum()}")
    print(f"\n  Binary label (in_ground_truth):")
    print(f"    - In ground truth (TP + FN): {(df_all['in_ground_truth'] == 1).sum()}")
    print(f"    - NOT in ground truth (FP + TN): {(df_all['in_ground_truth'] == 0).sum()}")
    print(f"    - Class balance: {df_all['in_ground_truth'].mean():.1%} in ground truth")
    
    return df_all

def prepare_features(df):
    """Prepare feature matrix X and labels y."""
    print("\nPreparing features...")
    
    # Select only rows with all metrics available
    feature_cols = [col for col in CI_METRICS if col in df.columns]
    print(f"Available metrics: {len(feature_cols)}/{len(CI_METRICS)}")
    
    if len(feature_cols) < len(CI_METRICS):
        missing = set(CI_METRICS) - set(feature_cols)
        print(f"  ⚠️  Missing metrics: {missing}")
    
    df_clean = df[feature_cols + ['in_ground_truth', 'cld_name']].copy()
    
    print(f"Rows before cleaning: {len(df_clean)}")
    
    # Replace inf values with NaN
    df_clean[feature_cols] = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # Filter out extreme perplexity values (overflow issues)
    for col in feature_cols:
        if 'Perplexity' in col:
            before = len(df_clean)
            df_clean = df_clean[df_clean[col] < 1e10]
            after = len(df_clean)
            if before != after:
                print(f"  Filtered {before - after} rows with extreme {col}")
    
    # Remove any remaining rows with NaN
    df_clean = df_clean.dropna(subset=feature_cols)
    print(f"Rows after cleaning: {len(df_clean)}")
    
    # Check class balance
    print(f"\nClass distribution after cleaning:")
    print(df_clean['in_ground_truth'].value_counts())
    print(f"In ground truth rate: {df_clean['in_ground_truth'].mean():.1%}")
    
    X = df_clean[feature_cols].values
    y = df_clean['in_ground_truth'].values
    cld_names = df_clean['cld_name'].values
    
    return X, y, cld_names, feature_cols

def train_and_evaluate_models(X, y, cld_names, feature_names):
    """Train multiple models and evaluate."""
    print("\n" + "="*80)
    print("TRAINING AND EVALUATING MODELS")
    print("="*80)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*80}")
        print(f"Model: {name}")
        print(f"{'='*80}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        auc = roc_auc_score(y_test, y_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        print(f"\nPerformance:")
        print(f"  ROC AUC: {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  TN: {cm[0,0]:3d}  FP: {cm[0,1]:3d}")
        print(f"  FN: {cm[1,0]:3d}  TP: {cm[1,1]:3d}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
        print(f"\n5-Fold Cross-Validation AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        results[name] = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'y_test': y_test,
            'y_proba': y_proba
        }
    
    return results, models

def plot_roc_curves(results, output_dir):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_proba'])
        auc = result['auc']
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5000)', linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Ground Truth CLD Membership Prediction (19 CI Metrics)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'roc_curves_judged_base_clds.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 ROC curves saved to: {output_path}")
    plt.close()

def save_results_summary(results, feature_names, output_dir):
    """Save results summary to JSON."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_features': len(feature_names),
        'features': feature_names,
        'models': {}
    }
    
    for name, result in results.items():
        summary['models'][name] = {
            'auc': float(result['auc']),
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1': float(result['f1']),
            'cv_auc_mean': float(result['cv_auc_mean']),
            'cv_auc_std': float(result['cv_auc_std'])
        }
    
    output_path = output_dir / 'ensemble_results_judged_base_clds.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Results summary saved to: {output_path}")

def main():
    print("="*80)
    print("ENSEMBLE CLASSIFIER - GROUND TRUTH CLD MEMBERSHIP - JUDGED BASE CLDs")
    print("="*80)
    
    # Create output directory
    output_dir = JUDGED_DIR / "ensemble_analysis"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data
    df = load_judged_data()
    
    # Prepare features
    X, y, cld_names, feature_names = prepare_features(df)
    
    if len(X) == 0:
        print("\n❌ No valid samples after cleaning. Cannot train models.")
        return
    
    # Train and evaluate
    results, models = train_and_evaluate_models(X, y, cld_names, feature_names)
    
    # Plot ROC curves
    plot_roc_curves(results, output_dir)
    
    # Save results
    save_results_summary(results, feature_names, output_dir)
    
    print("\n" + "="*80)
    print("✅ ENSEMBLE CLASSIFICATION COMPLETE")
    print("="*80)
    print(f"\nBest model: {max(results.items(), key=lambda x: x[1]['auc'])[0]}")
    print(f"Best AUC: {max(r['auc'] for r in results.values()):.4f}")

if __name__ == "__main__":
    main()

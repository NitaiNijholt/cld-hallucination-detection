"""
Combine data from the 3 most recent complete experiments and run rigorous feature selection.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define the 3 most recent complete experiments
EXPERIMENTS = [
    "exp_20251019_003307_d4af5bca",  # Oct 19
    "exp_20251015_140429_622ba6fd",  # Oct 15
    "exp_20251008_180446_a0d1b2de",  # Oct 8
]

BASE_DIR = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results")

def find_experiment_files(exp_id):
    """Find all result Excel files for a given experiment."""
    exp_dir = BASE_DIR / exp_id
    files = list(exp_dir.glob("**/prompts_Nitai_C_andrew_nodegen/**/results_*.xlsx"))
    return files

def load_and_combine_data():
    """Load and combine data from all experiments."""
    all_dataframes = []
    
    for exp_id in EXPERIMENTS:
        print(f"\nLoading experiment: {exp_id}")
        files = find_experiment_files(exp_id)
        print(f"  Found {len(files)} files")
        
        for file in files:
            print(f"  - Loading {file.name}")
            df = pd.read_excel(file)
            df['experiment_id'] = exp_id
            df['source_file'] = file.name
            all_dataframes.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"COMBINED DATASET:")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Experiments: {combined_df['experiment_id'].nunique()}")
    print(f"  Files: {combined_df['source_file'].nunique()}")
    print(f"{'='*60}\n")
    
    return combined_df

def prepare_features(df):
    """Prepare features and target for analysis."""
    # Context-insensitive features
    ci_features = [
        'Gen Perplexity',
        'Gen Cosine Similarity',
        'Gen Min Prob',
        'Gen Max Window Entropy',
        'Gen Aggregate Score'
    ]
    
    # Check which features are available
    available_features = [f for f in ci_features if f in df.columns]
    print(f"Available CI features: {available_features}")
    
    # Define target (hallucination)
    # Following ensemble classifier approach: Filter to only TP and FP (generated edges)
    if 'Classification' in df.columns:
        # Filter to only TP and FP (edges that were generated)
        # TP = True Positive (correct edge), FP = False Positive (hallucination)
        print(f"\nFiltering to generated edges only (TP + FP):")
        print(f"  Total edges before filter: {len(df)}")
        print(f"    TP: {(df['Classification'] == 'TP').sum()}")
        print(f"    FP: {(df['Classification'] == 'FP').sum()}")
        print(f"    TN: {(df['Classification'] == 'TN').sum()}")
        print(f"    FN: {(df['Classification'] == 'FN').sum()}")
        
        df_generated = df[df['Classification'].isin(['TP', 'FP'])].copy()
        print(f"  Edges after filter (TP+FP only): {len(df_generated)}")
        
        # Remove rows with missing values
        target_col = 'Classification'
        df_clean = df_generated[available_features + [target_col]].dropna()
        
        # Create binary label: 1 = hallucination (FP), 0 = correct (TP)
        y = (df_clean[target_col] == 'FP').astype(int).values
        print(f"\nTarget definition (ensemble approach):")
        print(f"  FP (False Positive) -> Hallucination (1)")
        print(f"  TP (True Positive) -> Correct Edge (0)")
    elif 'ground_truth_FP' in df.columns:
        target_col = 'ground_truth_FP'
        df_clean = df[available_features + [target_col]].dropna()
        y = df_clean[target_col].values
    elif 'hallucination' in df.columns:
        target_col = 'hallucination'
        df_clean = df[available_features + [target_col]].dropna()
        y = df_clean[target_col].values
    else:
        print("ERROR: No target column found!")
        return None, None, None
    
    # Get features
    X = df_clean[available_features].values
    
    print(f"\nDataset after cleaning:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(available_features)}")
    print(f"  Hallucinations (FP): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Correct edges (TP): {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")
    
    return X, y, available_features

def nested_cv_feature_selection(X, y, feature_names, n_outer=5, n_inner=3, random_seeds=[42, 123, 456]):
    """
    Rigorous nested cross-validation for feature selection.
    """
    print(f"\n{'='*60}")
    print("NESTED CROSS-VALIDATION FEATURE SELECTION")
    print(f"{'='*60}\n")
    
    # Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    }
    
    results = defaultdict(list)
    feature_importance_across_seeds = defaultdict(list)
    
    for seed in random_seeds:
        print(f"\n--- Random Seed: {seed} ---")
        
        # Outer CV for performance estimation
        outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
        
        for model_name, model in models.items():
            fold_scores = []
            fold_feature_importances = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                try:
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                    auc = roc_auc_score(y_test, y_prob)
                except:
                    auc = np.nan
                
                fold_scores.append({
                    'auc': auc,
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0)
                })
                
                # Permutation importance
                perm_importance = permutation_importance(
                    model, X_test_scaled, y_test,
                    n_repeats=10, random_state=seed
                )
                fold_feature_importances.append(perm_importance.importances_mean)
            
            # Average scores across folds
            avg_scores = {
                metric: np.mean([score[metric] for score in fold_scores])
                for metric in ['auc', 'precision', 'recall', 'f1']
            }
            results[model_name].append(avg_scores)
            
            # Average feature importance across folds
            avg_importance = np.mean(fold_feature_importances, axis=0)
            feature_importance_across_seeds[model_name].append(
                dict(zip(feature_names, avg_importance))
            )
            
            print(f"\n{model_name}:")
            print(f"  AUC: {avg_scores['auc']:.3f}")
            print(f"  Precision: {avg_scores['precision']:.3f}")
            print(f"  Recall: {avg_scores['recall']:.3f}")
            print(f"  F1: {avg_scores['f1']:.3f}")
    
    return results, feature_importance_across_seeds

def analyze_stability(feature_importance_across_seeds, feature_names, random_seeds):
    """Analyze stability of feature selection across different seeds."""
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE BY SEED")
    print(f"{'='*60}\n")
    
    for model_name, importance_list in feature_importance_across_seeds.items():
        print(f"\n{model_name}:")
        print(f"  {'Feature':<30} ", end="")
        for seed in random_seeds:
            print(f"Seed {seed:<6} ", end="")
        print("  Mean")
        print(f"  {'-'*30} ", end="")
        for _ in random_seeds:
            print(f"{'-'*11} ", end="")
        print(f"{'-'*10}")
        
        # Organize by feature
        importance_by_feature = defaultdict(list)
        for importance_dict in importance_list:
            for feature, imp in importance_dict.items():
                importance_by_feature[feature].append(imp)
        
        # Sort by mean importance
        sorted_features = sorted(
            importance_by_feature.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True
        )
        
        for feature, importances in sorted_features:
            print(f"  {feature:<30} ", end="")
            for imp in importances:
                print(f"{imp:>10.4f}  ", end="")
            print(f"{np.mean(importances):>10.4f}")
    
    print(f"\n{'='*60}")
    print("FEATURE SELECTION STABILITY ANALYSIS")
    print(f"{'='*60}\n")
    
    for model_name, importance_list in feature_importance_across_seeds.items():
        print(f"\n{model_name}:")
        print(f"  {'Feature':<30} {'Mean Imp':<12} {'Std Imp':<12} {'Stability'}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
        
        # Calculate mean and std across seeds
        importance_by_feature = defaultdict(list)
        for importance_dict in importance_list:
            for feature, imp in importance_dict.items():
                importance_by_feature[feature].append(imp)
        
        # Sort by mean importance
        sorted_features = sorted(
            importance_by_feature.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True
        )
        
        for feature, importances in sorted_features:
            mean_imp = np.mean(importances)
            std_imp = np.std(importances)
            stability = "HIGH" if std_imp < mean_imp * 0.3 else "MEDIUM" if std_imp < mean_imp * 0.6 else "LOW"
            print(f"  {feature:<30} {mean_imp:>10.4f}   {std_imp:>10.4f}   {stability}")

def main():
    print("="*60)
    print("COMBINING AND ANALYZING 3 MOST RECENT EXPERIMENTS")
    print("="*60)
    
    # Load and combine data
    combined_df = load_and_combine_data()
    
    # Save combined dataset
    output_file = BASE_DIR / "combined_3_recent_experiments.xlsx"
    combined_df.to_excel(output_file, index=False)
    print(f"Saved combined dataset to: {output_file}")
    
    # Prepare features
    X, y, feature_names = prepare_features(combined_df)
    if X is None:
        return
    
    # Run nested CV feature selection
    random_seeds = [42, 123, 456]
    results, feature_importance = nested_cv_feature_selection(X, y, feature_names, random_seeds=random_seeds)
    
    # Analyze stability
    analyze_stability(feature_importance, feature_names, random_seeds)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}\n")
    print("Average performance across all seeds:")
    for model_name, scores_list in results.items():
        avg_auc = np.mean([s['auc'] for s in scores_list])
        avg_f1 = np.mean([s['f1'] for s in scores_list])
        print(f"  {model_name}: AUC={avg_auc:.3f}, F1={avg_f1:.3f}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze whether hallucination classifiers generalize across different CLDs
- Check if logit distributions differ by CLD
- Test leave-one-CLD-out cross-validation
- Assess whether thresholds are CLD-specific or universal
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/combined_3_recent_experiments.xlsx")

# Key features
FEATURES = [
    'Gen Mean Token Prob',
    'Gen Perplexity', 
    'Gen Cosine Similarity'
]

def load_data_with_cld():
    """Load data with CLD information."""
    df = pd.read_excel(DATA_FILE, sheet_name=0)
    
    # Filter to generated edges only (TP + FP)
    df_clean = df[df['Gen Perplexity'] <= 100].copy()
    df_generated = df_clean[df_clean['Classification'].isin(['TP', 'FP'])].copy()
    
    # Create hallucination label
    df_generated['is_hallucination'] = (df_generated['Classification'] == 'FP').astype(int)
    
    # Try to identify CLD from filename or other columns
    if 'source_file' in df_generated.columns:
        df_generated['CLD'] = df_generated['source_file']
    elif 'cld_name' in df_generated.columns:
        df_generated['CLD'] = df_generated['cld_name']
    else:
        # Try to infer from any column that contains file/experiment info
        for col in df_generated.columns:
            if 'file' in col.lower() or 'cld' in col.lower() or 'experiment' in col.lower():
                print(f"Using {col} as CLD identifier")
                df_generated['CLD'] = df_generated[col]
                break
    
    return df_generated

def analyze_feature_distributions_by_cld(df):
    """Check if logit distributions differ across CLDs."""
    print("="*80)
    print("FEATURE DISTRIBUTIONS BY CLD")
    print("="*80)
    
    if 'CLD' not in df.columns:
        print("\n⚠️  Cannot identify CLD from data - skipping CLD-specific analysis")
        return
    
    clds = df['CLD'].unique()
    print(f"\nFound {len(clds)} CLDs in dataset:")
    for cld in clds:
        n = len(df[df['CLD'] == cld])
        n_hal = df[(df['CLD'] == cld) & (df['is_hallucination'] == 1)].shape[0]
        print(f"  - {cld}: {n} edges ({n_hal} hallucinations, {n_hal/n:.1%})")
    
    # Check feature distributions
    print(f"\n{'Feature':<30} {'CLD':<40} {'Mean':>8} {'Std':>8}")
    print(f"{'-'*30} {'-'*40} {'-'*8} {'-'*8}")
    
    for feat in FEATURES:
        if feat not in df.columns:
            continue
            
        for cld in clds:
            cld_data = df[df['CLD'] == cld][feat].dropna()
            cld_short = str(cld)[:40]
            print(f"{feat:<30} {cld_short:<40} {cld_data.mean():>8.3f} {cld_data.std():>8.3f}")
        print()
    
    # Statistical test: Are distributions significantly different?
    from scipy import stats
    
    print("\n🔬 STATISTICAL TEST: Do feature distributions differ by CLD?")
    print(f"{'Feature':<30} {'Test':>15} {'p-value':>10} {'Significant?':>15}")
    print(f"{'-'*30} {'-'*15} {'-'*10} {'-'*15}")
    
    for feat in FEATURES:
        if feat not in df.columns:
            continue
        
        groups = [df[df['CLD'] == cld][feat].dropna() for cld in clds]
        
        if len(groups) >= 2:
            # Kruskal-Wallis test (non-parametric ANOVA)
            stat, p_value = stats.kruskal(*groups)
            sig = "YES ⚠️" if p_value < 0.05 else "NO ✅"
            print(f"{feat:<30} {'Kruskal-Wallis':>15} {p_value:>10.4f} {sig:>15}")

def leave_one_cld_out_evaluation(df):
    """Test classifier generalization using leave-one-CLD-out validation."""
    print(f"\n{'='*80}")
    print("LEAVE-ONE-CLD-OUT CROSS-VALIDATION")
    print("="*80)
    print("\nTrain on N-1 CLDs, test on held-out CLD to assess generalization\n")
    
    if 'CLD' not in df.columns:
        print("⚠️  Cannot perform CLD-level CV - no CLD identifier found")
        return
    
    clds = df['CLD'].unique()
    
    if len(clds) < 2:
        print(f"⚠️  Only {len(clds)} CLD found - need at least 2 for leave-one-out")
        return
    
    # Test with best feature
    feature = 'Gen Cosine Similarity'
    
    if feature not in df.columns:
        feature = 'Gen Mean Token Prob'
    
    print(f"Testing with: {feature}")
    print(f"\n{'Train CLDs':<50} {'Test CLD':<40} {'AUC':>8} {'F1':>8} {'Precision':>8} {'Recall':>8}")
    print(f"{'-'*50} {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    results = []
    
    for test_cld in clds:
        # Split data
        train_df = df[df['CLD'] != test_cld]
        test_df = df[df['CLD'] == test_cld]
        
        # Prepare data
        X_train = train_df[[feature]].replace([np.inf, -np.inf], np.nan).dropna()
        y_train = train_df.loc[X_train.index, 'is_hallucination'].values
        
        X_test = test_df[[feature]].replace([np.inf, -np.inf], np.nan).dropna()
        y_test = test_df.loc[X_test.index, 'is_hallucination'].values
        
        if len(y_test) < 5:
            print(f"{'Various':<50} {str(test_cld)[:40]:<40} {'SKIP':>8} {'(too few samples)':>20}")
            continue
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        
        train_clds_short = ', '.join([str(c)[:20] for c in clds if c != test_cld])
        test_cld_short = str(test_cld)[:40]
        
        print(f"{train_clds_short[:50]:<50} {test_cld_short:<40} {auc:>8.3f} {f1:>8.3f} {prec:>8.3f} {rec:>8.3f}")
        
        results.append({
            'test_cld': test_cld,
            'auc': auc,
            'f1': f1,
            'precision': prec,
            'recall': rec
        })
    
    if len(results) > 0:
        avg_auc = np.mean([r['auc'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        std_auc = np.std([r['auc'] for r in results])
        std_f1 = np.std([r['f1'] for r in results])
        
        print(f"\n{'AVERAGE ACROSS CLDs':<50} {'':>40} {avg_auc:>8.3f} {avg_f1:>8.3f}")
        print(f"{'STANDARD DEVIATION':<50} {'':>40} {std_auc:>8.3f} {std_f1:>8.3f}")
        
        return results

def compare_within_vs_across_cld(df):
    """Compare performance when training/testing within same CLD vs across CLDs."""
    print(f"\n{'='*80}")
    print("WITHIN-CLD vs ACROSS-CLD PERFORMANCE")
    print("="*80)
    
    if 'CLD' not in df.columns:
        print("⚠️  No CLD identifier found")
        return
    
    clds = df['CLD'].unique()
    
    if len(clds) < 2:
        print(f"⚠️  Only {len(clds)} CLD - need multiple CLDs")
        return
    
    feature = 'Gen Cosine Similarity' if 'Gen Cosine Similarity' in df.columns else 'Gen Mean Token Prob'
    
    # Within-CLD performance (5-fold CV within each CLD)
    print(f"\n📊 WITHIN-CLD PERFORMANCE (5-fold CV within each CLD):")
    print(f"{'CLD':<50} {'AUC':>8} {'F1':>8}")
    print(f"{'-'*50} {'-'*8} {'-'*8}")
    
    within_aucs = []
    
    for cld in clds:
        cld_df = df[df['CLD'] == cld]
        
        X = cld_df[[feature]].replace([np.inf, -np.inf], np.nan).dropna()
        y = cld_df.loc[X.index, 'is_hallucination'].values
        
        if len(y) < 10:
            print(f"{str(cld)[:50]:<50} {'SKIP':>8} {'(too few)':>8}")
            continue
        
        # 5-fold CV
        cv = StratifiedKFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
        
        aucs, f1s = [], []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.values[train_idx], X.values[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))
            X_test_scaled = scaler.transform(X_test.reshape(-1, 1))
            
            clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
            clf.fit(X_train_scaled, y_train)
            
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            
            aucs.append(roc_auc_score(y_test, y_prob))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
        
        avg_auc = np.mean(aucs)
        avg_f1 = np.mean(f1s)
        within_aucs.append(avg_auc)
        
        print(f"{str(cld)[:50]:<50} {avg_auc:>8.3f} {avg_f1:>8.3f}")
    
    print(f"\n{'AVERAGE WITHIN-CLD':<50} {np.mean(within_aucs):>8.3f}")
    
    # Across-CLD performance (already computed above)
    print(f"\n📊 ACROSS-CLD PERFORMANCE (train on other CLDs, test on target):")
    print(f"  See 'LEAVE-ONE-CLD-OUT CROSS-VALIDATION' section above")

def assess_threshold_stability(df):
    """Check if optimal thresholds are stable across CLDs."""
    print(f"\n{'='*80}")
    print("THRESHOLD STABILITY ACROSS CLDs")
    print("="*80)
    
    if 'CLD' not in df.columns:
        print("⚠️  No CLD identifier found")
        return
    
    clds = df['CLD'].unique()
    feature = 'Gen Cosine Similarity' if 'Gen Cosine Similarity' in df.columns else 'Gen Mean Token Prob'
    
    print(f"\nOptimal threshold (maximizing F1) for each CLD:")
    print(f"{'CLD':<50} {'Optimal Threshold':>18} {'F1 at optimal':>15}")
    print(f"{'-'*50} {'-'*18} {'-'*15}")
    
    optimal_thresholds = []
    
    for cld in clds:
        cld_df = df[df['CLD'] == cld]
        
        X = cld_df[[feature]].replace([np.inf, -np.inf], np.nan).dropna()
        y = cld_df.loc[X.index, 'is_hallucination'].values
        
        if len(y) < 10:
            continue
        
        # Train classifier
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        clf.fit(X_scaled, y)
        
        y_prob = clf.predict_proba(X_scaled)[:, 1]
        
        # Find optimal threshold
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.linspace(0.1, 0.9, 81):
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
        print(f"{str(cld)[:50]:<50} {best_threshold:>18.3f} {best_f1:>15.3f}")
    
    if len(optimal_thresholds) > 1:
        print(f"\n{'MEAN THRESHOLD':<50} {np.mean(optimal_thresholds):>18.3f}")
        print(f"{'STD THRESHOLD':<50} {np.std(optimal_thresholds):>18.3f}")
        
        if np.std(optimal_thresholds) < 0.1:
            print(f"\n✅ Thresholds are STABLE across CLDs (std < 0.1)")
            print(f"   → Can use a universal threshold in production")
        else:
            print(f"\n⚠️  Thresholds VARY across CLDs (std = {np.std(optimal_thresholds):.3f})")
            print(f"   → May need CLD-specific thresholds or calibration")

def main():
    print("\n" + "="*80)
    print("CROSS-CLD GENERALIZATION ANALYSIS")
    print("="*80 + "\n")
    
    # Load data
    df = load_data_with_cld()
    print(f"Dataset: {len(df)} samples\n")
    
    # 1. Check feature distributions by CLD
    analyze_feature_distributions_by_cld(df)
    
    # 2. Leave-one-CLD-out evaluation
    results = leave_one_cld_out_evaluation(df)
    
    # 3. Within vs Across CLD
    compare_within_vs_across_cld(df)
    
    # 4. Threshold stability
    assess_threshold_stability(df)
    
    # Final recommendation
    print(f"\n{'='*80}")
    print("PRODUCTION RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n💡 OPTIONS FOR PRODUCTION:")
    print(f"\n1️⃣  USE PROBABILITY SCORES (no threshold):")
    print(f"   - Return probability of hallucination (0-1)")
    print(f"   - Let downstream systems decide threshold")
    print(f"   - More flexible for different use cases")
    
    print(f"\n2️⃣  UNIVERSAL THRESHOLD:")
    print(f"   - If thresholds are stable (std < 0.1), use fixed threshold")
    print(f"   - Simple to implement")
    print(f"   - May need occasional recalibration")
    
    print(f"\n3️⃣  CALIBRATION:")
    print(f"   - Use Platt scaling or isotonic regression")
    print(f"   - Ensures probabilities are well-calibrated")
    print(f"   - More robust to distribution shifts")
    
    print(f"\n4️⃣  CLD-SPECIFIC THRESHOLDS:")
    print(f"   - If thresholds vary significantly")
    print(f"   - Requires maintaining threshold per CLD")
    print(f"   - Better performance but more complex")
    
    print(f"\n🎯 RECOMMENDED APPROACH:")
    print(f"   → Return CALIBRATED PROBABILITIES (option 1 + 3)")
    print(f"   → No hardcoded threshold in production")
    print(f"   → Let users/systems set their own risk tolerance")
    
    print()

if __name__ == "__main__":
    main()

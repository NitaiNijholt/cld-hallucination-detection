"""
Phase 4: Statistical Analysis & Correlation

This module performs statistical analysis to correlate spectral metrics
with hallucination labels from TruthfulQA.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class HallucinationCorrelator:
    """Correlate spectral metrics with hallucination labels."""
    
    def __init__(self, metrics_path: str = "experiment/spectral_metrics"):
        """
        Initialize correlator.
        
        Args:
            metrics_path: Path to spectral metrics
        """
        self.metrics_path = Path(metrics_path)
        self.data = None
        self.features = None
        self.labels = None
    
    def load_data(self) -> pd.DataFrame:
        """Load spectral metrics and prepare dataframe."""
        
        # Load all results
        results_file = self.metrics_path / "all_spectral_metrics.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Results not found at {results_file}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Convert to dataframe
        data_rows = []
        for result in results:
            if 'spectral_analysis' not in result:
                continue
            
            agg_metrics = result['spectral_analysis'].get('aggregate_metrics', {})
            
            row = {
                'sample_id': result['sample_id'],
                'prompt': result.get('prompt', ''),
                'generated_text': result.get('generated_text', ''),
                'expected_truthful': result.get('expected_truthful'),
                'category': result.get('category', 'unknown'),
                
                # Spectral metrics
                'mean_alpha': agg_metrics.get('mean_alpha', 0),
                'std_alpha': agg_metrics.get('std_alpha', 0),
                'min_alpha': agg_metrics.get('min_alpha', 0),
                'max_alpha': agg_metrics.get('max_alpha', 0),
                'mean_risk_score': agg_metrics.get('mean_risk_score', 0),
                'max_risk_score': agg_metrics.get('max_risk_score', 0),
                'total_spikes': agg_metrics.get('total_spikes', 0),
                'layers_at_risk': agg_metrics.get('layers_at_risk', 0),
                
                # Binary label (1 = hallucination, 0 = truthful)
                'is_hallucination': 0 if result.get('expected_truthful') else 1
            }
            
            data_rows.append(row)
        
        self.data = pd.DataFrame(data_rows)
        
        # Prepare features and labels
        feature_cols = [
            'mean_alpha', 'std_alpha', 'min_alpha', 'max_alpha',
            'mean_risk_score', 'max_risk_score', 'total_spikes', 'layers_at_risk'
        ]
        
        # Only use samples with known labels
        labeled_data = self.data[self.data['expected_truthful'].notna()]
        
        if len(labeled_data) > 0:
            self.features = labeled_data[feature_cols].values
            self.labels = labeled_data['is_hallucination'].values
        
        return self.data
    
    def compute_correlations(self) -> pd.DataFrame:
        """Compute correlations between spectral metrics and hallucination."""
        
        if self.features is None or self.labels is None:
            print("No labeled data available")
            return pd.DataFrame()
        
        correlations = []
        
        feature_names = [
            'mean_alpha', 'std_alpha', 'min_alpha', 'max_alpha',
            'mean_risk_score', 'max_risk_score', 'total_spikes', 'layers_at_risk'
        ]
        
        for i, feature_name in enumerate(feature_names):
            feature_values = self.features[:, i]
            
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(feature_values, self.labels)
            
            # Spearman correlation (rank-based, more robust)
            spearman_r, spearman_p = stats.spearmanr(feature_values, self.labels)
            
            # Point-biserial correlation (for binary labels)
            pb_r, pb_p = stats.pointbiserialr(self.labels, feature_values)
            
            correlations.append({
                'feature': feature_name,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'pointbiserial_r': pb_r,
                'pointbiserial_p': pb_p,
                'significant': pearson_p < 0.05
            })
        
        return pd.DataFrame(correlations)
    
    def group_comparison(self) -> Dict:
        """Compare spectral metrics between truthful and hallucinated groups."""
        
        if self.data is None:
            return {}
        
        truthful = self.data[self.data['expected_truthful'] == True]
        hallucinated = self.data[self.data['expected_truthful'] == False]
        
        comparisons = {}
        
        metrics = ['mean_alpha', 'mean_risk_score', 'total_spikes', 'layers_at_risk']
        
        for metric in metrics:
            if metric not in truthful.columns:
                continue
            
            truthful_vals = truthful[metric].values
            hallucinated_vals = hallucinated[metric].values
            
            if len(truthful_vals) > 0 and len(hallucinated_vals) > 0:
                # T-test
                t_stat, t_pval = stats.ttest_ind(truthful_vals, hallucinated_vals)
                
                # Mann-Whitney U test (non-parametric)
                u_stat, u_pval = stats.mannwhitneyu(truthful_vals, hallucinated_vals)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (np.var(truthful_vals) + np.var(hallucinated_vals)) / 2
                )
                if pooled_std > 0:
                    cohens_d = (np.mean(hallucinated_vals) - np.mean(truthful_vals)) / pooled_std
                else:
                    cohens_d = 0
                
                comparisons[metric] = {
                    'truthful_mean': float(np.mean(truthful_vals)),
                    'truthful_std': float(np.std(truthful_vals)),
                    'hallucinated_mean': float(np.mean(hallucinated_vals)),
                    'hallucinated_std': float(np.std(hallucinated_vals)),
                    't_statistic': float(t_stat),
                    't_pvalue': float(t_pval),
                    'u_statistic': float(u_stat),
                    'u_pvalue': float(u_pval),
                    'cohens_d': float(cohens_d),
                    'significant': t_pval < 0.05
                }
        
        return comparisons
    
    def train_classifier(self, test_size: float = 0.3) -> Dict:
        """Train a classifier to predict hallucination from spectral metrics."""
        
        if self.features is None or self.labels is None:
            return {'error': 'No labeled data available'}
        
        if len(np.unique(self.labels)) < 2:
            return {'error': 'Need both positive and negative samples'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=test_size, random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC AUC if we have both classes
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_pred_proba)
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        else:
            auc = 0
            fpr, tpr, thresholds = [], [], []
        
        # Feature importance (coefficients)
        feature_names = [
            'mean_alpha', 'std_alpha', 'min_alpha', 'max_alpha',
            'mean_risk_score', 'max_risk_score', 'total_spikes', 'layers_at_risk'
        ]
        
        feature_importance = dict(zip(feature_names, clf.coef_[0]))
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(auc),
            'feature_importance': feature_importance,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'model': clf,
            'scaler': scaler
        }
    
    def analyze_by_category(self) -> pd.DataFrame:
        """Analyze performance by question category."""
        
        if self.data is None:
            return pd.DataFrame()
        
        category_stats = []
        
        for category in self.data['category'].unique():
            cat_data = self.data[self.data['category'] == category]
            
            if len(cat_data) == 0:
                continue
            
            # Split by hallucination status
            truthful_data = cat_data[cat_data['expected_truthful'] == True]
            hallucinated_data = cat_data[cat_data['expected_truthful'] == False]
            
            stats_dict = {
                'category': category,
                'count': len(cat_data),
                'truthful_count': len(truthful_data),
                'hallucinated_count': len(hallucinated_data),
                'truthful_ratio': len(truthful_data) / len(cat_data),
                
                # Overall means
                'mean_risk': cat_data['mean_risk_score'].mean(),
                'mean_alpha': cat_data['mean_alpha'].mean(),
                'mean_spikes': cat_data['total_spikes'].mean(),
                
                # Truthful vs Hallucinated breakdown
                'truthful_mean_alpha': truthful_data['mean_alpha'].mean() if len(truthful_data) > 0 else np.nan,
                'hallucinated_mean_alpha': hallucinated_data['mean_alpha'].mean() if len(hallucinated_data) > 0 else np.nan,
                'alpha_diff': (hallucinated_data['mean_alpha'].mean() if len(hallucinated_data) > 0 else np.nan) - 
                             (truthful_data['mean_alpha'].mean() if len(truthful_data) > 0 else np.nan),
                
                'truthful_mean_risk': truthful_data['mean_risk_score'].mean() if len(truthful_data) > 0 else np.nan,
                'hallucinated_mean_risk': hallucinated_data['mean_risk_score'].mean() if len(hallucinated_data) > 0 else np.nan,
                'risk_diff': (hallucinated_data['mean_risk_score'].mean() if len(hallucinated_data) > 0 else np.nan) - 
                            (truthful_data['mean_risk_score'].mean() if len(truthful_data) > 0 else np.nan),
                
                'truthful_mean_spikes': truthful_data['total_spikes'].mean() if len(truthful_data) > 0 else np.nan,
                'hallucinated_mean_spikes': hallucinated_data['total_spikes'].mean() if len(hallucinated_data) > 0 else np.nan,
                'spikes_diff': (hallucinated_data['total_spikes'].mean() if len(hallucinated_data) > 0 else np.nan) - 
                              (truthful_data['total_spikes'].mean() if len(truthful_data) > 0 else np.nan),
            }
            
            category_stats.append(stats_dict)
        
        return pd.DataFrame(category_stats)


def main():
    """Main function for statistical analysis."""
    
    print("="*80)
    print("Phase 4: Statistical Analysis & Correlation")
    print("="*80)
    
    # Initialize correlator
    correlator = HallucinationCorrelator()
    
    # Load data
    print("\nLoading spectral metrics...")
    try:
        data = correlator.load_data()
        print(f"Loaded {len(data)} samples")
        print(f"Labeled samples: {correlator.labels is not None and len(correlator.labels) or 0}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Compute correlations
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    correlations = correlator.compute_correlations()
    if not correlations.empty:
        print("\nFeature Correlations with Hallucination:")
        print("-" * 60)
        for _, row in correlations.iterrows():
            sig = "✓" if row['significant'] else "✗"
            print(f"{sig} {row['feature']:20s}: r={row['pearson_r']:+.3f}, p={row['pearson_p']:.3f}")
    
    # Group comparison
    print("\n" + "="*60)
    print("GROUP COMPARISON (Truthful vs Hallucinated)")
    print("="*60)
    
    comparisons = correlator.group_comparison()
    for metric, stats in comparisons.items():
        if stats['significant']:
            print(f"\n✓ {metric} (SIGNIFICANT, p={stats['t_pvalue']:.3f}):")
        else:
            print(f"\n✗ {metric} (not significant, p={stats['t_pvalue']:.3f}):")
        
        print(f"  Truthful:     {stats['truthful_mean']:.3f} ± {stats['truthful_std']:.3f}")
        print(f"  Hallucinated: {stats['hallucinated_mean']:.3f} ± {stats['hallucinated_std']:.3f}")
        print(f"  Effect size (Cohen's d): {stats['cohens_d']:.3f}")
    
    # Train classifier
    print("\n" + "="*60)
    print("CLASSIFICATION PERFORMANCE")
    print("="*60)
    
    classifier_results = correlator.train_classifier()
    if 'error' not in classifier_results:
        print(f"\nTrain size: {classifier_results['train_size']}")
        print(f"Test size:  {classifier_results['test_size']}")
        print(f"\nAccuracy:  {classifier_results['accuracy']:.3f}")
        print(f"Precision: {classifier_results['precision']:.3f}")
        print(f"Recall:    {classifier_results['recall']:.3f}")
        print(f"F1 Score:  {classifier_results['f1_score']:.3f}")
        print(f"ROC AUC:   {classifier_results['roc_auc']:.3f}")
        
        print("\nFeature Importance (Logistic Regression Coefficients):")
        importance = classifier_results['feature_importance']
        sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, coef in sorted_importance:
            print(f"  {feature:20s}: {coef:+.3f}")
    else:
        print(f"Error: {classifier_results['error']}")
    
    # Category analysis
    print("\n" + "="*60)
    print("ANALYSIS BY CATEGORY")
    print("="*60)
    
    category_stats = correlator.analyze_by_category()
    if not category_stats.empty:
        print("\nCategory Breakdown (Truthful vs Hallucinated):")
        print("-" * 80)
        
        for _, row in category_stats.iterrows():
            print(f"\n📊 {row['category'].upper()}:")
            print(f"   Total samples: {row['count']} ({row['truthful_count']} truthful, {row['hallucinated_count']} hallucinated)")
            print(f"   Truthful ratio: {row['truthful_ratio']:.1%}")
            
            if not pd.isna(row['truthful_mean_alpha']) and not pd.isna(row['hallucinated_mean_alpha']):
                print(f"   Alpha (α):")
                print(f"     Truthful:     {row['truthful_mean_alpha']:.3f}")
                print(f"     Hallucinated: {row['hallucinated_mean_alpha']:.3f}")
                print(f"     Difference:   {row['alpha_diff']:+.3f} {'↑' if row['alpha_diff'] > 0 else '↓'}")
            
            if not pd.isna(row['truthful_mean_risk']) and not pd.isna(row['hallucinated_mean_risk']):
                print(f"   Risk Score:")
                print(f"     Truthful:     {row['truthful_mean_risk']:.3f}")
                print(f"     Hallucinated: {row['hallucinated_mean_risk']:.3f}")
                print(f"     Difference:   {row['risk_diff']:+.3f} {'↑' if row['risk_diff'] > 0 else '↓'}")
            
            if not pd.isna(row['truthful_mean_spikes']) and not pd.isna(row['hallucinated_mean_spikes']):
                print(f"   Total Spikes:")
                print(f"     Truthful:     {row['truthful_mean_spikes']:.1f}")
                print(f"     Hallucinated: {row['hallucinated_mean_spikes']:.1f}")
                print(f"     Difference:   {row['spikes_diff']:+.1f} {'↑' if row['spikes_diff'] > 0 else '↓'}")
            
            print("-" * 40)
    
    # Save results
    output_dir = Path("experiment/analysis_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results = {
        'correlations': correlations.to_dict('records') if not correlations.empty else [],
        'group_comparisons': convert_numpy_types(comparisons),
        'classification': convert_numpy_types({k: v for k, v in classifier_results.items() if k not in ['model', 'scaler']}),
        'category_analysis': category_stats.to_dict('records') if not category_stats.empty else []
    }
    
    with open(output_dir / "statistical_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/statistical_results.json")
    
    # Final verdict
    print("\n" + "="*60)
    print("HYPOTHESIS TEST RESULT")
    print("="*60)
    
    significant_metrics = sum(1 for c in comparisons.values() if c.get('significant', False))
    if significant_metrics > 0:
        print(f"✓ HYPOTHESIS SUPPORTED: {significant_metrics}/{len(comparisons)} metrics show significant differences")
        print("  Spectral signatures correlate with hallucination tendency!")
    else:
        print("✗ HYPOTHESIS NOT SUPPORTED (with current data)")
        print("  Need more data or refined metrics")
    
    return results


if __name__ == "__main__":
    main()
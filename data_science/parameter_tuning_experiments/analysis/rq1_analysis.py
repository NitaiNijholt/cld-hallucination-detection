"""
RQ1 Analysis - Hallucination Detection Performance

Analyzes judge performance in detecting corrupted edges.
Similar structure to RQ2 analysis but focuses on judging metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    precision_score, recall_score, f1_score, accuracy_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class RQ1Analysis:
    """
    Analyzes judge performance in detecting hallucinations (corrupted edges).
    
    Treats judge scores as binary classifiers and computes:
    - Confusion matrices
    - Precision, Recall, F1, Accuracy
    - ROC curves and AUC
    - Optimal thresholds
    """
    
    def __init__(self, phase2_dir: str):
        """
        Initialize with Phase 2 results directory.
        
        Args:
            phase2_dir: Path to Phase 2 results directory
        """
        self.phase2_dir = Path(phase2_dir)
        self.results = {}
        self.summary_df = None
        
        # Load all judged Excel files
        self.load_data()
    
    def load_data(self):
        """Load all judged Excel files from Phase 2."""
        
        judged_files = list(self.phase2_dir.glob("judged_*.xlsx"))
        
        if not judged_files:
            raise FileNotFoundError(f"No judged Excel files found in {self.phase2_dir}")
        
        print(f"Found {len(judged_files)} judged result files")
        
        # Load each file
        all_data = []
        for file in judged_files:
            # Extract strategy from filename
            parts = file.stem.split('_')
            if 'serial' in file.stem:
                strategy = 'serial_baseline'
            elif 'homogeneous' in file.stem:
                strategy = 'homogeneous_ensemble'
            elif 'heterogeneous' in file.stem:
                strategy = 'heterogeneous_ensemble'
            else:
                strategy = 'unknown'
            
            # Load Excel file - 'All Edges' sheet contains the data
            try:
                df = pd.read_excel(file, sheet_name='All Edges')
                df['judging_strategy'] = strategy
                df['result_file'] = file.name
                all_data.append(df)
                print(f"  Loaded {len(df)} edges from {strategy}")
            except Exception as e:
                print(f"  Warning: Could not load {file.name}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data could be loaded from Excel files")
        
        # Combine all data
        self.df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal edges loaded: {len(self.df)}")
        
        # Identify relevant columns
        self._identify_columns()
    
    def _identify_columns(self):
        """Identify key columns in the data."""
        
        cols = self.df.columns.tolist()
        
        # Ground truth column (corruption status)
        self.ground_truth_col = None
        for col in ['is_corrupted', 'corrupted', 'corruption_flag']:
            if col in cols:
                self.ground_truth_col = col
                break
        
        if self.ground_truth_col is None:
            print("Warning: No ground truth column found. Looking for alternatives...")
            # Check if there's a motivation_corrupted or similar
            corruption_cols = [c for c in cols if 'corrupt' in c.lower()]
            if corruption_cols:
                self.ground_truth_col = corruption_cols[0]
                print(f"  Using {self.ground_truth_col} as ground truth")
        
        # Judge score column (case-insensitive search)
        self.judge_score_col = None
        cols_lower = {c.lower(): c for c in cols}
        for col_lower in ['aggregate_score', 'judge_score', 'verdict_score', 'aggregate score']:
            if col_lower in cols_lower:
                self.judge_score_col = cols_lower[col_lower]
                break
        
        # Judge verdict column (case-insensitive search)
        self.judge_verdict_col = None
        for col_lower in ['aggregate_verdict', 'judge_verdict', 'verdict', 'judge verdict', 'aggregate verdict']:
            if col_lower in cols_lower:
                self.judge_verdict_col = cols_lower[col_lower]
                break
        
        print(f"\nIdentified columns:")
        print(f"  Ground truth: {self.ground_truth_col}")
        print(f"  Judge score: {self.judge_score_col}")
        print(f"  Judge verdict: {self.judge_verdict_col}")
        print(f"\nAvailable columns: {cols[:20]}...")
    
    def compute_classification_metrics(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Compute classification metrics for each judging strategy.
        
        Args:
            threshold: Threshold for converting judge scores to binary predictions
            
        Returns:
            DataFrame with metrics for each strategy
        """
        
        if self.ground_truth_col is None or self.judge_score_col is None:
            raise ValueError("Missing required columns for classification analysis")
        
        results = []
        
        for strategy in self.df['judging_strategy'].unique():
            strategy_df = self.df[self.df['judging_strategy'] == strategy].copy()
            
            # Remove NaN values
            valid_df = strategy_df[[self.ground_truth_col, self.judge_score_col]].dropna()
            
            if len(valid_df) == 0:
                print(f"Warning: No valid data for {strategy}")
                continue
            
            y_true = valid_df[self.ground_truth_col].astype(int)
            y_score = valid_df[self.judge_score_col]
            
            # Convert scores to binary predictions
            # Assuming lower score = not supported = hallucination
            y_pred = (y_score < threshold).astype(int)
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            # Compute metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_accuracy = (recall + specificity) / 2
            
            results.append({
                'judging_strategy': strategy,
                'threshold': threshold,
                'n_edges': len(valid_df),
                'n_corrupted': int(y_true.sum()),
                'n_not_corrupted': int((~y_true.astype(bool)).sum()),
                'TP': int(tp),
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'specificity': specificity,
                'balanced_accuracy': balanced_accuracy
            })
        
        self.classification_results = pd.DataFrame(results)
        return self.classification_results
    
    def compute_roc_curves(self) -> Dict:
        """
        Compute ROC curves and AUC for each judging strategy.
        
        Returns:
            Dictionary with ROC data for each strategy
        """
        
        if self.ground_truth_col is None or self.judge_score_col is None:
            raise ValueError("Missing required columns for ROC analysis")
        
        roc_results = {}
        
        for strategy in self.df['judging_strategy'].unique():
            strategy_df = self.df[self.df['judging_strategy'] == strategy].copy()
            
            # Remove NaN values
            valid_df = strategy_df[[self.ground_truth_col, self.judge_score_col]].dropna()
            
            if len(valid_df) < 10:
                print(f"Warning: Insufficient data for ROC curve ({len(valid_df)} edges) for {strategy}")
                continue
            
            y_true = valid_df[self.ground_truth_col].astype(int)
            y_score = valid_df[self.judge_score_col]
            
            # Invert score: lower score = higher risk of hallucination
            y_score_inverted = -y_score
            
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_score_inverted)
            roc_auc = auc(fpr, tpr)
            
            roc_results[strategy] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc,
                'n_samples': len(valid_df)
            }
            
            print(f"  {strategy}: AUC = {roc_auc:.3f} (n={len(valid_df)})")
        
        self.roc_results = roc_results
        return roc_results
    
    def find_optimal_thresholds(self, criterion: str = 'f1') -> pd.DataFrame:
        """
        Find optimal thresholds for each strategy based on criterion.
        
        Args:
            criterion: Optimization criterion ('f1', 'accuracy', 'balanced_accuracy')
            
        Returns:
            DataFrame with optimal thresholds and metrics
        """
        
        if self.ground_truth_col is None or self.judge_score_col is None:
            raise ValueError("Missing required columns for threshold analysis")
        
        results = []
        
        for strategy in self.df['judging_strategy'].unique():
            strategy_df = self.df[self.df['judging_strategy'] == strategy].copy()
            
            # Remove NaN values
            valid_df = strategy_df[[self.ground_truth_col, self.judge_score_col]].dropna()
            
            if len(valid_df) < 10:
                continue
            
            y_true = valid_df[self.ground_truth_col].astype(int)
            y_score = valid_df[self.judge_score_col]
            
            # Try different threshold percentiles
            thresholds = np.percentile(y_score, np.arange(10, 91, 5))
            
            best_threshold = None
            best_score = -np.inf
            best_metrics = None
            
            for threshold in thresholds:
                # Lower score = hallucination
                y_pred = (y_score < threshold).astype(int)
                
                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                balanced_accuracy = (recall + specificity) / 2
                
                # Select best based on criterion
                if criterion == 'f1':
                    score = f1
                elif criterion == 'accuracy':
                    score = accuracy
                elif criterion == 'balanced_accuracy':
                    score = balanced_accuracy
                else:
                    score = f1
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_metrics = {
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'accuracy': accuracy,
                        'specificity': specificity,
                        'balanced_accuracy': balanced_accuracy,
                        'TP': int(tp),
                        'TN': int(tn),
                        'FP': int(fp),
                        'FN': int(fn)
                    }
            
            if best_metrics:
                results.append({
                    'judging_strategy': strategy,
                    'optimal_threshold': best_threshold,
                    'criterion': criterion,
                    **best_metrics
                })
        
        self.optimal_thresholds = pd.DataFrame(results)
        return self.optimal_thresholds
    
    def compute_baseline_comparison(self) -> pd.DataFrame:
        """
        Compare judge scores for corrupted vs non-corrupted edges.
        Tests if judges assign significantly different scores to each group.
        
        Returns:
            DataFrame with statistical test results for each strategy
        """
        
        if self.ground_truth_col is None or self.judge_score_col is None:
            raise ValueError("Missing required columns for baseline comparison")
        
        results = []
        
        for strategy in self.df['judging_strategy'].unique():
            strategy_df = self.df[self.df['judging_strategy'] == strategy].copy()
            
            # Remove NaN values
            valid_df = strategy_df[[self.ground_truth_col, self.judge_score_col]].dropna()
            
            if len(valid_df) < 10:
                continue
            
            # Split into corrupted and non-corrupted
            corrupted_scores = valid_df[valid_df[self.ground_truth_col] == 1][self.judge_score_col]
            non_corrupted_scores = valid_df[valid_df[self.ground_truth_col] == 0][self.judge_score_col]
            
            # Compute descriptive statistics
            corr_mean = corrupted_scores.mean()
            corr_std = corrupted_scores.std()
            corr_median = corrupted_scores.median()
            
            non_corr_mean = non_corrupted_scores.mean()
            non_corr_std = non_corrupted_scores.std()
            non_corr_median = non_corrupted_scores.median()
            
            # Mann-Whitney U test (non-parametric, no normality assumption)
            u_stat, p_value_mw = stats.mannwhitneyu(
                corrupted_scores, non_corrupted_scores, alternative='less'
            )
            
            # T-test (parametric)
            t_stat, p_value_t = stats.ttest_ind(
                corrupted_scores, non_corrupted_scores, equal_var=False
            )
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((corr_std**2 + non_corr_std**2) / 2)
            cohens_d = (corr_mean - non_corr_mean) / pooled_std if pooled_std > 0 else 0
            
            results.append({
                'judging_strategy': strategy,
                'n_corrupted': len(corrupted_scores),
                'n_non_corrupted': len(non_corrupted_scores),
                'corrupted_mean': corr_mean,
                'corrupted_median': corr_median,
                'corrupted_std': corr_std,
                'non_corrupted_mean': non_corr_mean,
                'non_corrupted_median': non_corr_median,
                'non_corrupted_std': non_corr_std,
                'mean_difference': corr_mean - non_corr_mean,
                'cohens_d': cohens_d,
                'mannwhitney_u': u_stat,
                'mannwhitney_p': p_value_mw,
                'ttest_t': t_stat,
                'ttest_p': p_value_t
            })
        
        self.baseline_comparison = pd.DataFrame(results)
        return self.baseline_comparison
    
    def compute_classification_correlation(self) -> pd.DataFrame:
        """
        Analyze correlation between judge scores and ground truth classifications (FP/FN/TP/TN).
        
        Tests if:
        - FP edges (hallucinations) get lower scores than TP edges (correct predictions)
        - FN edges get similar or lower scores than TN edges
        
        Returns:
            DataFrame with score statistics for each classification type
        """
        
        # Check if Classification column exists
        if 'Classification' not in self.df.columns:
            print("Warning: No 'Classification' column found. Skipping FP/FN/TP/TN analysis.")
            return pd.DataFrame()
        
        if self.judge_score_col is None:
            raise ValueError("Missing judge score column for classification correlation")
        
        results = []
        
        for strategy in self.df['judging_strategy'].unique():
            strategy_df = self.df[self.df['judging_strategy'] == strategy].copy()
            
            # Remove NaN values
            valid_df = strategy_df[['Classification', self.judge_score_col]].dropna()
            
            if len(valid_df) < 4:  # Need at least some data
                continue
            
            # Group by classification
            for classification in ['TP', 'FP', 'TN', 'FN']:
                class_scores = valid_df[valid_df['Classification'] == classification][self.judge_score_col]
                
                if len(class_scores) > 0:
                    results.append({
                        'judging_strategy': strategy,
                        'classification': classification,
                        'n': len(class_scores),
                        'mean_score': class_scores.mean(),
                        'median_score': class_scores.median(),
                        'std_score': class_scores.std(),
                        'min_score': class_scores.min(),
                        'max_score': class_scores.max()
                    })
            
            # Compute pairwise comparisons
            tp_scores = valid_df[valid_df['Classification'] == 'TP'][self.judge_score_col]
            fp_scores = valid_df[valid_df['Classification'] == 'FP'][self.judge_score_col]
            tn_scores = valid_df[valid_df['Classification'] == 'TN'][self.judge_score_col]
            fn_scores = valid_df[valid_df['Classification'] == 'FN'][self.judge_score_col]
            
            # TP vs FP: Do hallucinations (FP) get lower scores?
            if len(tp_scores) > 0 and len(fp_scores) > 0:
                u_stat, p_value = stats.mannwhitneyu(fp_scores, tp_scores, alternative='less')
                cohens_d = (fp_scores.mean() - tp_scores.mean()) / np.sqrt((fp_scores.std()**2 + tp_scores.std()**2) / 2)
                
                results.append({
                    'judging_strategy': strategy,
                    'classification': 'FP_vs_TP',
                    'n': len(fp_scores) + len(tp_scores),
                    'mean_score': fp_scores.mean() - tp_scores.mean(),
                    'median_score': fp_scores.median() - tp_scores.median(),
                    'std_score': np.nan,
                    'min_score': np.nan,
                    'max_score': np.nan,
                    'mann_whitney_u': u_stat,
                    'mann_whitney_p': p_value,
                    'cohens_d': cohens_d,
                    'interpretation': 'FP < TP (lower scores for hallucinations)' if p_value < 0.05 else 'No significant difference'
                })
            
            # TN vs FN: Similar analysis
            if len(tn_scores) > 0 and len(fn_scores) > 0:
                u_stat, p_value = stats.mannwhitneyu(fn_scores, tn_scores, alternative='less')
                cohens_d = (fn_scores.mean() - tn_scores.mean()) / np.sqrt((fn_scores.std()**2 + tn_scores.std()**2) / 2)
                
                results.append({
                    'judging_strategy': strategy,
                    'classification': 'FN_vs_TN',
                    'n': len(fn_scores) + len(tn_scores),
                    'mean_score': fn_scores.mean() - tn_scores.mean(),
                    'median_score': fn_scores.median() - tn_scores.median(),
                    'std_score': np.nan,
                    'min_score': np.nan,
                    'max_score': np.nan,
                    'mann_whitney_u': u_stat,
                    'mann_whitney_p': p_value,
                    'cohens_d': cohens_d,
                    'interpretation': 'FN < TN' if p_value < 0.05 else 'No significant difference'
                })
        
        self.classification_correlation = pd.DataFrame(results)
        return self.classification_correlation
    
    def plot_score_distributions(self, output_dir: Path = None):
        """Plot distribution of judge scores for corrupted vs non-corrupted edges."""
        
        if self.ground_truth_col is None or self.judge_score_col is None:
            raise ValueError("Missing required columns for distribution plot")
        
        strategies = self.df['judging_strategy'].unique()
        n_strategies = len(strategies)
        
        fig, axes = plt.subplots(1, n_strategies, figsize=(6*n_strategies, 5))
        if n_strategies == 1:
            axes = [axes]
        
        colors = {'corrupted': 'red', 'non_corrupted': 'blue'}
        
        for idx, strategy in enumerate(strategies):
            ax = axes[idx]
            strategy_df = self.df[self.df['judging_strategy'] == strategy].copy()
            valid_df = strategy_df[[self.ground_truth_col, self.judge_score_col]].dropna()
            
            # Split data
            corrupted_scores = valid_df[valid_df[self.ground_truth_col] == 1][self.judge_score_col]
            non_corrupted_scores = valid_df[valid_df[self.ground_truth_col] == 0][self.judge_score_col]
            
            # Plot distributions
            ax.hist(corrupted_scores, bins=20, alpha=0.6, label='Corrupted', color='red', density=True)
            ax.hist(non_corrupted_scores, bins=20, alpha=0.6, label='Non-Corrupted', color='blue', density=True)
            
            # Add vertical lines for means
            ax.axvline(corrupted_scores.mean(), color='darkred', linestyle='--', linewidth=2, 
                      label=f'Corrupted Mean: {corrupted_scores.mean():.3f}')
            ax.axvline(non_corrupted_scores.mean(), color='darkblue', linestyle='--', linewidth=2,
                      label=f'Non-Corrupted Mean: {non_corrupted_scores.mean():.3f}')
            
            ax.set_xlabel('Aggregate Score')
            ax.set_ylabel('Density')
            ax.set_title(f'{strategy}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_file = Path(output_dir) / 'rq1_score_distributions.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved distribution plot to {output_file}")
        
        plt.show()
    
    def plot_roc_curves(self, output_dir: Path = None):
        """Plot ROC curves for all strategies."""
        
        if not hasattr(self, 'roc_results'):
            self.compute_roc_curves()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = {'serial_baseline': 'blue', 'homogeneous_ensemble': 'green', 'heterogeneous_ensemble': 'red'}
        
        for strategy, data in self.roc_results.items():
            color = colors.get(strategy, 'black')
            label = f"{strategy} (AUC = {data['auc']:.3f})"
            ax.plot(data['fpr'], data['tpr'], color=color, lw=2, label=label)
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves: Judge Performance in Detecting Corrupted Edges')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_file = Path(output_dir) / 'rq1_roc_curves.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved ROC curve plot to {output_file}")
        
        plt.show()
    
    def generate_summary_report(self, output_dir: Path = None) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Summary report as string
        """
        
        if output_dir is None:
            output_dir = self.phase2_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute all analyses
        print("\nComputing baseline comparison (Corrupted vs Non-Corrupted)...")
        baseline_comp = self.compute_baseline_comparison()
        
        print("\nComputing classification metrics...")
        class_metrics = self.compute_classification_metrics(threshold=0.5)
        
        print("\nComputing ROC curves...")
        roc_results = self.compute_roc_curves()
        
        print("\nFinding optimal thresholds...")
        optimal_thresholds = self.find_optimal_thresholds(criterion='f1')
        
        # Generate report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("RQ1 ANALYSIS REPORT: Judge Performance in Hallucination Detection")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Data summary
        report_lines.append("## DATA SUMMARY")
        report_lines.append(f"Phase 2 Directory: {self.phase2_dir}")
        report_lines.append(f"Total Edges Analyzed: {len(self.df)}")
        report_lines.append(f"Judging Strategies: {', '.join(self.df['judging_strategy'].unique())}")
        report_lines.append("")
        
        # Baseline comparison - KEY FINDING!
        report_lines.append("## BASELINE COMPARISON: Corrupted vs Non-Corrupted Edges")
        report_lines.append("")
        report_lines.append("Do judges assign significantly different scores to corrupted vs non-corrupted edges?")
        report_lines.append("")
        
        for _, row in baseline_comp.iterrows():
            strategy = row['judging_strategy']
            report_lines.append(f"{strategy}:")
            report_lines.append(f"  Corrupted edges (n={row['n_corrupted']}):")
            report_lines.append(f"    Mean score: {row['corrupted_mean']:.4f} ± {row['corrupted_std']:.4f}")
            report_lines.append(f"    Median score: {row['corrupted_median']:.4f}")
            report_lines.append(f"  Non-corrupted edges (n={row['n_non_corrupted']}):")
            report_lines.append(f"    Mean score: {row['non_corrupted_mean']:.4f} ± {row['non_corrupted_std']:.4f}")
            report_lines.append(f"    Median score: {row['non_corrupted_median']:.4f}")
            report_lines.append(f"  Difference: {row['mean_difference']:.4f} (Cohen's d = {row['cohens_d']:.3f})")
            report_lines.append(f"  Mann-Whitney U test: U = {row['mannwhitney_u']:.2f}, p = {row['mannwhitney_p']:.4f}")
            report_lines.append(f"  T-test: t = {row['ttest_t']:.3f}, p = {row['ttest_p']:.4f}")
            
            # Interpretation
            if row['mannwhitney_p'] < 0.001:
                sig_level = "*** (p < 0.001)"
            elif row['mannwhitney_p'] < 0.01:
                sig_level = "** (p < 0.01)"
            elif row['mannwhitney_p'] < 0.05:
                sig_level = "* (p < 0.05)"
            else:
                sig_level = "n.s. (not significant)"
            
            report_lines.append(f"  RESULT: Corrupted edges have LOWER scores {sig_level}")
            report_lines.append("")
        
        report_lines.append("")
        
        # FP/FN/TP/TN correlation
        print("\nComputing FP/FN/TP/TN correlation...")
        classification_corr = self.compute_classification_correlation()
        
        if not classification_corr.empty:
            report_lines.append("## FP/FN/TP/TN CORRELATION: Judge Scores vs Ground Truth Classifications")
            report_lines.append("")
            report_lines.append("Do judges assign different scores to False Positives (FP) vs True Positives (TP)?")
            report_lines.append("")
            
            # Display per-classification scores
            basic_rows = classification_corr[classification_corr['classification'].isin(['TP', 'FP', 'TN', 'FN'])]
            if not basic_rows.empty:
                report_lines.append("### Score Distribution by Classification:")
                report_lines.append("")
                for _, row in basic_rows.iterrows():
                    report_lines.append(f"  {row['judging_strategy']} - {row['classification']} (n={row['n']}):")
                    report_lines.append(f"    Mean: {row['mean_score']:.3f}, Median: {row['median_score']:.3f}, Std: {row['std_score']:.3f}")
                report_lines.append("")
            
            # Display pairwise comparisons
            comparison_rows = classification_corr[classification_corr['classification'].isin(['FP_vs_TP', 'FN_vs_TN'])]
            if not comparison_rows.empty:
                report_lines.append("### Pairwise Comparisons:")
                report_lines.append("")
                for _, row in comparison_rows.iterrows():
                    report_lines.append(f"  {row['judging_strategy']} - {row['classification']}:")
                    report_lines.append(f"    Mean difference: {row['mean_score']:.3f}")
                    report_lines.append(f"    Cohen's d: {row['cohens_d']:.3f}")
                    report_lines.append(f"    Mann-Whitney U: U={row['mann_whitney_u']:.2f}, p={row['mann_whitney_p']:.4f}")
                    report_lines.append(f"    Result: {row['interpretation']}")
                    report_lines.append("")
            
            report_lines.append("")
        
        # Classification metrics
        report_lines.append("## CLASSIFICATION METRICS (Threshold = 0.5)")
        report_lines.append("")
        report_lines.append(class_metrics.to_string(index=False))
        report_lines.append("")
        
        # ROC AUC scores
        report_lines.append("## ROC CURVE ANALYSIS")
        report_lines.append("")
        for strategy, data in roc_results.items():
            report_lines.append(f"  {strategy}:")
            report_lines.append(f"    AUC: {data['auc']:.4f}")
            report_lines.append(f"    Samples: {data['n_samples']}")
        report_lines.append("")
        
        # Optimal thresholds
        report_lines.append("## OPTIMAL THRESHOLDS (F1-optimized)")
        report_lines.append("")
        report_lines.append(optimal_thresholds.to_string(index=False))
        report_lines.append("")
        
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_file = output_dir / 'rq1_analysis_summary.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
        print(f"\nSaved summary report to {report_file}")
        
        # Save DataFrames as CSV
        baseline_comp.to_csv(output_dir / 'rq1_baseline_comparison.csv', index=False)
        class_metrics.to_csv(output_dir / 'rq1_classification_metrics.csv', index=False)
        optimal_thresholds.to_csv(output_dir / 'rq1_optimal_thresholds.csv', index=False)
        
        if not classification_corr.empty:
            classification_corr.to_csv(output_dir / 'rq1_classification_correlation.csv', index=False)
        
        print(f"Saved CSV files to {output_dir}")
        
        return report_text


def main():
    """Main entry point for RQ1 analysis."""
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rq1_analysis.py <phase2_results_dir>")
        print("Example: python rq1_analysis.py parameter_tuning_experiments/results/rq1_phase2_20251010_014921")
        sys.exit(1)
    
    phase2_dir = sys.argv[1]
    
    print("="*80)
    print("RQ1 ANALYSIS: Judge Performance in Hallucination Detection")
    print("="*80)
    print()
    
    # Initialize analysis
    analyzer = RQ1Analysis(phase2_dir)
    
    # Generate comprehensive report
    report = analyzer.generate_summary_report()
    
    # Plot score distributions
    print("\nGenerating score distribution plot...")
    analyzer.plot_score_distributions(output_dir=analyzer.phase2_dir)
    
    # Plot ROC curves
    print("\nGenerating ROC curve plot...")
    analyzer.plot_roc_curves(output_dir=analyzer.phase2_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {analyzer.phase2_dir}")
    print("\nFiles generated:")
    print("  - rq1_analysis_summary.txt")
    print("  - rq1_baseline_comparison.csv")
    print("  - rq1_classification_metrics.csv")
    print("  - rq1_optimal_thresholds.csv")
    print("  - rq1_score_distributions.png")
    print("  - rq1_roc_curves.png")


if __name__ == '__main__':
    main()

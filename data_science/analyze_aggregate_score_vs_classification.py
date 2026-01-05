#!/usr/bin/env python3
"""
Analyze Aggregate Score vs Ground Truth Classification
Check if judge's aggregate score discriminates between TP/FP/TN/FN
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Input files
INPUT_DIR = Path("parameter_tuning_experiments/results/rq1_base_judging_correctness_20251012_213637")
INPUT_FILES = {
    'Social_norms': INPUT_DIR / 'judged_Social_norms_and_obesity_prevalence_citations_20251012_213910.xlsx',
    'Depressive_symptoms': INPUT_DIR / 'judged_Depressive_symptoms_in_response_to_a_stressor_citations_20251012_221924.xlsx',
    'Older_persons': INPUT_DIR / 'judged_older_persons_emergency_department_visits_citations_20251012_221219.xlsx'
}

def load_all_data():
    """Load all judged CLDs."""
    all_data = []
    
    for cld_name, file_path in INPUT_FILES.items():
        df = pd.read_excel(file_path, sheet_name='All Edges')
        df['cld_name'] = cld_name
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Filter out rows without Aggregate Score
    combined = combined[combined['Aggregate Score'].notna()].copy()
    
    return combined

def analyze_by_classification(df):
    """Analyze aggregate score distribution by classification."""
    print("=" * 80)
    print("AGGREGATE SCORE vs GROUND TRUTH CLASSIFICATION")
    print("=" * 80)
    
    print(f"\nTotal edges with Aggregate Score: {len(df)}\n")
    
    # Statistics by classification
    classifications = ['TP', 'FP', 'FN', 'TN']
    
    print("Distribution by Classification:")
    print("-" * 80)
    print(f"{'Class':<8} {'Count':<8} {'Mean Score':<12} {'Median':<10} {'Std':<10} {'Min':<8} {'Max':<8}")
    print("-" * 80)
    
    stats_data = []
    
    for cls in classifications:
        cls_data = df[df['Classification'] == cls]['Aggregate Score']
        if len(cls_data) > 0:
            stats_data.append({
                'Classification': cls,
                'Count': len(cls_data),
                'Mean': cls_data.mean(),
                'Median': cls_data.median(),
                'Std': cls_data.std(),
                'Min': cls_data.min(),
                'Max': cls_data.max()
            })
            print(f"{cls:<8} {len(cls_data):<8} {cls_data.mean():<12.4f} {cls_data.median():<10.4f} {cls_data.std():<10.4f} {cls_data.min():<8.4f} {cls_data.max():<8.4f}")
        else:
            print(f"{cls:<8} {0:<8} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<8}")
    
    print("-" * 80)
    
    # Key comparisons
    print("\n" + "=" * 80)
    print("KEY DISCRIMINATION TESTS")
    print("=" * 80)
    
    # TP vs FP (most important)
    tp_scores = df[df['Classification'] == 'TP']['Aggregate Score']
    fp_scores = df[df['Classification'] == 'FP']['Aggregate Score']
    
    if len(tp_scores) > 0 and len(fp_scores) > 0:
        print("\n1. TP vs FP (Can judge distinguish TRUE from FALSE positives?)")
        print("-" * 80)
        print(f"   TP mean: {tp_scores.mean():.4f}")
        print(f"   FP mean: {fp_scores.mean():.4f}")
        print(f"   Difference: {tp_scores.mean() - fp_scores.mean():.4f}")
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(tp_scores, fp_scores)
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"   ✅ SIGNIFICANT difference (p < 0.05)")
        else:
            print(f"   ❌ NOT significant (p >= 0.05)")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((tp_scores.std()**2 + fp_scores.std()**2) / 2)
        cohens_d = (tp_scores.mean() - fp_scores.mean()) / pooled_std
        print(f"   Cohen's d: {cohens_d:.4f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")
    
    # FN vs TN
    fn_scores = df[df['Classification'] == 'FN']['Aggregate Score']
    tn_scores = df[df['Classification'] == 'TN']['Aggregate Score']
    
    if len(fn_scores) > 0 and len(tn_scores) > 0:
        print("\n2. FN vs TN (Ground truth edges rejected vs edges not in ground truth)")
        print("-" * 80)
        print(f"   FN mean: {fn_scores.mean():.4f}")
        print(f"   TN mean: {tn_scores.mean():.4f}")
        print(f"   Difference: {fn_scores.mean() - tn_scores.mean():.4f}")
        
        t_stat, p_value = stats.ttest_ind(fn_scores, tn_scores)
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"   ✅ SIGNIFICANT difference (p < 0.05)")
        else:
            print(f"   ❌ NOT significant (p >= 0.05)")
    
    # Generated vs Not Generated (TP+FP vs FN+TN)
    generated_scores = df[df['Classification'].isin(['TP', 'FP'])]['Aggregate Score']
    not_generated_scores = df[df['Classification'].isin(['FN', 'TN'])]['Aggregate Score']
    
    if len(generated_scores) > 0 and len(not_generated_scores) > 0:
        print("\n3. Generated (TP+FP) vs Not Generated (FN+TN)")
        print("-" * 80)
        print(f"   Generated mean: {generated_scores.mean():.4f}")
        print(f"   Not Generated mean: {not_generated_scores.mean():.4f}")
        print(f"   Difference: {generated_scores.mean() - not_generated_scores.mean():.4f}")
        
        t_stat, p_value = stats.ttest_ind(generated_scores, not_generated_scores)
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"   ✅ SIGNIFICANT difference (p < 0.05)")
        else:
            print(f"   ❌ NOT significant (p >= 0.05)")
    
    # ANOVA across all 4 classes
    print("\n4. ANOVA: All 4 classifications")
    print("-" * 80)
    
    groups = [df[df['Classification'] == cls]['Aggregate Score'].values 
              for cls in classifications if len(df[df['Classification'] == cls]) > 0]
    
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"   F-statistic: {f_stat:.4f}")
        print(f"   p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"   ✅ SIGNIFICANT differences exist (p < 0.05)")
        else:
            print(f"   ❌ NO significant differences (p >= 0.05)")
    
    return pd.DataFrame(stats_data)

def plot_distributions(df, output_dir):
    """Plot aggregate score distributions by classification."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Box plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # All classifications
    ax = axes[0, 0]
    classifications = ['TP', 'FP', 'FN', 'TN']
    data_to_plot = [df[df['Classification'] == cls]['Aggregate Score'].values 
                    for cls in classifications if len(df[df['Classification'] == cls]) > 0]
    labels = [cls for cls in classifications if len(df[df['Classification'] == cls]) > 0]
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['lightgreen', 'lightcoral', 'lightyellow', 'lightblue']
    for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Aggregate Score', fontsize=12)
    ax.set_title('Aggregate Score by Classification', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # TP vs FP only
    ax = axes[0, 1]
    tp_fp_data = [df[df['Classification'] == cls]['Aggregate Score'].values 
                  for cls in ['TP', 'FP'] if len(df[df['Classification'] == cls]) > 0]
    tp_fp_labels = [cls for cls in ['TP', 'FP'] if len(df[df['Classification'] == cls]) > 0]
    
    if len(tp_fp_data) == 2:
        bp = ax.boxplot(tp_fp_data, labels=tp_fp_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_ylabel('Aggregate Score', fontsize=12)
        ax.set_title('TP vs FP (Key Discrimination)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
    
    # Histogram
    ax = axes[1, 0]
    for cls, color in zip(classifications, ['green', 'red', 'orange', 'blue']):
        cls_data = df[df['Classification'] == cls]['Aggregate Score']
        if len(cls_data) > 0:
            ax.hist(cls_data, bins=20, alpha=0.5, label=cls, color=color, edgecolor='black')
    
    ax.set_xlabel('Aggregate Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Histogram by Classification', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Violin plot
    ax = axes[1, 1]
    plot_data = []
    plot_labels = []
    for cls in classifications:
        cls_data = df[df['Classification'] == cls]['Aggregate Score']
        if len(cls_data) > 0:
            plot_data.extend(cls_data.values)
            plot_labels.extend([cls] * len(cls_data))
    
    if plot_data:
        plot_df = pd.DataFrame({'Classification': plot_labels, 'Aggregate Score': plot_data})
        parts = ax.violinplot([plot_df[plot_df['Classification'] == cls]['Aggregate Score'].values 
                               for cls in classifications if cls in plot_df['Classification'].values],
                              positions=range(len([cls for cls in classifications if cls in plot_df['Classification'].values])),
                              showmeans=True, showmedians=True)
        
        ax.set_xticks(range(len([cls for cls in classifications if cls in plot_df['Classification'].values])))
        ax.set_xticklabels([cls for cls in classifications if cls in plot_df['Classification'].values])
        ax.set_ylabel('Aggregate Score', fontsize=12)
        ax.set_title('Violin Plot by Classification', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'aggregate_score_by_classification.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved plot: {output_path}")

def main():
    print("\n" + "=" * 80)
    print("ANALYZING AGGREGATE SCORE vs CLASSIFICATION")
    print("=" * 80 + "\n")
    
    # Load data
    df = load_all_data()
    
    # Analyze
    stats_df = analyze_by_classification(df)
    
    # Plot
    output_dir = "parameter_tuning_experiments/rq2_analyses/aggregate_score_analysis"
    plot_distributions(df, output_dir)
    
    # Save statistics
    output_path = Path(output_dir) / 'aggregate_score_statistics.csv'
    stats_df.to_csv(output_path, index=False)
    print(f"✓ Saved statistics: {output_path}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()





















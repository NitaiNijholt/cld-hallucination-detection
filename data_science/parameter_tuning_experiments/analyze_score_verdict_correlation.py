#!/usr/bin/env python3
"""
Analyze correlation between judge aggregate scores (0-1) and categorical verdicts.

Tests whether continuous scores align with categorical judgments.
This validates the coherence of the judging system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_judge_data(excel_path: Path) -> pd.DataFrame:
    """Extract aggregate scores and verdicts from Excel."""
    try:
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        
        # Filter to causal edges only
        df = df[df['Relationship Type'] != 'NONE'].copy()
        
        # Extract aggregate score and verdict
        scores = []
        for _, row in df.iterrows():
            verdict = row.get('Judge Verdict', '')
            aggregate_score = row.get('Aggregate Score', np.nan)
            
            # If aggregate score not directly available, try to parse from Judge Message
            if pd.isna(aggregate_score):
                judge_msg = row.get('Judge Message', '')
                try:
                    judge_data = json.loads(judge_msg)
                    aggregate_score = judge_data.get('aggregate_score', np.nan)
                except:
                    pass
            
            scores.append({
                'Source': row['Source'],
                'Target': row['Target'],
                'Verdict': verdict,
                'Aggregate_Score': aggregate_score,
                'Is_Corrupted': row.get('Is Corrupted', False),
                'Classification': row.get('Classification', '')
            })
        
        return pd.DataFrame(scores)
    
    except Exception as e:
        print(f"Error loading {excel_path}: {e}")
        return pd.DataFrame()


def analyze_score_verdict_relationship(df: pd.DataFrame, cld_name: str):
    """Analyze relationship between aggregate scores and categorical verdicts."""
    
    print(f"\n{'='*80}")
    print(f"SCORE-VERDICT CORRELATION ANALYSIS: {cld_name}")
    print(f"{'='*80}")
    
    # Remove rows with missing scores
    df_clean = df[df['Aggregate_Score'].notna()].copy()
    
    if len(df_clean) == 0:
        print("❌ No aggregate scores found")
        return None
    
    print(f"\nTotal edges analyzed: {len(df_clean)}")
    print(f"Verdict distribution:")
    print(df_clean['Verdict'].value_counts())
    
    # 1. DESCRIPTIVE STATISTICS BY VERDICT
    print(f"\n📊 AGGREGATE SCORE STATISTICS BY VERDICT:")
    print("-" * 80)
    
    score_by_verdict = df_clean.groupby('Verdict')['Aggregate_Score'].agg([
        ('Count', 'count'),
        ('Mean', 'mean'),
        ('Median', 'median'),
        ('Std', 'std'),
        ('Min', 'min'),
        ('Max', 'max')
    ])
    print(score_by_verdict)
    
    # 2. ONE-WAY ANOVA / KRUSKAL-WALLIS TEST
    print(f"\n📈 STATISTICAL TEST: Kruskal-Wallis (non-parametric ANOVA)")
    print("-" * 80)
    
    # Group scores by verdict
    verdict_groups = [group['Aggregate_Score'].values 
                      for name, group in df_clean.groupby('Verdict')]
    
    if len(verdict_groups) >= 2:
        from scipy import stats
        
        # Kruskal-Wallis test (non-parametric alternative to ANOVA)
        h_stat, p_value = stats.kruskal(*verdict_groups)
        
        print(f"  H-statistic: {h_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        
        if p_value < 0.001:
            print(f"  ✅ HIGHLY SIGNIFICANT (p < 0.001)")
            print(f"     Aggregate scores differ significantly across verdict categories")
        elif p_value < 0.01:
            print(f"  ✅ VERY SIGNIFICANT (p < 0.01)")
        elif p_value < 0.05:
            print(f"  ✅ SIGNIFICANT (p < 0.05)")
        else:
            print(f"  ❌ NOT SIGNIFICANT (p >= 0.05)")
            print(f"     Scores do not differ meaningfully across verdicts")
        
        # Effect size (eta-squared approximation)
        n = len(df_clean)
        eta_squared = (h_stat - len(verdict_groups) + 1) / (n - len(verdict_groups))
        print(f"  Effect size (η²): {eta_squared:.4f}")
        
        if eta_squared > 0.14:
            print(f"     Large effect")
        elif eta_squared > 0.06:
            print(f"     Medium effect")
        elif eta_squared > 0.01:
            print(f"     Small effect")
        else:
            print(f"     Negligible effect")
    
    # 3. PAIRWISE COMPARISONS (if 3 categories)
    verdicts_ordered = ['INCORRECT', 'PARTIALLY_CORRECT', 'CORRECT']
    available_verdicts = [v for v in verdicts_ordered if v in df_clean['Verdict'].unique()]
    
    if len(available_verdicts) >= 2:
        print(f"\n📊 PAIRWISE COMPARISONS (Mann-Whitney U):")
        print("-" * 80)
        
        from scipy import stats
        from itertools import combinations
        
        for v1, v2 in combinations(available_verdicts, 2):
            scores_v1 = df_clean[df_clean['Verdict'] == v1]['Aggregate_Score']
            scores_v2 = df_clean[df_clean['Verdict'] == v2]['Aggregate_Score']
            
            u_stat, p_value = stats.mannwhitneyu(scores_v1, scores_v2, alternative='two-sided')
            
            mean_diff = scores_v1.mean() - scores_v2.mean()
            
            # Cohen's d effect size
            pooled_std = np.sqrt((scores_v1.std()**2 + scores_v2.std()**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            print(f"  {v1} vs {v2}:")
            print(f"    Mean difference: {mean_diff:+.3f} ({scores_v1.mean():.3f} - {scores_v2.mean():.3f})")
            print(f"    Mann-Whitney U: {u_stat:.1f}, p = {p_value:.6f} {sig_marker}")
            print(f"    Cohen's d: {cohens_d:.3f}")
            print()
    
    # 4. ORDINAL CORRELATION (Spearman)
    print(f"\n📈 ORDINAL CORRELATION (Spearman's ρ):")
    print("-" * 80)
    
    # Map verdicts to ordinal values
    verdict_mapping = {
        'INCORRECT': 0,
        'PARTIALLY_CORRECT': 1,
        'CORRECT': 2,
        'ERROR': 0,  # treat as incorrect
    }
    
    df_clean['Verdict_Ordinal'] = df_clean['Verdict'].map(verdict_mapping)
    df_clean = df_clean[df_clean['Verdict_Ordinal'].notna()]
    
    if len(df_clean) > 0:
        from scipy import stats
        rho, p_value = stats.spearmanr(df_clean['Verdict_Ordinal'], df_clean['Aggregate_Score'])
        
        print(f"  Spearman's ρ: {rho:.4f}")
        print(f"  P-value: {p_value:.6f}")
        
        if p_value < 0.001:
            print(f"  ✅ HIGHLY SIGNIFICANT (p < 0.001)")
        elif p_value < 0.01:
            print(f"  ✅ VERY SIGNIFICANT (p < 0.01)")
        elif p_value < 0.05:
            print(f"  ✅ SIGNIFICANT (p < 0.05)")
        else:
            print(f"  ❌ NOT SIGNIFICANT")
        
        if abs(rho) > 0.7:
            print(f"  Strong positive correlation - verdicts highly consistent with scores")
        elif abs(rho) > 0.5:
            print(f"  Moderate positive correlation - verdicts generally consistent with scores")
        elif abs(rho) > 0.3:
            print(f"  Weak positive correlation - some consistency")
        else:
            print(f"  Very weak correlation - verdicts inconsistent with scores")
    
    # 5. ANALYZE BY CORRUPTION STATUS (if available)
    if 'Is_Corrupted' in df_clean.columns:
        print(f"\n📊 SCORES BY CORRUPTION STATUS:")
        print("-" * 80)
        
        for is_corrupted in [True, False]:
            subset = df_clean[df_clean['Is_Corrupted'] == is_corrupted]
            if len(subset) > 0:
                label = "CORRUPTED" if is_corrupted else "CLEAN"
                print(f"\n{label} edges (n={len(subset)}):")
                
                for verdict in available_verdicts:
                    verdict_subset = subset[subset['Verdict'] == verdict]
                    if len(verdict_subset) > 0:
                        mean_score = verdict_subset['Aggregate_Score'].mean()
                        count = len(verdict_subset)
                        pct = 100 * count / len(subset)
                        print(f"  {verdict:20s}: {count:3d} ({pct:5.1f}%) - Mean score: {mean_score:.3f}")
    
    return {
        'cld_name': cld_name,
        'n_edges': len(df_clean),
        'score_by_verdict': score_by_verdict.to_dict(),
        'spearman_rho': rho if 'rho' in locals() else None,
        'spearman_p': p_value if 'p_value' in locals() and 'rho' in locals() else None
    }


def create_visualizations(df: pd.DataFrame, cld_name: str, output_dir: Path):
    """Create visualizations of score-verdict relationships."""
    
    df_clean = df[df['Aggregate_Score'].notna()].copy()
    
    if len(df_clean) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Box plot: Scores by Verdict
    ax1 = axes[0, 0]
    verdict_order = ['INCORRECT', 'PARTIALLY_CORRECT', 'CORRECT']
    available_verdicts = [v for v in verdict_order if v in df_clean['Verdict'].unique()]
    
    sns.boxplot(data=df_clean, x='Verdict', y='Aggregate_Score', 
                order=available_verdicts, ax=ax1, palette='Set2')
    ax1.set_title(f'Aggregate Score Distribution by Verdict\n{cld_name}', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Aggregate Score', fontsize=12)
    ax1.set_xlabel('Verdict', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add mean markers
    for i, verdict in enumerate(available_verdicts):
        mean_score = df_clean[df_clean['Verdict'] == verdict]['Aggregate_Score'].mean()
        ax1.plot(i, mean_score, 'D', color='red', markersize=10, label='Mean' if i == 0 else '')
    ax1.legend()
    
    # 2. Violin plot: Scores by Verdict
    ax2 = axes[0, 1]
    sns.violinplot(data=df_clean, x='Verdict', y='Aggregate_Score', 
                   order=available_verdicts, ax=ax2, palette='Set2')
    ax2.set_title(f'Score Distribution (Violin)\n{cld_name}', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('Aggregate Score', fontsize=12)
    ax2.set_xlabel('Verdict', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Histogram: Score distributions
    ax3 = axes[1, 0]
    for verdict in available_verdicts:
        scores = df_clean[df_clean['Verdict'] == verdict]['Aggregate_Score']
        ax3.hist(scores, bins=20, alpha=0.5, label=verdict, edgecolor='black')
    ax3.set_title(f'Score Distributions by Verdict\n{cld_name}', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Aggregate Score', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Scatter: Score vs Verdict (with jitter) colored by corruption
    ax4 = axes[1, 1]
    
    # Map verdicts to numeric for x-axis
    verdict_mapping = {v: i for i, v in enumerate(available_verdicts)}
    df_clean['Verdict_Numeric'] = df_clean['Verdict'].map(verdict_mapping)
    
    # Add jitter for visibility
    jitter = np.random.normal(0, 0.05, len(df_clean))
    df_clean['Verdict_Jittered'] = df_clean['Verdict_Numeric'] + jitter
    
    if 'Is_Corrupted' in df_clean.columns:
        for is_corrupted, color, label in [(True, 'red', 'Corrupted'), (False, 'blue', 'Clean')]:
            subset = df_clean[df_clean['Is_Corrupted'] == is_corrupted]
            ax4.scatter(subset['Verdict_Jittered'], subset['Aggregate_Score'], 
                       alpha=0.6, s=50, color=color, label=label, edgecolor='black', linewidth=0.5)
    else:
        ax4.scatter(df_clean['Verdict_Jittered'], df_clean['Aggregate_Score'], 
                   alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
    
    ax4.set_xticks(range(len(available_verdicts)))
    ax4.set_xticklabels(available_verdicts, rotation=15, ha='right')
    ax4.set_title(f'Score vs Verdict (Scatter)\n{cld_name}', 
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('Verdict', fontsize=12)
    ax4.set_ylabel('Aggregate Score', fontsize=12)
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / f"score_verdict_analysis_{cld_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_file}")
    plt.close()


def main():
    """Main execution."""
    
    # Base directory
    base_dir = Path("parameter_tuning_experiments/results/rq1_multi_cld_20251105_205040")
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return
    
    # Load summary
    with open(base_dir / "experiment_summary.json", 'r') as f:
        summary = json.load(f)
    
    print("="*80)
    print("SCORE-VERDICT CORRELATION ANALYSIS")
    print("="*80)
    print("Question: Are categorical verdicts consistent with continuous scores?")
    print()
    
    all_results = []
    
    for cld_result in summary['results']:
        cld_name = cld_result['cld_name']
        output_dir = Path(cld_result['output_dir'])
        
        # Analyze judged_no_correction file (has original judge scores)
        judged_files = list(output_dir.glob("judged_no_correction_*.xlsx"))
        
        if not judged_files:
            print(f"❌ No judged file found for {cld_name}")
            continue
        
        judged_file = judged_files[0]
        
        print(f"\nProcessing: {cld_name}")
        print(f"File: {judged_file.name}")
        
        # Load data
        df = load_judge_data(judged_file)
        
        if df.empty:
            print(f"❌ Failed to load data for {cld_name}")
            continue
        
        # Analyze
        result = analyze_score_verdict_relationship(df, cld_name)
        if result:
            all_results.append(result)
        
        # Create visualizations
        create_visualizations(df, cld_name, output_dir)
    
    # Summary
    if all_results:
        print(f"\n{'='*80}")
        print("SUMMARY ACROSS ALL CLDs")
        print(f"{'='*80}\n")
        
        summary_df = pd.DataFrame(all_results)
        print(summary_df[['cld_name', 'n_edges', 'spearman_rho', 'spearman_p']].to_string(index=False))
        
        avg_rho = summary_df['spearman_rho'].mean()
        print(f"\nAverage Spearman's ρ: {avg_rho:.4f}")
        
        if avg_rho > 0.7:
            print("✅ Strong consistency between scores and verdicts")
        elif avg_rho > 0.5:
            print("✅ Moderate consistency between scores and verdicts")
        elif avg_rho > 0.3:
            print("⚠️  Weak consistency between scores and verdicts")
        else:
            print("❌ Very weak consistency - scoring system may be unreliable")
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()




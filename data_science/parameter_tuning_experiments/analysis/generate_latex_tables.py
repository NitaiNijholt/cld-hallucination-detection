#!/usr/bin/env python3
"""
Generate LaTeX Tables for Thesis

This script generates publication-ready LaTeX tables from RQ2 analysis results
for inclusion in the thesis Experiments & Results chapter.

Usage:
    python generate_latex_tables.py
"""

import pandas as pd
from pathlib import Path
import sys


def generate_latex_tables(
    tables_dir: str = "tables",
    output_dir: str = "thesis_outputs/tables"
):
    """
    Generate LaTeX tables for thesis from RQ2 analysis results.
    
    Args:
        tables_dir: Directory containing CSV result files
        output_dir: Directory to save LaTeX tables
    """
    
    # Get base directory (parameter_tuning_experiments)
    base_dir = Path(__file__).parent.parent
    tables_path = base_dir / tables_dir
    output_path = base_dir / output_dir
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Generating LaTeX Tables for Thesis - RQ2")
    print("=" * 80)
    
    # Table 1: Correlation Results
    print("\n[1/4] Generating correlation table...")
    try:
        corr_df = pd.read_csv(tables_path / 'rq2_correlations.csv')
        
        # Format for thesis
        corr_formatted = corr_df[['metric', 'n_samples', 'pearson_r', 'pearson_p', 'significant']].copy()
        corr_formatted['metric'] = corr_formatted['metric'].str.replace('_', '\\_')
        corr_formatted.columns = ['Metric', 'N', 'r', 'p-value', 'Sig.']
        corr_formatted['Sig.'] = corr_formatted['Sig.'].map({True: '✓', False: '✗'})
        
        latex_table = corr_formatted.to_latex(
            index=False,
            float_format="%.3f",
            caption="Pearson correlation coefficients between context-insensitive metrics and hallucination labels. Statistical significance determined at α=0.05.",
            label="tab:rq2_correlations",
            escape=False,
            column_format='lrrrr'
        )
        
        with open(output_path / 'rq2_correlations.tex', 'w') as f:
            f.write(latex_table)
        
        print(f"  ✓ Saved: {output_path / 'rq2_correlations.tex'}")
        print(f"    {len(corr_formatted)} metrics included")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Table 2: ROC AUC Summary
    print("\n[2/4] Generating ROC AUC table...")
    try:
        roc_df = pd.read_csv(tables_path / 'rq2_roc_auc.csv')
        
        # Format for thesis
        roc_formatted = roc_df.copy()
        roc_formatted['metric'] = roc_formatted['metric'].str.replace('_', '\\_')
        roc_formatted.columns = ['Metric', 'AUC']
        roc_formatted = roc_formatted.sort_values('AUC', ascending=False)
        
        latex_table = roc_formatted.to_latex(
            index=False,
            float_format="%.3f",
            caption="Area Under the ROC Curve (AUC) for each CI metric as a binary classifier for hallucination detection. AUC > 0.5 indicates better than random performance.",
            label="tab:rq2_roc_auc",
            escape=False,
            column_format='lr'
        )
        
        with open(output_path / 'rq2_roc_auc.tex', 'w') as f:
            f.write(latex_table)
        
        print(f"  ✓ Saved: {output_path / 'rq2_roc_auc.tex'}")
        best_metric = roc_formatted.iloc[0]
        print(f"    Best: {best_metric['Metric']} (AUC={best_metric['AUC']:.3f})")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Table 3: Optimal Thresholds
    print("\n[3/4] Generating optimal thresholds table...")
    try:
        thresh_df = pd.read_csv(tables_path / 'rq2_optimal_thresholds.csv')
        
        # Format for thesis
        thresh_formatted = thresh_df[['metric', 'optimal_threshold', 'precision', 'recall', 'f1', 'accuracy']].copy()
        thresh_formatted['metric'] = thresh_formatted['metric'].str.replace('_', '\\_')
        thresh_formatted.columns = ['Metric', 'Threshold', 'Precision', 'Recall', 'F1', 'Accuracy']
        thresh_formatted = thresh_formatted.sort_values('F1', ascending=False)
        
        latex_table = thresh_formatted.to_latex(
            index=False,
            float_format="%.3f",
            caption="Optimal thresholds and classification performance for CI metrics. Thresholds optimized for maximum F1 score. These thresholds can be used for RQ3 smart test-time compute allocation.",
            label="tab:rq2_optimal_thresholds",
            escape=False,
            column_format='lrrrrrr'
        )
        
        with open(output_path / 'rq2_optimal_thresholds.tex', 'w') as f:
            f.write(latex_table)
        
        print(f"  ✓ Saved: {output_path / 'rq2_optimal_thresholds.tex'}")
        print(f"    {len(thresh_formatted)} thresholds identified")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Table 4: Statistical Validation Summary
    print("\n[4/4] Generating statistical validation table...")
    try:
        perm_df = pd.read_csv(tables_path / 'rq2_permutation_tests.csv')
        boot_df = pd.read_csv(tables_path / 'rq2_bootstrap_ci.csv')
        
        # Merge permutation and bootstrap results
        stat_merged = pd.merge(
            perm_df[['metric', 'observed_r', 'permutation_p_value', 'significant_corrected']],
            boot_df[['metric', 'ci_lower', 'ci_upper']],
            on='metric'
        )
        
        # Format for thesis
        stat_formatted = stat_merged.copy()
        stat_formatted['metric'] = stat_formatted['metric'].str.replace('_', '\\_')
        stat_formatted['ci'] = stat_formatted.apply(
            lambda row: f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]", axis=1
        )
        stat_formatted = stat_formatted[['metric', 'observed_r', 'permutation_p_value', 'ci', 'significant_corrected']]
        stat_formatted.columns = ['Metric', 'r', 'p-value', '95\\% CI (AUC)', 'Sig.']
        stat_formatted['Sig.'] = stat_formatted['Sig.'].map({True: '✓', False: '✗'})
        
        latex_table = stat_formatted.to_latex(
            index=False,
            float_format="%.3f",
            caption="Statistical validation of CI metrics with permutation tests and bootstrap confidence intervals. Significance after Holm-Bonferroni correction for multiple comparisons (α=0.05).",
            label="tab:rq2_statistical_validation",
            escape=False,
            column_format='lrrlr'
        )
        
        with open(output_path / 'rq2_statistical_validation.tex', 'w') as f:
            f.write(latex_table)
        
        print(f"  ✓ Saved: {output_path / 'rq2_statistical_validation.tex'}")
        n_sig = stat_formatted['Sig.'].value_counts().get('✓', 0)
        print(f"    {n_sig}/{len(stat_formatted)} metrics statistically significant")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Generate summary README
    print("\n[Bonus] Generating README for thesis outputs...")
    readme_content = f"""# RQ2 LaTeX Tables for Thesis

This directory contains publication-ready LaTeX tables generated from RQ2 analysis.

## Files

1. **rq2_correlations.tex**
   - Table: Correlation coefficients between CI metrics and hallucination labels
   - Use in: Experiments & Results > RQ2 > Correlation Analysis section
   - Label: \\ref{{tab:rq2_correlations}}

2. **rq2_roc_auc.tex**
   - Table: ROC AUC scores for CI metrics as classifiers
   - Use in: Experiments & Results > RQ2 > Classification Performance section
   - Label: \\ref{{tab:rq2_roc_auc}}

3. **rq2_optimal_thresholds.tex**
   - Table: Optimal thresholds and performance metrics
   - Use in: Experiments & Results > RQ2 > Threshold Optimization section
   - Label: \\ref{{tab:rq2_optimal_thresholds}}
   - **Important**: These thresholds are used in RQ3!

4. **rq2_statistical_validation.tex**
   - Table: Permutation tests and bootstrap CIs
   - Use in: Experiments & Results > RQ2 > Statistical Validation section
   - Label: \\ref{{tab:rq2_statistical_validation}}

## Usage in LaTeX

Add to your thesis .tex file:

```latex
\\input{{tables/rq2_correlations.tex}}
\\input{{tables/rq2_roc_auc.tex}}
\\input{{tables/rq2_optimal_thresholds.tex}}
\\input{{tables/rq2_statistical_validation.tex}}
```

## Generated on

{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

From analysis results in: {tables_path}
"""
    
    with open(output_path / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"  ✓ Saved: {output_path / 'README.md'}")
    
    print("\n" + "=" * 80)
    print("LaTeX Table Generation Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {output_path}")
    print("\nNext steps:")
    print("  1. Review generated .tex files")
    print("  2. Copy tables to thesis/tables/ directory")
    print("  3. Include in thesis using \\input{} commands")
    print("  4. Generate figure captions for the 6 PNG files")
    print("\n")


if __name__ == '__main__':
    generate_latex_tables()

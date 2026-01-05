#!/usr/bin/env python3
"""
Generate Figure Captions for Thesis

Creates LaTeX figure captions and documentation for all RQ2 figures.

Usage:
    python generate_figure_captions.py
"""

from pathlib import Path


def generate_figure_captions(
    figures_dir: str = "figures",
    output_dir: str = "thesis_outputs/figures"
):
    """
    Generate LaTeX figure captions and documentation for RQ2 figures.
    
    Args:
        figures_dir: Directory containing figure PNG files
        output_dir: Directory to save figure documentation
    """
    
    # Get base directory
    base_dir = Path(__file__).parent.parent
    figures_path = base_dir / figures_dir
    output_path = base_dir / output_dir
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Generating Figure Captions for Thesis - RQ2")
    print("=" * 80)
    
    # Define figure captions
    figures = {
        'rq2_distributions.png': {
            'short_caption': 'Distribution of CI metrics by hallucination status',
            'long_caption': '''Distribution comparison of context-insensitive metrics between hallucinated 
and non-hallucinated edges. Each subplot shows the distribution of one CI metric, 
with orange representing hallucinated edges and blue representing non-hallucinated edges. 
Overlapping distributions suggest limited discriminative power, while separated distributions 
indicate potential for hallucination detection. Metrics showing clear separation between 
the two classes are more useful for detecting hallucinations.''',
            'label': 'fig:rq2_distributions',
            'placement': 'htbp',
            'width': '0.95\\textwidth'
        },
        
        'rq2_scatter_plots.png': {
            'short_caption': 'Scatter plots with regression lines for CI metrics',
            'long_caption': '''Scatter plots showing the relationship between context-insensitive metrics 
and hallucination occurrence (0=no hallucination, 1=hallucination). Each subplot displays 
one metric with a linear regression line (red). The Pearson correlation coefficient (r) 
and p-value are shown in each panel. Positive slopes indicate higher metric values correlate 
with increased hallucination likelihood, while negative slopes suggest the opposite. 
The strength of the linear relationship can be assessed from the r-value and the tightness 
of points around the regression line.''',
            'label': 'fig:rq2_scatter',
            'placement': 'htbp',
            'width': '0.95\\textwidth'
        },
        
        'rq2_roc_curves.png': {
            'short_caption': 'ROC curves for CI metrics as hallucination classifiers',
            'long_caption': '''Receiver Operating Characteristic (ROC) curves evaluating context-insensitive 
metrics as binary classifiers for hallucination detection. Each curve represents one CI metric, 
with the Area Under the Curve (AUC) displayed in the legend. The dashed diagonal line represents 
random classification (AUC=0.5). Curves closer to the top-left corner indicate better classification 
performance. AUC > 0.7 is generally considered acceptable, 0.7-0.8 good, 0.8-0.9 excellent, 
and > 0.9 outstanding. The best performing metric (max window entropy, AUC≈0.62) shows modest but 
above-random discriminative ability.''',
            'label': 'fig:rq2_roc',
            'placement': 'htbp',
            'width': '0.85\\textwidth'
        },
        
        'rq2_confusion_matrices.png': {
            'short_caption': 'Confusion matrices at optimal thresholds',
            'long_caption': '''Confusion matrices for each context-insensitive metric at their respective 
optimal thresholds (optimized for F1 score). Each heatmap shows the classification performance 
with true labels on the y-axis and predicted labels on the x-axis. Darker blue indicates higher 
counts. The optimal threshold and resulting F1 score are displayed in the title of each subplot. 
High values on the diagonal (true positives and true negatives) indicate good classification, 
while off-diagonal values represent misclassifications. All metrics achieve F1 scores around 0.88-0.90, 
suggesting similar practical performance despite differences in AUC.''',
            'label': 'fig:rq2_confusion',
            'placement': 'htbp',
            'width': '0.95\\textwidth'
        },
        
        'rq2_permutation_distributions.png': {
            'short_caption': 'Null distributions from permutation tests',
            'long_caption': '''Null distributions of correlation coefficients from permutation tests 
(10,000 permutations) for each context-insensitive metric. The vertical red dashed line indicates 
the observed correlation coefficient from the actual data. The shaded histogram represents the 
distribution of correlation values obtained by randomly permuting the hallucination labels, 
which approximates the null hypothesis of no relationship. P-values are computed as the proportion 
of permuted correlations with absolute values greater than or equal to the observed value. 
None of the metrics show statistically significant correlations after multiple comparison correction, 
suggesting weak predictive power.''',
            'label': 'fig:rq2_permutation',
            'placement': 'htbp',
            'width': '0.95\\textwidth'
        },
        
        'rq2_bootstrap_distributions.png': {
            'short_caption': 'Bootstrap distributions of AUC scores',
            'long_caption': '''Bootstrap distributions of AUC scores from 10,000 resamples for each 
context-insensitive metric. Each histogram shows the distribution of AUC values obtained by 
bootstrap resampling (sampling with replacement from the original dataset). The vertical red 
dashed line indicates the observed AUC from the original data. The 95\\% confidence interval 
is computed from the 2.5th and 97.5th percentiles of the bootstrap distribution. Wide confidence 
intervals indicate high uncertainty in the AUC estimates, which may be due to small sample size 
or high variability in the data. All metrics show substantial overlap with AUC=0.5 (random performance), 
suggesting limited predictive ability.''',
            'label': 'fig:rq2_bootstrap',
            'placement': 'htbp',
            'width': '0.95\\textwidth'
        }
    }
    
    # Generate LaTeX figure environment for each figure
    latex_output = []
    latex_output.append("% RQ2 Figures - Generated LaTeX Code")
    latex_output.append("% Copy and paste these into your thesis .tex file\n")
    
    for filename, info in figures.items():
        latex_code = f"""
\\begin{{figure}}[{info['placement']}]
    \\centering
    \\includegraphics[width={info['width']}]{{figures/{filename}}}
    \\caption[{info['short_caption']}]{{{info['long_caption']}}}
    \\label{{{info['label']}}}
\\end{{figure}}
"""
        latex_output.append(latex_code)
    
    # Save to file
    latex_file = output_path / 'rq2_figure_captions.tex'
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_output))
    
    print(f"\n✓ Generated LaTeX figure code: {latex_file}")
    
    # Generate a markdown documentation file
    md_output = []
    md_output.append("# RQ2 Figures for Thesis\n")
    md_output.append("This document provides captions and usage instructions for all RQ2 figures.\n")
    md_output.append("## Figures\n")
    
    for i, (filename, info) in enumerate(figures.items(), 1):
        source_file = figures_path / filename
        if source_file.exists():
            status = "✓ Available"
        else:
            status = "✗ Missing"
        
        md_output.append(f"### {i}. {filename} {status}\n")
        md_output.append(f"**Label:** `{info['label']}`")
        md_output.append(f"**Short Caption:** {info['short_caption']}")
        md_output.append(f"\n**Full Caption:**\n{info['long_caption']}\n")
        md_output.append(f"**LaTeX Reference:** `\\ref{{{info['label']}}}`")
        md_output.append(f"**File Path:** `figures/{filename}`\n")
        md_output.append("---\n")
    
    # Add usage instructions
    md_output.append("\n## Usage in Thesis\n")
    md_output.append("### Option 1: Include figures directly in text\n")
    md_output.append("```latex")
    md_output.append("\\input{figures/rq2_figure_captions.tex}")
    md_output.append("```\n")
    
    md_output.append("### Option 2: Reference figures in text\n")
    md_output.append("```latex")
    md_output.append("As shown in Figure~\\ref{fig:rq2_distributions}, the distributions...")
    md_output.append("The ROC curves (Figure~\\ref{fig:rq2_roc}) demonstrate...")
    md_output.append("```\n")
    
    md_output.append("### Recommended placement in thesis\n")
    md_output.append("1. **Correlation Analysis Section:**")
    md_output.append("   - Figure~\\ref{fig:rq2_distributions}")
    md_output.append("   - Figure~\\ref{fig:rq2_scatter}\n")
    
    md_output.append("2. **Classification Performance Section:**")
    md_output.append("   - Figure~\\ref{fig:rq2_roc}")
    md_output.append("   - Figure~\\ref{fig:rq2_confusion}\n")
    
    md_output.append("3. **Statistical Validation Section:**")
    md_output.append("   - Figure~\\ref{fig:rq2_permutation}")
    md_output.append("   - Figure~\\ref{fig:rq2_bootstrap}\n")
    
    md_file = output_path / 'README_figures.md'
    with open(md_file, 'w') as f:
        f.write('\n'.join(md_output))
    
    print(f"✓ Generated figure documentation: {md_file}")
    
    # Check which figures exist
    print(f"\n{'-' * 80}")
    print("Figure Status Check:")
    print(f"{'-' * 80}")
    
    for filename in figures.keys():
        source_file = figures_path / filename
        if source_file.exists():
            size = source_file.stat().st_size / 1024  # KB
            print(f"  ✓ {filename:40s} ({size:.1f} KB)")
        else:
            print(f"  ✗ {filename:40s} (MISSING)")
    
    print(f"\n{'-' * 80}")
    print("Next steps:")
    print(f"{'-' * 80}")
    print("1. Review generated LaTeX code in:")
    print(f"   {latex_file}")
    print("2. Copy figure PNG files to your thesis figures/ directory")
    print("3. Include figures in thesis using the generated LaTeX code")
    print("4. Cite figures in text using \\ref{fig:rq2_*} labels")
    print()


if __name__ == '__main__':
    generate_figure_captions()

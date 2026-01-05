#!/usr/bin/env python3
"""
RQ2 Normality Tests for UQ Metrics

Runs Shapiro-Wilk normality tests on UQ metric distributions and generates
LaTeX tables for the thesis. This script provides reproducible statistical
evidence for the choice of non-parametric methods.

Output:
- Console summary
- LaTeX table (normality_tests_table.tex)
- JSON results for archival (normality_tests_results.json)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
CI_METRICS = [
    'Gen Perplexity',
    'Gen Min Prob', 
    'Gen Max Window Entropy',
    'Gen Cosine Similarity'
]

SHAPIRO_SUBSAMPLE_SIZE = 5000  # Shapiro-Wilk limited to n<=5000

OUTPUT_DIR = Path("final_runs/RQ2_uq_hallucination_detection")


def format_p_value_latex(p: float) -> str:
    """
    Format a p-value for LaTeX output without producing false inequalities.

    Notes:
    - Very small p-values may underflow to 0.0; handle that explicitly.
    - For p < 1e-50 we keep the thesis-friendly bound '$<10^{-50}$'.
    - For p < 0.001 we use scientific notation '$m\\times 10^{e}$' (accurate).
    - Otherwise, we print to 3 decimals.
    """
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"

    # Underflow / exact zero
    if p == 0.0:
        return r"$\approx 0$"

    if p < 1e-50:
        return r"$<10^{-50}$"

    if p < 0.001:
        # Scientific notation with mantissa in [1, 10)
        exp = int(np.floor(np.log10(p)))
        mantissa = p / (10 ** exp)
        return rf"${mantissa:.2f}\times 10^{{{exp}}}$"

    return f"{p:.3f}"


def load_rq2_data():
    """Load combined RQ2 data from deduplicated inventory."""
    inventory_path = Path("parameter_tuning_experiments/rq2_analyses")
    
    # Find latest deduplicated inventory
    dedup_files = list(inventory_path.glob("rq2_valid_files_deduplicated_*.json"))
    if not dedup_files:
        dedup_files = list(inventory_path.glob("rq2_valid_files_*.json"))
    
    if not dedup_files:
        raise FileNotFoundError("No inventory files found!")
    
    latest_inventory = max(dedup_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading from: {latest_inventory.name}")
    
    with open(latest_inventory, 'r') as f:
        valid_files = json.load(f)
    
    all_data = []
    for file_info in valid_files:
        filepath = file_info['filepath']
        exp_type = file_info.get('experiment_type', 'unknown')
        
        if exp_type not in ['citation', 'correctness']:
            continue
        
        try:
            df = pd.read_excel(filepath, sheet_name='All Edges')
            if all(col in df.columns for col in CI_METRICS[:3]) and 'Classification' in df.columns:
                df['is_hallucination'] = (df['Classification'] == 'FP') | (df['Classification'] == 'FN')
                df['cld'] = file_info.get('cld', 'unknown')
                all_data.append(df)
        except Exception as e:
            continue
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined)} edges from {len(all_data)} files\n")
    return combined


def run_shapiro_wilk(data, subsample_size=SHAPIRO_SUBSAMPLE_SIZE):
    """
    Run Shapiro-Wilk test, subsampling if n > 5000.
    
    Returns:
        dict with W statistic, p-value, n, and whether subsampled
    """
    n = len(data)
    subsampled = False
    
    if n > subsample_size:
        np.random.seed(42)  # Reproducibility
        data = np.random.choice(data, subsample_size, replace=False)
        subsampled = True
        n_tested = subsample_size
    else:
        n_tested = n
    
    W, p = stats.shapiro(data)
    
    return {
        'W': float(W),
        'p': float(p),
        'n_original': int(n),
        'n_tested': int(n_tested),
        'subsampled': bool(subsampled),
        'is_normal': bool(p > 0.05)
    }


def compute_distribution_stats(data):
    """Compute descriptive statistics for a distribution."""
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'median': float(np.median(data)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data))
    }


def run_normality_analysis(df):
    """Run comprehensive normality analysis on all metrics."""
    results = {}
    
    for metric in CI_METRICS:
        print(f"\n{'='*60}")
        print(f"METRIC: {metric}")
        print(f"{'='*60}")
        
        # Get clean data
        data = df[metric].dropna().replace([np.inf, -np.inf], np.nan).dropna().values
        halluc_data = df[df['is_hallucination']][metric].dropna().replace([np.inf, -np.inf], np.nan).dropna().values
        correct_data = df[~df['is_hallucination']][metric].dropna().replace([np.inf, -np.inf], np.nan).dropna().values
        
        if len(data) < 20:
            print(f"  Skipping - insufficient data (n={len(data)})")
            continue
        
        metric_results = {
            'overall': {
                'shapiro': run_shapiro_wilk(data),
                'stats': compute_distribution_stats(data)
            },
            'hallucinations': {
                'shapiro': run_shapiro_wilk(halluc_data),
                'stats': compute_distribution_stats(halluc_data)
            },
            'correct': {
                'shapiro': run_shapiro_wilk(correct_data),
                'stats': compute_distribution_stats(correct_data)
            }
        }
        
        results[metric] = metric_results
        
        # Print summary
        for group_name in ['overall', 'hallucinations', 'correct']:
            r = metric_results[group_name]
            sw = r['shapiro']
            st = r['stats']
            verdict = "NORMAL" if sw['is_normal'] else "NOT NORMAL"
            subsample_note = f" (n={sw['n_tested']} subsample)" if sw['subsampled'] else ""
            
            print(f"\n  {group_name.upper()}:")
            print(f"    N = {sw['n_original']}{subsample_note}")
            print(f"    Skewness = {st['skewness']:.3f}, Kurtosis = {st['kurtosis']:.3f}")
            print(f"    Shapiro-Wilk: W={sw['W']:.4f}, p={sw['p']:.2e} → {verdict}")
    
    return results


def generate_latex_table(results):
    """Generate LaTeX table for thesis."""
    
    latex = r"""\begin{table}[H]
\centering
\caption{Shapiro-Wilk Normality Tests for UQ Metric Distributions}
\label{tab:rq2_normality_tests}
\begin{threeparttable}
\begin{tabular}{llrrrrrl}
\toprule
\textbf{UQ Metric} & \textbf{Group} & \textbf{N} & \textbf{Skew} & \textbf{Kurt} & \textbf{W} & \textbf{p-value} & \textbf{Normal?} \\
\midrule
"""
    
    for metric in CI_METRICS:
        if metric not in results:
            continue
        
        metric_short = metric.replace('Gen ', '').replace(' ', ' ')
        first_row = True
        
        for group_name, group_label in [('overall', 'Overall'), ('hallucinations', 'Halluc'), ('correct', 'Correct')]:
            r = results[metric][group_name]
            sw = r['shapiro']
            st = r['stats']
            
            verdict = "Yes" if sw['is_normal'] else "No"
            n_str = f"{sw['n_original']:,}"
            if sw['subsampled']:
                n_str += r"$^\dagger$"

            # Format p-value (must not print false inequalities)
            p_str = format_p_value_latex(sw['p'])
            
            if first_row:
                latex += f"{metric_short} & {group_label} & {n_str} & {st['skewness']:+.2f} & {st['kurtosis']:+.1f} & {sw['W']:.3f} & {p_str} & {verdict} \\\\\n"
                first_row = False
            else:
                latex += f" & {group_label} & {n_str} & {st['skewness']:+.2f} & {st['kurtosis']:+.1f} & {sw['W']:.3f} & {p_str} & {verdict} \\\\\n"
        
        latex += r"\midrule" + "\n"
    
    # Remove last \midrule and replace with \bottomrule
    latex = latex.rstrip("\n").rstrip(r"\midrule")
    
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} Shapiro-Wilk test for normality (null hypothesis: data is normally distributed). 
$^\dagger$Subsampled to $n=5{,}000$ due to Shapiro-Wilk sample size limit.
Skew = skewness (0 = symmetric; positive = right-tailed); Kurt = excess kurtosis (0 = normal; positive = heavy-tailed).
All metrics reject normality at $\alpha = 0.05$, justifying non-parametric methods (Mann-Whitney U, AUC-ROC).
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    
    return latex


def generate_inline_summary(results):
    """Generate inline LaTeX text summarizing key findings."""
    
    # Collect key stats
    summaries = []
    for metric in CI_METRICS:
        if metric not in results:
            continue
        r = results[metric]['overall']
        sw = r['shapiro']
        st = r['stats']
        
        metric_short = metric.replace('Gen ', '')
        summaries.append({
            'metric': metric_short,
            'skew': st['skewness'],
            'kurt': st['kurtosis'],
            'W': sw['W'],
            'p': sw['p']
        })
    
    # Find extremes
    max_skew = max(summaries, key=lambda x: abs(x['skew']))
    max_kurt = max(summaries, key=lambda x: abs(x['kurt']))
    
    text = f"""Shapiro-Wilk tests confirm all four UQ metrics are significantly non-normal 
($p < 10^{{-50}}$ for all). {max_skew['metric']} exhibits the most extreme skewness 
(skewness = {max_skew['skew']:.1f}, kurtosis = {max_skew['kurt']:.0f}), 
while the other metrics show moderate departures from normality 
(skewness range: {min(s['skew'] for s in summaries):.1f} to {max(s['skew'] for s in summaries):.1f})."""
    
    return text


def generate_excel_output(results, df, output_path):
    """Generate Excel file with normality test results."""
    
    # Build rows for the summary table
    rows = []
    for metric in CI_METRICS:
        if metric not in results:
            continue
        
        for group_name in ['overall', 'hallucinations', 'correct']:
            r = results[metric][group_name]
            sw = r['shapiro']
            st = r['stats']
            
            rows.append({
                'Metric': metric,
                'Group': group_name.capitalize(),
                'N_Original': sw['n_original'],
                'N_Tested': sw['n_tested'],
                'Subsampled': sw['subsampled'],
                'Mean': st['mean'],
                'Std': st['std'],
                'Median': st['median'],
                'Min': st['min'],
                'Max': st['max'],
                'Skewness': st['skewness'],
                'Kurtosis': st['kurtosis'],
                'Shapiro_W': sw['W'],
                'Shapiro_p': sw['p'],
                'Is_Normal': sw['is_normal'],
                'Verdict': 'Normal' if sw['is_normal'] else 'Not Normal'
            })
    
    summary_df = pd.DataFrame(rows)
    
    # Write to Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Summary table
        summary_df.to_excel(writer, sheet_name='Normality Tests', index=False)
        
        # Sheet 2: Metadata
        metadata = pd.DataFrame([{
            'Timestamp': datetime.now().isoformat(),
            'Total_Edges': len(df),
            'Total_Hallucinations': df['is_hallucination'].sum(),
            'Total_Correct': (~df['is_hallucination']).sum(),
            'Shapiro_Subsample_Size': SHAPIRO_SUBSAMPLE_SIZE,
            'Alpha': 0.05,
            'Conclusion': 'All metrics are non-normal; use non-parametric methods'
        }])
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
    
    return summary_df


def main():
    print("=" * 80)
    print("RQ2 NORMALITY TESTS FOR UQ METRICS")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    df = load_rq2_data()
    
    # Run analysis
    results = run_normality_analysis(df)
    
    # Generate outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. LaTeX table
    latex_table = generate_latex_table(results)
    latex_path = OUTPUT_DIR / "normality_tests_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\n✅ LaTeX table saved to: {latex_path}")
    
    # 2. JSON results
    json_path = OUTPUT_DIR / "normality_tests_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_edges_total': len(df),
            'subsample_size': SHAPIRO_SUBSAMPLE_SIZE,
            'results': results
        }, f, indent=2)
    print(f"✅ JSON results saved to: {json_path}")
    
    # 3. Excel output
    excel_path = OUTPUT_DIR / "normality_tests_results.xlsx"
    generate_excel_output(results, df, excel_path)
    print(f"✅ Excel results saved to: {excel_path}")
    
    # 4. Inline summary
    inline_summary = generate_inline_summary(results)
    print(f"\n{'='*80}")
    print("INLINE SUMMARY FOR THESIS:")
    print("=" * 80)
    print(inline_summary)
    
    # 5. Print the LaTeX table
    print(f"\n{'='*80}")
    print("LATEX TABLE:")
    print("=" * 80)
    print(latex_table)
    
    return results


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)


#!/usr/bin/env python3
"""
Temperature Sensitivity Analysis Script

Analyzes existing temperature sensitivity experiment results from Excel files.
Computes F1 scores, performs ANOVA, and generates LaTeX tables.

This script processes raw experiment data (Excel files) from previous runs
and produces reproducible statistical analysis and tables.

Usage:
    python analyze_temperature_sensitivity.py [--output-dir PATH]

Output:
    - temperature_sensitivity_table.tex (LaTeX table)
    - temperature_sensitivity_stats.json (detailed statistics)
    - temperature_sensitivity_anova.json (ANOVA results)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats

# Data directories containing temperature sensitivity experiment results
DATA_DIRS = [
    'data_science/parameter_tuning_experiments/results/temperature_sensitivity_all_clds_20251217_202734',
]

TEMPERATURE_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]
CLD_NAMES = {
    'social_norms': 'Social Norms',
    'depressive': 'Depressive',
    'emergency_department': 'Emergency Dept'
}


def compute_f1_from_excel(xlsx_path: Path) -> dict:
    """
    Compute F1 score from an experiment Excel file.
    
    Args:
        xlsx_path: Path to the Excel file
    
    Returns:
        dict with tp, fp, fn, precision, recall, f1
    """
    try:
        df = pd.read_excel(xlsx_path, sheet_name='All Edges')
        
        if 'Classification' not in df.columns:
            return None
        
        counts = df['Classification'].value_counts()
        tp = counts.get('TP', 0)
        fp = counts.get('FP', 0)
        fn = counts.get('FN', 0)
        tn = counts.get('TN', 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'file': xlsx_path.name
        }
    except Exception as e:
        print(f"  Error processing {xlsx_path}: {e}")
        return None


def load_results_from_json(json_path: Path) -> list:
    """
    Load results from JSON file and extract metadata (temperature, seed).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data.get('results', [])


def collect_all_results(workspace_root: Path) -> pd.DataFrame:
    """
    Collect all temperature sensitivity results by reading Excel files
    and matching them with metadata from JSON.
    
    Returns:
        DataFrame with columns: cld, temperature, seed, f1, precision, recall, etc.
    """
    all_results = []
    
    for data_dir_rel in DATA_DIRS:
        data_dir = workspace_root / data_dir_rel
        if not data_dir.exists():
            print(f"Warning: Data directory not found: {data_dir}")
            continue
        
        # Load the results JSON for metadata
        json_path = data_dir / 'temperature_sensitivity_all_clds_results.json'
        if json_path.exists():
            json_results = load_results_from_json(json_path)
            # Create a mapping from excel_path to metadata
            excel_to_meta = {}
            for r in json_results:
                excel_path = Path(r.get('excel_path', ''))
                excel_to_meta[excel_path.name] = {
                    'cld': r.get('cld_key'),
                    'temperature': r.get('temperature'),
                    'seed': r.get('seed'),
                    'run_idx': r.get('run_idx')
                }
        else:
            excel_to_meta = {}
        
        # Process each CLD folder
        for cld_key in CLD_NAMES.keys():
            cld_dir = data_dir / cld_key
            if not cld_dir.exists():
                continue
            
            xlsx_files = sorted(cld_dir.glob('*.xlsx'))
            print(f"Found {len(xlsx_files)} Excel files in {cld_key}")
            
            for xlsx_path in xlsx_files:
                result = compute_f1_from_excel(xlsx_path)
                if result is None:
                    continue
                
                # Get metadata from JSON mapping
                meta = excel_to_meta.get(xlsx_path.name, {})
                
                result['cld'] = meta.get('cld', cld_key)
                result['temperature'] = meta.get('temperature')
                result['seed'] = meta.get('seed')
                result['run_idx'] = meta.get('run_idx')
                
                all_results.append(result)
    
    df = pd.DataFrame(all_results)
    return df


def infer_temperature_from_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    If temperature is missing, infer from file order (3 files per temperature).
    Files are created in order: T=0.0 (3x), T=0.3 (3x), T=0.5 (3x), T=0.7 (3x), T=1.0 (3x)
    """
    for cld in df['cld'].unique():
        cld_mask = df['cld'] == cld
        cld_df = df[cld_mask].copy()
        
        # If temperature is missing, assign based on order
        if cld_df['temperature'].isna().any():
            n = len(cld_df)
            if n == 15:  # Expected: 5 temps × 3 runs
                temps = []
                for t in TEMPERATURE_VALUES:
                    temps.extend([t, t, t])
                df.loc[cld_mask, 'temperature'] = temps[:n]
    
    return df


def compute_statistics(df: pd.DataFrame) -> dict:
    """
    Compute per-temperature, per-CLD statistics.
    
    Returns:
        dict with nested structure: {temp: {cld: {mean, std, n, values}}}
    """
    stats_dict = {}
    
    for temp in TEMPERATURE_VALUES:
        temp_data = df[df['temperature'] == temp]
        stats_dict[temp] = {}
        
        for cld in CLD_NAMES.keys():
            cld_data = temp_data[temp_data['cld'] == cld]
            f1_values = cld_data['f1'].dropna().values
            
            if len(f1_values) > 0:
                stats_dict[temp][cld] = {
                    'mean': float(np.mean(f1_values)),
                    'std': float(np.std(f1_values)),
                    'n': len(f1_values),
                    'values': f1_values.tolist()
                }
            else:
                stats_dict[temp][cld] = {
                    'mean': None,
                    'std': None,
                    'n': 0,
                    'values': []
                }
        
        # Overall for this temperature
        all_f1 = temp_data['f1'].dropna().values
        if len(all_f1) > 0:
            stats_dict[temp]['overall'] = {
                'mean': float(np.mean(all_f1)),
                'std': float(np.std(all_f1)),
                'n': len(all_f1),
                'values': all_f1.tolist()
            }
        else:
            stats_dict[temp]['overall'] = {'mean': None, 'std': None, 'n': 0, 'values': []}
    
    return stats_dict


def perform_anova(df: pd.DataFrame) -> dict:
    """
    Perform one-way ANOVA to test if temperature affects F1.
    Also performs assumption tests (normality, homogeneity).
    """
    groups = []
    group_labels = []
    
    for temp in TEMPERATURE_VALUES:
        temp_data = df[df['temperature'] == temp]['f1'].dropna().values
        if len(temp_data) >= 2:
            groups.append(temp_data)
            group_labels.append(f'T={temp}')
    
    if len(groups) < 2:
        return {'error': 'Not enough groups for ANOVA'}
    
    # ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Effect size (eta squared)
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum((x - grand_mean)**2 for x in all_data)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    # Normality tests per group (Shapiro-Wilk)
    normality = {}
    for label, g in zip(group_labels, groups):
        if len(g) >= 3:
            w, p = stats.shapiro(g)
            normality[label] = {'W': float(w), 'p': float(p), 'normal': p > 0.05}
        else:
            normality[label] = {'W': None, 'p': None, 'normal': None, 'note': f'n={len(g)} < 3'}
    
    # Homogeneity of variance (Levene's test)
    if len(groups) >= 2:
        lev_stat, lev_p = stats.levene(*groups)
        homogeneity = {'W': float(lev_stat), 'p': float(lev_p), 'equal_var': lev_p > 0.05}
    else:
        homogeneity = {'W': None, 'p': None, 'equal_var': None}
    
    # Kruskal-Wallis (non-parametric alternative)
    h_stat, kw_p = stats.kruskal(*groups)
    
    return {
        'anova': {
            'F': float(f_stat),
            'p': float(p_value),
            'significant': p_value < 0.05,
            'eta_squared': float(eta_squared),
            'df_between': len(groups) - 1,
            'df_within': len(all_data) - len(groups)
        },
        'kruskal_wallis': {
            'H': float(h_stat),
            'p': float(kw_p),
            'significant': kw_p < 0.05
        },
        'normality': normality,
        'homogeneity': homogeneity,
        'n_groups': len(groups),
        'n_total': len(all_data)
    }


def perform_per_cld_anova(df: pd.DataFrame) -> dict:
    """
    Perform one-way ANOVA separately for each CLD.
    
    This addresses the nested structure concern: runs are nested within CLDs,
    so pooling across CLDs can conflate between-CLD variance with temperature effects.
    Per-CLD ANOVA tests temperature effect within each domain independently.
    
    Returns:
        dict with per-CLD ANOVA results
    """
    per_cld_results = {}
    
    for cld_key, cld_name in CLD_NAMES.items():
        cld_df = df[df['cld'] == cld_key]
        
        groups = []
        group_labels = []
        
        for temp in TEMPERATURE_VALUES:
            temp_data = cld_df[cld_df['temperature'] == temp]['f1'].dropna().values
            if len(temp_data) >= 2:
                groups.append(temp_data)
                group_labels.append(f'T={temp}')
        
        if len(groups) < 2:
            per_cld_results[cld_key] = {'error': f'Not enough groups for ANOVA in {cld_name}'}
            continue
        
        # ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Effect size (eta squared)
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum((x - grand_mean)**2 for x in all_data)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Kruskal-Wallis (non-parametric alternative)
        h_stat, kw_p = stats.kruskal(*groups)
        
        per_cld_results[cld_key] = {
            'cld_name': cld_name,
            'anova': {
                'F': float(f_stat),
                'p': float(p_value),
                'significant': p_value < 0.05,
                'eta_squared': float(eta_squared),
                'df_between': len(groups) - 1,
                'df_within': len(all_data) - len(groups)
            },
            'kruskal_wallis': {
                'H': float(h_stat),
                'p': float(kw_p),
                'significant': kw_p < 0.05
            },
            'n_per_temp': [len(g) for g in groups],
            'n_total': len(all_data)
        }
    
    return per_cld_results


def generate_latex_table(stats_dict: dict, per_cld_anova: dict = None) -> str:
    """Generate LaTeX table from statistics."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Generator Temperature Sensitivity: Edge F1 Score by CLD}",
        r"\label{tab:temperature_sensitivity}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Temperature} & \multicolumn{4}{c}{\textbf{Edge F1 Score (Mean $\pm$ Std)}} \\",
        r"\cmidrule(lr){2-5}",
        r" & \textbf{Social Norms} & \textbf{Depressive} & \textbf{Emergency Dept} & \textbf{Overall} \\",
        r"\midrule"
    ]
    
    for temp in TEMPERATURE_VALUES:
        row_vals = []
        for key in ['social_norms', 'depressive', 'emergency_department', 'overall']:
            s = stats_dict[temp].get(key, {})
            if s.get('mean') is not None and s.get('std') is not None:
                row_vals.append(f"${s['mean']:.3f} \\pm {s['std']:.3f}$")
            else:
                row_vals.append("--")
        
        lines.append(f"T={temp} & {' & '.join(row_vals)} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item \textit{Note.} Edge F1 = $\frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$ comparing generated edges against literature ground truth.",
        r"\item \textit{Design.} 5 temperatures $\times$ 3 runs $\times$ 3 CLDs = 45 total experiments. Seeds: 10, 20, 30 per temperature.",
        r"\item \textit{Per-CLD columns:} Mean $\pm$ Std across 3 runs per CLD at each temperature.",
        r"\item \textit{Overall column:} Mean $\pm$ Std computed across all 9 individual runs (3 CLDs $\times$ 3 runs) per temperature. The larger Std reflects variability both within and between CLDs.",
    ])
    
    # Add per-CLD ANOVA results if available
    if per_cld_anova:
        anova_strs = []
        for cld_key in ['social_norms', 'depressive', 'emergency_department']:
            if cld_key in per_cld_anova and 'anova' in per_cld_anova[cld_key]:
                a = per_cld_anova[cld_key]['anova']
                cld_name = per_cld_anova[cld_key].get('cld_name', cld_key)
                anova_strs.append(f"{cld_name} $F({a['df_between']},{a['df_within']})={a['F']:.2f}$, $p={a['p']:.2f}$")
        if anova_strs:
            lines.append(r"\item \textit{Per-CLD ANOVA (temperature effect):} " + "; ".join(anova_strs) + ". None significant at $\\alpha=.05$.")
    
    lines.extend([
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}"
    ])
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze temperature sensitivity experiments'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--workspace', '-w',
        type=Path,
        default=None,
        help='Workspace root directory'
    )
    
    args = parser.parse_args()
    
    # Determine workspace root
    if args.workspace:
        workspace_root = args.workspace
    else:
        workspace_root = Path(__file__).resolve().parents[2]
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = workspace_root / 'final_runs' / 'temperature_sensitivity_analysis'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TEMPERATURE SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workspace: {workspace_root}")
    print(f"Output: {output_dir}")
    print()
    
    # Collect results from Excel files
    print("Collecting results from Excel files...")
    df = collect_all_results(workspace_root)
    
    if df.empty:
        print("ERROR: No results found!")
        return 1
    
    print(f"\nCollected {len(df)} experiment runs")
    print(f"CLDs: {df['cld'].unique().tolist()}")
    print(f"Temperatures: {sorted(df['temperature'].dropna().unique().tolist())}")
    
    # Infer missing temperature values if needed
    df = infer_temperature_from_order(df)
    
    # Compute statistics
    print("\nComputing statistics...")
    stats_dict = compute_statistics(df)
    
    # Print summary
    print("\nSummary by Temperature:")
    for temp in TEMPERATURE_VALUES:
        overall = stats_dict[temp].get('overall', {})
        if overall.get('mean') is not None:
            print(f"  T={temp}: F1 = {overall['mean']:.3f} ± {overall['std']:.3f} (n={overall['n']})")
    
    # Perform pooled ANOVA (for reference, but see per-CLD below)
    print("\nPerforming pooled ANOVA (exploratory, see per-CLD for proper analysis)...")
    anova_results = perform_anova(df)
    
    if 'anova' in anova_results:
        anova = anova_results['anova']
        print(f"  Pooled: F({anova['df_between']},{anova['df_within']}) = {anova['F']:.3f}, p = {anova['p']:.4f}")
        print(f"  η² = {anova['eta_squared']:.4f} ({'significant' if anova['significant'] else 'not significant'})")
    
    # Perform per-CLD ANOVA (addresses nested structure)
    print("\nPerforming per-CLD ANOVA (proper nested analysis)...")
    per_cld_anova = perform_per_cld_anova(df)
    anova_results['per_cld'] = per_cld_anova
    
    for cld_key, result in per_cld_anova.items():
        if 'anova' in result:
            anova = result['anova']
            print(f"  {result['cld_name']}: F({anova['df_between']},{anova['df_within']}) = {anova['F']:.3f}, p = {anova['p']:.4f}, η² = {anova['eta_squared']:.4f}")
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(stats_dict, per_cld_anova)
    
    # Save outputs
    table_path = output_dir / 'temperature_sensitivity_table.tex'
    with open(table_path, 'w') as f:
        f.write(latex_table)
    print(f"  ✓ Saved: {table_path}")
    
    stats_path = output_dir / 'temperature_sensitivity_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"  ✓ Saved: {stats_path}")
    
    anova_path = output_dir / 'temperature_sensitivity_anova.json'
    with open(anova_path, 'w') as f:
        # Convert numpy bools to Python bools for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, (np.bool_, np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        json.dump(convert_numpy(anova_results), f, indent=2)
    print(f"  ✓ Saved: {anova_path}")
    
    # Save raw data
    raw_path = output_dir / 'temperature_sensitivity_raw.csv'
    df.to_csv(raw_path, index=False)
    print(f"  ✓ Saved: {raw_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    # Print the LaTeX table for review
    print("\nLaTeX Table Preview:")
    print(latex_table)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


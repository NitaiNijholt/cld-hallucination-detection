#!/usr/bin/env python3
"""
Generate Human Validation Distribution Tables for RQ3

This script generates LaTeX tables showing:
1. Evidence verdict distribution by edge classification (TP/FP/FN)
2. Applicability score distribution by edge classification

These tables are for the human-validated sample (n=50) of DR edges.

Author: Generated for RQ3 Deep Research validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

VALIDATION_FILE = Path(__file__).parent.parent / "validation" / "deep_research_edge_validation_sample_20251226_055509_nitai_validated_cleaned.xlsx"
OUTPUT_DIR = Path(__file__).parent.parent / "validation"
THESIS_GENERATED_DIR = Path(__file__).parent.parent.parent.parent / "thesis" / "reproducible_version" / "generated"

# Applicability score bins
BINS = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
BIN_LABELS = ['[0.0, 0.5)', '[0.5, 0.6)', '[0.6, 0.7)', '[0.7, 0.8)', '[0.8, 0.9)', '[0.9, 1.0]']

# =============================================================================
# DATA LOADING
# =============================================================================

def load_validation_data(filepath: Path) -> pd.DataFrame:
    """Load and clean the validation data."""
    df = pd.read_excel(filepath, engine='openpyxl')
    
    # Clean human_verdict (strip whitespace)
    df['human_verdict'] = df['human_verdict'].astype(str).str.strip()
    
    # Standardize verdicts
    def standardize_verdict(v):
        v = str(v).strip().lower()
        if v == 'causal':
            return 'causal'
        elif v in ['different vars, causal', 'different vars causal']:
            return 'different_vars_causal'
        else:
            return 'invalid'
    
    df['verdict_std'] = df['human_verdict'].apply(standardize_verdict)
    df['is_causal'] = df['verdict_std'] == 'causal'
    df['is_any_causal'] = df['verdict_std'].isin(['causal', 'different_vars_causal'])
    
    return df


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_verdict_distribution(df: pd.DataFrame) -> dict:
    """Generate verdict distribution statistics."""
    results = {'by_class': {}, 'totals': {}}
    
    for cls in ['TP', 'FP', 'FN']:
        subset = df[df['classification'] == cls]
        n = len(subset)
        
        causal = (subset['verdict_std'] == 'causal').sum()
        diff_vars = (subset['verdict_std'] == 'different_vars_causal').sum()
        invalid = (subset['verdict_std'] == 'invalid').sum()
        any_causal = causal + diff_vars
        
        results['by_class'][cls] = {
            'n': n,
            'causal': causal,
            'causal_pct': causal / n * 100 if n > 0 else 0,
            'different_vars_causal': diff_vars,
            'different_vars_causal_pct': diff_vars / n * 100 if n > 0 else 0,
            'invalid': invalid,
            'invalid_pct': invalid / n * 100 if n > 0 else 0,
            'any_causal': any_causal,
            'any_causal_pct': any_causal / n * 100 if n > 0 else 0,
        }
    
    # Totals
    n_total = len(df)
    results['totals'] = {
        'n': n_total,
        'causal': (df['verdict_std'] == 'causal').sum(),
        'different_vars_causal': (df['verdict_std'] == 'different_vars_causal').sum(),
        'invalid': (df['verdict_std'] == 'invalid').sum(),
        'any_causal': df['is_any_causal'].sum(),
    }
    for k in ['causal', 'different_vars_causal', 'invalid', 'any_causal']:
        results['totals'][f'{k}_pct'] = results['totals'][k] / n_total * 100
    
    return results


def generate_applicability_distribution(df: pd.DataFrame) -> dict:
    """Generate applicability score distribution statistics."""
    df['app_bucket'] = pd.cut(df['Fully applicable'], bins=BINS, labels=BIN_LABELS, right=False)
    
    results = {'by_class': {}, 'totals': {}}
    
    for cls in ['TP', 'FP', 'FN']:
        subset = df[df['classification'] == cls]
        n = len(subset)
        
        bucket_counts = subset['app_bucket'].value_counts()
        results['by_class'][cls] = {
            'n': n,
            'mean': subset['Fully applicable'].mean(),
            'std': subset['Fully applicable'].std(),
            'buckets': {label: int(bucket_counts.get(label, 0)) for label in BIN_LABELS}
        }
    
    # Totals
    n_total = len(df)
    bucket_counts = df['app_bucket'].value_counts()
    results['totals'] = {
        'n': n_total,
        'mean': df['Fully applicable'].mean(),
        'std': df['Fully applicable'].std(),
        'buckets': {label: int(bucket_counts.get(label, 0)) for label in BIN_LABELS}
    }
    
    return results


def write_verdict_latex_table(results: dict, output_path: Path):
    """Write verdict distribution LaTeX table."""
    lines = [
        "% Auto-generated by generate_validation_distribution_tables.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Human Validation: Evidence Verdict Distribution by Edge Classification ($n=50$)}",
        "\\label{tab:rq3_verdict_distribution}",
        "\\begin{threeparttable}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Verdict} & \\textbf{TP} ($n=9$) & \\textbf{FP} ($n=32$) & \\textbf{FN} ($n=9$) & \\textbf{Total} \\\\",
        "\\midrule",
    ]
    
    # Causal row
    tp = results['by_class']['TP']
    fp = results['by_class']['FP']
    fn = results['by_class']['FN']
    tot = results['totals']
    
    lines.append(f"\\texttt{{causal}} & {tp['causal']} ({tp['causal_pct']:.1f}\\%) & {fp['causal']} ({fp['causal_pct']:.1f}\\%) & {fn['causal']} ({fn['causal_pct']:.1f}\\%) & {tot['causal']} ({tot['causal_pct']:.1f}\\%) \\\\")
    lines.append(f"\\texttt{{different\\_vars, causal}} & {tp['different_vars_causal']} ({tp['different_vars_causal_pct']:.1f}\\%) & {fp['different_vars_causal']} ({fp['different_vars_causal_pct']:.1f}\\%) & {fn['different_vars_causal']} ({fn['different_vars_causal_pct']:.1f}\\%) & {tot['different_vars_causal']} ({tot['different_vars_causal_pct']:.1f}\\%) \\\\")
    lines.append(f"\\texttt{{Not relevant / non-causal}} & {tp['invalid']} ({tp['invalid_pct']:.1f}\\%) & {fp['invalid']} ({fp['invalid_pct']:.1f}\\%) & {fn['invalid']} ({fn['invalid_pct']:.1f}\\%) & {tot['invalid']} ({tot['invalid_pct']:.1f}\\%) \\\\")
    lines.append("\\midrule")
    lines.append(f"\\textbf{{Any causal}} & {tp['any_causal']} ({tp['any_causal_pct']:.1f}\\%) & {fp['any_causal']} ({fp['any_causal_pct']:.1f}\\%) & {fn['any_causal']} ({fn['any_causal_pct']:.1f}\\%) & {tot['any_causal']} ({tot['any_causal_pct']:.1f}\\%) \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item \\textit{Note.} \\texttt{causal} = exact match to edge claim; \\texttt{different\\_vars, causal} = causal evidence for related but not identical variables/constructs; \\texttt{Not relevant / non-causal} = evidence does not support a causal relationship. ``Any causal'' = \\texttt{causal} $\\cup$ \\texttt{different\\_vars, causal}.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
        "",
    ])
    
    output_path.write_text("\n".join(lines))
    print(f"Saved: {output_path}")


def write_applicability_latex_table(results: dict, output_path: Path):
    """Write applicability distribution LaTeX table."""
    lines = [
        "% Auto-generated by generate_validation_distribution_tables.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{RQ3 Human Validation: Evidence Applicability Score Distribution by Edge Classification ($n=50$)}",
        "\\label{tab:rq3_applicability_distribution}",
        "\\begin{threeparttable}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Applicability Range} & \\textbf{TP} ($n=9$) & \\textbf{FP} ($n=32$) & \\textbf{FN} ($n=9$) & \\textbf{Total} \\\\",
        "\\midrule",
    ]
    
    tp = results['by_class']['TP']
    fp = results['by_class']['FP']
    fn = results['by_class']['FN']
    tot = results['totals']
    
    for label in BIN_LABELS:
        tp_count = tp['buckets'][label]
        fp_count = fp['buckets'][label]
        fn_count = fn['buckets'][label]
        tot_count = tot['buckets'][label]
        tot_pct = tot_count / tot['n'] * 100
        lines.append(f"${label}$ & {tp_count} & {fp_count} & {fn_count} & {tot_count} ({tot_pct:.1f}\\%) \\\\")
    
    lines.append("\\midrule")
    lines.append(f"\\textbf{{Mean}} $\\pm$ \\textbf{{SD}} & ${tp['mean']:.2f} \\pm {tp['std']:.2f}$ & ${fp['mean']:.2f} \\pm {fp['std']:.2f}$ & ${fn['mean']:.2f} \\pm {fn['std']:.2f}$ & ${tot['mean']:.2f} \\pm {tot['std']:.2f}$ \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item \\textit{Note.} Applicability score (\\texttt{Fully applicable}) measures how well retrieved evidence matches the specific (source, target, mechanism) claim. Higher scores indicate more applicable evidence. TP edges have the highest mean applicability and lowest variance, consistent with literature-grounded relationships.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
        "",
    ])
    
    output_path.write_text("\n".join(lines))
    print(f"Saved: {output_path}")


def print_summary(verdict_results: dict, app_results: dict):
    """Print summary to console."""
    print("\n" + "=" * 80)
    print("HUMAN VALIDATION DISTRIBUTION SUMMARY")
    print("=" * 80)
    
    print("\n--- VERDICT DISTRIBUTION ---")
    print(f"{'Verdict':<30} {'TP':>10} {'FP':>10} {'FN':>10} {'Total':>10}")
    print("-" * 70)
    
    for verdict in ['causal', 'different_vars_causal', 'invalid']:
        tp = verdict_results['by_class']['TP']
        fp = verdict_results['by_class']['FP']
        fn = verdict_results['by_class']['FN']
        tot = verdict_results['totals']
        print(f"{verdict:<30} {tp[verdict]:>10} {fp[verdict]:>10} {fn[verdict]:>10} {tot[verdict]:>10}")
    
    print("-" * 70)
    print(f"{'Any causal':<30} {tp['any_causal']:>10} {fp['any_causal']:>10} {fn['any_causal']:>10} {tot['any_causal']:>10}")
    
    print("\n--- APPLICABILITY DISTRIBUTION ---")
    print(f"{'Range':<15} {'TP':>8} {'FP':>8} {'FN':>8} {'Total':>8}")
    print("-" * 50)
    
    for label in BIN_LABELS:
        tp = app_results['by_class']['TP']['buckets'][label]
        fp = app_results['by_class']['FP']['buckets'][label]
        fn = app_results['by_class']['FN']['buckets'][label]
        tot = app_results['totals']['buckets'][label]
        print(f"{label:<15} {tp:>8} {fp:>8} {fn:>8} {tot:>8}")
    
    print("-" * 50)
    tp = app_results['by_class']['TP']
    fp = app_results['by_class']['FP']
    fn = app_results['by_class']['FN']
    tot = app_results['totals']
    print(f"{'Mean ± SD':<15} {tp['mean']:.2f}±{tp['std']:.2f} {fp['mean']:.2f}±{fp['std']:.2f} {fn['mean']:.2f}±{fn['std']:.2f} {tot['mean']:.2f}±{tot['std']:.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("GENERATING HUMAN VALIDATION DISTRIBUTION TABLES")
    print("=" * 80)
    print(f"Validation file: {VALIDATION_FILE}")
    
    # Load data
    print("\nLoading validation data...")
    df = load_validation_data(VALIDATION_FILE)
    print(f"Loaded {len(df)} validated edges")
    
    # Generate distributions
    print("\nGenerating verdict distribution...")
    verdict_results = generate_verdict_distribution(df)
    
    print("Generating applicability distribution...")
    app_results = generate_applicability_distribution(df)
    
    # Print summary
    print_summary(verdict_results, app_results)
    
    # Write LaTeX tables
    print("\n" + "=" * 80)
    print("WRITING LATEX TABLES")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Write to validation folder
    write_verdict_latex_table(verdict_results, OUTPUT_DIR / f"verdict_distribution_table_{timestamp}.tex")
    write_applicability_latex_table(app_results, OUTPUT_DIR / f"applicability_distribution_table_{timestamp}.tex")
    
    # Write to thesis generated folder (without timestamp for stable includes)
    write_verdict_latex_table(verdict_results, THESIS_GENERATED_DIR / "rq3_verdict_distribution_table.tex")
    write_applicability_latex_table(app_results, THESIS_GENERATED_DIR / "rq3_applicability_distribution_table.tex")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()





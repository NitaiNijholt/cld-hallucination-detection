#!/usr/bin/env python3
"""
Generator Model Comparison Analysis Script
Compares GPT-4.1, Sonar-Pro, and Claude-Sonnet-4 for CLD generation.

Usage:
    python analyze_generator_models.py

Outputs:
    - generator_model_comparison.pdf (bar chart)
    - generator_model_comparison.png (for presentations)
    - generator_model_stats.xlsx (detailed stats)
    - generator_model_stats.csv (for further analysis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent / "data"

# Run directories for each model (data is stored in the analysis folder)
RUNS = {
    "GPT-4.1": DATA_DIR / "gpt-4.1",
    "Sonar-Pro": DATA_DIR / "sonar-pro",
    "Claude-Sonnet-4": DATA_DIR / "claude-sonnet-4"
}

CLD_NAMES = ["social_norms", "depressive", "emergency_dept"]
CLD_DISPLAY = {
    "social_norms": "Social Norms",
    "depressive": "Depressive",
    "emergency_dept": "Emergency Dept"
}

def collect_stats():
    """
    Collect statistics from all model runs by reading xlsx files.

    If multiple run workbooks exist for a given model×CLD, we:
    - compute per-run TP/FP/FN and derived metrics for each workbook
    - return a per-CLD *pooled-count* summary (TP/FP/FN/Edges summed across runs)

    Rationale: the thesis table reports integer TP/FP/FN and "micro-averaged (pooled TP/FP/FN)",
    so pooling counts across runs keeps the table interpretable and mathematically consistent.
    """
    per_run_results = []
    
    for model, base_path in RUNS.items():
        for cld in CLD_NAMES:
            cld_path = base_path / cld
            if not cld_path.exists():
                print(f"Warning: {cld_path} not found")
                continue
            
            # Find the result xlsx
            xlsx_files = sorted(list(cld_path.glob("*_result_*.xlsx")))
            if not xlsx_files:
                print(f"Warning: No xlsx in {cld_path}")
                continue

            for xlsx_file in xlsx_files:
                try:
                    df = pd.read_excel(xlsx_file, sheet_name='All Edges')

                    # Count by classification
                    tp = (df['Classification'] == 'TP').sum()
                    fp = (df['Classification'] == 'FP').sum()
                    fn = (df['Classification'] == 'FN').sum()

                    # Count by relationship type
                    rel_counts = df['Relationship Type'].value_counts().to_dict() if 'Relationship Type' in df.columns else {}

                    total = len(df)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    per_run_results.append({
                        'Model': model,
                        'CLD': cld,
                        'CLD_Display': CLD_DISPLAY[cld],
                        'Total_Edges': total,
                        'TP': tp,
                        'FP': fp,
                        'FN': fn,
                        'Precision': precision,
                        'Recall': recall,
                        'F1': f1,
                        'POSITIVE': rel_counts.get('POSITIVE', 0),
                        'NEGATIVE': rel_counts.get('NEGATIVE', 0),
                        'NONE': rel_counts.get('NONE', 0),
                        'COMPLEX': rel_counts.get('COMPLEX', 0),
                        'Source_File': str(xlsx_file),
                    })
                except Exception as e:
                    print(f"Error processing {xlsx_file}: {e}")

            print(f"  Loaded: {model} / {cld} - runs={len(xlsx_files)}")

    df_runs = pd.DataFrame(per_run_results)
    if len(df_runs) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Summarize to one row per model×CLD by pooling counts across runs
    grouped = df_runs.groupby(['Model', 'CLD', 'CLD_Display'], as_index=False)
    df_pooled = grouped.agg({
        'Total_Edges': 'sum',
        'TP': 'sum',
        'FP': 'sum',
        'FN': 'sum',
        'POSITIVE': 'sum',
        'NEGATIVE': 'sum',
        'NONE': 'sum',
        'COMPLEX': 'sum',
    })
    df_pooled['N_Runs'] = grouped.size()['size']

    # Derived metrics from pooled counts (micro-averaged within the CLD across runs)
    df_pooled['Precision'] = df_pooled.apply(
        lambda r: (r['TP'] / (r['TP'] + r['FP'])) if (r['TP'] + r['FP']) > 0 else 0.0, axis=1
    )
    df_pooled['Recall'] = df_pooled.apply(
        lambda r: (r['TP'] / (r['TP'] + r['FN'])) if (r['TP'] + r['FN']) > 0 else 0.0, axis=1
    )
    df_pooled['F1'] = df_pooled.apply(
        lambda r: (2 * r['Precision'] * r['Recall'] / (r['Precision'] + r['Recall']))
        if (r['Precision'] + r['Recall']) > 0 else 0.0,
        axis=1,
    )

    # Keep a compact provenance string (folder + n)
    df_pooled['Source_File'] = df_pooled.apply(
        lambda r: f"{RUNS[r['Model']] / r['CLD']} (n={int(r['N_Runs'])})",
        axis=1,
    )

    return df_pooled, df_runs

def compute_aggregates(df):
    """Compute aggregate stats per model."""
    agg_results = []
    
    for model in RUNS.keys():
        model_data = df[df['Model'] == model]
        if len(model_data) == 0:
            continue
            
        # Pool TP/FP/FN across all CLDs (and across runs if df is pooled-per-CLD)
        total_tp = int(model_data['TP'].sum())
        total_fp = int(model_data['FP'].sum())
        total_fn = int(model_data['FN'].sum())
        total_edges = int(model_data['Total_Edges'].sum())
        
        agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        agg_f1 = 2 * agg_precision * agg_recall / (agg_precision + agg_recall) if (agg_precision + agg_recall) > 0 else 0
        
        agg_results.append({
            'Model': model,
            'CLD': 'Total',
            'CLD_Display': 'Total',
            'Total_Edges': total_edges,
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Precision': agg_precision,
            'Recall': agg_recall,
            'F1': agg_f1
        })
    
    return pd.DataFrame(agg_results)

def create_comparison_chart(df, agg_df):
    """Create bar chart comparing models across CLDs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(RUNS.keys())
    clds = list(CLD_DISPLAY.values()) + ['Total']
    
    x = np.arange(len(clds))
    width = 0.25
    
    colors = {'GPT-4.1': '#4CAF50', 'Sonar-Pro': '#2196F3', 'Claude-Sonnet-4': '#FF9800'}
    
    for i, model in enumerate(models):
        f1_scores = []
        for cld in list(CLD_NAMES) + ['Total']:
            if cld == 'Total':
                row = agg_df[agg_df['Model'] == model]
            else:
                row = df[(df['Model'] == model) & (df['CLD'] == cld)]
            
            if len(row) > 0:
                f1_scores.append(row['F1'].values[0])
            else:
                f1_scores.append(0)
        
        bars = ax.bar(x + i * width, f1_scores, width, label=model, color=colors[model], alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, f1_scores):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('CLD Dataset', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Generator Model Comparison: F1 Score by CLD', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(clds)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 0.7)
    ax.grid(axis='y', alpha=0.3)
    
    # Add a vertical line before Total
    ax.axvline(x=2.5 + width, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figures
    fig.savefig(OUTPUT_DIR / 'generator_model_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'generator_model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'generator_model_comparison.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'generator_model_comparison.png'}")
    
    plt.close()

def create_precision_recall_chart(df, agg_df):
    """Create grouped bar chart for Precision and Recall."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = list(RUNS.keys())
    
    # Precision chart
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.6
    
    precisions = [agg_df[agg_df['Model'] == m]['Precision'].values[0] for m in models]
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    bars = ax.bar(x, precisions, width, color=colors, alpha=0.8)
    
    for bar, val in zip(bars, precisions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Aggregate Precision', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 0.25)
    ax.grid(axis='y', alpha=0.3)
    
    # Recall chart
    ax = axes[1]
    recalls = [agg_df[agg_df['Model'] == m]['Recall'].values[0] for m in models]
    bars = ax.bar(x, recalls, width, color=colors, alpha=0.8)
    
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Aggregate Recall', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / 'generator_precision_recall.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'generator_precision_recall.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'generator_precision_recall.pdf'}")
    
    plt.close()

def generate_latex_table(df, agg_df):
    """
    Generate LaTeX table with intelligent bolding:
    - Per CLD: Bold highest F1 among models for that CLD
    - Aggregate: Bold highest Precision, Recall, and F1 separately
    
    Returns the LaTeX table as a string and saves to file.
    """
    models = ["GPT-4.1", "Sonar-Pro", "Claude-Sonnet-4"]
    clds = ["social_norms", "depressive", "emergency_dept"]
    cld_display = {"social_norms": "Social Norms", "depressive": "Depressive", "emergency_dept": "Emergency Dept"}
    
    # Find best F1 per CLD
    best_f1_per_cld = {}
    for cld in clds:
        cld_data = df[df['CLD'] == cld]
        if len(cld_data) > 0:
            best_f1_per_cld[cld] = cld_data['F1'].max()
    
    # Find best metrics in aggregate
    best_agg_precision = agg_df['Precision'].max()
    best_agg_recall = agg_df['Recall'].max()
    best_agg_f1 = agg_df['F1'].max()
    
    def fmt_bold(val, is_best, decimals=3):
        """Format value, bold if best."""
        formatted = f"{val:.{decimals}f}"
        return f"\\textbf{{{formatted}}}" if is_best else formatted
    
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Generator Model Comparison: Edge Recovery Performance. Bold indicates highest F1 per CLD; in aggregate row, bold indicates highest Precision, Recall, and F1 separately. \textbf{Edges processed} refers to the number of ordered (source, target) edge slots evaluated (including \texttt{NONE}).}",
        r"\label{tab:generator_model_comparison}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{CLD} & \textbf{Edges processed} & \textbf{TP} & \textbf{FP} & \textbf{FN} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\",
        r"\midrule"
    ]
    
    # Per-model, per-CLD rows
    for i, model in enumerate(models):
        for cld in clds:
            row = df[(df['Model'] == model) & (df['CLD'] == cld)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            
            # Check if this is best F1 for this CLD
            is_best_f1 = abs(row['F1'] - best_f1_per_cld.get(cld, -1)) < 0.0001
            
            f1_str = fmt_bold(row['F1'], is_best_f1)
            
            lines.append(
                f"{model} & {cld_display[cld]} & {int(row['Total_Edges'])} & "
                f"{int(row['TP'])} & {int(row['FP'])} & {int(row['FN'])} & "
                f"{row['Precision']:.3f} & {row['Recall']:.3f} & {f1_str} \\\\"
            )
        
        # Add midrule after each model except the last
        if i < len(models) - 1:
            lines.append(r"\midrule")
    
    # Aggregate section
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{l}{\textit{Micro-averaged (pooled TP/FP/FN across all CLDs):}} \\")
    
    for model in models:
        row = agg_df[agg_df['Model'] == model]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        
        # Check if this model has best aggregate metrics
        is_best_precision = abs(row['Precision'] - best_agg_precision) < 0.0001
        is_best_recall = abs(row['Recall'] - best_agg_recall) < 0.0001
        is_best_f1 = abs(row['F1'] - best_agg_f1) < 0.0001
        
        precision_str = fmt_bold(row['Precision'], is_best_precision)
        recall_str = fmt_bold(row['Recall'], is_best_recall)
        f1_str = fmt_bold(row['F1'], is_best_f1)
        
        # Bold model name if it has best F1
        model_str = f"\\textbf{{{model}}}" if is_best_f1 else model
        
        lines.append(
            f"{model_str} & Total & {int(row['Total_Edges'])} & "
            f"{int(row['TP'])} & {int(row['FP'])} & {int(row['FN'])} & "
            f"{precision_str} & {recall_str} & {f1_str} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        # Note: runs are pooled as counts; this matches the micro-averaged definition used in the caption.
        rf"\item \textit{{Note.}} All models used identical prompts (Nitai\_C variant) and temperature (T=0.7). Results pool TP/FP/FN counts across {int(df['N_Runs'].min())} independent runs per CLD (same configuration; stochastic sampling). Nodes (variables) were directly retrieved from the \textbf{{GT lit CLDs}}. \textbf{{Edges processed}} denotes the number of ordered (source, target) edge slots evaluated (including \texttt{{NONE}}), i.e., ideally $|V|(|V|-1)$; it can be slightly lower if some rows are missing due to parsing/processing errors.",
        rf"\item \textbf{{Interpretation:}} {agg_df.loc[agg_df['F1'].idxmax(), 'Model']} achieves the best aggregate F1 ({agg_df['F1'].max():.3f}).",
        rf"\item {agg_df.loc[agg_df['Recall'].idxmax(), 'Model']} shows the highest aggregate recall ({agg_df['Recall'].max():.3f}), while {agg_df.loc[agg_df['Precision'].idxmax(), 'Model']} has the highest aggregate precision ({agg_df['Precision'].max():.3f}).",
        rf"\item Based on best micro-average F1 score, \textbf{{{agg_df.loc[agg_df['F1'].idxmax(), 'Model']} was selected}} as the generator model for all subsequent experiments. Additionally, GPT-4.1 is the only model that provides token-level log probabilities (logprobs), enabling confidence-based edge filtering which is necessary for RQ2 experiments.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}"
    ])
    
    latex_str = "\n".join(lines)
    
    # Save to file
    output_path = OUTPUT_DIR / 'generator_model_comparison.tex'
    with open(output_path, 'w') as f:
        f.write(latex_str)
    print(f"Saved: {output_path}")
    
    return latex_str


def main():
    print("=" * 60)
    print("Generator Model Comparison Analysis")
    print("=" * 60)
    
    # Collect stats
    df, df_runs = collect_stats()
    if len(df) == 0:
        print("\nNo data found. Did you run the generator model comparison experiments?")
        return
    print(f"\nCollected {len(df_runs)} per-run points; pooled to {len(df)} model×CLD rows")
    
    # Compute aggregates
    agg_df = compute_aggregates(df)
    
    # Combine for full stats
    full_df = pd.concat([df, agg_df], ignore_index=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Per-CLD Results:")
    print("=" * 60)
    cols = ['Model', 'CLD_Display', 'N_Runs', 'Total_Edges', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
    print(df[cols].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Aggregate Results:")
    print("=" * 60)
    print(agg_df[['Model', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']].to_string(index=False))
    
    # Create charts
    print("\n" + "=" * 60)
    print("Generating Charts...")
    print("=" * 60)
    create_comparison_chart(df, agg_df)
    create_precision_recall_chart(df, agg_df)
    
    # Save data
    full_df.to_excel(OUTPUT_DIR / 'generator_model_stats.xlsx', index=False)
    full_df.to_csv(OUTPUT_DIR / 'generator_model_stats.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'generator_model_stats.xlsx'}")
    print(f"Saved: {OUTPUT_DIR / 'generator_model_stats.csv'}")

    # Save per-run data separately (for robustness inspection)
    df_runs.to_excel(OUTPUT_DIR / 'generator_model_stats_runs.xlsx', index=False)
    df_runs.to_csv(OUTPUT_DIR / 'generator_model_stats_runs.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'generator_model_stats_runs.xlsx'}")
    
    # Save JSON for programmatic access
    results_json = {
        'per_cld_mean': df.to_dict(orient='records'),
        'per_run': df_runs.to_dict(orient='records'),
        'aggregate': agg_df.to_dict(orient='records'),
    }
    with open(OUTPUT_DIR / 'generator_model_stats.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'generator_model_stats.json'}")
    
    # Generate LaTeX table with proper bolding
    print("\n" + "=" * 60)
    print("Generating LaTeX Table...")
    print("=" * 60)
    latex_table = generate_latex_table(df, agg_df)
    print("\nLaTeX Table Preview:")
    print(latex_table)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()






